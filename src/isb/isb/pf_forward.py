"""Implements trainable Deterministic Particle Filtering (no terminal constraints)."""

from scipy.misc import electrocardiogram
import torch
import numpy as np
import math
from isb.isb import SDEModelFilter
from isb.isb.optimal_transport import transport_resample
from torch.utils.data import TensorDataset
from TrajectoryNet.optimal_transport.emd import earth_mover_distance


class ParticleFilter():
    """Filtering with discrete observation."""

    def __init__(self, config, sde_filter_model: SDEModelFilter, time_forward=True, device='cpu'):
        self.n_dim = config.dataset.n_dim
        self.time_diff = config.filter.time_diff
        self.n_steps = config.filter.n_steps
        self.config = config
        self.transport_eps = torch.Tensor([config.filter.transport_eps]).to(device)[0]

        if 'n_near' in self.config.filter:
            self.n_near = self.config.filter.n_near
        else:
            self.n_near = None

        time_max = self.n_steps*self.time_diff - self.time_diff
        self.time_stamps = torch.linspace(0, time_max, self.n_steps).to(device)   # fixed linear time, to be expanded later
        if not time_forward:
            self.time_stamps = torch.flip(self.time_stamps, dims=[0])
        self.time_forward = time_forward
        self.sde_filter_model = sde_filter_model
        self.model_type = self.sde_filter_model.model_type
        self.device = device
        print(f'Obs noise level: {self.sde_filter_model.obs_noise_level}')
    
    def ode_flow_step(self, t, t_next, x, dt=0.01, obs=None):
        """Single step in the ODE flow."""
        drift = self.sde_filter_model.eval_drift(t, x)
        score = self.sde_filter_model.score_fn(t, x)
        diffusion = self.sde_filter_model.eval_diffusion(t)
        diffusion_2 = diffusion**2

        new_y = x + dt*(drift - 0.5*diffusion_2*score)   # was 0.25
        diff_particles = self.sde_filter_model.eval_drift(t_next, new_y)
        dt_stacked = dt.unsqueeze(0).repeat(x.shape[0], *self.n_dim)

        if obs is None:
            return new_y, diff_particles, dt_stacked
        particles = self.reweight(t, new_y, obs)
        return particles, diff_particles, dt_stacked

    def print_metric(self,t, particles, samples, report_data):
        """Print metric between generated particles and true data."""
        if report_data is not None:
            comp_time = float(t.cpu().numpy())
            emd = earth_mover_distance(particles, report_data[comp_time])
            print(f'EMD at time {t}: {emd}')
        elif samples.shape[0] != 0:
            emd = earth_mover_distance(particles, samples)
            print(f'EMD at time {t}: {emd}')

    def reweight(self, t, particles, obs):
        """Reweights the uncontrolled particles."""
        if 'image' in self.model_type:
            particles_enc = particles
            obs_enc = obs
            n_sum = tuple([-3, -2, -1])
            if self.config.dataset.distance == 'cosine':
                log_weights = torch.nn.functional.cosine_similarity(particles_enc.unsqueeze(0), obs_enc.unsqueeze(1), dim=-1)
            elif self.config.dataset.distance == 'euclidean':
                particle_error = particles_enc.unsqueeze(0) - obs_enc.unsqueeze(1) 
                if self.config.dataset.masking:
                    if self.config.dataset.mask_type == 'data_mask':
                        masks = torch.gt(obs_enc, 0.1).unsqueeze(1).repeat(1, particles_enc.shape[0],  1, 1, 1).to(dtype=torch.float32)
                    elif self.config.dataset.mask_type == 'block_mask':
                        masks = torch.ones(obs.shape[0], particles.shape[0], *self.n_dim, device=self.device)
                        half_size = int(self.n_dim[1]/2)
                        masks[:, :, :, half_size:] = 0        # was :half_size
                    particle_error = torch.sum(masks*(particle_error**2), dim=n_sum)
                    print(particle_error.shape)
                else:
                    particle_error = torch.sum((particle_error)**2, dim=n_sum)
                obs_noise = self.sde_filter_model.eval_obs_noise(t)
                obs_noise_mult = -0.5/(obs_noise**2)
                log_weights = obs_noise_mult*particle_error
            log_weights_unnorm, _ = torch.max(log_weights, dim=0)
            log_weights = log_weights_unnorm - torch.logsumexp(log_weights_unnorm, dim=0)
            particles, ot_matrix = self.resample(particles, log_weights)
            return particles, ot_matrix, log_weights_unnorm, None
        else:
            particle_error = particles.unsqueeze(0) - obs.unsqueeze(1) 
            obs_noise = self.sde_filter_model.eval_obs_noise(t)
            obs_noise_mult = -0.5/(obs_noise**2)
            log_weights = obs_noise_mult*torch.sum((particle_error)**2, dim=-1)
            weights_sorted, _ = torch.sort(log_weights, dim=0, descending=True)
            if self.n_near is None:
                k_nearest = max(int(np.floor(np.sqrt(obs.shape[0]))), 1)
            else:
                k_nearest = self.n_near
            log_weights_unnorm = torch.log(torch.mean(torch.exp(weights_sorted[:k_nearest]), dim=0))    # TODO: fix this
          #  log_weights_unnorm, _ = torch.max(log_weights, dim=0) # take the weight to the closest observation point!
            log_weights = log_weights_unnorm - torch.logsumexp(log_weights_unnorm, dim=0)
            weights = torch.exp(log_weights)
            particles, ot_matrix = self.resample(particles, log_weights)
            return particles, ot_matrix, log_weights_unnorm

    def sde_flow_step(self, t, t_next, x, dt=0.01, obs=None, wght_report=False):
        """Stochastic step."""
        drift = self.sde_filter_model.eval_drift(t, x)
        diffusions = self.sde_filter_model.eval_diffusion(t)
        rand = torch.sqrt(dt)*diffusions*torch.randn(*x.shape, device=self.device)

        new_mean = x + dt*drift
        new_y = new_mean + rand
        next_mean = new_y + self.sde_filter_model.eval_drift(t, new_y)*dt

        diff_particles = new_mean - next_mean
        dt_stacked = dt.unsqueeze(0).repeat(x.shape[0], *self.n_dim)

        if obs is None:
            return new_y, diff_particles, dt_stacked, None
        
        if obs is not None:
            particles, ot_matrix, weights = self.reweight(t, new_y, obs)
            new_mean_flat = torch.flatten(new_mean, start_dim=-(len(self.n_dim)))
            new_mean = torch.bmm(ot_matrix, new_mean_flat.unsqueeze(0)).squeeze(0)
            new_mean = new_mean.reshape(-1, *self.n_dim)
            next_mean = particles + self.sde_filter_model.eval_drift(t, particles)*dt  # notice time
            diff_particles = new_mean - next_mean

            wght_sum = torch.mean(weights)
            if wght_report:
                return new_y, diff_particles, dt_stacked, wght_sum
            else:
                return particles, diff_particles, dt_stacked, None

    def resample(self, particles, log_weights):
        if self.model_type == 'flat':
            return self.ot_resample(particles, log_weights)
        elif 'image' in self.model_type:
            return self.ot_resample(particles, log_weights)

    def rng_resample(self, particles, log_weights):
        """Random sampling -based resampling."""
        dist = torch.distributions.Categorical(logits=log_weights)
        indices = dist.sample((particles.shape[0],))
        resampled_particles = particles[indices]
        return resampled_particles

    def ot_resample(self, particles, log_weights, stable=True):
        """Perform a resampling step.
        
        Uses entropy-regularized OT to transform weights and particles
        to uniformly-weighted particles.
        """
        if self.model_type == 'flat':
            ot_input = particles.unsqueeze(0)
        elif 'image' in self.model_type:
            ot_input = torch.flatten(particles, start_dim=-len(self.n_dim)).unsqueeze(0)
        transported_particles, _, transport_matrix = transport_resample(ot_input, log_weights.unsqueeze(0), eps=self.transport_eps, stable=stable)
        if self.model_type == 'flat':
            new_particles = transported_particles.squeeze(0)
        elif 'image' in self.model_type:
            particles_flat = torch.flatten(particles, start_dim=-len(self.n_dim))
            new_particles = torch.bmm(transport_matrix, particles_flat.unsqueeze(0))
            new_particles = new_particles.unsqueeze(0).reshape(particles.shape[0], *self.n_dim)
            print('Weight report:')
            print(f'log_weight median: {torch.median(log_weights)}')
            print(f'Log weight max: {torch.max(log_weights)}')
            print(f'Log weight min: {torch.min(log_weights)}')
         #   assert 1 == 0   # forcing an error for faster log writing...
        return new_particles, transport_matrix

    def check_time(self, t, obs_times, obs_idx, obs_samples):
        if obs_idx >= len(obs_times):
            obs_time = False
        else:
            comp_time = obs_times[obs_idx]
            obs_time = torch.isclose(t, comp_time)
        if obs_time:
            obs = obs_samples[:, obs_idx]
        else:
            obs = None
        return obs_time, obs
        
    def generate_particles(self, obs_times, obs_samples, n_particles, stochastic=False, report_weights=False, init_points=None, report_times=[], report_data=None):
        """Generate particles.
        
        Assumption: obs_times is a subset of self.time_stamps. Has option to use pre-determined random 
        samples, so that this step only performes reparametrization. Outputs particles and observation losses.

        Reparametrized version. If random numbers not given, they are generated.
        """
        if init_points is None:
            particles = self.sde_filter_model.generate_init_samples(n_samples=n_particles)
        else:
            particles = init_points
        particle_output = torch.empty((len(self.time_stamps), n_particles, *self.n_dim), device=self.device, dtype=particles.dtype)
        diff_output = torch.empty((len(self.time_stamps) - 1, n_particles, *self.n_dim),  device=self.device, dtype=particles.dtype)
        dt_output = torch.empty((len(self.time_stamps) - 1, n_particles, *self.n_dim),  device=self.device, dtype=particles.dtype)

        wght_list = []
        if len(obs_times) != 0: 
            self.time_stamps = self.time_stamps.to(obs_times.dtype)
        obs_idx = 0
        report_idx = 0
        particle_output[0] = particles

        for i, t in enumerate(self.time_stamps[:-1]):
            if self.time_forward:
                dt = self.time_stamps[i+1] - t
                t_next = t + dt
            else:
                dt =  t - self.time_stamps[i+1]
                t_next = t - dt
            obs_time, obs = self.check_time(t, obs_times, obs_idx, obs_samples)
            report_time, report_obs = self.check_time(t, report_times, report_idx, obs_samples)

            if not stochastic:
                particles, diff_particle, dt_stacked = self.ode_flow_step(t, t_next, particles, dt=dt, obs=obs)
            else:
                particles, diff_particle, dt_stacked, wght_sum = self.sde_flow_step(t, t_next, particles, 
                                                                dt=dt, obs=obs, wght_report=report_weights)
                
            if obs_time:
                obs_idx += 1
                if report_weights:
                    wght_list.append(wght_sum.cpu().numpy())
            
            if report_time:
                report_idx += 1
                self.print_metric(t, particles, report_obs, report_data)    # TODO: confirm indexing !

            diff_output[i] = diff_particle
            dt_output[i] = dt_stacked
            particle_output[i+1] = particles
        if report_weights:
            print(np.mean(wght_list))
        times_batched = self.time_stamps.unsqueeze(1).repeat(1, n_particles)
        return particle_output, diff_output, times_batched, dt_output

    def estimate_loss(self, t, x, true_diff, dts):
        """Estimate the loss using the current score."""
        score = self.sde_filter_model.eval_bwd_drift(t, x)
        dt_normalized = true_diff/dts
        loss = torch.mean(torch.sum((dt_normalized-score)**2, axis=-1))
        return loss

    def create_loader(self, obs_times, obs_samples, n_particles, batch_size):
        """Create particle trajectories, save results to loader."""
        with torch.no_grad():
            all_particles, true_diffs, times_batched, dts = self.generate_particles(
                    obs_times, obs_samples, n_particles, stochastic=True)
            times_batched = times_batched[1:] # only evaluation times
            all_particles = all_particles[1:]  # only at evaluation times
            all_particles = torch.flatten(all_particles, start_dim=0, end_dim=1).detach()
            dts = torch.flatten(dts, start_dim=0, end_dim=1)
            times_batched = torch.flatten(times_batched).detach()
            true_diffs = torch.flatten(true_diffs, start_dim=0, end_dim=1).detach()
            dataset = TensorDataset(all_particles, true_diffs, times_batched, dts)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            print('Re-created loader')
            return loader

    def generate_random_steps(self, n_steps, n_particles, n_samples, loader=True):
        """Generate the stochastic part of the dynamics.
        
        To be used later for reparametrized learning.
        """
        with torch.no_grad():
            particles = self.sde_filter_model.generate_init_samples(1)
            normal_mean = torch.zeros((n_steps, *self.n_dim), device=self.device, dtype=particles.dtype)
            if 'image' in self.sde_filter_model.model_type:
                cov_diag = torch.ones((n_steps, *self.n_dim), device=self.device, dtype=particles.dtype)
                cov_factor = torch.zeros(*self.n_dim, device=self.device, dtype=particles.dtype)
                rand_normal = torch.distributions.LowRankMultivariateNormal(normal_mean, cov_diag=cov_diag, cov_factor=cov_factor)
            else:
                normal_var = torch.eye(*self.n_dim, device=self.device, dtype=particles.dtype).unsqueeze(0).repeat(n_steps, 1, 1)
                rand_normal = torch.distributions.MultivariateNormal(normal_mean, normal_var)
            samples = rand_normal.sample((n_samples, ))
            if loader:
                dataset = TensorDataset(samples)
                loader = torch.utils.data.DataLoader(dataset, batch_size=n_particles, shuffle=True)
                return loader
            else:
                return samples

    def train_drift(self, obs_times, obs_samples, n_particles, n_iters=10, lr=0.01):
        """Train the drift function to match the observations."""
        self.n_refresh = self.config.train.n_refresh
        optimizer = torch.optim.Adam(self.sde_filter_model.drift.parameters(), lr=lr)
        loader = self.generate_random_steps(len(self.time_stamps), n_particles,  n_samples=10*n_particles, loader=True)
        iter_count, refresh_count = 0, 0
        epoch_losses = []
        while iter_count < n_iters:
            for rand_eps, in loader:     # batches of random samples
                iter_count += 1
                refresh_count += 1
                optimizer.zero_grad()
                _, _, _, loss = self.generate_particles(obs_times, obs_samples, n_particles, stochastic=True, rand_eps=rand_eps)
                loss = torch.mean(loss)
                loss.backward()
                epoch_losses.append(loss.detach().cpu().numpy())
                if iter_count % 2== 0:
                    print(f'Epoch loss after {iter_count} iterations: {np.mean(epoch_losses)}')
                    epoch_losses = []
                if refresh_count > self.n_refresh:
                    torch.cuda.empty_cache()
                    loader = self.generate_random_steps(len(self.time_stamps), n_particles, n_samples=10*n_particles, loader=True)
                    refresh_count = 0
                    break
                if iter_count >= n_iters:
                    break
            self.sde_filter_model.save_drift_model_pickle()

        
            