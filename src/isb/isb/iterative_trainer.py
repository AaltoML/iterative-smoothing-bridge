"""Class for iterative smoother training."""

import torch
import os
import time
import pickle
from collections import defaultdict
import numpy as np
from torch.utils.data import TensorDataset
from TrajectoryNet.optimal_transport.emd import earth_mover_distance
from isb.isb import ParticleFilter, SDENNFilter
from isb.plot import plot_trajectory_video, plot_1d_trajectory, plot_trajectory_video_img
from isb.data_loaders import TensorSampler

class IterativeTrainer:
    """Train the smoother model iteratively.
    
    Does not proppagate the particles, only uses lower-level classes
    for particle filtering.
    """

    def __init__(self, config, base_folder, plot_folder,  init_dist, fwd_drift_model_path, bwd_drift_model_path, init_drift_model_path, device='cpu'):
        self.config = config
        self.model_type = config.model.model_type   # used only in plots
        self.init_dist = init_dist
        self.device = device
        self.base_folder = base_folder
        self.data_name = config.dataset.dataset_name
        self.fwd_drift_model_path = fwd_drift_model_path    # fwd model will be overwritten on each iteration
        self.bwd_drift_model_path = bwd_drift_model_path     # bwd model will be overwritten on each iteration
        self.init_drift_model_path = init_drift_model_path
        self.init_loop = True

        self.plot_folder = plot_folder

        if hasattr(config.dataset, 'plot_x_min'):
            self.plot_xlim = [config.dataset.plot_x_min, config.dataset.plot_x_max]
            self.plot_ylim = [config.dataset.plot_y_min, config.dataset.plot_y_max]

        self.transport_eps = config.filter.transport_eps
        self.n_particles = config.filter.n_particles
        self.n_dim = self.config.dataset.n_dim
        self.time_forward = self.config.filter.time_forward
        self.n_steps = config.filter.n_steps
        self.uniform_weights =  torch.ones((self.n_steps, self.n_particles), device=device)/self.n_particles

        self.n_refresh = config.train.n_refresh
        self.n_iters = config.train.n_iters

    def init_particle_filter_fwd(self, all_data=None):
        """Initialize particle filter for forward pass."""
        if self.init_loop:
            file_name = self.init_drift_model_path
            warm_start = False
            drift_type = self.config.model.init_drift
        else:
            file_name = self.bwd_drift_model_path
            drift_type = 'nn'
            warm_start = self.config.model.warm_start 

        self.sde_model = SDENNFilter(self.config, self.base_folder, self.init_dist, drift_type=drift_type, score_model_path=self.fwd_drift_model_path, drift_model_path=file_name, warm_start=warm_start,load_score_path=self.fwd_drift_model_path, device=self.device, all_data=all_data)   # never save forward model in this step
        self.pf = ParticleFilter(self.config, sde_filter_model=self.sde_model, time_forward=not self.time_forward, device=self.device)

    def init_particle_filter_bwd(self, terminal_dist):
        """Initialize particle filter for backward pass."""
        if self.init_loop and self.config.model.init_drift == 'nn':
            load_score_path = self.init_drift_model_path
        elif self.init_loop:    # when initializing with zero drift, should not use the Schrödinger bridge model
            load_score_path = None
        else:
            load_score_path = self.bwd_drift_model_path
        warm_start = self.config.model.warm_start
        drift_type = 'nn'
        self.sde_model = SDENNFilter(self.config, self.base_folder, terminal_dist, drift_type=drift_type, score_model_path=self.bwd_drift_model_path, drift_model_path=self.fwd_drift_model_path, warm_start=warm_start,load_score_path=load_score_path, device=self.device)    # never save backward model in this step
        self.pf = ParticleFilter(self.config, sde_filter_model=self.sde_model, time_forward=self.time_forward, device=self.device)
        self.init_loop = False

    def noise_schedule(self, t, tprime):
        obs_noise1 = self.config.filter.obs_noise_1
        obs_noise2 = self.config.filter.obs_noise_2
        if t < tprime:
            return (obs_noise1*(tprime - t) + obs_noise2*(t - 0)) / tprime
        else:
            return (obs_noise2*(self.config.train.n_iftp - t) + obs_noise1*(t - tprime)) / (self.config.train.n_iftp - tprime)

    def train(self, obs_times, obs_samples, terminal_dist, rand_select=None, obs_ts=None, plot=True, batch_size=256, lr=0.01, store_pickle=False, report_data=None, all_data=None):
        """Full iterative training loop."""

        tprime = round(self.config.train.n_iftp / 2)

        n_rounds = self.config.train.n_iftp
        for i in range(n_rounds):
            self.iteration = i
            filter_on = i < self.config.train.n_smooth
            print(f'Starting train step {i}, with obs noise: {self.config.filter.obs_noise}')
            plot_collections = self.train_step(obs_times, obs_samples, terminal_dist, batch_size, lr, filter_on=filter_on, report_data=report_data, all_data=all_data)
            if 'schedule_type' in self.config.filter:
                sched_type = self.config.filter.schedule_type
            else:
                sched_type = 'linear'
            if sched_type == 'linear':
                self.config.filter.obs_noise = self.noise_schedule(i, tprime)
            elif sched_type == 'mult':
                self.config.filter.obs_noise *= self.config.filter.obs_noise_mult
            if plot:
                with torch.no_grad():
                    self.create_all_plots(plot_collections, obs_ts, rand_select, i)
            if store_pickle:
                with torch.no_grad():
                    self.pickle_for_plots(plot_collections, obs_samples, obs_times, i)

    def pickle_for_plots(self, plot_collections, obs_samples, obs_times, iteration=0):
        """Pickle some relevant objects."""
        pickle_folder = os.path.join(self.plot_folder, 'pickles')
        if not os.path.isdir(pickle_folder):
            os.makedirs(pickle_folder)
        
        # forward
        for plot_name in plot_collections['fwd_plot']:
            particle_file = os.path.join(pickle_folder, f'particles_fwd_{plot_name}_{iteration}.pkl')
            with open(particle_file, 'wb') as handle:
                pickle.dump(plot_collections['fwd_plot'][plot_name].cpu().numpy(), handle)

        # backward
        for plot_name in plot_collections['bwd_plot']:
            particle_file = os.path.join(pickle_folder, f'particles_bwd_{plot_name}_{iteration}.pkl')
            with open(particle_file, 'wb') as handle:
                pickle.dump(plot_collections['bwd_plot'][plot_name].cpu().numpy(), handle)
        

    def create_all_plots(self, plot_collections, obs_ts=None, rand_select=None, iteration=0):
        if obs_ts is not None:
            rev_obs_ts = torch.flip(obs_ts.cpu(), dims=[1])
            obs_ts_cpu = obs_ts.cpu()
        else:
            obs_ts_cpu = None
            rev_obs_ts = None
        if rand_select is not None:
            rev_indices = self.config.filter.n_steps - 1 - rand_select
        else:
            rev_indices = None
        for plot_name in plot_collections['fwd_plot']:
            if 'image' not in self.model_type:
                plot_data = plot_collections['fwd_plot'][plot_name]
                if self.n_dim[0] == 5:
                    if obs_ts is not None:
                        obs_ts_cpu = obs_ts_cpu[:, :, :2]
                    plot_data = plot_data[:, :, :2]
                if plot_data.shape[-1] == 2:
                    plot_trajectory_video(self.plot_folder,  rand_select, obs_ts_cpu, plot_data,
                                    xlim=self.plot_xlim, ylim=self.plot_ylim, filename=f'{plot_name}_{iteration}',
                                    base_folder=self.base_folder, map_version=(self.data_name=='birds'))
                else:
                    plot_1d_trajectory(os.path.join(self.plot_folder, 'trajectories'), plot_data,
                                        data_name=f'{plot_name}_{iteration}')
            else:
                plot_trajectory_video_img(self.plot_folder, plot_collections['fwd_plot'][plot_name][:, :9],
                                        filename=f'{plot_name}_{iteration}')
        for plot_name in plot_collections['bwd_plot']:
            if 'image' not in self.model_type:
                plot_data = plot_collections['bwd_plot'][plot_name]
                if self.n_dim[0] == 5:
                    if rev_obs_ts is not None:
                        rev_obs_ts = rev_obs_ts[:, :, :2]
                    plot_data = plot_data[:, :, :2]
                if plot_data.shape[-1] == 2:
                    plot_trajectory_video(self.plot_folder, rev_indices, rev_obs_ts, plot_data,
                                    xlim=self.plot_xlim, ylim=self.plot_ylim, filename=f'{plot_name}_{iteration}',
                                    base_folder=self.base_folder, map_version=(self.data_name=='birds'))
                elif plot_data.shape[-1] == 1:    
                    plot_1d_trajectory(os.path.join(self.plot_folder, 'trajectories'), plot_data,
                                        data_name=f'{plot_name}_{iteration}')
            else:
                plot_trajectory_video_img(self.plot_folder, plot_collections['bwd_plot'][plot_name][:, :9],
                                        filename=f'{plot_name}_{iteration}')

    
    def print_metric(self,t, samples, particles):
        """Print metric between generated particles and true data."""
        emd = earth_mover_distance(samples, particles)
        print(f'EMD at time {t}: {emd}')

    def train_step(self, obs_times, obs_samples, terminal_dist, batch_size, lr, filter_on=True, report_data=None, all_data=None):
        plot_collections = defaultdict(dict)

        # learn "noise to data", t in [T, 0]
        self.init_particle_filter_fwd(all_data=all_data)
        rev_samples = torch.flip(obs_samples, dims=[1])
        rev_times = torch.flip(obs_times, dims=[0])
        report_times = []   #rev_times
        init_uncontrolled_particles, _, _, _ = self.generate_smoothed_particles([], rev_samples,report_times=report_times, report_data=report_data)   # Report performance !
        if 'image' not in self.model_type:
            self.print_metric('max', init_uncontrolled_particles[-1], terminal_dist.sample((init_uncontrolled_particles.shape[1], )))
   #     rev_times = []  # TODO: Remove this
        init_filter, _, _, _ = self.generate_smoothed_particles(rev_times, rev_samples)    
        if not filter_on:
            rev_times = []
            obs_times = []
        self.learn_bwd_drift(rev_times, rev_samples, batch_size=batch_size, lr=lr)
        

         # learn "data to noise", t in [0, T]
        self.init_particle_filter_bwd(terminal_dist)
        report_times = [] # obs_times
        bwd_particles, _, _, _ = self.generate_smoothed_particles([], obs_samples, report_times=report_times, report_data=report_data) # Report again !
      #  obs_times = []  # TODO: Remove this
        if 'image' not in self.model_type:
            self.print_metric('min', bwd_particles[-1], self.init_dist.sample((bwd_particles.shape[1], )))
        plot_collections['fwd_plot']['bwd_from_term'] = bwd_particles
        self.learn_fwd_drift(obs_times, obs_samples, batch_size=batch_size, lr=lr)

        plot_collections['bwd_plot']['init_uncontrolled'] = init_uncontrolled_particles
        plot_collections['bwd_plot']['init_filter'] = init_filter
        return plot_collections

    def generate_smoothed_particles(self, obs_times, obs_samples, report_weights=False, report_times=[], report_data=None):
        with torch.no_grad():
            bwd_particles, bwd_diffs, bwd_times, bwd_dts = self.pf.generate_particles(obs_times, obs_samples, n_particles=self.n_particles,
                                                 stochastic=True, report_weights=report_weights, report_times=report_times, report_data=report_data)
            return bwd_particles, bwd_diffs, bwd_times, bwd_dts

    def create_loader(self, obs_times, obs_samples, batch_size):
        """Create particle trajectories, save results to loader."""
        with torch.no_grad():
            all_particles, true_diffs, times_batched, dts = self.generate_smoothed_particles(obs_times, obs_samples) 
            times_batched = times_batched[1:] # only evaluation times
            all_particles = torch.flatten(all_particles[1:], start_dim=0, end_dim=1).detach()
            dts = torch.flatten(dts, start_dim=0, end_dim=1)   # TODO: figure this out?
            times_batched = torch.flatten(times_batched).detach()
            true_diffs = torch.flatten(true_diffs, start_dim=0, end_dim=1).detach()
            dataset = TensorDataset(all_particles, true_diffs, times_batched, dts)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            print('Re-created loader')
            return loader

    def evaluate_loss(self, times, y, diffs, dts):
        """Compute batch loss."""
        drift = self.sde_model.eval_bwd_drift(times, y)
        loss = torch.nn.functional.mse_loss(diffs/dts, drift)  
        return loss

    def learn_bwd_drift(self, obs_times, obs_samples, batch_size, lr=0.01):

        optimizer = torch.optim.Adam(self.sde_model.bwd_drift_function.parameters(), lr=lr)
        loader = self.create_loader(obs_times, obs_samples, batch_size=batch_size)

        iter_count, refresh_count = 0, 0
        epoch_losses = []
        while iter_count < self.n_iters:
            for (y, diffs, time_batched, dts) in loader:
                iter_count += 1
                refresh_count += 1
                optimizer.zero_grad()
                loss = self.evaluate_loss(time_batched, y, diffs, dts)
                loss.backward()
                epoch_losses.append(loss.detach().cpu().numpy())
                optimizer.step()
                loss = loss.detach()
                if iter_count % 100 == 0:
                    print(f'Epoch loss after {iter_count} iterations: {np.mean(epoch_losses)}')
                    epoch_losses = []
                if refresh_count > self.n_refresh:
                    torch.cuda.empty_cache()
                    loader = self.create_loader(obs_times, obs_samples, batch_size=batch_size)
                    refresh_count = 0
                    self.sde_model.save_score_model_pickle()
                    break
                if iter_count >= self.n_iters:
                    self.sde_model.save_score_model_pickle()
                    break

    def learn_fwd_drift(self, obs_times, obs_samples, batch_size, lr=0.01):
        optimizer = torch.optim.Adam(self.sde_model.bwd_drift_function.parameters(), lr=lr)
        loader = self.create_loader(obs_times, obs_samples, batch_size=batch_size)

        iter_count, refresh_count = 0, 0
        epoch_losses = []
        while iter_count < self.n_iters:
            for (y, diffs, time_batched, dts) in loader:
                iter_count += 1
                refresh_count += 1
                optimizer.zero_grad()
                loss = self.evaluate_loss(time_batched, y, diffs, dts)
                loss.backward()
                epoch_losses.append(loss.detach().cpu().numpy())
                optimizer.step()
                loss = loss.detach()
                if iter_count % 100 == 0:
                    print(f'Epoch loss after {iter_count} iterations: {np.mean(epoch_losses)}')
                    epoch_losses = []
                if refresh_count > self.n_refresh:
                    torch.cuda.empty_cache()
                    loader = self.create_loader(obs_times, obs_samples, batch_size=batch_size)
                    self.sde_model.save_score_model_pickle()
                    refresh_count = 0
                    break
                if iter_count >= self.n_iters:
                    self.sde_model.save_score_model_pickle()
                    break
