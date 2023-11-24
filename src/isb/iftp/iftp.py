"""Classes for the Schrödinger bridge problem."""

import torch
from torch.utils.data import TensorDataset
import numpy as np
import os
from os.path import join
from isb.iftp import LinearSDESampler
from isb.neural_nets import DSPSmall, UNetModel
from torch_ema import ExponentialMovingAverage


class IFTP(torch.nn.Module):

    def __init__(self, config, data, device='cpu', 
                forward_path=None, backward_path=None, 
                forward_path_load=None, backward_path_load=None,
                load_models='none', save_models=True, init_loader=None, offset=0, train=True):
        super(IFTP, self).__init__()
        self.config = config
        self.n_dim = config.dataset.n_dim
        self.device = device
        self.save_models = save_models
        self.init_loader = init_loader

        self.num_steps = config.model.num_steps

        self.set_gammas(config)
        self.set_prior_gammas(config)
        self.reverse_time_stamps = torch.flip(self.time_stamps, dims=[0])
        self.prior_reverse_time_stamps = torch.flip(self.prior_time_stamps, dims=[0])
        self.final_time = self.time_stamps[-1]
        assert self.prior_time_stamps[-1] == self.final_time, 'The prior process final time does not match'

        self.prior_model_type = config.prior.prior_model
        self.train = train

        if train:   # used when the model is trained, and not when it is instantiated only for use with trained NNs.
            self.n_iters = config.train.nn_iters
            self.n_refresh = config.train.nn_refresh
            self.cache_size = config.train.cache_size

            self.lr = config.train.lr
            self.batch_size = config.train.batch_size
            self.start_type = config.model.start_type
        self.forward_drift_path = forward_path
        self.backward_drift_path = backward_path
        self.forward_path_load = forward_path_load
        self.backward_path_load = backward_path_load

        self.model_type = config.model.model_type
        self.mean_final, self.var_final = self.get_init_dist(data, unit_var=config.model.unit_var, zero_mean=config.model.zero_mean)
        self.mean_final = self.mean_final - offset
        self.model_output_shift = config.model.model_output_shift
        self.stepsize_scaling = config.model.stepsize_scaling
        
        self.backward_nn = None
        self.forward_nn = None
        self.init_backward_nn()
        self.init_forward_nn()
        self.load_models(load_models)

    def set_gammas(self, config):
        gamma_min = config.model.gamma_min
        gamma_max = config.model.gamma_max
        assert gamma_min <= gamma_max
        self.time_diff = config.model.time_diff
        self.time_stamps = torch.linspace(0.0, self.time_diff*(self.num_steps - 1), self.num_steps) 
        self.gammas = self.get_gamma_values(gamma_min, gamma_max, config.model.gamma_type, self.num_steps)
        self.gammas = self.gammas
    
    def set_prior_gammas(self, config):
        """Form the sampling times for the prior (may differ from self.gammas)."""
        if config.prior.prior_type == 'match':
            self.prior_gammas = self.gammas
            self.prior_time_stamps = self.time_stamps
        elif config.prior.prior_type == 'no_match':
            prior_gamma_min = config.prior.gamma_min
            prior_gamma_max = config.prior.gamma_max
            assert prior_gamma_min <= prior_gamma_max
            prior_time_diff = config.prior.time_diff
            self.prior_time_stamps = torch.linspace(0.0, prior_time_diff*(config.prior.num_steps - 1), config.prior.num_steps)
            self.prior_gammas = self.get_gamma_values(prior_gamma_min, prior_gamma_max, config.prior.gamma_type, config.prior.num_steps)
            self.prior_gammas = self.prior_gammas

    def get_gamma_values(self, gamma_min, gamma_max, gamma_type, num_steps):
        if gamma_type == 'linear_halves':
            gamma_half = np.linspace(gamma_min, gamma_max, (num_steps-1) //2)
            gammas = torch.Tensor(np.concatenate([gamma_half, np.flip(gamma_half)])).to(self.device)
        elif gamma_type == 'linear':
            gammas = np.linspace(gamma_min, gamma_max, num_steps - 1)
            gammas = torch.Tensor(gammas).to(self.device)
        elif gamma_type == 'cosine':
            pass
        return gammas

    def get_model_instance(self):
        if self.model_type == 'flat':
            model =  DSPSmall(input_dim=self.n_dim, device=self.device)
        elif self.model_type == 'imagev2':
            model = UNetModel(self.config)
        model = torch.nn.DataParallel(model).to(self.device)
        return model

    def get_sampler(self, mean_function):

        self.linear_sampler = LinearSDESampler(mean_function=mean_function, mean_final=self.mean_final, var_final=self.var_final, model_output_shift=self.model_output_shift,
                                               gammas=self.gammas, prior_gammas=self.prior_gammas, n_dim=self.n_dim, device=self.device, stepsize_scaling=self.stepsize_scaling)
        self.sampler = self.linear_sampler.to(self.device)

    def get_forward_mean(self, x, t):
        drift = self.forward_nn(x, t)
        return drift

    def get_backward_mean(self, x, t):
        drift = self.backward_nn(x, t)
        return drift

    def get_init_dist(self, data, unit_var=False, zero_mean=False):
        if zero_mean and 'image' not in self.model_type:
            mean = torch.zeros(*data.shape[1:], device=self.device, dtype=data.dtype)
        elif zero_mean and 'image' in self.model_type:
             mean = 0.5*torch.ones(*data.shape[1:], device=self.device, dtype=data.dtype)
        else:
            mean = torch.mean(data, axis=0)
        if unit_var:
            var = torch.ones(*data.shape[1:], device=self.device, dtype=data.dtype)
        else:
            var = torch.ones(*data.shape[1:], device=self.device, dtype=data.dtype)*(np.std(data.detach().cpu().numpy())**2)
        return mean, var

    def sample_init_dist(self, n_samples):
        if self.init_loader is None:
            samples = self.mean_final + torch.sqrt(self.var_final)*torch.randn((n_samples, *self.mean_final.shape), device=self.device)
        else:
            samples = next(self.init_loader)
        return samples

    def init_diffusion(self, x, t):
        return x

    def forward_backward(self, final_data, n_traj=5, first_iter=False):
        self.init_backward_nn()
        self.train_backward(final_data, first_iter)
        self.init_forward_nn()
        self.train_forward()

    def generate_forward_trajectories(self, dist, n_traj=1, first_iter=False):
        self.get_sampler(self.get_forward_mean)
        if first_iter:
            time_stamps = self.prior_time_stamps
        else:
            time_stamps = self.time_stamps

        if (not first_iter) or (self.prior_model_type=='trained'):
            samples, diffs, time_batched, gammas = self.sampler.sample_discrete(dist, time_stamps, n_traj, prior=first_iter)
        else:
            samples, diffs, time_batched, gammas = self.sampler.sample_initial(dist, time_stamps, n_traj)

        if first_iter and len(self.prior_gammas) == 2*len(self.gammas):
            samples = samples[:, 1::2]
            diffs_2d = diffs.reshape(diffs.shape[0], diffs.shape[1]//2, 2, *diffs.shape[2:])
            diffs = diffs_2d.sum(axis=2)
            time_batched = time_batched[:,1::2]
            gammas = gammas[:, 1::2]*2  
        print('Forward samples generated')
        return samples, diffs, time_batched, gammas

    def generate_backward_trajectories(self, dist, n_traj=1, prior=False):
        self.get_sampler(self.get_backward_mean)
        if prior:
            time_stamps = self.prior_reverse_time_stamps
        else:
            time_stamps = self.reverse_time_stamps
        samples, diffs, time_batched, gammas = self.sampler.sample_discrete(dist, time_stamps, n_traj, backward=True, prior=prior)
        print('Backward samples generated')
        return samples, diffs, time_batched, gammas

    def get_final_samples(self, n_trajectories, prior=False):
        """Get the last samples from the trajectory."""
        with torch.no_grad():
            init_samples = self.sample_init_dist(self.cache_size)
            samples, _, _, _ = self.generate_backward_trajectories(init_samples, n_trajectories, prior)
        return samples[:, -1, :]

    def get_init_samples(self, data, n_traj, first_iter=False):
        with torch.no_grad():
            samples, _, _, _ = self.generate_forward_trajectories(data, n_traj, first_iter=first_iter)
        return samples[:, -1, :]

    def load_models(self, load_models):
        if load_models == 'forward':
            self.forward_nn.load_state_dict(torch.load(join(self.forward_path_load, 'model_dict_iter', 'model.pkl')))
        elif load_models == 'all':
            self.backward_nn.load_state_dict(torch.load(join(self.backward_path_load, 'model_dict_iter', 'model.pkl')))
            self.forward_nn.load_state_dict(torch.load(join(self.forward_path_load, 'model_dict_iter', 'model.pkl')))

    def save_forward(self):
        if not self.save_models:
            return
        dir = join(self.forward_drift_path, f'model_dict_iter')
        if not os.path.isdir(dir):
            os.makedirs(dir)
        torch.save(self.forward_nn.state_dict(), join(dir, 'model.pkl'))

    def save_backward(self):
        if not self.save_models:
            return
        dir = join(self.backward_drift_path, f'model_dict_iter')
        if not os.path.isdir(dir):
            os.makedirs(dir)
        torch.save(self.backward_nn.state_dict(), join(dir, 'model.pkl'))

    def init_forward_nn(self):
        if not self.train or self.start_type != 'warm' or not isinstance(self.forward_nn, torch.nn.Module):
            self.forward_nn = self.get_model_instance()
            self.forward_master_params = self.forward_nn.parameters()

    def init_backward_nn(self):
        if not self.train or self.start_type != 'warm' or not isinstance(self.backward_nn, torch.nn.Module):
            self.backward_nn = self.get_model_instance()
            self.backward_master_params = self.backward_nn.parameters() # TODO: Use this instead of the separate mea package (neater) 

    def get_forward_loader(self):
        with torch.no_grad():
            init_data = self.sample_init_dist(self.cache_size)
            data, diffs, time_batched, gammas = self.generate_backward_trajectories(init_data)
            data = data.flatten(start_dim=0, end_dim=1)
            diffs = diffs.flatten(start_dim=0, end_dim=1)
            time_batched = time_batched.flatten(start_dim=0, end_dim=1)
            gammas = gammas.flatten(start_dim=0, end_dim=1)
            dataset = TensorDataset(data, diffs, time_batched, gammas)
            loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return loader

    def train_forward(self):
        loader = self.get_forward_loader()
        ema = ExponentialMovingAverage(self.forward_nn.parameters(), decay=0.999)
        optimizer = torch.optim.Adam(self.forward_nn.parameters(), lr=self.lr)

        iter_count = 0
        refresh_count = 0
        while iter_count < self.n_iters:
            epoch_losses = []
            for (y, diffs, time_batched, gammas) in loader:
                iter_count += 1
                refresh_count += 1
                optimizer.zero_grad()
                times = torch.flatten(time_batched)
                if self.model_output_shift:
                    pred_diffs = self.forward_nn(y, times)
                    if self.stepsize_scaling:
                        diffs = diffs/gammas
                else:
                    pred_diffs = self.forward_nn(y, times) - y
                loss = torch.nn.functional.mse_loss(diffs, pred_diffs)
                loss.backward()
                epoch_losses.append(loss.detach().cpu().numpy())
                optimizer.step()
                ema.update()
                if iter_count % 100 == 0:
                    self.save_forward()
                    print(f'Epoch loss after {iter_count} iterations: {np.mean(epoch_losses)}')
                    epoch_losses = []
                if refresh_count > self.n_refresh:
                    torch.cuda.empty_cache()
                    loader = self.get_forward_loader()
                    refresh_count = 0
                    break
                if iter_count >= self.n_iters:
                    break
        self.save_forward()

    def get_backward_loader(self, final_data, first_iter):
        with torch.no_grad():
            rand_choices = next(final_data).to(self.device)
            data, diffs, time_batched, gammas = self.generate_forward_trajectories(rand_choices, first_iter=first_iter)
            data = data.flatten(start_dim=0, end_dim=1)
            diffs = diffs.flatten(start_dim=0, end_dim=1)
            time_batched = time_batched.flatten(start_dim=0, end_dim=1)
            gammas = gammas.flatten(start_dim=0, end_dim=1)
            dataset = TensorDataset(data, diffs, time_batched, gammas)
            loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return loader

    def train_backward(self, final_data, first_iter=False):
        loader = self.get_backward_loader(final_data, first_iter)
        optimizer = torch.optim.Adam(self.backward_nn.parameters(), lr=self.lr)
        ema = ExponentialMovingAverage(self.backward_nn.parameters(), decay=0.999)
        iter_count, refresh_count = 0, 0

        while iter_count < self.n_iters:
            epoch_losses = []
            for (y, diffs, time_batched, gammas) in loader:
                iter_count += 1
                refresh_count += 1
                optimizer.zero_grad()
                times = torch.flatten(time_batched)
                if self.model_output_shift:
                    pred_diffs = self.backward_nn(y, times)
                    if self.stepsize_scaling:
                        diffs = diffs/gammas
                else:
                    pred_diffs = self.backward_nn(y, times) - y
                
                loss = torch.nn.functional.mse_loss(diffs, pred_diffs)
                loss.backward()
                optimizer.step()
                ema.update()
                epoch_losses.append(loss.detach().cpu().numpy())
                if iter_count % 100 == 0:
                    self.save_backward()
                    print(f'Epoch loss after {iter_count} iterations: {np.mean(epoch_losses)}')
                    epoch_losses = []
                if refresh_count > self.n_refresh:
                    torch.cuda.empty_cache()
                    loader = self.get_backward_loader(final_data, first_iter=first_iter)
                    refresh_count = 0
                    break
                if iter_count >= self.n_iters:
                    break
        self.save_backward()