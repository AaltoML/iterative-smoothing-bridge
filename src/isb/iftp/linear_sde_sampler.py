"""Sample from SDEs with Gaussian transitions."""

import torch

class LinearSDESampler(torch.nn.Module):

    def __init__(self, mean_function, mean_final, var_final, n_dim, gammas, prior_gammas, device='cpu',
        model_output_shift=True, stepsize_scaling=False):
        super(LinearSDESampler, self).__init__()
        self.mean_function = mean_function
        self.mean_final = mean_final
        self.var_final = var_final
        self.model_output_shift = model_output_shift
        self.gammas = gammas
        self.time_stamps = torch.cumsum(gammas, axis=0)
        self.prior_gammas = prior_gammas
        self.prior_time_stamps = torch.cumsum(prior_gammas, axis=0)
        self.n_dim = n_dim
        self.device = device
        self.stepsize_scaling = stepsize_scaling

    def grad_gauss(self, x):
        """As in the codebase for Schrödinger bridge."""
        xout = (x - self.mean_final[None, :]) / self.var_final[None, :]
        return -xout

    def get_gamma_t(self, index, prior=False, backward=False):
        if prior:
            gammas = self.prior_gammas
        else:
            gammas = self.gammas
        if backward:
            gammas = torch.flip(gammas, dims=[0])
        return gammas[index]

    def set_mean_function(self, mean_func):
        self.mean_function = mean_func

    def sample_discrete(self, init_points, time_stamps, n_trajectories=5, prior=False, backward=False):
        """Sample trajectory from given initial point at given observation times.
        
        Uses the preset self.mean_function to compute the mean shift of the transition.
        """
        batch_size = init_points.shape[0]

        init_stacked = torch.stack([init_points]*n_trajectories)
        init_stacked = torch.reshape(init_stacked, (batch_size*n_trajectories, *self.n_dim))

        time_stamps_batched = time_stamps.reshape((1, len(time_stamps), 1)).repeat(init_stacked.shape[0], 1, 1)
        samples = torch.empty(n_trajectories*batch_size, len(time_stamps) - 1, *self.n_dim, device=self.device, dtype=init_points.dtype)
        gammas = torch.empty((n_trajectories*batch_size, len(time_stamps) -1, *self.n_dim), device=self.device, dtype=init_points.dtype)
        sample = init_stacked

        diffs = torch.empty(n_trajectories*batch_size, len(time_stamps) - 1, *self.n_dim, device=self.device, dtype=init_points.dtype)
        for i in range(len(time_stamps) - 1):
            time_prev, time_next = time_stamps_batched[:, i], time_stamps_batched[:, i+1]
            sample, diff, gamma_t = self.sample_normal(sample, time_prev, time_next, i, prior=prior, backward=backward)
            samples[:, i] = sample
            diffs[:, i] = diff
            gammas[:, i] = gamma_t
        return samples, diffs, time_stamps_batched[:, 1:], gammas

    def sample_initial(self, init_points, time_stamps, n_trajectories=5):
        """Sample prior trajectory from given initial point at given observation times.
        
        Applies the function self.grad_gauss to compute the transition mean shift.
        """
        batch_size = init_points.shape[0]

        init_stacked = torch.stack([init_points]*n_trajectories)
        init_stacked = torch.reshape(init_stacked, (batch_size*n_trajectories, *self.n_dim))
        time_stamps_batched = time_stamps.reshape((1, len(time_stamps), 1)).repeat(init_stacked.shape[0], 1, 1)
        samples = torch.empty(n_trajectories*batch_size, len(time_stamps) - 1, *self.n_dim, device=self.device, dtype=init_points.dtype)
        gammas = torch.empty((n_trajectories*batch_size, len(time_stamps) -1, *self.n_dim), device=self.device, dtype=init_points.dtype)
        sample = init_stacked
        diffs = torch.empty(n_trajectories*batch_size, len(time_stamps) - 1, *self.n_dim, device=self.device, dtype=init_points.dtype)

        for i in range(len(time_stamps) - 1):
            time_prev, time_next = time_stamps_batched[:, i], time_stamps_batched[:, i+1]
            sample, diff, gamma_t = self.sample_init_normal(sample, time_prev, time_next, i)
            samples[:, i] = sample
            diffs[:, i] = diff
            gammas[:, i] = gamma_t
        return samples, diffs, time_stamps_batched[:, 1:], gammas

    def sample_init_normal(self, x, t_prev, t_next, i):
        time_diff = torch.abs(t_next - t_prev)[0][0]
        gamma_t = self.get_gamma_t(i, prior=True, backward=False)
        mean_prev = x + time_diff*self.grad_gauss(x)
        random_samples = torch.randn(mean_prev.shape, device=self.device)
        var = gamma_t*2*(time_diff)
        new_sample = mean_prev + torch.sqrt(var)*random_samples

        mean_next = new_sample + time_diff*self.grad_gauss(new_sample)
        diff = (mean_prev - mean_next)

        return new_sample, diff, time_diff # assuminh constant time diffs

    def sample_normal(self, x, t_prev, t_next, i, prior, backward=False):
        gamma_t = self.get_gamma_t(i, prior, backward=backward)
        time_diff = torch.abs(t_next - t_prev)[0][0]
        nn_output = self.mean_function(x, torch.flatten(t_prev))
        if self.stepsize_scaling:
            nn_output *= time_diff

        if self.model_output_shift:
            mean_prev = x + nn_output
        else:
            mean_prev = nn_output

        random_samples = torch.randn(mean_prev.shape, device=self.device)
        var = gamma_t*2*(time_diff)
        scaled = torch.sqrt(var)*random_samples
        samples = mean_prev + scaled

        nn_output2 = self.mean_function(samples, torch.flatten(t_prev))
        if self.stepsize_scaling:
            nn_output2 *= time_diff
        
        if self.model_output_shift:
            mean_next = samples + nn_output2
        else:
            mean_next = nn_output2
        diff = (mean_prev - mean_next)
        return samples, diff, time_diff