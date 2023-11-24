"""An interface for score-based models."""

import torch
import math


class SDEModelFilter(torch.nn.Module):
    """A class describing all the required functionalities for DPF.

    This class should be able to compute:
        1. The drift function of the SDE
        2. The diffusion function of the SDE
        3. The score function matching the density
        4. An initial distribution.
    """

    def __init__(self):
        super(SDEModelFilter, self).__init__()

    def set_functions(self, drift, diffusion, bwd_drift_function, init_dist, obs_noise, device='cpu'):
        """Set the relevant functions.
        
        Use when using this class directly instead of inheriting it.
        """
        self.drift = drift
        self.diffusion = diffusion
        self.bwd_drift_function = bwd_drift_function
        self.init_dist = init_dist
        self.obs_noise = obs_noise
        self.device = device

    def eval_drift(self, t, x):
        if t.ndim == 0:
            batch_size = x.shape[0]
            t = t.unsqueeze(0).repeat(batch_size, 1)
        return self.drift(x, t).to(x.dtype) # this is ugly, looks like I coded (t, x) in new code, (x, t) in old code

    def eval_diffusion(self, t):
        return self.diffusion(t).to(t.dtype)

    def eval_obs_noise(self, t):
        return self.obs_noise(t).to(t.dtype)
    
    def eval_bwd_drift(self, t, x):
        if t.ndim == 0:
            batch_size = x.shape[0]
            t = t.unsqueeze(0).repeat(batch_size, 1)
        return self.bwd_drift_function(x, t)

    def score_fn(self, t, x):
        """Computes the score function based on the learned nerwork."""
        backward_drift = self.eval_bwd_drift(t, x)
        diffusion = math.sqrt(2)*self.eval_diffusion(t)
        bwd_fwd = self.eval_drift(t, x) + backward_drift   # was a sum
        diffusion_2 = diffusion**2
        diffuse_inv = 1/diffusion_2
        output = diffuse_inv*bwd_fwd
        return output

    def generate_init_samples(self, n_samples):
        return self.init_dist.sample((n_samples,))

    def eval_state_transition_density(self, x_old, x_new, t_old, t_new):
        """Evaluates the state transition density from x_old to x_new. """
        state_mean = (x_old - self.eval_drift(t_old, x_old)*(t_new - t_old))
        diff = math.sqrt(2)*self.eval_diffusion(t_old)
        state_mean = torch.flatten(state_mean, start_dim=-len(self.n_dim))
        x_new_flat = torch.flatten(x_new, start_dim=-len(self.n_dim))

        state_cov = diff**2*torch.abs(t_new - t_old)*torch.eye(state_mean.shape[-1], device=self.device, dtype=x_new.dtype)
        state_cov = state_cov.unsqueeze(0).repeat(x_old.shape[0], 1, 1) # add batch dimension

        dist = torch.distributions.MultivariateNormal(loc=state_mean, covariance_matrix=state_cov)
        log_prob = dist.log_prob(x_new_flat)
        return log_prob