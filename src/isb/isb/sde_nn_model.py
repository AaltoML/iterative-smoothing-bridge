"""A neural network model for SDE filtering."""

from isb.isb import SDEModelFilter
from isb.neural_nets import DSPSmall, UNetModel
import torch
from os.path import join
import os


class SDENNFilter(SDEModelFilter):
    """A NN-score, fixed drift model for the particle filtering problem."""

    def __init__(self, config, base_folder, init_dist, drift_type=None, drift_model_path=None, score_model_path=None, load_score_path=None, warm_start=False,  device='cpu', all_data=None):
        super(SDENNFilter, self).__init__()
        self.n_dim = config.dataset.n_dim
        self.device = device
        self.config = config
        if drift_type is None:
            drift_type = config.model.drift
        self.model_path = join(base_folder, 'model')
        self.init_noise_mult = 1
        self.model_type = config.model.model_type # supported options: flat or imagev2

        if drift_type == 'zero':
            self.init_noise_mult = config.model.init_gamma_multiplier
            self.drift = lambda x, t: torch.zeros(x.shape[0], *self.n_dim, device=self.device)
        elif drift_type == 'neg_linear':
            self.drift = lambda x, t: -0.1*x
        elif drift_type == 'tanh':
            self.drift = self.tanh_drift
        elif drift_type == 'nn':
            self.drift = self.create_model_instance()
            self.drift = torch.nn.DataParallel(self.drift).to(self.device)
            self.drift_model_path = drift_model_path 
            self.save_drift_model = config.model.save_drift_model
            if config.model.load_drift_model:
                self.load_drift_model()
        else:
            raise NotImplementedError(f'Drift of type {config.model.drift} not implemented')

        self.bwd_drift_function = self.create_model_instance()
        self.bwd_drift_function = torch.nn.DataParallel(self.bwd_drift_function).to(self.device)
        self.init_dist = init_dist
        self.bwd_drift_path = score_model_path   # join(self.model_path, config.model.score_file_name)
        self.load_score_path = load_score_path
        self.save_score_model = config.model.save_score_model
        self.obs_noise_level = config.filter.obs_noise
        self.half_time = config.filter.time_diff*config.filter.n_steps/2
        self.set_gamma_t(config)

        if (config.model.load_score_model or warm_start) and self.load_score_path is not None:
            self.load_bwd_drift_model()

    def tanh_drift(self, x, t):
        
        if t[0][0] > self.half_time:
            return -x/(2*self.half_time - t)
        else:
            return torch.tanh(x)

    def create_model_instance(self):
        if self.model_type == 'flat':
            model = DSPSmall(*self.n_dim, device=self.device)
        elif self.model_type == 'imagev2':
            model = UNetModel(self.config)
        else:
            raise NotImplementedError(f'Model type {self.model_type} not implemented')
        return model
        
    def set_gamma_t(self, config):
        """Sets time and diffusion constants."""
        self.t_min = config.filter.t_min
        self.gamma_min = config.model.gamma_min
        self.gamma_max = config.model.gamma_max
        self.t_max = self.t_min + (config.filter.n_steps - 1)*config.filter.time_diff

    def obs_noise(self, t):
        return torch.Tensor([float(self.obs_noise_level)])[0]

    def diffusion(self, t):
        """Compute SDE diffusion (based on gammas)."""
        diffusion = (self.gamma_min + (t - self.t_min)*(self.gamma_max - self.gamma_min)/(self.t_max - self.t_min))
        diffusion = torch.sqrt(2*diffusion)*self.init_noise_mult
        return diffusion

    def load_bwd_drift_model(self):
        """Read score model from pickle."""
        self.bwd_drift_function.load_state_dict(torch.load(join(self.load_score_path, 'model_dict_iter', 'model.pkl')))

    def load_drift_model(self):
        """Read drift model form pickle."""
        self.drift.load_state_dict(torch.load(join(self.drift_model_path, 'model_dict_iter', 'model.pkl')))

    def save_score_model_pickle(self):
        """Save score model to pickle."""
        if not self.save_score_model:
            return
        dir = join(self.bwd_drift_path, f'model_dict_iter')
        if not os.path.isdir(dir):
            os.makedirs(dir)
        torch.save(self.bwd_drift_function.state_dict(), join(dir, 'model.pkl'))

    def save_drift_model_pickle(self):
        """Save drift model to pickle."""
        if not self.save_drift_model:
            return
        dir = join(self.drift_model_path, f'model_dict_iter')
        if not os.path.isdir(dir):
            os.makedirs(dir)
        torch.save(self.drift.state_dict(), join(dir, 'model.pkl'))
