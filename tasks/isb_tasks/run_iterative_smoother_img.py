"""Image dataset specific iterative smoother."""
import torch
import os
from pathlib import Path
from isb.isb import IterativeTrainer
import hydra
import numpy as np
import time
from isb.data_loaders import get_mnist_loader, generate_blocked_mnist,  TensorSampler
from isb.plot import plot_trajectory_video_img, plot_img_grid, plot_img


@hydra.main(config_path="../../configs/isb")
def train_iterative(config):
    cwd = os.getcwd()
    if cwd[-7:] != '-bridge' :
        base_folder = Path(os.getcwd()).parent.parent.parent 
    else:
        base_folder = cwd   # 
    print(config.model.fwd_drift_file_name)

    class GaussSampler(torch.nn.Module):
        def __init__(self, mean, shape, std=None, device='cpu'):
            super(GaussSampler, self).__init__()
            self.mean = mean
            self.std = std
            if self.std is None:
                self.std = 1
            self.shape = shape
            self.device = device

        def sample(self, n_samples):
            samples = self.mean + self.std*torch.randn(*n_samples, *self.shape, device=self.device)
            return samples

    torch.random.manual_seed(42)    # matters for selection of observations and for initial sample.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config.dataset.n_dim = tuple(eval(config.dataset.n_dim))    # allow a (n_channels, n_width, n_height) format!
    config.model.attn_resolutions = eval(config.model.attn_resolutions)
    config.model.ch_mult = eval(config.model.ch_mult)

    print(device)
    if torch.cuda.is_available():
        print(f'Device: {torch.cuda.get_device_name(0)}')

    model_class = config.model.score
    dataset_name = config.dataset.dataset_name
    data_subset = config.dataset.data_subset
    if dataset_name == 'mnist':
        class_names = [data_subset, '8 - eight']
        loader = get_mnist_loader(base_folder, class_names=class_names, batch_size=config.filter.n_particles)
        terminal_loader = get_mnist_loader(base_folder, class_names=class_names, batch_size=1000)
    plot_folder = os.path.join(base_folder, 'plots', 'dpf_iftp', dataset_name, 'iterative_training', time.strftime("%Y-%m-%d"), time.strftime("%H-%M"))
    data = next(loader)

    # define observation dataset
    if config.dataset.obs_dataset == 'other_class':
        obs_subset = config.dataset.obs_subset
        if dataset_name == 'mnist':
            obs_loader = get_mnist_loader(base_folder, class_names=obs_subset, batch_size=config.filter.n_ts)
        elif dataset_name == 'cifar10':
            obs_loader = get_cifar_loader(base_folder, class_name=obs_subset, batch_size=config.filter.n_ts)
        obs_samples = next(obs_loader).to(device)
        if config.dataset.augment_noise > 0:
            noise_additive = torch.Tensor(np.random.randn(*list(obs_samples.shape))).to(device)
            obs_samples = obs_samples + config.dataset.augment_noise*noise_additive
        obs_samples = obs_samples.unsqueeze(1)  # single ts
        obs_times = torch.Tensor([0.12]).to(device)  
    elif config.dataset.obs_dataset == 'red':
        obs_samples = generate_red_torch_images(config.dataset.n_dim, n_images=config.filter.n_ts, device=device)
        obs_times = torch.Tensor([0.5]).to(device)  
    elif config.dataset.obs_dataset == 'block':
        obs_samples = generate_blocked_mnist(config.dataset.n_dim,  n_images=config.filter.n_ts, device=device)
        obs_times = torch.Tensor([0.25]).to(device) 
    elif config.dataset.obs_dataset == 'block_self':
        obs_samples = data[:config.filter.n_ts].unsqueeze(1)
        obs_samples = obs_samples[:, :, :, ]
        half_size = int(config.dataset.n_dim[1]/2)
        obs_samples[:, :, :, :half_size] = 0
        obs_samples = obs_samples.to(device)
        obs_times = torch.Tensor([0.12]).to(device)  # was 0.12
    elif config.dataset.obs_dataset == 'block_other':
        obs_subset = [config.dataset.obs_subset]
        loader2 = get_mnist_loader(base_folder, class_names=obs_subset, batch_size=config.filter.n_particles)
        obs_data = next(loader2)
        obs_samples = obs_data[:config.filter.n_ts].unsqueeze(1)
        obs_samples = obs_samples[:, :, :, ]
        half_size = int(config.dataset.n_dim[1]/2)
        obs_samples[:, :, :, half_size:] = 0    # was :half_size (lower half)
        obs_samples = obs_samples.to(device)
        obs_times = torch.Tensor([0.2]).to(device)  # was 0.12
    else:
        raise NotImplementedError(f'Observational dataset {config.dataset.obs_dataset} not recognized.')

    # define initial distribution
    if config.model.init_dist == 'gauss':
        if config.filter.zero_mean and 'image' not in config.model.model_type:
            mean = torch.zeros(config.dataset.n_dim, device=device, dtype=data.dtype)
        elif config.filter.zero_mean and 'image' in config.model.model_type:
             mean = 0.5*torch.ones(*config.dataset.n_dim, device=device, dtype=data.dtype)
        else:
            mean = torch.mean(data, axis=0)

        if config.filter.unit_var:
            std = 1
            cov_diag = torch.ones(*config.dataset.n_dim, device=device, dtype=data.dtype)
        else:
            std = np.std(data.detach().cpu().numpy())**2
            cov_diag = torch.ones(*config.dataset.n_dim, device=device, dtype=data.dtype)*(np.std(data.detach().cpu().numpy())**2)
       
        cov_factor = torch.zeros(*config.dataset.n_dim, device=device, dtype=data.dtype)
        init_dist = GaussSampler(mean=mean, shape=config.dataset.n_dim, std=std, device=device)
       # init_dist = torch.distributions.LowRankMultivariateNormal(mean, cov_diag=cov_diag, cov_factor=cov_factor)
    else:
        raise NotImplementedError(f'Initial distribution type {config.model.init_dist} not implemented')

    # set model paths
    if model_class == 'nn':
        init_drift_model_path = os.path.join(base_folder, 'bridge_models', 'backward_drift', config.model.drift_file_name)
        fwd_drift_model_path = os.path.join(base_folder, 'isb_models',  config.model.fwd_drift_file_name)
        bwd_drift_model_path = os.path.join(base_folder, 'isb_models',  config.model.bwd_drift_file_name)
    else:
        raise NotImplementedError(f'Type of score-based model {model_class} not implemented')

    plot_img_grid(obs_samples[:, 0], plot_folder, 'observational_data', nrows=3)
    terminal_dist = TensorSampler(next(terminal_loader).to(device))  # using more samples than only batch_size!
    iterative_trainer = IterativeTrainer(config,
                                        plot_folder=plot_folder,
                                        init_dist=init_dist,
                                        fwd_drift_model_path=fwd_drift_model_path,
                                        bwd_drift_model_path=bwd_drift_model_path,
                                        init_drift_model_path=init_drift_model_path,
                                        device=device)
    print('In this script?')
    if not config.train.skip_train:
        iterative_trainer.train(obs_times, obs_samples, terminal_dist, rand_select=None, obs_ts=None, plot=True, batch_size=config.train.batch_size, lr=config.train.lr)
    with torch.no_grad():
        rev_times = torch.flip(obs_times, dims=[0])
        rev_samples = torch.flip(obs_samples, dims=[1])
        iterative_trainer.init_loop = False
        iterative_trainer.init_particle_filter_fwd()
        rev_times = []
        smooth_particles, _ , _, _ = iterative_trainer.generate_smoothed_particles(rev_times, rev_samples, smoothing=False)
    plot_trajectory_video_img(plot_folder, smooth_particles[:, :30])
    for i in range(30):
        for j in range(smooth_particles.shape[0]):
            plot_img(smooth_particles[j, i], os.path.join(plot_folder, 'img_trajectories'),
            data_name=f'mnist_8_final_sample_{i}_step_{j}', time_in_name=False, grayscale=True)
 
if __name__ == '__main__':
    train_iterative()