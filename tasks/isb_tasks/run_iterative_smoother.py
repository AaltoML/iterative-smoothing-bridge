"""Train a model iteratively, using smoother distributions."""

import torch
import os
from pathlib import Path
from isb.isb import IterativeTrainer
import hydra
import numpy as np
import time
import pickle
from isb.data_loaders import get_ssm_data, prep_ssm_data, get_rna_data, read_bridge_observations, load_toy_data, TensorSampler
from isb.plot import plot_trajectory_video, plot_problem_constraints, plot_1d_trajectory, plot_2d_hist


@hydra.main(config_path="../../configs/isb", config_name="circle_iterative_smoother")
def train_iterative(config):
    cwd = os.getcwd()
    if cwd[-7:] != '-bridge' :
        base_folder = Path(os.getcwd()).parent.parent.parent 
    else:
        base_folder = cwd   # 

    torch.random.manual_seed(335)    # matters for selection of observations and for initial sample. was 42
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config.dataset.n_dim = tuple(eval(config.dataset.n_dim))

    print(device)
    if torch.cuda.is_available():
        print(f'Device: {torch.cuda.get_device_name(0)}')

    # read observations
    model_class = config.model.score
    dataset_name = config.dataset.dataset_name
    if dataset_name != 'rna':
        all_times, all_samples = get_ssm_data(base_folder, f'{dataset_name}_bridge', device, n_steps=config.filter.n_steps)
        if dataset_name != 'circles':
            obs_ts, obs_times, obs_samples, rand_select = prep_ssm_data(config, all_times, all_samples)
    
    # configure plotting
    plot_folder = os.path.join(base_folder, 'plots', 'isb', dataset_name, 'iterative_training', time.strftime("%Y-%m-%d"), time.strftime("%H-%M"))
    plot_xlim = [config.dataset.plot_x_min, config.dataset.plot_x_max]
    plot_ylim = [config.dataset.plot_y_min, config.dataset.plot_y_max]

    # get terminal data
    if dataset_name  != 'rna':
        loaders, _ = load_toy_data([dataset_name], batch_size=10000, device=device)
        loader = loaders[dataset_name]
        term_data = next(loader)
    elif dataset_name == 'rna':
        init_loader, term_data, obs_samples, obs_times, obs_ts, rand_select, report_data, all_data = get_rna_data(base_folder, config, n_init_points=1000, device=device)

    # define obs dataset based on read data
    if config.dataset.obs_dataset == 'circle':
        mean = torch.Tensor([1.5, 0]).to(device)
        radius = 3
        n_points = config.filter.n_ts
        n_obs_t = config.filter.n_obs
        obs_samples_single = torch.Tensor([mean.cpu().numpy() + radius*np.array([np.sin(2*np.pi*i/n_points), np.cos(2*np.pi*i/n_points)]) for i in range(n_points)]).to(device)
        obs_samples = obs_samples_single.unsqueeze(1).repeat(1, n_obs_t, 1)
        obs_ts = obs_samples_single.unsqueeze(1).repeat(1, all_times.shape[1], 1)
        obs_times = torch.Tensor([0.5]*n_obs_t).to(device)
        rand_select = torch.Tensor([50]*n_obs_t).to(device)
        if n_points == 0:
            obs_ts = None
            obs_times = torch.Tensor([])
            rand_select = torch.Tensor([])
        report_data = None 
    elif config.dataset.obs_dataset == 'dynamics':
        print('Using the dynamical system observations directly!')
        report_data = obs_samples
    elif config.dataset.obs_dataset == 'bridge':
        obs_ts, obs_samples, obs_times, rand_select = read_bridge_observations(base_folder, config, device=device) # split bridge observations are in a separate file 
        report_data = obs_samples    
    elif config.dataset.obs_dataset != 'rna':
        raise NotImplementedError(f'Observational dataset {config.dataset.obs_dataset} not recognized.')

    if dataset_name != 'rna':
        all_data = None

    # set initial distribution
    if config.model.init_dist == 'gauss':
        data = all_samples[:, -1]
        if config.filter.zero_mean:
            mean = torch.zeros(*config.dataset.n_dim, device=device, dtype=torch.float64)
        else:
            mean = torch.mean(data, axis=0)
        if config.filter.unit_var:
            cov = torch.eye(*config.dataset.n_dim, device=device, dtype=torch.float64)
        else:
            cov = torch.eye(*config.dataset.n_dim, device=device, dtype=torch.float64)*(np.std(data.detach().cpu().numpy())**2)
        init_dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
    elif config.model.init_dist == 'data':      # initial distribution is non-Gaussian, use the data directly
        data = all_samples[:, 0]
        init_dist = TensorSampler(data)
    elif config.model.init_dist == 'loader':
        data = next(init_loader)
        init_dist = TensorSampler(data)
    elif config.model.init_dist == 'zero':
        zeros = torch.zeros((1000, *config.dataset.n_dim), device=device, dtype=torch.float64)
        init_dist = TensorSampler(zeros)
    else:
        raise NotImplementedError(f'Initial distribution type {config.model.init_dist} not implemented')

    # check where the neural network files are if necessary
    if model_class == 'nn':
        if 'drift_file_name' in config.model:
            init_drift_model_path = os.path.join(base_folder, 'bridge_models', 'backward_drift', config.model.drift_file_name)
        else:
            init_drift_model_path = None
        fwd_drift_model_path = os.path.join(base_folder, 'isb_models',  config.model.fwd_drift_file_name)
        bwd_drift_model_path = os.path.join(base_folder, 'isb_models',  config.model.bwd_drift_file_name)
    else:
        raise NotImplementedError(f'Type of score-based model {model_class} not implemented')

    terminal_dist = TensorSampler(term_data)

    # Make RNA plots
    if config.dataset.n_dim[0] == 5:    # applies only to the RNA dataset
        _  = plot_2d_hist(data[:, :2], base_folder=os.path.join(plot_folder,  'true_snapshots'), data_name=f'init_distribution_first', xlim=plot_xlim, ylim=plot_ylim, obs=None, marker_mode='')
        _  = plot_2d_hist(term_data[:,  :2], base_folder=os.path.join(plot_folder,  'true_snapshots'), data_name=f'term_distribution_first', xlim=plot_xlim, ylim=plot_ylim, obs=None, marker_mode='')

        _  = plot_2d_hist(data[:, 2:4], base_folder=os.path.join(plot_folder,  'true_snapshots'), data_name=f'init_distribution_last', xlim=plot_xlim, ylim=plot_ylim, obs=None, marker_mode='')
        _  = plot_2d_hist(term_data[:,  2:4], base_folder=os.path.join(plot_folder,  'true_snapshots'), data_name=f'term_distribution_last', xlim=plot_xlim, ylim=plot_ylim, obs=None, marker_mode='')

        _  = plot_2d_hist(all_data[:, :2], base_folder=os.path.join(plot_folder,  'true_snapshots'), data_name=f'all_data_first', xlim=plot_xlim, ylim=plot_ylim, obs=None, marker_mode='')

        for obs_idx in range(obs_samples.shape[1]):
            _  = plot_2d_hist(obs_samples[:, obs_idx, :2], base_folder=os.path.join(plot_folder,  'true_snapshots'), data_name=f'middle_distribution_first_{obs_idx}', xlim=plot_xlim, ylim=plot_ylim, obs=None, marker_mode='')
            _  = plot_2d_hist(obs_samples[:, obs_idx, 2:4], base_folder=os.path.join(plot_folder,  'true_snapshots'), data_name=f'middle_distribution_last_{obs_idx}', xlim=plot_xlim, ylim=plot_ylim, obs=None, marker_mode='')
       
    # define iterative trainer class (ISB) and train the model
    iterative_trainer = IterativeTrainer(config,
                                        base_folder=base_folder,
                                        plot_folder=plot_folder,
                                        init_dist=init_dist,
                                        fwd_drift_model_path=fwd_drift_model_path,
                                        bwd_drift_model_path=bwd_drift_model_path,
                                        init_drift_model_path=init_drift_model_path,
                                        device=device)
    iterative_trainer.train(obs_times, obs_samples, terminal_dist, rand_select, obs_ts, plot=config.train.plot, batch_size=config.train.batch_size, lr=config.train.lr, store_pickle=config.train.savepickle, report_data=report_data, all_data=all_data)

    print('Generating final results for fwd model...')
    # use trained model to generate final results
    with torch.no_grad():
        iterative_trainer.init_loop = False
        iterative_trainer.init_particle_filter_fwd()
        rev_obs = torch.flip(obs_samples, dims=[1])
        rev_times = torch.flip(obs_times, dims=[0])
        rev_times = []
        smooth_particles, _ , times_batched, _ = iterative_trainer.generate_smoothed_particles(rev_times, rev_obs, report_times=torch.flip(obs_times, dims=[0]), report_data=report_data)
        rev_times = torch.flip(obs_times, dims=[0])
        _, _, _, _, = iterative_trainer.generate_smoothed_particles(rev_times, rev_obs,  report_weights=True)

    def pickle_for_plots(base_folder, particles, obs_samples, obs_times, init_data=None, term_data=None):
        """Pickle some relevant objects."""
        pickle_folder = os.path.join(base_folder, 'pickles')
        if not os.path.isdir(pickle_folder):
            os.makedirs(pickle_folder)
        particle_file = os.path.join(pickle_folder, 'particles.pkl')
        obs_file = os.path.join(pickle_folder, 'obs_data.pkl')
        init_file = os.path.join(pickle_folder, 'init_data.pkl')
        term_file = os.path.join(pickle_folder, 'term_data.pkl')
        obs_samples_flat = torch.flatten(obs_samples, start_dim=0, end_dim=1).cpu().numpy()
        obs_times_batched = torch.flatten(obs_times.unsqueeze(0).repeat(obs_samples.shape[0], 1)).cpu().numpy()
        with open(particle_file, 'wb') as handle:
            pickle.dump(particles.cpu().numpy(), handle)
        with open(obs_file, 'wb') as handle:
            pickle.dump((obs_samples_flat, obs_times_batched), handle)
        
        if init_data is not None:
            with open(init_file, 'wb') as handle:
                pickle.dump(init_data.cpu().numpy(), handle)
        
        if term_data is not None:
            with open(term_file, 'wb') as handle:
                pickle.dump(term_data.cpu().numpy(), handle)

    # pickle relevant data and plot results
    max_time = config.filter.time_diff*(config.filter.n_steps - 1)
    pickle_for_plots(plot_folder, particles=smooth_particles, obs_samples=obs_samples, obs_times=max_time-obs_times, init_data=(data if dataset_name=='rna' else None), term_data=(term_data if dataset_name=='rna' else None))
    if config.dataset.n_dim[0] == 2:
        plot_problem_constraints(plot_folder, all_samples[:, 0], all_samples[:, -1], obs_samples, filename='problem_statement', xlim=plot_xlim, ylim=plot_ylim)
        plot_trajectory_video(plot_folder, config.filter.n_steps - rand_select - 1, torch.flip(obs_ts.cpu(), dims=[1]), smooth_particles, xlim=plot_xlim, ylim=plot_ylim, filename='final_particles', base_folder=base_folder, map_version=(dataset_name=='birds'))
    elif config.dataset.n_dim[0] == 1:
        plot_1d_trajectory(plot_folder, smooth_particles, data_name=dataset_name)
    elif config.dataset.n_dim[0] == 5:
        pca_12_particles = smooth_particles[:, :, :2]
        plot_trajectory_video(plot_folder, config.filter.n_steps - rand_select - 1, None, pca_12_particles, xlim=plot_xlim, ylim=plot_ylim, filename='final_particles', base_folder=base_folder, map_version=(dataset_name=='birds'))

 
if __name__ == '__main__':
    train_iterative()