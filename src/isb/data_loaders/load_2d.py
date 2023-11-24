"""Load data for experiments."""
from torch.utils.data import TensorDataset, Dataset
import torch
import numpy as np
from sklearn import datasets
from matplotlib import image
import os
import pandas as pd


def cycle(iterable):
    while True:
        for x in iterable:
            yield x[0]


def load_toy_data(data_names, batch_size, device='cpu', offset=0.5):
    """Load all listed datasets."""
    loaders = dict()
    dataset_dict = dict()
    for data_name in data_names:
        if data_name == 'moons':
            data, _ = datasets.make_moons(n_samples=100000,
                                           noise=0.01)
            data = data *7  # scaling used in De Bortoli et al.
        elif data_name =='moons_offset':
            data, _ = datasets.make_moons(n_samples=100000,
                                        noise=0.01)
            data = data*7
            data = data + offset
        elif data_name == 'circles':
            data, _ = datasets.make_circles(n_samples=100000,
                                               noise=0.01)
            data = data * 10    # scaling used in De Bortoli et al.
        elif data_name == 's_shape':
            data, _ = datasets.make_s_curve(n_samples=10000,
                                            noise=0.01)
            data = data[:,[0,2]]*3 # use first and last dimension only
        elif data_name == 'two_gaussians':
            mean_1 = np.array([3, 6])
            mean_2 = np.array([3, -6])
            data_1 = mean_1 + 2*np.random.randn(5000, 2)
            data_2 = mean_2 + 2*np.random.randn(5000, 2)
            data = np.concatenate([data_1, data_2], axis=0)
        elif data_name == 'single_gaussian':       
            mean_1 = np.array([10, 0])
            data = mean_1 + np.random.randn(10000, 2)
        dataset = TensorDataset(torch.Tensor(data, device=device).double())
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        loaders[data_name] = iter(cycle(loader))
        dataset_dict[data_name] = dataset
    return loaders, dataset_dict


def create_2d_obs_data(data, config, device='cpu'):
    """Create the obs datasets from a pandas dataframe."""
    n_steps = config.filter.n_steps
    time_diff = config.filter.time_diff
    time_forward = config.filter.time_forward
    data = data.iloc[:, :3]
    obs_times = torch.Tensor(np.array(data.times)).to(device)
    n_dims = data.shape[1] - 1
    obs_samples = torch.empty((data.shape[0], n_dims), device=device, dtype=torch.float64)
    for i in range(n_dims):
        data[f'obs_{i}'] = data[f'obs_{i}'].str.replace(',', '.').astype(float)
        obs_samples[:, i] = torch.Tensor(np.array(data[f'obs_{i}'])).to(device)

    obs_samples = obs_samples.reshape(config.filter.n_ts, -1, obs_samples.shape[-1])
    obs_times = obs_times.reshape(config.filter.n_ts, -1)
    obs_times = obs_times[0]

    obs_times, index_sort = torch.sort(obs_times)
    obs_samples = obs_samples[:, index_sort]
    obs_ts = torch.zeros((config.filter.n_ts, n_steps, n_dims), device=device)
    time_max = n_steps*time_diff - time_diff
    time_linspace = torch.linspace(0, time_max, n_steps).to(device) 
    obs_idx = 0
    rand_select = []
    for i, t in enumerate(time_linspace[:-1]):
        t_next = time_linspace[i+1]
        time_match = any([t_obs < t_next and t_obs >=t for t_obs in obs_times])
        if time_match:
            rand_select.append(i)
            obs_ts[:, i] = obs_samples[:, obs_idx]
            obs_idx += 1
    
    rand_select = torch.Tensor(rand_select).to(device)
    if not time_forward:
        rand_select = n_steps - 1 - rand_select
        obs_times = torch.flip(obs_times, dims=[0])
        obs_samples = torch.flip(obs_samples, dims=[1])
        obs_ts = torch.flip(obs_ts, dims=[1])
    return obs_ts, obs_samples, obs_times, rand_select

def read_bridge_observations(base_folder,config, device='cpu'):
    """Read bridge data for Gaussian transport."""
    obs_filename = os.path.join(base_folder, 'data', 'one_gaussian', 'observations', 'bridge.csv')
    data = pd.read_csv(obs_filename, header=0, sep=';')
    return create_2d_obs_data(data, config, device)
