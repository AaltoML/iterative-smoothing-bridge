import os
import pandas as pd
import torch
import numpy as np


class TensorSampler:
    def __init__(self, tensor):
        self.tensor = tensor
        self.size = self.tensor.shape[0]
        probs = torch.ones(self.size)/self.size
        self.categorical = torch.distributions.Categorical(probs=probs)

    def sample(self, n_samples):
        indices = self.categorical.sample(n_samples)
        output = self.tensor[indices]
        return output


def get_ssm_data(folder, model_type, device, n_steps=100):
    """Retrieve SSM data df."""
    data_folder = os.path.join(folder, 'data', model_type)
    data = pd.read_csv(os.path.join(data_folder, 'obs.csv'))
    obs_times = torch.Tensor(np.array(data.times)).to(device)
    n_dims = data.shape[1] - 1
    obs_samples = torch.empty((data.shape[0], n_dims), device=device, dtype=torch.float64)
    for i in range(n_dims):
        obs_samples[:, i] = torch.Tensor(np.array(data[f'obs_{i}'])).to(device)

    obs_samples = obs_samples.reshape( -1, n_steps - 1, obs_samples.shape[-1])
    obs_times = obs_times.reshape(-1, n_steps - 1)
    return obs_times.double(), obs_samples.double()


def prep_ssm_data(config, all_times, all_samples):
    n_steps = config.filter.n_steps
    obs_times = all_times[:config.filter.n_ts] # select the first timeseries
    obs_ts = all_samples[:config.filter.n_ts]

    n_obs_total = config.filter.n_obs*config.filter.n_subs
    grp_size = int(config.filter.n_ts/config.filter.n_subs)

    rand_select = torch.randperm(n_steps - 2)[:n_obs_total] + 1
  #  rand_select = torch.Tensor([25, 50, 75]).long()
    obs_times = obs_times[0, rand_select]   # assuming that the obs times match for all time series
    obs_samples = obs_ts[:, rand_select]
    obs_samples1 = obs_samples[:grp_size, ::config.filter.n_subs]
    obs_samples2 = obs_samples[grp_size:, 1::config.filter.n_subs]
    if obs_samples2.shape[0] != 0:
        obs_samples = torch.cat([obs_samples1, obs_samples2], dim=1)

    # sort by times
    obs_times, order_idx = torch.sort(obs_times)
    obs_samples = obs_samples[:, order_idx]

    if config.filter.time_forward:
        rand_select = n_steps - 1 - rand_select
        obs_ts = torch.flip(obs_ts, dims=[1])
    else:
        obs_times = torch.flip(obs_times, dims=[0])
        obs_samples = torch.flip(obs_samples, dims=[1])
        
    return obs_ts, obs_times, obs_samples, rand_select

def save_ssm_data(obs_times, obs_samples, folder, model_type):
    """Store SSM data in a data frame, and save as csv."""
    df = pd.DataFrame(data={'times': obs_times.numpy()})
    for i in range(obs_samples.shape[1]):
        df[f'obs_{i}'] = obs_samples[:, i].numpy()
    data_folder = os.path.join(folder, 'data', model_type)
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)
    df.to_csv(os.path.join(data_folder, 'obs.csv'), index=False)


def get_ssm_moments(folder, model_type, device, task):
    """Retrieve SSM solved moments df."""
    data_folder = os.path.join(folder, 'data', f'{model_type}_ssm')
    data = pd.read_csv(os.path.join(data_folder, f'moments_{task}.csv'))
    obs_times = torch.Tensor(np.array(data.times), device=device)
    data['means'] = data['means'].str.strip('[]').str.split(' ')
    means = torch.Tensor([[float(x) for x in arr] for arr in data['means'].values])
    data['vars'] = data['vars'].str.replace('[', '').str.replace(']', '').str.split('\n ')
    vars = torch.Tensor([[[float(val) for val in arr.split(' ') if len(val) > 0] for arr in matrix] for matrix in data['vars'].values])
    return obs_times, means, vars


def save_moments(obs_times, means, vars, folder, model_type, task):
    """Store smoothing/filtering moments in df, and save it as csv."""
    df = pd.DataFrame(data={'times': obs_times.numpy()})

    df['means'] = list(means.numpy())
    df['vars'] = list(vars.numpy())
    data_folder = os.path.join(folder, 'data', f'{model_type}_ssm')
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)
    df.to_csv(os.path.join(data_folder, f'moments_{task}.csv'), index=False)