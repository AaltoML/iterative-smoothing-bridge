"""Torchvision dataset loaders."""

from torchvision import datasets, transforms
import torch
import numpy as np
import os
import ndjson
import pandas as pd


def cycle(iterable):
    while True:
        for x in iterable:
            yield x[0]


def save_moments(obs_times, means, vars, folder, model_type, task):
    """Store smoothing/filtering moments in df, and save it as csv."""
    df = pd.DataFrame(data={'times': obs_times.numpy()})

    df['means'] = list(means.numpy())
    df['vars'] = list(vars.numpy())
    data_folder = os.path.join(folder, 'data', f'{model_type}_ssm')
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)
    df.to_csv(os.path.join(data_folder, f'moments_{task}.csv'), index=False)


def get_mnist_loader(folder, class_names, batch_size):
    dataset = datasets.MNIST(
        root=os.path.join(folder, 'data', 'MNIST'),
        train=True,
        transform=transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.CenterCrop((28, 28)),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32)
        ]),
        download=True)
    selected_idxs = [dataset.class_to_idx[class_name] for class_name in class_names]
    tensor_targets = torch.tensor(dataset.targets)
    class_indices = sum(tensor_targets==i for i in selected_idxs).bool()
   # class_indices = torch.tensor(dataset.targets) in selected_idxs
    new_dataset = torch.utils.data.dataset.Subset(dataset, np.where(class_indices==1)[0])
    loader = torch.utils.data.DataLoader(new_dataset, batch_size=batch_size)
    return iter(cycle(loader))


def generate_blocked_mnist(n_dim, n_images, device='cpu'):
    """Generate images with upper half white, lower half black."""
    block_imgs = torch.zeros(*n_dim, device=device, dtype=torch.float32).unsqueeze(0).repeat(n_images, 1, 1, 1)
    block_imgs = block_imgs.unsqueeze(1)
    half_size = int(n_dim[1]/2)
    block_imgs[:, :, :, :,  :half_size] = 1
    return block_imgs

def get_rna_data(base_folder, config, n_init_points=1000,  device='cpu'):
    """Load RNA data initial and terminal distributions and observational data."""
    data_folder = os.path.join(base_folder, 'data', 'processed_rna')
    init_data = torch.load(os.path.join(data_folder, 'init_dist.pkl')).to(device)
    term_data = torch.load(os.path.join(data_folder, 'terminal_dist.pkl')).to(device)
    obs_samples_in = torch.load(os.path.join(data_folder, 'obs_samples.pkl')).to(device)
    obs_times_in = torch.load(os.path.join(data_folder, 'obs_times.pkl')).to(device)

    chng_points = []
    for i in range(obs_samples_in.shape[0] - 1):
        if obs_times_in[i] != obs_times_in[i+1]:
            chng_points.append(i)
    chng_1 = chng_points[0]
    chng_2 = chng_points[1]
    chng_3 = obs_samples_in.shape[0] - chng_points[1]
    
    # TODO: implement a varying number of observations
    n_ts = config.filter.n_ts
    assert n_ts < min(chng_1, chng_2 - chng_1, chng_3)
    obs_samples = torch.empty((n_ts,3, obs_samples_in.shape[-1]), device=device)
    obs_samples[:, 0] = obs_samples_in[:n_ts]
    obs_samples[:, 1] = obs_samples_in[chng_1: chng_1+n_ts]
    obs_samples[:, 2] = obs_samples_in[chng_2: chng_2+n_ts]
    report_data = dict()
    report_data[1.0] = obs_samples_in[:chng_1]
    report_data[2.0] = obs_samples_in[chng_1:chng_2]
    report_data[3.0] = obs_samples_in[chng_2:]
    obs_times = torch.unique(obs_times_in).float()

    dataset = torch.utils.data.TensorDataset(init_data)
    init_loader = torch.utils.data.DataLoader(dataset, batch_size=n_init_points, shuffle=True)
    rand_select = np.array([int(x*100) - 1 for x in obs_times])

    n_steps = config.filter.n_steps
    time_forward = config.filter.time_forward
    obs_ts = None

    if not time_forward:
        rand_select = n_steps - 1 - rand_select
        obs_times = torch.flip(obs_times, dims=[0])
        obs_samples = torch.flip(obs_samples, dims=[1])

    all_data = torch.cat([init_data, obs_samples_in, term_data], dim=0)
    return iter(cycle(init_loader)), term_data, obs_samples, obs_times, obs_ts, rand_select, report_data, all_data