"""Solve the Schrödinger bridge problem on flat data."""


import sys
import torch
import pandas as pd
import os
from pathlib import Path
import time
from isb.data_loaders import load_toy_data
from isb.plot import plot_2d_hist, plot_as_video
from isb.iftp import IFTP
import hydra


@hydra.main(config_path="../../configs/bridge", config_name="circle2d")
def run_training(config):
    # the cwd is hydra's output folder, need to go 3 levels up
    cwd = os.getcwd()
    if cwd[-7:] != '-bridge' :
        base_folder = Path(os.getcwd()).parent.parent.parent 
    else:
        base_folder = cwd   # 

    torch.manual_seed(42)
    data_name = config.dataset.dataset_name
    config.dataset.n_dim = tuple(eval(config.dataset.n_dim))   # evaluate as tuple
    plot_xlim = [config.dataset.plot_x_min, config.dataset.plot_x_max]
    plot_ylim = [config.dataset.plot_y_min, config.dataset.plot_y_max]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    if torch.cuda.is_available():
        print(f'Device: {torch.cuda.get_device_name(0)}')

    # get data
    loaders, _ = load_toy_data([data_name], batch_size=config.train.cache_size, device=device)
    loader = loaders[data_name]
    init_loader = None
    data = next(loader)

    # get model paths
    forward_output = os.path.join(base_folder, 'bridge_models', 'forward_drift',  config.model.forward_name)
    backward_output = os.path.join(base_folder, 'bridge_models', 'backward_drift', config.model.backward_name)

    # define model and prior based on data
    iftp = IFTP(config=config,
                data=data,
                device=device,
                forward_path=forward_output,
                backward_path=backward_output,
                save_models=True,
                init_loader=init_loader,
                load_models=False).to(device)


    # plot true distribution for reference
    plot_folder = os.path.join(base_folder, 'plots', data_name, time.strftime("%Y-%m-%d"), time.strftime("%H-%M"))
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    plot_xlim = [config.dataset.plot_x_min, config.dataset.plot_x_max]
    plot_ylim = [config.dataset.plot_y_min, config.dataset.plot_y_max]
    _ = plot_2d_hist(data[:config.train.nn_refresh], base_folder=plot_folder, data_name=f'{data_name}_true_samples', time_in_name=False, xlim=plot_xlim, ylim=plot_ylim)

    # iftp outer loop
    for i in range(config.train.iftp_epochs):
        first_iter = (i == 0)
        init_samples = iftp.get_init_samples(data[:10000], 1, first_iter=first_iter)
        _ = plot_2d_hist(
                init_samples, base_folder=os.path.join(plot_folder,'forward_pass'),  data_name=f'{data_name}_forward_samples_{i}', xlim=plot_xlim, ylim=plot_ylim)
        iftp.forward_backward(loader, n_traj=config.train.n_trajectories, first_iter=first_iter)
        samples = iftp.get_final_samples(n_trajectories=1)
        _ = plot_2d_hist(
                samples, base_folder=os.path.join(plot_folder,'backward_pass'), data_name=f'{data_name}_fit_IFTP_epoch_{i}', xlim=plot_xlim, ylim=plot_ylim)

    # plot final forward trajectory
    init_samples = iftp.get_init_samples(data[:10000], 1, first_iter=False)
    _ = plot_2d_hist(
                init_samples, base_folder=os.path.join(plot_folder,'forward_pass'),  data_name=f'{data_name}_forward_samples_final', xlim=plot_xlim, ylim=plot_ylim)

    # plot the trajectories
    init_data = iftp.sample_init_dist(10000)
    trajectory_samples, _, times_batched, _ = iftp.generate_backward_trajectories(init_data)
    plot_files = []
    for i in range(trajectory_samples.shape[1]):
        sample = trajectory_samples[:, i]
        img_filename = plot_2d_hist(sample, base_folder=os.path.join(plot_folder,  'trajectories'), data_name=f'{data_name}_final_fit_trajectory_step_{i}', xlim=plot_xlim, ylim=plot_ylim)
        plot_files.append(img_filename)

    # combine to gif
    plot_as_video(os.path.join(plot_folder, 'videos'), plot_files, 'final_trajectory')

    # save the trajectory data (can be used as observations)
    df = pd.DataFrame(data={'times': times_batched.flatten()})
    for i in range(trajectory_samples.shape[2]):
        df[f'obs_{i}'] = trajectory_samples[:, :, i].flatten().detach().numpy()
    data_folder = os.path.join(base_folder, 'data', f'{data_name}_bridge')
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)
    df.to_csv(os.path.join(data_folder, 'obs.csv'), index=False)


if __name__ == '__main__':
    run_training()
