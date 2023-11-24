"""Fit a Schrödinger bridge model to image datasets."""
import sys
import torch
import os
from pathlib import Path
from isb.data_loaders import get_mnist_loader
from isb.plot import plot_img, plot_img_grid, plot_as_video
import time
from isb.iftp import IFTP
import hydra


@hydra.main(config_path="../../configs/bridge", config_name="mnist_bortoli")
def run_training(config):
    cwd = os.getcwd()
    if cwd[-7:] != '-bridge' :
        base_folder = Path(os.getcwd()).parent.parent.parent 
    else:
        base_folder = cwd   # 
    data_name = config.dataset.dataset_name
    data_subset = config.dataset.data_subset
    config.dataset.n_dim = tuple(eval(config.dataset.n_dim))   # evaluate as tuple
    config.model.attn_resolutions = eval(config.model.attn_resolutions)
    config.model.ch_mult = eval(config.model.ch_mult)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    if torch.cuda.is_available():
        print(f'Device: {torch.cuda.get_device_name(0)}')

    # get data
    if data_name == 'mnist':
        class_names = [data_subset]
        loader = get_mnist_loader(base_folder, class_names=class_names, batch_size=config.train.cache_size)
    else:
        raise NotImplementedError(f'Dataset {data_name} not recognized')
    data = next(loader).to(device)

    # get model paths
    forward_output = os.path.join(base_folder,'bridge_models', 'forward_drift',  config.model.forward_name)
    backward_output = os.path.join(base_folder, 'bridge_models', 'backward_drift', config.model.backward_name)

    # define model and prior based on data
    iftp = IFTP(config=config,
                data=data,
                device=device,
                forward_path=forward_output,
                backward_path=backward_output).to(device)


    # plot true distribution for reference
    plot_folder = os.path.join(base_folder, 'plots', f'iftp_{data_name}', time.strftime("%Y-%m-%d"), time.strftime("%H-%M"))
    plot_img_grid(data[:25], base_folder=plot_folder, data_name=f'{data_name}-{data_subset}-true_sample', nrows=5)

    # iftp outer loop
    for i in range(config.train.iftp_epochs):
        first_iter = (i == 0)
        init_samples = iftp.get_init_samples(data, 1, first_iter=first_iter)
        plot_img(
                init_samples[0], base_folder=os.path.join(plot_folder, 'forward_pass'), data_name=f'{data_name}-{data_subset}_forward_samples_{i}', time_in_name=False)
        iftp.forward_backward(loader, n_traj=config.train.n_trajectories, first_iter=first_iter)
        samples = iftp.get_final_samples(n_trajectories=1)
        plot_img_grid(samples[:25], base_folder=os.path.join(plot_folder,'backward_pass'), data_name=f'{data_name}-{data_subset}_fit_IFTP_loop_{i}', nrows=5)

         # plot the trajectories
        with torch.no_grad():
            init_data = iftp.sample_init_dist(9)
            trajectory_samples, _, _ , _ = iftp.generate_backward_trajectories(init_data)
            plot_files = []
            for j in range(trajectory_samples.shape[1]):
                sample = trajectory_samples[:, j]
                img_filename = plot_img_grid(sample, base_folder=os.path.join(plot_folder,  'trajectories'), data_name=f'{data_subset}_final_fit_trajectory_step_{j}', nrows=3)
                plot_files.append(img_filename)

            print(plot_files)
            # combine to gif
            plot_as_video(os.path.join(plot_folder, 'videos'), plot_files, f'image_trajectory_{i}')


if __name__ == '__main__':
    print('Job started')
    run_training()
