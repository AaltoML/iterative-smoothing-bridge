"""Create 2D density plots."""
import matplotlib.pyplot as plt
from matplotlib import image
from pylab import text
import time
import os
import torch
import numpy as np
import cv2     #TODO: Fix OS X issue Segmentation fault 11


def plot_2d_hist(data, base_folder, data_name, time_in_name=True, xlim=None, ylim=None, obs=None, marker_mode='yellow', weights=None):
    """Create 2D histogram plot."""
    if time_in_name:
        filename = os.path.join(base_folder,  f'{data_name}_2D_hist_{time.time()}.png')
    else:
        filename = os.path.join(base_folder,  f'{data_name}_2D_hist.png')
    if not os.path.isdir(base_folder):
        os.makedirs(base_folder)
    np_data = data.cpu().detach().numpy()
    if xlim is not None and ylim is not None:
        ranges = np.array(np.stack([xlim, ylim]))
    else:
        ranges = None
    plt.hist2d(np_data[:, 0], np_data[:, 1], bins=(100, 100), cmap=plt.cm.jet,  range=ranges, weights=weights)

    if obs is not None:
        if marker_mode == 'yellow':
            marker = '+'
            color = 'yellow'
            display_text = 'Observation time'
        else:
            marker = 'o'
            color = 'red'
            display_text = ''
        for obs_i in obs:
            plt.plot(obs_i[0], obs_i[1],marker=marker, markersize='50', markeredgecolor=color, markerfacecolor='none', fillstyle='none') 
        text(obs[0, 0], obs[0, 1],display_text,
            fontsize=15,
            color='yellow',
            horizontalalignment='center',
            verticalalignment='center')
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()
    return filename


def plot_as_video(base_folder, images, video_name):
    img_array = []
    video_file_name = os.path.join(base_folder, f'{video_name}_{time.time()}.mp4')
    for filename in images:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    if not os.path.isdir(base_folder):
        os.makedirs(base_folder)
    out = cv2.VideoWriter(video_file_name, fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=5, frameSize=size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    cv2.destroyAllWindows()
    out.release()
    print(f'Video written to file {video_file_name}')


def plot_scatter(base_folder, x, y, data_name):
    filename = os.path.join(base_folder,  f'{data_name}_scatter.png')
    if not os.path.isdir(base_folder):
        os.makedirs(base_folder)
    ndim = y.shape[1]
    if ndim > 1:
        fig, axs = plt.subplots(ndim)
        for i in range(ndim):
            axs[i].set_title(f'Dimension {i}')
            axs[i].scatter(x, y[:, i])
    else:
        plt.scatter(x, y)
    print(f'Saving scatter plot to file {filename}')
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()
    return filename


def plot_trajectory_paths(plot_folder, rand_select, obs_ts, particles,  xlim=None, ylim=None, filename='', map_version=False, base_folder=None):
    """Plot trajectory paths as video."""
    plot_files = []
    for i in range(particles.shape[0] - 1):
        if i in rand_select:
            obs = obs_ts[:, i]
        else:
            obs = None
        sample = particles[:i]  # using all the particles
        if not map_version:
           return    # not implemented! but also not distruptive
        else:
            img_filename = plot_europe(base_folder, plot_folder, sample, file_name=f'{filename}_{i}', xlim=xlim, ylim=ylim, obs=obs if i in rand_select else None, plot_type='paths')
        plot_files.append(img_filename)
    plot_as_video(os.path.join(plot_folder, 'videos'), plot_files, f'paths_{filename}')
    for file in plot_files: # cleanup
        os.remove(file)


def plot_problem_constraints(plot_folder, init_pts, term_pts, obs_samples, filename, xlim, ylim):
    """Plot the initial and terminal distributions and the observations on a 2d plane."""

    # plot initial and terminal distributions
    plt.scatter(init_pts[:, 0], init_pts[:, 1], alpha=0.1, color='cadetblue')
    plt.scatter(term_pts[:, 0], term_pts[:, 1], alpha=0.1, color='lightsalmon')

    # plot observations
    obs_samples_flat = torch.flatten(obs_samples, start_dim=0, end_dim=1).cpu().numpy()
    plt.scatter(obs_samples_flat[:, 0], obs_samples_flat[:, 1], color='black', alpha=0.9, marker='o', s=100)

    plt.xlim = xlim
    plt.ylim = ylim

    if not os.path.isdir(plot_folder):
        os.makedirs(plot_folder)
    plt.savefig(os.path.join(plot_folder, filename), bbox_inches='tight')
    plt.axis('off')
    plt.clf()


def plot_trajectory_3d(plot_folder, particles, obs_times, obs_samples, filename, xlim, ylim):
    """Plot trajectories as a 3D plot."""
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(projection='3d')
    time_dist = 15

    # Points start
    particle_init = particles[0]
    y0 = np.zeros((particles.shape[1], 1))
    ax.scatter(particle_init[:,0], y0, particle_init[:,1], alpha=.1)

    # Draw axes #1
    ax.plot3D(xlim,[0, 0],[0,0],'black')
    ax.plot3D([0, 0],[0,0],ylim,'black')

    # Draw axes #2
    ax.plot3D(xlim,[time_dist, time_dist],[0,0],'black')
    ax.plot3D([0, 0],[time_dist,time_dist],ylim,'black')

    # Points end
    particle_term = particles[-1]
    y = y0 + time_dist
    ax.scatter(particle_term[:,0], y, particle_term[:,1], alpha=.1)

    # Connect with lines
    n_lines = 20
    t = np.linspace(0,time_dist,num=100)
    for i in range(n_lines):
        linedata = [particles[:, i, 0], t, particles[:, i, 1]]
        ax.plot3D(linedata[0],linedata[1],linedata[2], color='green')

    # add observations
    obs_samples_flat = torch.flatten(obs_samples, start_dim=0, end_dim=1).cpu().numpy()
    obs_times_batched = time_dist * torch.flatten(obs_times.unsqueeze(0).repeat(obs_samples.shape[0], 1)).cpu().numpy()
    ax.scatter(obs_samples_flat[:, 0], obs_times_batched, obs_samples_flat[:, 1], marker='o', color='black')

    # This is the view angle
    ax.view_init(20, -40)
    plt.axis('off')
    if not os.path.isdir(plot_folder):
        os.makedirs(plot_folder)
    plt.savefig(os.path.join(plot_folder, filename), bbox_inches='tight')
    plt.clf()

def plot_trajectory_video(plot_folder, rand_select, obs_ts=None, particles=None, xlim=None, ylim=None, filename='', weights=None, map_version=False, base_folder=None):
    """Very specific plotting."""
    plot_files = []
    for i in range(particles.shape[0] - 1):
        if i in rand_select:
            marker_mode = 'yellow'
        else:
            marker_mode = 'red'
        if obs_ts is not None:
            obs = obs_ts[:, i]
        else:
            obs = None
        sample = particles[i]
        if weights is not None:
            plot_weights = weights[i]
        else:
            plot_weights = None
        if not map_version:
            img_filename = plot_2d_hist(sample, base_folder=os.path.join(plot_folder,  'trajectories'), data_name=f'{filename}_{i}', xlim=xlim, ylim=ylim, obs=obs if i in rand_select else None, marker_mode=marker_mode, weights=plot_weights)
        else:
            img_filename = plot_europe(base_folder, plot_folder, sample, file_name=f'{filename}_{i}', xlim=xlim, ylim=ylim, obs=obs if i in rand_select else None)
        plot_files.append(img_filename)
    plot_as_video(os.path.join(plot_folder, 'videos'), plot_files, f'final_trajectory_{filename}')
    for file in plot_files: # cleanup
        os.remove(file)

