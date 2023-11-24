import os
import time
import torchvision
from isb.plot.plot_2D import plot_as_video


def plot_img(img_data, base_folder, data_name, time_in_name=True, grayscale=False):
    if not os.path.isdir(base_folder):
        os.makedirs(base_folder)
    if time_in_name:
        filename = os.path.join(base_folder,  f'{data_name}_image_{time.time()}.png')
    else:
        filename = os.path.join(base_folder,  f'{data_name}_image.png')
    print(f'Saving Image to file {filename}')
    if not grayscale:
        torchvision.utils.save_image(img_data, filename)
    else:
        img_data_with_channel = img_data.reshape((1, *img_data.shape))
        torchvision.utils.save_image(img_data_with_channel, filename)
    return filename


def plot_img_grid(img_datas, base_folder, data_name, nrows=5, grayscale=False):
    if not os.path.isdir(base_folder):
        os.makedirs(base_folder)
    filename = os.path.join(base_folder,  f'{data_name}_image_grid.png')
    if not grayscale:   
        grid_img = torchvision.utils.make_grid(img_datas, nrow=nrows)
        torchvision.utils.save_image(grid_img, filename)
    else:
        img_datas_with_channel = [x.reshape((1, *x.shape)) for x in img_datas]
        grid_img = torchvision.utils.make_grid(img_datas_with_channel, nrow=nrows)
        torchvision.utils.save_image(grid_img, filename)
    return filename


def plot_trajectory_video_img(plot_folder, trajectory_samples, filename=None):
    plot_files = []
    if filename is None:
        filename = 'image_trajectory'
    for j in range(trajectory_samples.shape[0]):
        sample = trajectory_samples[j]
        img_filename = plot_img_grid(sample,
                                     base_folder=os.path.join(plot_folder,  'trajectories'),
                                     data_name=f'trajectory_step_{j}', nrows=3)
        plot_files.append(img_filename)
    plot_as_video(os.path.join(plot_folder, 'video'), plot_files, filename)
    for file in plot_files:
        os.remove(file)
