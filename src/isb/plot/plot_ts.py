"""Plotting functions for time series."""

import matplotlib.pyplot as plt
import os
import time

def plot_ts_graphs(data, base_folder, data_name):
    filename = os.path.join(base_folder, 'plots', f'{data_name}_2D_hist_{time.time()}.png')
    np_data = data.detach().numpy()
    for i in range(data.shape[0]):
        plt.plot(np_data[i, :])
    print(f'Saving sample paths to file {filename}')
    plt.savefig(filename, bbox_inches='tight')