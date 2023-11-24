import torch
import os
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

def plot_1d_dist(plot_folder, sample_times, means, covs, samples_true=None, samples=None, filename=None):
    if not os.path.isdir(plot_folder):
        os.makedirs(plot_folder)
    if samples is not None:
        plt.scatter(sample_times, samples)
    if samples_true is not None:
        plt.scatter(sample_times, samples_true, c='r')
    if filename is None:
        filename = 'dist.png'
    q_10 = np.zeros(means.shape)
    q_90 = np.zeros(means.shape)
    for i in range(len(sample_times)):
        var = covs[i]
        mu = means[i]
        [q_ij_10, q_ij_90] = stats.norm.interval(0.8, loc=mu, scale=np.sqrt(var))
        q_10[i] = q_ij_10
        q_90[i] = q_ij_90
    fig, ax = plt.subplots()
    ax.plot(sample_times, means)
    ax.fill_between(sample_times, q_10[:], q_90[:], color='b', alpha=.1)
    plt.savefig(os.path.join(plot_folder, filename))
    plt.clf()

def plot_1d_dist_samples(plot_folder, sample_times, samples, weights, filename=None):
    if not os.path.isdir(plot_folder):
        os.makedirs(plot_folder)

    if filename is None:
        filename = 'dist_from_samples.png'
    q_10 = np.zeros((len(sample_times), samples.shape[-1]))
    q_90 = np.zeros((len(sample_times), samples.shape[-1]))
    for i in range(len(sample_times)):
        q_10[i] = weighted_quantile(samples[i, :], quantiles=[0.10], sample_weight=weights[i])
        q_90[i] = weighted_quantile(samples[i, :], quantiles=[0.90], sample_weight=weights[i])
    means = np.sum(samples*weights, axis=1)
    fig, ax = plt.subplots()
    ax.plot(sample_times, means)
    ax.fill_between(sample_times, q_10[:, 0], q_90[:, 0], color='b', alpha=.1)
    plt.savefig(os.path.join(plot_folder, filename))
    plt.clf()

def plot_1d_hist(plot_folder, samples, n_bins=100):
    """Plot 1D histogram."""
    if not os.path.isdir(plot_folder):
        os.makedirs(plot_folder)
    plt.hist(samples, n_bins=n_bins)
    plt.savefig(os.path.join(plot_folder, '1d-hist.png'))
    plt.clf()


def plot_1d_trajectory(base_folder, particles, data_name):
    n_steps = particles.shape[0]
    times = np.linspace(0, (n_steps - 1)/100, n_steps)
    time_batched = np.stack([times]*particles.shape[1], axis=1)
    plt.scatter(time_batched, particles, alpha=0.1, s=2)
    filename = os.path.join(base_folder,  f'{data_name}_1D_trajectories.png')
    if not os.path.isdir(base_folder):
        os.makedirs(base_folder)
    plt.savefig(filename)
    plt.clf()


def weighted_quantile(values, quantiles, sample_weight=None, 
                      values_sorted=False):
    """ 
    Stolen code from stackoverflow, seems to work though.
    Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)