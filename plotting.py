import matplotlib
import numpy as np
import pandas as pd
from collections import namedtuple
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import matplotlib as mpl
label_size = 12
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'


def plot_episode_stats(episode_lengths,episode_rewards,fname, smoothing_window=25, noshow=False):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10,7))
    plt.plot(pd.Series(episode_lengths).rolling(smoothing_window, min_periods=smoothing_window).mean())
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time", y=1.04)
    plt.grid(True)
    plt.savefig("./{}_epi_length.png".format(fname),dpi = 150, bbox_inches='tight')
    if noshow:
        plt.close(fig1)
    else:
        plt.show(fig1)

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10,7))
    rewards_smoothed = pd.Series(episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window), y=1.04)
    plt.grid(True)
    plt.savefig("./{}_epi_reward.png".format(fname),dpi = 150, bbox_inches='tight')
    if noshow:
        plt.close(fig2)
    else:
        plt.show(fig2)


    return fig1, fig2




fname = "MC_tile_3"
path = 'tile'
episode_lengths = np.load("./{}/iterations/{}.npy".format(tile,fname))
episode_rewards = np.load("./{}/rewards/{}.npy".format(tile, fname))

plot_episode_stats(episode_lengths,episode_rewards,fname)