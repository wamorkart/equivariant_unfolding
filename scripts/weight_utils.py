import h5py
import torch
import torch.nn.functional as F
from torchinfo import summary
from sklearn.model_selection import train_test_split

import numpy as np
import uproot
import awkward as ak
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
import pandas as pd
import matplotlib.gridspec as gs

def plot_reweighting(
        source_data, 
        target_data, 
        source_weight_start, 
        source_weight_end=None,
        target_weight=None, 
        bins=150, 
        xlabel="", 
        linear_scale=False, 
        ylim=None, 
        rlim=None,
        names=None):
    """ plot_reweighting - This function will plot the quality of the reweighting
    along a given dimension. Inputs are the original data, the starting weights, the ending weights, and 
    the target data. Optional is the bins to be used in the plot, and the x-axis label.

    Arguments:
    source_data - numpy array of original data
    target_data - numpy array of target data
    source_weight_start - numpy array of starting weights for source, required
    source_weights_end - numpy array of ending weights for source. Optional if there are none
    target_weight - numpy array of weights for target, optional
    bins - number of bins to use in the plot, if not set use mpl default w/ 150 bins
    xlabel - label to use for the x-axis
    linear_scale - if true, use linear scale for y-axis, otherwise use log scale
    ylim - if set, use this as the y-axis limits
    rlim - if set, use this as the ratio axis limits
    names - an optional tuple of strings which sets the legends names for source and target data

    Returns:
    fig - matplotlib figure object
    """

    # Parse names
    if names is not None:
        name1, name2 = names
    else:
        name1 = 'MC'
        name2 = 'PseudoData'

    fig = plt.figure()
    ax, axr = add_ratios(fig)
    n_mc, bins, patches =  ax.hist(source_data, bins=bins, label=name1, density=True, alpha=0.5, weights=source_weight_start)
    if source_weight_end is not None:
        n_rw, bins, patches = ax.hist(source_data, bins=bins, label='Reweighted', density=True, histtype='step', color='black', weights=source_weight_end)
        print (n_rw)
    if target_weight is None:
        target_weight = np.ones_like(target_data)
    n_pd, bins, patches = ax.hist(target_data, bins=bins, label=name2, density=True, alpha=0.5, weights=target_weight)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_ylabel('A.U.')
    if linear_scale:
        ax.set_yscale('linear')
    else:
        ax.set_yscale('log')
    ax.legend()

    axr.hlines(1, bins[0], bins[-1], color='k', linestyle='--', alpha=0.8)
    axr.plot(bins[:-1], n_mc / n_pd, color='#1f77b4', drawstyle='steps-post')
    if source_weight_end is not None:
        axr.plot(bins[:-1], n_rw / n_pd, color='k', drawstyle='steps-post')
    axr.set_ylabel(f'{name1}/{name2}')
    axr.set_xlim(ax.get_xlim())
    axr.set_xlabel(xlabel)
    if rlim is not None:
        axr.set_ylim(rlim)
    return fig
    
def add_ratios(fig):
    """ add_ratios - This function adds ratio pads to a given matplotlib figure.

    Arguments:
    fig - matplotlib axis to add ratio pads to

    Returns:
    ax - main matplotlib axis
    axr - ratio matplotlib axis
    """

    this_grid = gs.GridSpec(2, 1, figure=fig, height_ratios=(7, 2), hspace=0.0)

    axr = fig.add_subplot(this_grid[1,0])
    ax = fig.add_subplot(this_grid[0,0])

    return ax, axr


def getweight_pelican(inputfile):
    prediction=torch.load(inputfile,"cpu")
    logits_MC = prediction['predict'][:, 1]
    logits_PD = prediction['predict'][:, 0]
    ## convert logits into probability by passing through the sigmoid function (essentially doing: prob = 1 / (1 + np.exp(-logit)) )
    prob_MC = torch.sigmoid(logits_MC)
    prob_PD = torch.sigmoid(logits_PD)
    weight = np.array(prob_MC/(1-prob_MC))


    # weight = np.array(prob_MC)
    # print(weight)
    # print(probs)
    # print(prob_MC)
    # print(weight.shape)


    # weight = np.array(prob_MC/prob_PD)
    targets = prediction["targets"]
    weight = weight[targets==1]
    # print(weight)
    # print(weight.shape)
    return prob_MC, prob_PD, weight

def getweight_LN(inputfile):
    prediction=np.load(inputfile)
    labels=prediction[:, 0]
    prob_PD=prediction[:,1]
    prob_MC=prediction[:,2] 
    weight=np.array(prob_MC/(1-prob_MC))
    weight=weight[labels==1]
    return prob_MC, prob_PD, weight

def originalweights(inputfile):
    inputfile=h5py.File(inputfile,"r")
    weight=np.array(inputfile["weight"])
    is_signal=np.array(inputfile["is_signal"])
    weight_original=weight[is_signal==1]
    return weight_original
