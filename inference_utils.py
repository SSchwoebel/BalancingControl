import torch

torch.set_num_threads(1)
print("torch threads", torch.get_num_threads())


import pyro
import pyro.distributions as dist
import world
import environment as env
import agent as agt
import perception as prc
import action_selection as asl
import inference as inf

import itertools
import matplotlib.pylab as plt
from matplotlib.animation import FuncAnimation
from multiprocessing import Pool
from matplotlib.colors import LinearSegmentedColormap
import jsonpickle as pickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import json
import seaborn as sns
import pandas as pd
import os
import glob
import scipy as sc
import scipy.signal as ss
from scipy.stats import pearsonr
import gc
import sys
from numpy import eye
from statsmodels.stats.multitest import multipletests
from scipy.io import loadmat

###################################
"""inference convenience functions"""

def infer(inferrer, iter_steps, fname_str, npart, base_dir):

    inferrer.infer_posterior(iter_steps=iter_steps, num_particles=npart, optim_kwargs={'lr': .01})#, param_dict

    storage_name = os.path.join(base_dir, fname_str+'.save')#h_recovered
    inferrer.save_parameters(storage_name)
    # inferrer.load_parameters(storage_name)

    loss = inferrer.loss
    plt.figure()
    plt.title("ELBO")
    plt.plot(loss)
    plt.ylabel("ELBO")
    plt.xlabel("iteration")
    plt.savefig(os.path.join(base_dir, fname_str+'_ELBO.svg'))
    plt.show()

def sample_posterior(inferrer, param_names, fname_str, base_dir, n_samples=500, true_vals=None):

    sample_df, locs_sample_df = inferrer.sample_posterior(n_samples=n_samples) #inferrer.plot_posteriors(n_samples=1000)
    # inferrer.plot_posteriors(n_samples=n_samples)
    if true_vals is not None:
        append_trues = True
    else:
        append_trues = False
    
    sample_file = os.path.join(base_dir, fname_str+'_sample_df.csv')
    sample_df.to_csv(sample_file)

    locs_file = os.path.join(base_dir, fname_str+'_locs_sample_df.csv')
    locs_sample_df.to_csv(locs_file)
    
    mean_df = pd.DataFrame()

    print(sample_df)

    for name in param_names:
        means = []
        if append_trues:
            trues = []
        subs = []
        for i in range(inferrer.nsubs):
            means.append(sample_df[sample_df['subject']==i][name].mean())
            if append_trues:
                trues.append(true_vals[name][true_vals['subject']==i])
            subs.append(i)

        mean_df["inferred "+name] = torch.tensor(means)
        if append_trues:
            mean_df["true "+name] = torch.tensor(trues)
        mean_df["subject"] = torch.tensor(subs)
        
    smaller_file = os.path.join(base_dir, fname_str+'_mean_df.csv')
    mean_df.to_csv(smaller_file)

    return mean_df, sample_df, locs_sample_df


def load_samples(base_dir, fname_str):

    sample_file = os.path.join(base_dir, fname_str+'_sample_df.csv')
    sample_df = pd.read_csv(sample_file)

    mean_file = os.path.join(base_dir, fname_str+'_mean_df.csv')
    mean_df = pd.read_csv(mean_file)

    locs_sample_file = os.path.join(base_dir, fname_str+'_locs_sample_df.csv')
    locs_sample_df = pd.read_csv(locs_sample_file)

    return mean_df, sample_df, locs_sample_df