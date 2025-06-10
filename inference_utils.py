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
from misc import annot_corrfunc

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

def big_custom_plot(plot_df, param_names, base_dir, fname_str, ELBO, param_ranges, fit_reg=False, annot=False):
    
    axes_names = param_names
    ranges = param_ranges
    positions = [[0,0], [1,0], [0,1], [1,1], [0,2], [1,2]]

    fig = plt.figure(layout='constrained', figsize=(14,12))
    axes = fig.subplots(3, 3)
    
    for i, name in enumerate(param_names):
    
        ax = axes[positions[i][0], positions[i][1]]
        ax.plot(ranges[i],ranges[i], linestyle='-', color="grey", alpha=0.6)
        # sns.scatterplot(data=plot_df, x="true "+name, y="inferred "+name, ax=ax)
        sns.regplot(data=plot_df, x="true "+name, y="inferred "+name, ax=ax,
                   line_kws = {'color': 'green', 'alpha': 0.3}, fit_reg=fit_reg)
        ax.set_xlim(ranges[i])
        ax.set_ylim(ranges[i])
        ax.set_xlabel("true "+axes_names[i])
        ax.set_ylabel("inferred "+axes_names[i])
        ax.annotate(axes_names[i], (0.+0.1*ranges[i][1], ranges[i][1]-0.1*ranges[i][1]), fontsize=16)
        
        if annot:
            (r, p) = pearsonr(plot_df["true "+name], plot_df["inferred "+name])
            ax.annotate("r = {:.2f} ".format(r)+"p = {:.3f}".format(p), 
                        (0.4*ranges[i][1], 0.05*ranges[i][1]), fontsize=16)
            # ax.annotate("p = {:.3f}".format(p),
            #             (0.7*ranges[i][1], 0.05*ranges[i][1]))
        
    ax = axes[2,0]
    # plt.title("ELBO")
    ax.plot(ELBO)
    ax.set_ylabel("ELBO", fontsize=16)
    ax.set_xlabel("iteration", fontsize=16)

    rho = plot_df.corr()
    pval = plot_df.corr(method=lambda x, y: pearsonr(x, y)[1]) - eye(*rho.shape)
    reject, pval_corrected, alphaS, alphaB = multipletests(pval, method='bonferroni')
    
    gs = axes[2, 1].get_gridspec()
    # remove the underlying axes
    for ax in axes[2, 1:]:
        ax.remove()
    axbig = fig.add_subplot(gs[2, 2])
    ax = axbig
    
    p_opacity = pval_corrected*0.5 +0.5

    sns.heatmap(plot_df.corr(), annot=True, fmt='.2f', alpha=p_opacity, 
                cmap='vlag', vmin=-1, vmax=1, ax=ax)
    
    # sns.heatmap(mean_df.corr(), annot=True, fmt='.2f', ax=ax)#[pval_corrected<alphaB]
        
    try:
        plt.tight_layout()
    except:
        pass
    
    if fit_reg:
        name_str = "_regression"
    else:
        name_str = ""
    if annot:
        name_str += "_annot"
    
    plt.savefig(os.path.join(base_dir, fname_str+"_big_plot"+name_str+".svg"))
    plt.show()



def plot_results(sample_df, param_names, fname_str, ELBO, mean_df, base_dir, max_dt, big_custom=True):
    
    plot_df = mean_df.drop('subject', axis=1)\
                        .reindex(["inferred "+name for name in param_names]\
                                 +["true "+name for name in param_names], axis=1)
        
    if big_custom:
        big_custom_plot(plot_df, param_names, base_dir, fname_str, ELBO, max_dt, fit_reg=True, annot=True)
        # big_custom_plot(plot_df, param_names, base_dir, fname_str, ELBO, max_dt, fit_reg=True, annot=False)
        # big_custom_plot(plot_df, param_names, base_dir, fname_str, ELBO, max_dt, fit_reg=False, annot=True)
        # big_custom_plot(plot_df, param_names, base_dir, fname_str, ELBO, max_dt, fit_reg=False, annot=False)
    
    # plt.figure()
    # sns.pairplot(sample_df, kind='reg')
    # plt.savefig(os.path.join(base_dir, fname_str+"_pairplot_sample.svg"))
    # plt.show()
    
    plt.figure()
    f = sns.pairplot(data=plot_df, kind='reg', 
                     diag_kind="kde", corner=True,
                     plot_kws={'line_kws': {'color': 'green', 'alpha': 0.6}})
    f.map(annot_corrfunc)
    plt.savefig(os.path.join(base_dir, fname_str+"_pairplot_means_all.svg"))
    plt.show()
    
    plt.figure()
    xvars_of_interest = ["true "+name for name in param_names]
    yvars_of_interest = ["inferred "+name for name in param_names]
    f = sns.pairplot(data=plot_df, kind='reg', diag_kind="kde", corner=True,
                     plot_kws={'line_kws': {'color': 'green', 'alpha': 0.6}},
                     x_vars=xvars_of_interest, y_vars=yvars_of_interest)
    f.map(annot_corrfunc)
    plt.savefig(os.path.join(base_dir, fname_str+"_pairplot_means.svg"))
    plt.show()
    
    plt.figure()
    vars_of_interest = ["inferred "+name for name in param_names]
    f = sns.pairplot(data=plot_df, kind='reg', diag_kind="kde", corner=True,
                     plot_kws={'line_kws': {'color': 'green', 'alpha': 0.6}},
                     x_vars=vars_of_interest, y_vars=vars_of_interest)
    f.map(annot_corrfunc)
    plt.savefig(os.path.join(base_dir, fname_str+"_pairplot_means_inferred_corr.svg"))
    plt.show()
    
    # p_opacity = pval_corrected*0.5 +0.5
    
    # plt.figure()
    # sns.heatmap(plot_df.corr(), annot=True, fmt='.2f', alpha=p_opacity, 
    #             cmap='vlag', vmin=-1, vmax=1)
    # plt.show()

