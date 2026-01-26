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
from scipy.stats import ttest_1samp
from misc import annot_corrfunc
import numpy as np

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

    sns.heatmap(plot_df.corr(), annot=True, fmt='.2f', #alpha=p_opacity, 
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

def plot_correlations(plot_df, x_vars_of_interest, y_vars_of_interest):

    if set(x_vars_of_interest) == set(y_vars_of_interest):
        square = True
        vars_of_interest = x_vars_of_interest
    else:
        square = False
        vars_of_interest = x_vars_of_interest+y_vars_of_interest
    rho = plot_df[vars_of_interest].corr()
    pval = plot_df[vars_of_interest].corr(method=lambda x, y: pearsonr(x, y)[1]) - eye(*rho.shape)
    reject, pval_corrected, alphaS, alphaB = multipletests(pval, method='bonferroni')

    if square:
        mask = np.triu(np.ones_like(rho, dtype=bool), k=1)
    else:
        mask = np.zeros_like(rho.loc[x_vars_of_interest,y_vars_of_interest], dtype=bool)

    plt.figure()

    alpha_plot = np.where(pval.loc[x_vars_of_interest,y_vars_of_interest]<0.05, 1, 0.1)
    # sns.set(font_scale=1.2)
    g = sns.heatmap(rho.loc[x_vars_of_interest,y_vars_of_interest], annot=rho.loc[x_vars_of_interest,y_vars_of_interest], fmt='.2f', cmap='Spectral_r', alpha=alpha_plot, 
                # palettes: crest, icefire, vlag, Greys, RdGy, RdBu, etc etc, and add _r for reversed
                vmin=-1, vmax=1, mask=mask, annot_kws={"size": 12})#, ax=ax)
    g.set_xticklabels(g.get_xmajorticklabels(), fontsize = 12)
    g.set_yticklabels(g.get_ymajorticklabels(), fontsize = 12)
    cbar = g.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)

    plt.show()


def calculate_log_likelihood(data, agent, locs_df, npars, trials, T):

    # question: what do with nan trials? If I ignore them, they will decrease the likelihood and make subject seem to fit better. If I make them uniform, itll decrease by a lot?
    # this is the likelihood of the average parameter

    n_agents = data["actions"].shape[-1]
    log_like = torch.zeros(n_agents)
    
    locs_list = []
    for k in range(npars):
        locs_subs = []
        for i in range(n_agents):
            locs_subs.append(locs_df[locs_df["subject"]==i]["locs"+str(k)].mean())
        locs_list.append(torch.tensor(locs_subs).float())

    locs = torch.stack(locs_list, dim=-1)
    
    assert(locs.shape[0]==n_agents)

    agent.reset(locs)

    for tau in pyro.markov(range(trials)):
        for t in range(T):

            if t==0:
                prev_response = None
                context = None
            else:
                prev_response = data["actions"][tau, t-1]
                context = None

            observation = data["observations"][tau, t]

            reward = data["rewards"][tau, t]

            agent.update_beliefs(tau, t, observation, reward, prev_response, context)

            if t < T-1:

                probs = agent.perception.posterior_actions[-1]
                if torch.any(torch.isnan(probs)):
                    print(probs)
                    #print(param_dict)
                    print(tau,t)

                curr_response = data["actions"][tau, t]*data["valid"][tau].long()

                one_hot_responses = torch.nn.functional.one_hot(curr_response, num_classes=2).permute((1,0))[:,None,:].float()

                likes = (probs * one_hot_responses).sum(dim=0)[0]

                masked_probs = torch.where(data["valid"][tau], likes, torch.tensor([0.5]))

                log_like += torch.log(masked_probs)

    return -log_like.clone().detach()

    
def calculate_BIC(data, agent, locs_df, npars, trials, T):

    # use bic to circumvent the number of trials problem

    n_agents = data["actions"].shape[-1]
    BIC = torch.zeros(n_agents)
    
    locs_list = []
    for k in range(npars):
        locs_subs = []
        for i in range(n_agents):
            locs_subs.append(locs_df[locs_df["subject"]==i]["locs"+str(k)].mean())
        locs_list.append(torch.tensor(locs_subs).float())

    locs = torch.stack(locs_list, dim=-1)
    
    assert(locs.shape[0]==n_agents)

    agent.reset(locs)

    for tau in pyro.markov(range(trials)):
        for t in range(T):

            if t==0:
                prev_response = None
                context = None
            else:
                prev_response = data["actions"][tau, t-1]
                context = None

            observation = data["observations"][tau, t]

            reward = data["rewards"][tau, t]

            agent.update_beliefs(tau, t, observation, reward, prev_response, context)

            if t < T-1:

                probs = agent.perception.posterior_actions[-1]
                if torch.any(torch.isnan(probs)):
                    print(probs)
                    #print(param_dict)
                    print(tau,t)

                curr_response = data["actions"][tau, t]*data["valid"][tau].long()

                one_hot_responses = torch.nn.functional.one_hot(curr_response, num_classes=2).permute((1,0))[:,None,:].float()

                likes = (probs * one_hot_responses).sum(dim=0)[0]

                masked_probs = torch.where(data["valid"][tau], likes, torch.tensor(0.5))

                print(masked_probs.shape)

                BIC -= 2*torch.log(masked_probs)

    # question: is it noraml that the first term of the BIC (k*ln(n)) is much smaller than the second (-2*ln(L))?
    BIC += npars*torch.log(data["valid"].sum(axis=0))

    return BIC
    

def calculate_lppd(data, agent, locs_df, npars, trials, T, max_samples=-1):

    # question: what do with nan trials? If I ignore them, they will decrease the likelihood and make subject seem to fit better. If I make them uniform, itll decrease by a lot?
    # Eqs (4,5) from here:
    # http://www.stat.columbia.edu/~gelman/research/published/waic_understand3.pdf
    # this is the average likelihood

    n_agents = data["actions"].shape[-1]
    
    locs_list = []
    for k in range(npars):
        locs_subs = []
        for i in range(n_agents):
            locs_subs.append(torch.tensor(locs_df[locs_df["subject"]==i]["locs"+str(k)].values[:max_samples]).float())
        locs_list.append(torch.stack(locs_subs, dim=-1))

    locs = torch.stack(locs_list, dim=-1)

    n_samples = locs.shape[0]

    likelihoods = []

    agent.reset(locs)

    for tau in pyro.markov(range(trials)):
        for t in range(T):

            if t==0:
                prev_response = None
                context = None
            else:
                prev_response = data["actions"][tau, t-1]
                context = None

            observation = data["observations"][tau, t]

            reward = data["rewards"][tau, t]

            agent.update_beliefs(tau, t, observation, reward, prev_response, context)

            if t < T-1:

                probs = agent.perception.posterior_actions[-1]
                #print("probs", probs.shape)
                if torch.any(torch.isnan(probs)):
                    print(probs)
                    #print(param_dict)
                    print(tau,t)

                curr_response = data["actions"][tau, t]*data["valid"][tau].long()

                one_hot_responses = torch.nn.functional.one_hot(curr_response, num_classes=2).permute((1,0))[:,None,:].float()

                likes = (probs * one_hot_responses).sum(dim=0)

                masked_probs = torch.where(data["valid"][tau], likes, torch.tensor([0.5]))
                #print("masked probs", masked_probs.shape)

                likelihoods.append(masked_probs)

                #print(tau,t)

    mean_like = torch.stack(likelihoods, dim=0)
    # print("mean like stacked", mean_like.shape)
    # print(mean_like)

    mean_like = mean_like.sum(dim=-2) / n_samples
    # print("mean like summed", mean_like.shape)
    # print(mean_like)

    mean_log_like = torch.log(mean_like).sum(dim=0)

    #print(mean_log_like)
    # print("mean log like", mean_log_like.shape)
    # print(mean_log_like)

    return mean_log_like
    

def calculate_waic(data, agent, locs_df, npars, trials, T, max_samples=-1):

    # question: what do with nan trials? If I ignore them, they will decrease the likelihood and make subject seem to fit better. If I make them uniform, itll decrease by a lot?
    # Eqs (12,13) from here:
    # http://www.stat.columbia.edu/~gelman/research/published/waic_understand3.pdf
    # this is the average WAIC
    # are the larger number of params handled correctly?

    n_agents = data["actions"].shape[-1]
    
    locs_list = []
    for k in range(npars):
        locs_subs = []
        for i in range(n_agents):
            locs_subs.append(torch.tensor(locs_df[locs_df["subject"]==i]["locs"+str(k)].values[:max_samples]).float())
        locs_list.append(torch.stack(locs_subs, dim=-1))

    locs = torch.stack(locs_list, dim=-1)

    n_samples = locs.shape[0]

    likelihoods = []

    agent.reset(locs)

    for tau in pyro.markov(range(trials)):
        for t in range(T):

            if t==0:
                prev_response = None
                context = None
            else:
                prev_response = data["actions"][tau, t-1]
                context = None

            observation = data["observations"][tau, t]

            reward = data["rewards"][tau, t]

            agent.update_beliefs(tau, t, observation, reward, prev_response, context)

            if t < T-1:

                probs = agent.perception.posterior_actions[-1]
                #print("probs", probs.shape)
                if torch.any(torch.isnan(probs)):
                    print(probs)
                    #print(param_dict)
                    print(tau,t)

                curr_response = data["actions"][tau, t]*data["valid"][tau].long()

                one_hot_responses = torch.nn.functional.one_hot(curr_response, num_classes=2).permute((1,0))[:,None,:].float()

                likes = (probs * one_hot_responses).sum(dim=0)

                masked_probs = torch.where(data["valid"][tau], likes, torch.tensor([0.5]))
                #print("masked probs", masked_probs.shape)

                likelihoods.append(masked_probs)

                #print(tau,t)



    mean_like = torch.stack(likelihoods, dim=0)
    # print("mean like stacked", mean_like.shape)
    # print(mean_like)

    mean_like_samples = mean_like.sum(dim=-2) / n_samples
    # print("mean like summed", mean_like.shape)
    # print(mean_like)

    lppd = torch.log(mean_like_samples).sum(dim=0)
    # print("mean log like", mean_log_like.shape)
    # print(mean_log_like)

    mean_log_like_samples = torch.log(mean_like.sum(dim=-2)) / n_samples

    V_s = ((torch.log(mean_like) - mean_log_like_samples[:,None,:])**2).sum(dim=-2) / (n_samples-1)

    p_waic = V_s.sum(dim=0)

    ellp_waic = lppd - p_waic

    # text says it needs to be -2 * eq 13.
    # minus is required to make lower better, and the 2 converts it to variance scale

    return -2*ellp_waic
    

def predictive_accuracy_mean_param(data, agent, locs_df, npars, trials, T):

    n_agents = data["actions"].shape[-1]

    predicted_accuracy = torch.zeros(n_agents)
    
    locs_list = []
    for k in range(npars):
        locs_subs = []
        for i in range(n_agents):
            locs_subs.append(locs_df[locs_df["subject"]==i]["locs"+str(k)].mean())
        locs_list.append(torch.tensor(locs_subs).float())

    locs = torch.stack(locs_list, dim=-1)
    
    assert(locs.shape[0]==n_agents)

    agent.reset(locs)

    num_valid_responses = torch.zeros(data["actions"].shape[-1])

    for tau in pyro.markov(range(trials)):
        for t in range(T):

            if t==0:
                prev_response = None
                context = None
            else:
                prev_response = data["actions"][tau, t-1]
                context = None

            observation = data["observations"][tau, t]

            reward = data["rewards"][tau, t]

            agent.update_beliefs(tau, t, observation, reward, prev_response, context)

            if t < T-1:
                #print(tau,t)

                probs = agent.perception.posterior_actions[-1]
                if torch.any(torch.isnan(probs)):
                    print(probs)
                    #print(param_dict)
                    print(tau,t)

                curr_response = data["actions"][tau, t]#*data["valid"][tau].long()
                #print(curr_response)

                predicted_response = torch.argmax(probs, dim=0)[0]

                #print(probs)

                #print(predicted_response)

                correct_response_predicted = (curr_response == predicted_response).int()

                #print(correct_response_predicted)

                predicted_accuracy += correct_response_predicted

                num_valid_responses += data["valid"][tau]

                #print(num_valid_responses)


    corrected_predicted_accuracy = predicted_accuracy / num_valid_responses

    return corrected_predicted_accuracy


def calculate_exceedance_prob(measure, n_exc_samples=500):
    
    p_model = torch.nn.functional.softmax(measure, dim=-1)

    print("p model mean according to measure", p_model.mean(dim=0))

    dirichlet_counts = p_model.sum(dim=0)

    model_prob_dirichlet = dist.Dirichlet(dirichlet_counts)

    n_exc_samples = 500

    dir_samples = model_prob_dirichlet.sample(sample_shape=torch.tensor([n_exc_samples]))

    avg_best_model = dir_samples.mean(dim=0).argmax()

    best_model = dir_samples.argmax(dim=1)

    exc_prob = (best_model == avg_best_model).sum()/n_exc_samples

    print("best model:", avg_best_model, "exceedance prob", exc_prob)

    significant_best_model = ttest_1samp(dir_samples[:,avg_best_model], 1./measure.shape[-1], alternative="greater")

    print("is significantly different from uniform?", significant_best_model)


def calculate_exceedance_prob_1D(measure, n_exc_samples=500):
    
    p_model = torch.nn.functional.softmax(measure, dim=-1)

    print("p model mean according to measure", p_model)

    dirichlet_counts = measure

    model_prob_dirichlet = dist.Dirichlet(dirichlet_counts)

    n_exc_samples = 500

    dir_samples = model_prob_dirichlet.sample(sample_shape=torch.tensor([n_exc_samples]))

    avg_best_model = dir_samples.mean(dim=0).argmax()

    best_model = dir_samples.argmax(dim=1)

    exc_prob = (best_model == avg_best_model).sum()/n_exc_samples

    print("best model:", avg_best_model, "exceedance prob", exc_prob)

    significant_best_model = ttest_1samp(dir_samples[:,avg_best_model], 1./measure.shape[-1], alternative="greater")

    print("is significantly different from uniform?", significant_best_model)