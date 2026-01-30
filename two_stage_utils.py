#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 14:09:11 2021

@author: sarah
"""


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
#device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#device = torch.device("cuda")
#device = torch.device("cpu")

from inference import device

#torch.autograd.set_detect_anomaly(True)
###################################
###################################


"""
run function
"""
def set_up_Bayesian_agent(agent_par_list, trials, T, ns, na, nr, nb, A, B, nsubs=1, **kwargs):

    #set parameters:
    #obs_unc: observation uncertainty condition
    #state_unc: state transition uncertainty condition
    #goal_pol: evaluate only policies that lead to the goal
    #utility: goal prior, preference p(o)
    avg, perception_args, learn_rewards, learn_habit, learn_cached, valid, use_h = agent_par_list
    
    utility = torch.tensor([0.01, 0.99])
    

    """
    create matrices
    """

    C_alphas = torch.zeros((nr, ns)) + 1
    C_alphas[0,:(ns-nb)] = 100
    for i in range(1,nr):
        C_alphas[i,0] = 1


    """
    create policies
    """

    pol = torch.tensor(list(itertools.product(list(range(na)), repeat=T-1)))

    #pol = pol[-2:]
    npi = pol.shape[0]



    """
    set state prior (where agent thinks it starts)
    """

    state_prior = torch.zeros((ns))

    state_prior[0] = 1.

    """
    set action selection method
    """

    if avg:

        sel = 'avg'

        ac_sel = asl.AveragedSelector(trials = trials, T = T,
                                      number_of_actions = na)
    else:

        sel = 'max'

        ac_sel = asl.MaxSelector(trials = trials, T = T,
                                      number_of_actions = na)


    """
    set up agent
    """
    
    pol_lambda = perception_args["policy rate"]
    r_lambda = perception_args["reward rate"]
    dec_temp = perception_args["dec temp"]    
    if use_h:
        alpha_0 = 1./perception_args["habitual tendency"]
    else:
        alpha_0 = perception_args["habitual tendency"]
    alphas = torch.zeros((npi)) + alpha_0
    cached_weight = perception_args["cached weight"]
    cached_r_lambda = perception_args["cached rate"]

    # print(use_h)
    # print(alpha_0)

    if learn_rewards:
        infer_decision_temp = True
        infer_reward_rate = True
    else:
        infer_decision_temp = False
        infer_reward_rate = False

    if learn_habit:
        infer_h = True
        infer_policy_rate = True
    else:
        infer_h = False
        infer_policy_rate = False

    if learn_cached:
        infer_cached_weight = True
        infer_cached_rate = True
    else:
        infer_cached_weight = False
        infer_cached_rate = False

    # bayes_prc = prc.Group2ContextPerception(A, B, torch.tensor([[1]]),
    #                                 state_prior, utility, torch.tensor([1]), pol,
    #                                 alpha_0=alpha_0, dirichlet_rew_params=C_alphas, 
    #                                 learn_habit = learn_habit, mask=valid, learn_cached_rewards=learn_cached,
    #                                 learn_rew = learn_rewards, T=T, trials=trials,
    #                                 pol_lambda=pol_lambda, r_lambda=r_lambda,
    #                                 non_decaying=(ns-nb), dec_temp=dec_temp, 
    #                                 cached_weight=cached_weight, cached_r_lambda=cached_r_lambda,
    #                                 nsubs=nsubs, infer_alpha_0=infer_h, use_h=use_h,
    #                                 infer_context=False, dirichlet_context_obs_params=torch.tensor([[1]]),
    #                                 infer_decision_temp=infer_decision_temp, infer_policy_rate=infer_policy_rate, 
    #                                 infer_reward_rate=infer_reward_rate, infer_cached_weight=infer_cached_weight, 
    #                                 infer_cached_rate=infer_cached_rate)

    # C_alphas = torch.zeros((nr, ns, 2)) + 1
    # C_alphas[0,:(ns-nb),:] = 100
    # for i in range(1,nr):
    #     C_alphas[i,0,:] = 1
    
    bayes_prc = prc.Group2Perception(A, B, 
                                state_prior, utility, pol,
                                alpha_0=alpha_0, dirichlet_rew_params=C_alphas, 
                                learn_habit = learn_habit, mask=valid, learn_cached_rewards=learn_cached,
                                learn_rew = learn_rewards, T=T, trials=trials,
                                pol_lambda=pol_lambda, r_lambda=r_lambda,
                                non_decaying=(ns-nb), dec_temp=dec_temp, 
                                nsubs=nsubs, infer_alpha_0=infer_h, use_h=use_h,
                                cached_weight=cached_weight, cached_r_lambda=cached_r_lambda,
                                infer_decision_temp=infer_decision_temp, infer_policy_rate=infer_policy_rate, 
                                infer_reward_rate=infer_reward_rate, infer_cached_weight=infer_cached_weight, 
                                infer_cached_rate=infer_cached_rate, which_rewards=[T-1])#list(range(1,T)))#

    print(bayes_prc)
    
    bayes_prc.set_parameters(par_dict=perception_args)
    bayes_prc.reset()

    bayes_pln = agt.FittingAgent(bayes_prc, ac_sel, pol,
                      trials = trials, T = T,
                      number_of_states = ns,
                      number_of_policies = npi,
                      number_of_rewards = nr,
                      nsubs = nsubs)
    
    
    return bayes_pln, bayes_prc


def set_up_mfmb_agent(agent_par_list, trials, T, ns, na, nr, nb, A, B, nsubs=1, **kwargs):

    #set parameters:
    #obs_unc: observation uncertainty condition
    #state_unc: state transition uncertainty condition
    #goal_pol: evaluate only policies that lead to the goal
    #utility: goal prior, preference p(o)
    avg, perception_args, use_orig, use_p, restrict_alpha, valid = agent_par_list

    utility = []
    
    #ut = [0.5, 0.6, 0.7, 0.8, 0.9, 1. - 1e-5]
    #ut = [0.95, 0.96, 0.98, 0.99]
    #ut = [0.985]
    ut = [0.999]
    for u in ut:
        utility.append(torch.zeros(nr).to(device))
        for i in range(1,nr):
            utility[-1][i] = u/(nr-1)#u/nr*i
        utility[-1][0] = (1.-u)
    
    utility = utility[-1]


    """
    create policies
    """

    pol = torch.tensor(list(itertools.product(list(range(na)), repeat=T-1)))

    #pol = pol[-2:]
    npi = pol.shape[0]


    """
    set state prior (where agent thinks it starts)
    """

    state_prior = torch.zeros((ns))

    state_prior[0] = 1.

    """
    set action selection method
    """

    if avg:

        sel = 'avg'

        ac_sel = asl.AveragedSelector(trials = trials, T = T,
                                      number_of_actions = na)
    else:

        sel = 'max'

        ac_sel = asl.MaxSelector(trials = trials, T = T,
                                      number_of_actions = na)

#    ac_sel = asl.AveragedPolicySelector(trials = trials, T = T,
#                                        number_of_policies = npi,
#                                        number_of_actions = na)

    prior_context = torch.tensor([1.])

#    prior_context[0] = 1.

    """
    set up agent
    """


    Q_mf_init = [torch.zeros((3,na)), torch.zeros((3,na))]
    Q_mb_init = [torch.zeros((3,na)), torch.zeros((3,na))]
    counts_init = torch.ones((3,na))

    # perception
    if use_orig:
        lamb = perception_args["discount"]
        alpha = perception_args["learning rate"]
        beta = perception_args["dec temp"]
        w = perception_args["weight"]
        p = perception_args["repetition"]
        max_dt = perception_args["max dt"]
        min_alpha = perception_args["min learning rate"]
        prior_lr = perception_args["prior lr"]
        prior_weight = perception_args["prior weight"]
        learn_prior = perception_args["learn_prior"]

        # attention mfmbOrig2Perception is not fully implemented yet.
        
        mbmf_prc = prc.mfmbOrig2Perception(B, pol, Q_mf_init, Q_mb_init, utility,
                                        lamb, alpha, beta, w,
                                        p, nsubs=1, use_p=use_p, mask=valid,
                                        restrict_alpha=restrict_alpha,
                                        max_dt=max_dt, min_alpha=min_alpha)
    else:
        lamb = perception_args["discount"]
        alpha = perception_args["learning rate"]
        beta_mf = perception_args["mf weight"]
        beta_mb = perception_args["mb weight"]
        p = perception_args["repetition"]
        max_dt = perception_args["max dt"]
        min_alpha = perception_args["min learning rate"]
        prior_lr = perception_args["prior lr"]
        prior_weight = perception_args["prior weight"]
        learn_prior = perception_args["learn_prior"]
        if learn_prior:
            infer_prior_weight = True
            infer_prior_lr = True
        else:
            infer_prior_weight = False
            infer_prior_lr = False
        
        mbmf_prc = prc.mfmb3Perception(B, pol, Q_mf_init, Q_mb_init, utility,
                                       counts_init=counts_init,
                                       lamb = lamb, alpha = alpha,
                                       lr_prior = prior_lr, learn_rep=learn_prior,
                                       beta_mf = beta_mf, beta_mb = beta_mb,
                                       beta_prior = prior_weight, p=p,
                                       nsubs=1, use_p=use_p, mask=valid,
                                       restrict_alpha=restrict_alpha,
                                       max_dt=max_dt, min_alpha=min_alpha, 
                                       infer_prior_weight=infer_prior_weight,
                                       infer_prior_lr=infer_prior_lr)
    mbmf_prc.reset()

    planner = agt.FittingAgent(mbmf_prc, ac_sel, pol,
                      trials = trials, T = T,
                      number_of_states = ns,
                      #save_everything = True,
                      number_of_policies = npi,
                      number_of_rewards = nr)

    return planner, mbmf_prc


def set_up_two_stage_env(Rho, trials, T, A, B):
    
        """
        create environment (two stage task)
        """

        environment = env.MultiArmedBandid(A, B, Rho, trials = trials, T = T)
        
        return environment

    
def simulate_BCC_behavior(par_list, trials, T, ns, na, nr, nb, A, B):
    
    avg, Rho, perception_args, learn_rewards, learn_habit, learn_cached, valid, use_h = par_list
    
    environment = set_up_two_stage_env(Rho, trials, T, A, B)
    
    agent_par_list = [avg, perception_args, learn_rewards, learn_habit, learn_cached, valid, use_h]
    planner, perception = set_up_Bayesian_agent(agent_par_list, trials, T, ns, na, nr, nb, A, B)
    
    """
    create world
    """

    w = world.GroupWorld(environment, planner, trials = trials, T = T)

    """
    simulate experiment
    """

    w.simulate_experiment(range(trials))


    return w


def simulate_mfmb_behavior(pars, trials, T, ns, na, nr, nb, A, B):
    
    avg, Rho, perception_args, use_orig, use_p, restrict_alpha, valid = pars
    
    environment = set_up_two_stage_env(Rho, trials, T, A, B)
    
    agent_par_list = [avg, perception_args, use_orig, use_p, restrict_alpha, valid]
    planner, perception = set_up_mfmb_agent(agent_par_list, trials, T, ns, na, nr, nb, A, B)
    
    """
    create world
    """

    w = world.GroupWorld(environment, planner, trials = trials, T = T)

    """
    simulate experiment
    """

    w.simulate_experiment(range(trials))


    return w


def plot_inferred(mean_df, fname_str, reg_fit=False):
    
    plot_df = mean_df.drop('subject', axis=1)
                        
    axes_names = ["policy forgetting rate lambda_pi", "reward forgetting rate lambda_r", "decision temp gamma", "habitual tendency h"]
    ranges = [[0,1], [0,1], [1, max_dt], [0,1], [0, 1]]

    
    for i, name in enumerate(param_names):

        plt.figure()
        plt.plot(ranges[i],ranges[i], linestyle='-', color="grey", alpha=0.6)
        # sns.scatterplot(data=plot_df, x="true "+name, y="inferred "+name, ax=ax)
        sns.regplot(data=plot_df, x="true "+name, y="inferred "+name,
                   line_kws = {'color': 'green', 'alpha': 0.3}, fit_reg=reg_fit)
        plt.xlim(ranges[i])
        plt.ylim(ranges[i])
        plt.xlabel("true "+axes_names[i])
        plt.ylabel("inferred "+axes_names[i])
        plt.annotate(axes_names[i], (0.+0.1*ranges[i][1], ranges[i][1]-0.1*ranges[i][1]))
        plt.show()
        
        


def annot_corrfunc(x, y, **kws):
    (r, p) = pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f} ".format(r),
                xy=(.1, .9), xycoords=ax.transAxes)
    ax.annotate("p = {:.3f}".format(p),
                xy=(.5, .9), xycoords=ax.transAxes)
    
    
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
    
def run_BCC_simulations(nsubs, learn_rewards, learn_habit, learn_cached, agent_type, n_pars, fname_base, base_dir, Rho, trials, T, 
                        nb, ns, no, na, npi, nr, never_reward, A, B, p_invalid,
                        mask=None, max_dt=6, remove_old=True, use_h=True):
    

    # if it does exist, empty previous results, if we want that (remove_old==True)
    if remove_old:
            
        svgs = glob.glob(os.path.join(base_dir,"*.svg"))
        for file in svgs:
            os.remove(file)
            
        csvs = glob.glob(os.path.join(base_dir,"*.csv"))
        for file in csvs:
            os.remove(file)
            
        saves = glob.glob(os.path.join(base_dir,"*.save"))
        for file in saves:
            os.remove(file)
            
        agents = glob.glob(os.path.join(base_dir,"twostage_agent*"))
        for file in agents:
            os.remove(file)
            
        outputs = glob.glob(os.path.join(base_dir,"*.json"))
        for file in outputs:
            os.remove(file)

    if learn_rewards:
        true_vals_rewards = torch.rand((nsubs,2,1))
    else:
        true_vals_rewards = torch.zeros((nsubs,2,1))
    if learn_habit:
        true_vals_repetition = torch.rand((nsubs,2,1))
    else:
        true_vals_repetition = torch.zeros((nsubs,2,1))
    if learn_cached:
        true_vals_cached = torch.rand((nsubs,2,1))
    else:
        true_vals_cached = torch.zeros((nsubs,2,1))
    
    true_values_tensor = torch.cat([true_vals_rewards, true_vals_repetition, true_vals_cached], dim=1)
        
    
    true_vals = []
    data = []
    
    stayed = []
    indices = []
    
    for k, pars in enumerate(true_values_tensor):

        norm_dt, rl, norm_h, pl, norm_cw, cl = pars
    
        if use_h or not learn_habit:
            tend = norm_h
        else:
            tend = (max_dt-1)*norm_h + 1
        
        if learn_rewards:
            dt = (max_dt-1)*norm_dt+1
        else:
            dt = norm_dt
        if learn_cached:
            cw = (max_dt-1)*norm_cw+1
        else:
            cw = norm_cw
        
        # print(pl, rl, dt, tend)
        
        perception_args = {"subject": torch.tensor([k]), 
                           "dec temp": dt, "reward rate": rl, 
                           "habitual tendency": tend, "policy rate": pl, 
                           "cached weight": cw, "cached rate": cl}
        
        print(perception_args)
        
        worlds = []
        l = []
        avg = True
        if mask is not None:
            valid = mask[:,[k]]
        else:
            prob_matrix = torch.zeros((trials,1)) + p_invalid
            valid = torch.bernoulli(prob_matrix).bool()

        if len(Rho.shape) > 3:
            Rho_subj = Rho[k]
        else:
            Rho_subj = Rho
            
        pars = [avg, Rho_subj,perception_args, learn_rewards, learn_habit, learn_cached, valid, use_h]
        
        worlds.append(simulate_BCC_behavior(pars, trials, T, ns, na, nr, nb, A, B))
        
        w = worlds[-1]
        
        rewarded = w.rewards[:trials-1,-1] == 1
        
        unrewarded = rewarded==False
        
        rare = torch.logical_or(torch.logical_and(w.environment.hidden_states[:trials-1,1]==2, w.actions[:trials-1,0] == 0),
                       torch.logical_and(w.environment.hidden_states[:trials-1,1]==1, w.actions[:trials-1,0] == 1))
        
        common = rare==False
        
        rewarded_common = torch.where(torch.logical_and(rewarded,common) == True)[0]
        rewarded_rare = torch.where(torch.logical_and(rewarded,rare) == True)[0]
        unrewarded_common = torch.where(torch.logical_and(unrewarded,common) == True)[0]
        unrewarded_rare = torch.where(torch.logical_and(unrewarded,rare) == True)[0]
        
        index_list = [rewarded_common, rewarded_rare,
                     unrewarded_common, unrewarded_rare]
        
        stayed_list = [(w.actions[index_list[i],0] == w.actions[index_list[i]+1,0]).sum()/float(len(index_list[i])) for i in range(4)]
        
        stayed.append(stayed_list)
        
        run_name = "twostage_agent_daw_"+agent_type+"_dt"+str(dt)+"_rl"+str(rl)+"_tend"+str(tend)+"_pl"+str(pl)+"_cw"+str(cw)+"_cl"+str(cl)+".json"
        fname_behavior = os.path.join(base_dir, run_name)
        
        data.append({"subject": torch.tensor([k]), "actions": w.actions, "observations": w.observations, "rewards": w.rewards, "states": w.environment.hidden_states, 'valid': valid})
        
        pickled_behavior = pickle.encode(data[-1])
        with open(fname_behavior, 'w') as outfile:
            json.dump(pickled_behavior, outfile)
        
        pickled_behavior = 0
        
        gc.collect()
    
        true_vals.append(perception_args)
    
    stayed_arr = torch.tensor(stayed)
    
    # structure data
    
    data_obs = torch.stack([d["observations"] for d in data], dim=-1)
    data_rew = torch.stack([d["rewards"] for d in data], dim=-1)
    data_act = torch.stack([d["actions"] for d in data], dim=-1)
    data_val = torch.cat([d["valid"] for d in data], dim=-1)
    data_ind = torch.stack([d["subject"] for d in data], dim=-1)

    structured_data = {"subject": data_ind, "observations": data_obs, "rewards": data_rew, "actions": data_act, "valid": data_val}
    
    # structure true vals
    
    true_pol_rate = torch.stack([t["policy rate"] for t in true_vals], dim=-1)
    true_rew_rate = torch.stack([t["reward rate"] for t in true_vals], dim=-1)
    true_dec_temp = torch.stack([t["dec temp"] for t in true_vals], dim=-1)
    true_hab_tend = torch.stack([t["habitual tendency"] for t in true_vals], dim=-1)
    true_cac_wght = torch.stack([t["cached weight"] for t in true_vals], dim=-1)
    true_cac_rate = torch.stack([t["cached rate"] for t in true_vals], dim=-1)
    true_ind = torch.stack([t["subject"] for t in true_vals], dim=-1)
    
    structured_true_vals = {"subject": true_ind, 
                            "dec temp": true_dec_temp, "reward rate": true_rew_rate, 
                            "habitual tendency": true_hab_tend, "policy rate": true_pol_rate,
                            "cached weight": true_cac_wght, "cached rate": true_cac_rate}
    
    # save to disk
    
    # stayed arr
    fname_stayed = os.path.join(base_dir, "twostage_agent_daw_"+agent_type+"_stayed_arr.json")
    pickled_stayed_arr = pickle.encode(stayed_arr)
    with open(fname_stayed, 'w') as outfile:
        json.dump(pickled_stayed_arr, outfile)
        
    # data 
    fname_data = os.path.join(base_dir, "twostage_agent_daw_"+agent_type+"_data.json")
    pickled_data = pickle.encode(structured_data)
    with open(fname_data, 'w') as outfile:
        json.dump(pickled_data, outfile)
        
    # true values 
    fname_true_vals = os.path.join(base_dir, "twostage_agent_daw_"+agent_type+"_true_vals.json")
    pickled_true_vals = pickle.encode(structured_true_vals)
    with open(fname_true_vals, 'w') as outfile:
        json.dump(pickled_true_vals, outfile)
    
    return stayed_arr, structured_true_vals, structured_data

def run_BCC_post_pred_simulations(nsubs, learn_rewards, learn_habit, learn_cached, parameter_values, agent_type, n_pars, fname_base, base_dir, Rho, trials, T, 
                        nb, ns, no, na, npi, nr, never_reward, A, B, p_invalid,
                        mask=None, max_dt=6, remove_old=True, use_h=True):
    

    # if it does exist, empty previous results, if we want that (remove_old==True)
    if remove_old:
            
        svgs = glob.glob(os.path.join(base_dir,"*.svg"))
        for file in svgs:
            os.remove(file)
            
        csvs = glob.glob(os.path.join(base_dir,"*.csv"))
        for file in csvs:
            os.remove(file)
            
        saves = glob.glob(os.path.join(base_dir,"*.save"))
        for file in saves:
            os.remove(file)
            
        agents = glob.glob(os.path.join(base_dir,"twostage_agent*"))
        for file in agents:
            os.remove(file)
            
        outputs = glob.glob(os.path.join(base_dir,"*.json"))
        for file in outputs:
            os.remove(file)

    if learn_rewards:
        true_dec_temp = torch.from_numpy(parameter_values["inferred dec temp"].to_numpy())
        true_reward_rate = torch.from_numpy(parameter_values["inferred reward rate"].to_numpy())
        true_vals_rewards = torch.stack([true_dec_temp, true_reward_rate], dim=1)[:,:,None]
    else:
        true_vals_rewards = torch.zeros((nsubs,2,1))

    if learn_habit:
        true_hab_tend = torch.from_numpy(parameter_values["inferred habitual tendency"].to_numpy())
        true_pol_rate = torch.from_numpy(parameter_values["inferred policy rate"].to_numpy())
        true_vals_repetition = torch.stack([true_hab_tend, true_pol_rate], dim=1)[:,:,None]
    else:
        true_vals_repetition = torch.zeros((nsubs,2,1))

    if learn_cached:
        true_cached_weight = torch.from_numpy(parameter_values["inferred cached weight"].to_numpy())
        true_cached_rate = torch.from_numpy(parameter_values["inferred cached rate"].to_numpy())
        true_vals_cached = torch.stack([true_cached_weight, true_cached_rate], dim=1)[:,:,None]
    else:
        true_vals_cached = torch.zeros((nsubs,2,1))
    
    true_values_tensor = torch.cat([true_vals_rewards, true_vals_repetition, true_vals_cached], dim=1).float()
        
    
    true_vals = []
    data = []
    
    stayed = []
    indices = []
    
    for k, pars in enumerate(true_values_tensor):

        dt, rl, tend, pl, cw, cl = pars
        
        # print(pl, rl, dt, tend)
        
        perception_args = {"subject": torch.tensor([k]), 
                           "dec temp": dt, "reward rate": rl, 
                           "habitual tendency": tend, "policy rate": pl, 
                           "cached weight": cw, "cached rate": cl}
        
        print(perception_args)
        
        worlds = []
        l = []
        avg = True
        if mask is not None:
            valid = mask[:,[k]]
        else:
            prob_matrix = torch.zeros((trials,1)) + p_invalid
            valid = torch.bernoulli(prob_matrix).bool()

        if len(Rho.shape) > 3:
            Rho_subj = Rho[k]
        else:
            Rho_subj = Rho
            
        pars = [avg, Rho_subj,perception_args, learn_rewards, learn_habit, learn_cached, valid, use_h]
        
        worlds.append(simulate_BCC_behavior(pars, trials, T, ns, na, nr, nb, A, B))
        
        w = worlds[-1]
        
        rewarded = w.rewards[:trials-1,-1] == 1
        
        unrewarded = rewarded==False
        
        rare = torch.logical_or(torch.logical_and(w.environment.hidden_states[:trials-1,1]==2, w.actions[:trials-1,0] == 0),
                       torch.logical_and(w.environment.hidden_states[:trials-1,1]==1, w.actions[:trials-1,0] == 1))
        
        common = rare==False
        
        rewarded_common = torch.where(torch.logical_and(rewarded,common) == True)[0]
        rewarded_rare = torch.where(torch.logical_and(rewarded,rare) == True)[0]
        unrewarded_common = torch.where(torch.logical_and(unrewarded,common) == True)[0]
        unrewarded_rare = torch.where(torch.logical_and(unrewarded,rare) == True)[0]
        
        index_list = [rewarded_common, rewarded_rare,
                     unrewarded_common, unrewarded_rare]
        
        stayed_list = [(w.actions[index_list[i],0] == w.actions[index_list[i]+1,0]).sum()/float(len(index_list[i])) for i in range(4)]
        
        stayed.append(stayed_list)
        
        run_name = "twostage_agent_daw_"+agent_type+"_dt"+str(dt)+"_rl"+str(rl)+"_tend"+str(tend)+"_pl"+str(pl)+"_cw"+str(cw)+"_cl"+str(cl)+".json"
        fname_behavior = os.path.join(base_dir, run_name)
        
        data.append({"subject": torch.tensor([k]), "actions": w.actions, "observations": w.observations, "rewards": w.rewards, "states": w.environment.hidden_states, 'valid': valid})
        
        pickled_behavior = pickle.encode(data[-1])
        with open(fname_behavior, 'w') as outfile:
            json.dump(pickled_behavior, outfile)
        
        pickled_behavior = 0
        
        gc.collect()
    
        true_vals.append(perception_args)
    
    stayed_arr = torch.tensor(stayed)
    
    # structure data
    
    data_obs = torch.stack([d["observations"] for d in data], dim=-1)
    data_rew = torch.stack([d["rewards"] for d in data], dim=-1)
    data_act = torch.stack([d["actions"] for d in data], dim=-1)
    data_val = torch.cat([d["valid"] for d in data], dim=-1)
    data_ind = torch.stack([d["subject"] for d in data], dim=-1)

    structured_data = {"subject": data_ind, "observations": data_obs, "rewards": data_rew, "actions": data_act, "valid": data_val}
    
    # structure true vals
    
    true_pol_rate = torch.stack([t["policy rate"] for t in true_vals], dim=-1)
    true_rew_rate = torch.stack([t["reward rate"] for t in true_vals], dim=-1)
    true_dec_temp = torch.stack([t["dec temp"] for t in true_vals], dim=-1)
    true_hab_tend = torch.stack([t["habitual tendency"] for t in true_vals], dim=-1)
    true_cac_wght = torch.stack([t["cached weight"] for t in true_vals], dim=-1)
    true_cac_rate = torch.stack([t["cached rate"] for t in true_vals], dim=-1)
    true_ind = torch.stack([t["subject"] for t in true_vals], dim=-1)
    
    structured_true_vals = {"subject": true_ind, 
                            "dec temp": true_dec_temp, "reward rate": true_rew_rate, 
                            "habitual tendency": true_hab_tend, "policy rate": true_pol_rate,
                            "cached weight": true_cac_wght, "cached rate": true_cac_rate}
    
    # save to disk
    
    # stayed arr
    fname_stayed = os.path.join(base_dir, "twostage_agent_daw_"+agent_type+"_stayed_arr.json")
    pickled_stayed_arr = pickle.encode(stayed_arr)
    with open(fname_stayed, 'w') as outfile:
        json.dump(pickled_stayed_arr, outfile)
        
    # data 
    fname_data = os.path.join(base_dir, "twostage_agent_daw_"+agent_type+"_data.json")
    pickled_data = pickle.encode(structured_data)
    with open(fname_data, 'w') as outfile:
        json.dump(pickled_data, outfile)
        
    # true values 
    fname_true_vals = os.path.join(base_dir, "twostage_agent_daw_"+agent_type+"_true_vals.json")
    pickled_true_vals = pickle.encode(structured_true_vals)
    with open(fname_true_vals, 'w') as outfile:
        json.dump(pickled_true_vals, outfile)
    
    return stayed_arr, structured_true_vals, structured_data


def run_mfmb_simulations(nsubs, agent_type, n_pars, learn_prior, use_orig, use_p, restrict_alpha, min_alpha, fname_base, base_dir, Rho, trials, T, 
                        nb, ns, no, na, npi, nr, never_reward, A, B, mask, p_invalid,
                        max_dt=6, remove_old=True):

    # if it does exist, empty previous results, if we want that (remove_old==True)
    if remove_old:
            
        svgs = glob.glob(os.path.join(base_dir,"*.svg"))
        for file in svgs:
            os.remove(file)
            
        csvs = glob.glob(os.path.join(base_dir,"*.csv"))
        for file in csvs:
            os.remove(file)
            
        saves = glob.glob(os.path.join(base_dir,"*.save"))
        for file in saves:
            os.remove(file)
            
        agents = glob.glob(os.path.join(base_dir,"twostage_agent*"))
        for file in agents:
            os.remove(file)
            
        outputs = glob.glob(os.path.join(base_dir,"*.json"))
        for file in outputs:
            os.remove(file)

    true_values_tensor_mfmb = torch.rand((nsubs,4,1))
    
    if learn_prior:
        true_vals_prior = torch.rand((nsubs,2,1))
    else:
        true_vals_prior = torch.zeros((nsubs,2,1))
    
    if use_p: 
        true_vals_p = torch.rand((nsubs,1,1))
    else:
        true_vals_p = torch.zeros((nsubs,1,1))

    true_values_tensor = torch.cat([true_values_tensor_mfmb, true_vals_prior, true_vals_p], dim=1)
    
    true_vals = []
    data = []
    
    stayed = []
    indices = []
    
    for i, pars in enumerate(true_values_tensor):
    
        # make parameters for original mb mf: discount lambda, learning rate, dec temp, balancing w, perserveration
        if use_orig:
            discount, norm_lr, norm_dt, weight, prior_lr, norm_prior_weight, perserv = pars
        
            dt = max_dt*norm_dt
            dt_prior = max_dt*norm_prior_weight
            if restrict_alpha:
                lr = min_alpha + norm_lr*(1.-min_alpha)
            else:
                lr = norm_lr
            perception_args = {"subject": torch.tensor([i]), "discount": discount, "learning rate": lr, "dec temp": dt, "weight": weight, "repetition": perserv, 
                                "prior lr": prior_lr, "prior weight": dt_prior, "max dt": max_dt, "min learning rate": min_alpha, "learn_prior": learn_prior}
            
        # make parameters for two beta mb mf: discount lambda, learning rate, mb dec temp, mf dec temp, perserveration
        else:

            discount, norm_lr, norm_dt_mf, norm_dt_mb, prior_lr, norm_prior_weight, norm_perserv = pars
        
            dt_mf = max_dt*norm_dt_mf
            dt_mb = max_dt*norm_dt_mb
            dt_prior = max_dt*norm_prior_weight
            perserv = max_dt*norm_perserv
            if restrict_alpha:
                lr = min_alpha + norm_lr*(1.-min_alpha)
            else:
                lr = norm_lr

            perception_args = {"subject": torch.tensor([i]), "discount": discount, "learning rate": lr, "mf weight": dt_mf, "mb weight": dt_mb, 
                               "prior lr": prior_lr, "prior weight": dt_prior, "repetition": perserv,  "learn_prior": learn_prior,
                               "max dt": max_dt, "min learning rate": min_alpha}
            
        print(perception_args)

        if len(Rho.shape) > 3:
            Rho_subj = Rho[i]
        else:
            Rho_subj = Rho
        
        worlds = []
        l = []
        avg = True
        valid = mask[:,[i]]
        pars = [avg, Rho_subj,perception_args, use_orig, use_p, restrict_alpha, valid]
        
        worlds.append(simulate_mfmb_behavior(pars, trials, T, ns, na, nr, nb, A, B))
        
        w = worlds[-1]
        
        rewarded = w.rewards[:trials-1,-1] == 1
        
        unrewarded = rewarded==False
        
        rare = torch.logical_or(torch.logical_and(w.environment.hidden_states[:trials-1,1]==2, w.actions[:trials-1,0] == 0),
                       torch.logical_and(w.environment.hidden_states[:trials-1,1]==1, w.actions[:trials-1,0] == 1))
        
        common = rare==False
        
        rewarded_common = torch.where(torch.logical_and(rewarded,common) == True)[0]
        rewarded_rare = torch.where(torch.logical_and(rewarded,rare) == True)[0]
        unrewarded_common = torch.where(torch.logical_and(unrewarded,common) == True)[0]
        unrewarded_rare = torch.where(torch.logical_and(unrewarded,rare) == True)[0]
        
        index_list = [rewarded_common, rewarded_rare,
                     unrewarded_common, unrewarded_rare]
        
        stayed_list = [(w.actions[index_list[i],0] == w.actions[index_list[i]+1,0]).sum()/float(len(index_list[i])) for i in range(4)]
        
        stayed.append(stayed_list)
        
        if use_orig:
            run_name = "twostage_agent_daw_"+agent_type+"_"+str(i)+"_disc"+str(discount)+"_lr"+str(lr)+"_dt"+str(dt)+"weight"+str(weight)+"_perserv"+str(perserv)+".json"
        else:
            run_name = "twostage_agent_daw_"+agent_type+"_"+str(i)+"_disc"+str(discount)+"_lr"+str(lr)+"_dt_mf"+str(dt_mf)+"_dt_mb"+str(dt_mb)+"_perserv"+str(perserv)+".json"
        fname_behavior = os.path.join(base_dir, run_name)
        
        data.append({"subject": torch.tensor([i]), "actions": w.actions, "observations": w.observations, "rewards": w.rewards, "states": w.environment.hidden_states, 'valid': valid})
        
        pickled_behavior = pickle.encode(data[-1])
        with open(fname_behavior, 'w') as outfile:
            json.dump(pickled_behavior, outfile)
        
        pickled_behavior = 0
        
        gc.collect()
    
        true_vals.append(perception_args)
    
    stayed_arr = torch.tensor(stayed)
    
    # structure data
    
    data_obs = torch.stack([d["observations"] for d in data], dim=-1)
    data_rew = torch.stack([d["rewards"] for d in data], dim=-1)
    data_act = torch.stack([d["actions"] for d in data], dim=-1)
    data_val = torch.cat([d["valid"] for d in data], dim=-1)
    data_ind = torch.stack([d["subject"] for d in data], dim=-1)

    structured_data = {"subject": data_ind, "observations": data_obs, "rewards": data_rew, "actions": data_act, "valid": data_val}
    
    # structure true vals
    
    if use_orig:
        true_discount = torch.stack([t["discount"] for t in true_vals], dim=-1)
        true_learn_rate = torch.stack([t["learning rate"] for t in true_vals], dim=-1)
        true_dec_temp = torch.stack([t["dec temp"] for t in true_vals], dim=-1)
        true_weight = torch.stack([t["weight"] for t in true_vals], dim=-1)
        true_repetition = torch.stack([t["repetition"] for t in true_vals], dim=-1)
        true_prior_lr = torch.stack([t["prior lr"] for t in true_vals], dim=-1)
        true_prior_weight = torch.stack([t["prior weight"] for t in true_vals], dim=-1)
        true_ind = torch.stack([t["subject"] for t in true_vals], dim=-1)
    
        structured_true_vals = {"subject": true_ind, "discount": true_discount, "learning rate": true_learn_rate, 
                            "dec temp": true_dec_temp, "weight": true_weight, "repetition": true_repetition, "prior lr": true_prior_lr, "prior weight": true_prior_weight}

    else:
        true_discount = torch.stack([t["discount"] for t in true_vals], dim=-1)
        true_learn_rate = torch.stack([t["learning rate"] for t in true_vals], dim=-1)
        true_mf_weight = torch.stack([t["mf weight"] for t in true_vals], dim=-1)
        true_mb_weight = torch.stack([t["mb weight"] for t in true_vals], dim=-1)
        true_repetition = torch.stack([t["repetition"] for t in true_vals], dim=-1)
        true_prior_lr = torch.stack([t["prior lr"] for t in true_vals], dim=-1)
        true_prior_weight = torch.stack([t["prior weight"] for t in true_vals], dim=-1)
        true_ind = torch.stack([t["subject"] for t in true_vals], dim=-1)
    
        structured_true_vals = {"subject": true_ind, "discount": true_discount, "learning rate": true_learn_rate, 
                            "mf weight": true_mf_weight, "mb weight": true_mb_weight, "repetition": true_repetition, "prior lr": true_prior_lr, "prior weight": true_prior_weight}
    
    # save to disk
    
    # stayed arr
    fname_stayed = os.path.join(base_dir, "twostage_agent_daw_"+agent_type+"_stayed_arr.json")
    pickled_stayed_arr = pickle.encode(stayed_arr)
    with open(fname_stayed, 'w') as outfile:
        json.dump(pickled_stayed_arr, outfile)
        
    # data 
    fname_data = os.path.join(base_dir, "twostage_agent_daw_"+agent_type+"_data.json")
    pickled_data = pickle.encode(structured_data)
    with open(fname_data, 'w') as outfile:
        json.dump(pickled_data, outfile)
        
    # true values 
    fname_true_vals = os.path.join(base_dir, "twostage_agent_daw_"+agent_type+"_true_vals.json")
    pickled_true_vals = pickle.encode(structured_true_vals)
    with open(fname_true_vals, 'w') as outfile:
        json.dump(pickled_true_vals, outfile)
    
    return stayed_arr, structured_true_vals, structured_data

def run_mfmb_post_pred_simulations(nsubs, agent_type, n_pars, learn_prior, use_orig, use_p, parameter_values, restrict_alpha, min_alpha, fname_base, base_dir, Rho, trials, T, 
                        nb, ns, no, na, npi, nr, never_reward, A, B, mask, p_invalid,
                        max_dt=6, remove_old=True):

    # if it does exist, empty previous results, if we want that (remove_old==True)
    if remove_old:
            
        svgs = glob.glob(os.path.join(base_dir,"*.svg"))
        for file in svgs:
            os.remove(file)
            
        csvs = glob.glob(os.path.join(base_dir,"*.csv"))
        for file in csvs:
            os.remove(file)
            
        saves = glob.glob(os.path.join(base_dir,"*.save"))
        for file in saves:
            os.remove(file)
            
        agents = glob.glob(os.path.join(base_dir,"twostage_agent*"))
        for file in agents:
            os.remove(file)
            
        outputs = glob.glob(os.path.join(base_dir,"*.json"))
        for file in outputs:
            os.remove(file)

    true_discount = torch.from_numpy(parameter_values["inferred discount"].to_numpy())
    true_learn_rate = torch.from_numpy(parameter_values["inferred learning rate"].to_numpy())
    true_mb_weight = torch.from_numpy(parameter_values["inferred mb weight"].to_numpy())
    true_mf_weight = torch.from_numpy(parameter_values["inferred mf weight"].to_numpy())
    true_vals_mfmb_min = torch.stack([true_discount, true_learn_rate, true_mb_weight, true_mf_weight], dim=1)[:,:,None]

    if learn_prior:
        true_prior_lr = torch.from_numpy(parameter_values["inferred prior lr"].to_numpy())
        true_prior_weight = torch.from_numpy(parameter_values["inferred prior weight"].to_numpy())
        true_vals_prior = torch.stack([true_prior_lr, true_prior_weight], dim=1)[:,:,None]
    else:
        true_vals_prior = torch.zeros((nsubs,2,1))

    if use_p:
        true_perserv = torch.from_numpy(parameter_values["inferred perserv"].to_numpy())[:,None,None]
    else:
        true_perserv = torch.zeros((nsubs,1,1))
    
    true_values_tensor = torch.cat([true_vals_mfmb_min, true_vals_prior, true_perserv], dim=1).float()
    
    true_vals = []
    data = []
    
    stayed = []
    indices = []
    
    for i, pars in enumerate(true_values_tensor):
    
        # make parameters for original mb mf: discount lambda, learning rate, dec temp, balancing w, perserveration
        if use_orig:
            discount, lr, dt, weight, prior_lr, dt_prior, perserv = pars
        
            perception_args = {"subject": torch.tensor([i]), "discount": discount, "learning rate": lr, "dec temp": dt, "weight": weight, "repetition": perserv, 
                                "prior lr": prior_lr, "prior weight": dt_prior, "max dt": max_dt, "min learning rate": min_alpha, "learn_prior": learn_prior}
            
        # make parameters for two beta mb mf: discount lambda, learning rate, mb dec temp, mf dec temp, perserveration
        else:

            discount, lr, dt_mf, dt_mb, prior_lr, dt_prior, perserv = pars

            perception_args = {"subject": torch.tensor([i]), "discount": discount, "learning rate": lr, "mf weight": dt_mf, "mb weight": dt_mb, 
                               "prior lr": prior_lr, "prior weight": dt_prior, "repetition": perserv,  "learn_prior": learn_prior,
                               "max dt": max_dt, "min learning rate": min_alpha}
            
        print(perception_args)

        if len(Rho.shape) > 3:
            Rho_subj = Rho[i]
        else:
            Rho_subj = Rho
        
        worlds = []
        l = []
        avg = True
        valid = mask[:,[i]]
        pars = [avg, Rho_subj,perception_args, use_orig, use_p, restrict_alpha, valid]
        
        worlds.append(simulate_mfmb_behavior(pars, trials, T, ns, na, nr, nb, A, B))
        
        w = worlds[-1]
        
        rewarded = w.rewards[:trials-1,-1] == 1
        
        unrewarded = rewarded==False
        
        rare = torch.logical_or(torch.logical_and(w.environment.hidden_states[:trials-1,1]==2, w.actions[:trials-1,0] == 0),
                       torch.logical_and(w.environment.hidden_states[:trials-1,1]==1, w.actions[:trials-1,0] == 1))
        
        common = rare==False
        
        rewarded_common = torch.where(torch.logical_and(rewarded,common) == True)[0]
        rewarded_rare = torch.where(torch.logical_and(rewarded,rare) == True)[0]
        unrewarded_common = torch.where(torch.logical_and(unrewarded,common) == True)[0]
        unrewarded_rare = torch.where(torch.logical_and(unrewarded,rare) == True)[0]
        
        index_list = [rewarded_common, rewarded_rare,
                     unrewarded_common, unrewarded_rare]
        
        stayed_list = [(w.actions[index_list[i],0] == w.actions[index_list[i]+1,0]).sum()/float(len(index_list[i])) for i in range(4)]
        
        stayed.append(stayed_list)
        
        if use_orig:
            run_name = "twostage_agent_daw_"+agent_type+"_"+str(i)+"_disc"+str(discount)+"_lr"+str(lr)+"_dt"+str(dt)+"weight"+str(weight)+"_perserv"+str(perserv)+".json"
        else:
            run_name = "twostage_agent_daw_"+agent_type+"_"+str(i)+"_disc"+str(discount)+"_lr"+str(lr)+"_dt_mf"+str(dt_mf)+"_dt_mb"+str(dt_mb)+"_perserv"+str(perserv)+".json"
        fname_behavior = os.path.join(base_dir, run_name)
        
        data.append({"subject": torch.tensor([i]), "actions": w.actions, "observations": w.observations, "rewards": w.rewards, "states": w.environment.hidden_states, 'valid': valid})
        
        pickled_behavior = pickle.encode(data[-1])
        with open(fname_behavior, 'w') as outfile:
            json.dump(pickled_behavior, outfile)
        
        pickled_behavior = 0
        
        gc.collect()
    
        true_vals.append(perception_args)
    
    stayed_arr = torch.tensor(stayed)
    
    # structure data
    
    data_obs = torch.stack([d["observations"] for d in data], dim=-1)
    data_rew = torch.stack([d["rewards"] for d in data], dim=-1)
    data_act = torch.stack([d["actions"] for d in data], dim=-1)
    data_val = torch.cat([d["valid"] for d in data], dim=-1)
    data_ind = torch.stack([d["subject"] for d in data], dim=-1)

    structured_data = {"subject": data_ind, "observations": data_obs, "rewards": data_rew, "actions": data_act, "valid": data_val}
    
    # structure true vals
    
    if use_orig:
        true_discount = torch.stack([t["discount"] for t in true_vals], dim=-1)
        true_learn_rate = torch.stack([t["learning rate"] for t in true_vals], dim=-1)
        true_dec_temp = torch.stack([t["dec temp"] for t in true_vals], dim=-1)
        true_weight = torch.stack([t["weight"] for t in true_vals], dim=-1)
        true_repetition = torch.stack([t["repetition"] for t in true_vals], dim=-1)
        true_prior_lr = torch.stack([t["prior lr"] for t in true_vals], dim=-1)
        true_prior_weight = torch.stack([t["prior weight"] for t in true_vals], dim=-1)
        true_ind = torch.stack([t["subject"] for t in true_vals], dim=-1)
    
        structured_true_vals = {"subject": true_ind, "discount": true_discount, "learning rate": true_learn_rate, 
                            "dec temp": true_dec_temp, "weight": true_weight, "repetition": true_repetition, "prior lr": true_prior_lr, "prior weight": true_prior_weight}

    else:
        true_discount = torch.stack([t["discount"] for t in true_vals], dim=-1)
        true_learn_rate = torch.stack([t["learning rate"] for t in true_vals], dim=-1)
        true_mf_weight = torch.stack([t["mf weight"] for t in true_vals], dim=-1)
        true_mb_weight = torch.stack([t["mb weight"] for t in true_vals], dim=-1)
        true_repetition = torch.stack([t["repetition"] for t in true_vals], dim=-1)
        true_prior_lr = torch.stack([t["prior lr"] for t in true_vals], dim=-1)
        true_prior_weight = torch.stack([t["prior weight"] for t in true_vals], dim=-1)
        true_ind = torch.stack([t["subject"] for t in true_vals], dim=-1)
    
        structured_true_vals = {"subject": true_ind, "discount": true_discount, "learning rate": true_learn_rate, 
                            "mf weight": true_mf_weight, "mb weight": true_mb_weight, "repetition": true_repetition, "prior lr": true_prior_lr, "prior weight": true_prior_weight}
    
    # save to disk
    
    # stayed arr
    fname_stayed = os.path.join(base_dir, "twostage_agent_daw_"+agent_type+"_stayed_arr.json")
    pickled_stayed_arr = pickle.encode(stayed_arr)
    with open(fname_stayed, 'w') as outfile:
        json.dump(pickled_stayed_arr, outfile)
        
    # data 
    fname_data = os.path.join(base_dir, "twostage_agent_daw_"+agent_type+"_data.json")
    pickled_data = pickle.encode(structured_data)
    with open(fname_data, 'w') as outfile:
        json.dump(pickled_data, outfile)
        
    # true values 
    fname_true_vals = os.path.join(base_dir, "twostage_agent_daw_"+agent_type+"_true_vals.json")
    pickled_true_vals = pickle.encode(structured_true_vals)
    with open(fname_true_vals, 'w') as outfile:
        json.dump(pickled_true_vals, outfile)
    
    return stayed_arr, structured_true_vals, structured_data


def load_simulation_outputs(base_dir, agent_type):

    # stayed arr
    fname_stayed = os.path.join(base_dir, "twostage_agent_daw_"+agent_type+"_stayed_arr.json")
    with open(fname_stayed, 'r') as infile:
        loaded_stayed = json.load(infile)
    stayed_arr = pickle.decode(loaded_stayed)
        
    # data 
    fname_data = os.path.join(base_dir, "twostage_agent_daw_"+agent_type+"_data.json")
    with open(fname_data, 'r') as infile:
        loaded_data = json.load(infile)
    structured_data = pickle.decode(loaded_data)
        
    # true values 
    fname_true_vals = os.path.join(base_dir, "twostage_agent_daw_"+agent_type+"_true_vals.json")
    with open(fname_true_vals, 'r') as infile:
        loaded_true_vals = json.load(infile)
    structured_true_vals = pickle.decode(loaded_true_vals)
    
    return stayed_arr, structured_true_vals, structured_data

def set_up_Bayesian_inference_agent(n_agents, learn_rewards, learn_habit, learn_cached, base_dir, global_experiment_parameters, valid, remove_old=True, use_h=False):

    # if it does exist, empty previous results, if we want that (remove_old==True)
    if remove_old:

        svgs = glob.glob(os.path.join(base_dir,"*.svg"))
        for file in svgs:
            os.remove(file)

        csvs = glob.glob(os.path.join(base_dir,"*.csv"))
        for file in csvs:
            os.remove(file)

        saves = glob.glob(os.path.join(base_dir,"*.save"))
        for file in saves:
            os.remove(file)

        agents = glob.glob(os.path.join(base_dir,"twostage_agent*"))
        for file in agents:
            os.remove(file)

        outputs = glob.glob(os.path.join(base_dir,"*.json"))
        for file in outputs:
            os.remove(file)

    # perception args for init, will instantly be over-written, but have to be set for initialization
    pol_lambda = torch.tensor([0.5])
    r_lambda = torch.tensor([0.5])
    dec_temp = torch.tensor([2.])   
    alpha_0 = torch.tensor([1.])
    c_weight = torch.tensor([1.])
    c_lambda = torch.tensor([0.5])

    perception_args = {"policy rate": pol_lambda, "reward rate": r_lambda, "dec temp": dec_temp, "habitual tendency": alpha_0}
    perception_args = {"dec temp": dec_temp, "reward rate": r_lambda, 
                       "habitual tendency": alpha_0, "policy rate": pol_lambda, 
                       "cached weight": c_weight, "cached rate": c_lambda}

    avg = True

    agent_par_list = [avg, perception_args, learn_rewards, learn_habit, learn_cached, valid, use_h]
    bayes_agent, bayes_perception = set_up_Bayesian_agent(agent_par_list, **global_experiment_parameters, nsubs=n_agents)

    return bayes_agent

def set_up_mbmf_inference_agent(n_agents, learn_prior,use_orig, use_p, restrict_alpha, max_dt, min_alpha, base_dir, global_experiment_parameters, valid, remove_old=True):

    # if it does exist, empty previous results, if we want that (remove_old==True)
    if remove_old:

        svgs = glob.glob(os.path.join(base_dir,"*.svg"))
        for file in svgs:
            os.remove(file)

        csvs = glob.glob(os.path.join(base_dir,"*.csv"))
        for file in csvs:
            os.remove(file)

        saves = glob.glob(os.path.join(base_dir,"*.save"))
        for file in saves:
            os.remove(file)

        agents = glob.glob(os.path.join(base_dir,"twostage_agent*"))
        for file in agents:
            os.remove(file)

        outputs = glob.glob(os.path.join(base_dir,"*.json"))
        for file in outputs:
            os.remove(file)

    # perception args for init, will instantly be over-written, but have to be set for initialization
    if use_orig:
        discount = torch.tensor([0.99])
        lr = torch.tensor([0.05])
        dt = torch.tensor([2.])
        weight = torch.tensor([0.5])
        dt_prior = torch.tensor([0.])
        prior_lr = torch.tensor([0.])
        perserv = torch.tensor([0.1])

        perception_args = {"discount": discount, "learning rate": lr, "dec temp": dt, "weight": weight, 
                                "prior lr": prior_lr, "prior weight": dt_prior, "repetition": perserv,  "learn_prior": learn_prior,
                                "max dt": max_dt, "min learning rate": min_alpha}
    else:
        discount = torch.tensor([0.99])
        lr = torch.tensor([0.05])
        dt_mf = torch.tensor([2.])
        dt_mb = torch.tensor([2.])
        dt_prior = torch.tensor([0.])
        prior_lr = torch.tensor([0.])
        perserv = torch.tensor([0.1])

        perception_args = {"discount": discount, "learning rate": lr, "mf weight": dt_mf, "mb weight": dt_mb, 
                                "prior lr": prior_lr, "prior weight": dt_prior, "repetition": perserv,  "learn_prior": learn_prior,
                                "max dt": max_dt, "min learning rate": min_alpha}
    
    avg = True

    agent_par_list = [avg, perception_args, use_orig, use_p, restrict_alpha, valid]
    mbmf_agent, perception = set_up_mfmb_agent(agent_par_list, **global_experiment_parameters, nsubs=n_agents)

    return mbmf_agent

