#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 21:18:35 2019

@author: sarah
"""

import os
import numpy as np
import jsonpickle as pickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import json
from misc import *
import scipy as sc
import matplotlib.pylab as plt
import scipy.special as scs
import seaborn as sns
from scipy import stats
import sklearn.linear_model as lm
import statsmodels.api as sm
import pandas as pd
#plt.style.use('seaborn-whitegrid')
#sns.set_style("whitegrid", {"axes.edgecolor": "0.15"})#, "axes.spines.top": "False", "axes.spines.right": "False"})
sns.set_style("ticks")


save_figs = False

folder = "data"


fit_functions = [sigmoid, exponential]

def analyze_run(fname, save=False):

    jsonpickle_numpy.register_handlers()

    with open(fname, 'r') as infile:
        data = json.load(infile)
    worlds = pickle.decode(data)

    w = worlds[0]

    T = w.T
    trials = w.trials
    t_trials = trials - 100

    Rho = w.environment.Rho

    repetitions = len(worlds)
    results_won = np.zeros((repetitions,3))
    results_chosen = np.zeros((repetitions,3))
    results_stayed = np.zeros(repetitions)
    results_context = np.zeros((repetitions,2))
    results_c_param = np.zeros((repetitions,4))
    results_c_param_type = np.zeros(repetitions, dtype=int)
    results_p_param = np.zeros((repetitions,4))
    results_p_param_type = np.zeros(repetitions, dtype=int)
    entropy_c = np.zeros(repetitions)
    entropy_p = np.zeros(repetitions)
    entropy_l = np.zeros(repetitions)

    times = np.arange(0.+1,trials+1,1.)

    best_options = np.amax(np.argmax(Rho, axis=1), axis=1)-1

    for i in range(repetitions):
        #print(i)
        w = worlds[i]
        results_won[i,0] = (w.rewards[trials//4:trials//2] >0).sum()/(trials//4*(T-1))
        results_won[i,1] = (w.rewards[3*trials//4:] >0).sum()/(trials//4*(T-1))
        stayed = np.array([((w.actions[i,0] - w.actions[i+1,0])==0) for i in range(trials-1)])
        results_stayed[i] = stayed.sum()/(trials * (T-1))
        results_chosen[i,0] = np.array([w.actions[i,j] == best_options[i] for i in range(trials-100) for j in range(T-1)]).sum()/((trials-100)*(T-1))
        results_chosen[i,1] = np.array([w.actions[i,j] == best_options[i] for i in range(trials-100,trials-100+15) for j in range(T-1)]).sum()/((15)*(T-1))
        results_chosen[i,2] = np.array([w.actions[i,j] == best_options[i] for i in range(trials-100+15,trials) for j in range(T-1)]).sum()/((85)*(T-1))
        results_context[i,0] = np.array([w.agent.posterior_context[i,j,0] for i in range(trials//2) for j in range(T-1)]).sum()/(trials//2*(T-1))
        results_context[i,1] = np.array([w.agent.posterior_context[i,j,0] for i in range(trials//2,trials) for j in range(T-1)]).sum()/(trials//2*(T-1))
        if T > 2:
            stayed = np.array([((w.actions[i,j] - w.actions[i,j+1])==0) for i in range(trials-1) for j in range(T-2)])
            results_stayed[i] += stayed.sum()/(trials * (T-1))

        posterior_context = w.agent.posterior_context[:,1,:]
        entropy_c[i] = -(posterior_context * ln(posterior_context)).sum(axis=1).mean()

        posterior_pol = (w.agent.posterior_policies[:,0,0:]*w.agent.posterior_context[:,0,np.newaxis,:]).sum(axis=-1)
        entropy_p[i] = -(posterior_pol * ln(posterior_pol)).sum(axis=1).mean()

        likelihood = (w.agent.likelihood[:,0,0:,:]*w.agent.posterior_context[:,0,np.newaxis,:]).sum(axis=-1)
        likelihood /= likelihood.sum(axis=1)[:,np.newaxis]
        entropy_l[i] = -(likelihood * ln(likelihood)).sum(axis=1).mean()

        try:
            posterior_context = w.agent.posterior_context[:,1,1]#bn.move_mean(w.agent.posterior_context[:,1,1],10,1)
            results_c_param[i], pcov = sc.optimize.curve_fit(sigmoid, times, posterior_context, p0=[1.,1.,t_trials,0.])#, sigma=[0.5]*200)#*40+[2]*100+[0.5]*50)
            results_c_param_type[i] = 0
        except RuntimeError:
            try:
                results_c_param[i,1:], pcov = sc.optimize.curve_fit(exponential, times, posterior_context, p0=[1.,t_trials,0.])#, sigma=[0.5]*40+[2]*100+[0.5]*50)
                results_c_param[i,0] = 1.
                results_c_param_type[i] = 1
            except RuntimeError:
                results_c_param[i] = np.nan
        try:
            posterior_pol = (w.agent.posterior_policies[:,0,1]*w.agent.posterior_context[:,0]).sum(axis=1)[10:]
            results_p_param[i], pcov = sc.optimize.curve_fit(sigmoid, times[10:], posterior_pol, p0=[1.,1.,t_trials,0.])#, sigma=[0.5]*40+[2]*100+[0.5]*50)
            results_p_param_type[i] = 0
        except RuntimeError:
            try:
                results_p_param[i,1:], pcov = sc.optimize.curve_fit(exponential, times[10:], posterior_pol, p0=[1.,t_trials,0.])#, sigma=[0.5]*40+[2]*100+[0.5]*50)
                results_p_param[i,0] = 1.
                results_p_param_type[i] = 1
            except RuntimeError:
                results_p_param[i] = np.nan

        if results_c_param[i,0] < 0.1 or results_c_param[i,1] < 0.0 or results_c_param[i,2] < 15 or results_c_param[i,2] > trials:
            results_c_param[i] = [0,0,trials+1,0]

        if results_p_param[i,0] < 0.1 or results_p_param[i,1] < 0.0 or results_p_param[i,2] < 15 or results_p_param[i,2] > trials:
            results_p_param[i] = [0,0,trials+1,0]

        if save:

            results = [results_won, results_chosen, results_context, \
                       results_c_param, results_c_param_type, entropy_c, \
                       results_p_param, results_p_param_type, entropy_p, entropy_l]

            analysis_name = fname[:-5] + "_ananlysis.json"

            jsonpickle_numpy.register_handlers()
            pickled = pickle.encode(results)
            with open(analysis_name, 'w') as outfile:
                json.dump(pickled, outfile)

def analyze_check(fname, reference_name, check=False, naive=False, save=False):

    jsonpickle_numpy.register_handlers()

    with open(fname, 'r') as infile:
        data = json.load(infile)
    worlds = pickle.decode(data)

    w = worlds[0]

    T = w.T
    trials = w.trials

    if check and naive:
        trials = trials//2

    Rho = w.environment.Rho

    with open(reference_name, 'r') as infile:
        data = json.load(infile)

    results_won_t, results_chosen_t, results_context_t, \
    results_c_param_t, results_c_param_type_t, entropy_c_t, \
    results_p_param_t, results_p_param_type_t, entropy_p_t, entropy_l_t = pickle.decode(data)


    repetitions = len(worlds)
    print(repetitions)
    results_won = np.zeros((repetitions,3))
    results_chosen = np.zeros((repetitions))
    results_stayed = np.zeros(repetitions)
    results_context = np.zeros((repetitions))
    results_c_param = np.zeros((repetitions,4))
    results_c_param_type = np.zeros(repetitions, dtype=int)
    results_p_param = np.zeros((repetitions,4))
    results_p_param_type = np.zeros(repetitions, dtype=int)
    entropy_c = np.zeros(repetitions)
    entropy_p = np.zeros(repetitions)
    entropy_l = np.zeros(repetitions)

    times = np.arange(0.+1,trials+1,1.)

    best_options = np.amax(np.argmax(Rho, axis=1), axis=1)-1

    for i in range(repetitions):
        #print(i)
        w = worlds[i]
        results_won[i,0] = (w.rewards[:] >0).sum()/(trials*(T-1))
        results_won[i,1] = (w.rewards[:] >0).sum()/(trials*(T-1))
        stayed = np.array([((w.actions[i,0] - w.actions[i+1,0])==0) for i in range(trials-1)])
        results_stayed[i] = stayed.sum()/(trials * (T-1))
        results_chosen[i] = np.array([w.actions[i,j] == best_options[i] for i in range(trials) for j in range(T-1)]).sum()/(trials*(T-1))
        results_context[i] = np.array([w.agent.posterior_context[i,j,0] for i in range(trials) for j in range(T-1)]).sum()/(trials*(T-1))
        if T > 2:
            stayed = np.array([((w.actions[i,j] - w.actions[i,j+1])==0) for i in range(trials-1) for j in range(T-2)])
            results_stayed[i] += stayed.sum()/(trials * (T-1))

        posterior_context = w.agent.posterior_context[:,1,:]
        entropy_c[i] = -(posterior_context * ln(posterior_context)).sum(axis=1).mean()

        posterior_pol = (w.agent.posterior_policies[:,0,0:]*w.agent.posterior_context[:,0,np.newaxis,:]).sum(axis=-1)
        entropy_p[i] = -(posterior_pol * ln(posterior_pol)).sum(axis=1).mean()

        likelihood = (w.agent.likelihood[:,0,0:,:]*w.agent.posterior_context[:,0,np.newaxis,:]).sum(axis=-1)
        likelihood /= likelihood.sum(axis=1)[:,np.newaxis]
        entropy_l[i] = -(likelihood * ln(likelihood)).sum(axis=1).mean()

        threshold_c = fit_functions[results_c_param_type_t[i]](0, *results_c_param_t[i,results_c_param_type_t[i]:])
        switch_time = results_c_param_t[i,2]
        if threshold_c < 0.5 and switch_time <=200:
            posterior_context = w.agent.posterior_context[:,1,1]#bn.move_mean(w.agent.posterior_context[:,1,1],10,1)
            if threshold_c < 0.001:
                threshold_c = 0.001
            time = np.where(posterior_context[:trials]<=threshold_c)[0]
            if len(time)>0:
                results_c_param[i,2] = time[0]
            else:
                results_c_param[i,2] = 101
        else:
            results_c_param[i,2] = np.nan

        threshold_p = fit_functions[results_p_param_type_t[i]](0, *results_p_param_t[i,results_p_param_type_t[i]:])
        switch_time = results_p_param_t[i,2]
        if threshold_p < 0.5 and switch_time <=200:
            posterior_pol = (w.agent.posterior_policies[:,0,1]*w.agent.posterior_context[:,0]).sum(axis=1)
            if threshold_p < 0.001:
                threshold_p = 0.001
            time = np.where(posterior_pol[:trials]<=threshold_p)[0]
            if len(time)>0:
                results_p_param[i,2] = time[0]
            else:
                results_p_param[i,2] = 101
        else:
            results_p_param[i,2] = np.nan


        if save:

            results = [results_won, results_chosen, results_context, \
                       results_c_param, results_c_param_type, entropy_c, \
                       results_p_param, results_p_param_type, entropy_p, entropy_l]

            analysis_name = fname[:-5] + "_ananlysis_check.json"

            jsonpickle_numpy.register_handlers()
            pickled = pickle.encode(results)
            with open(analysis_name, 'w') as outfile:
                json.dump(pickled, outfile)


def plot_beliefs(fname, plot_context=True, plot_policies=True, plot_actions=True, plot_prior_pol=True, fit_params=None, save_num=-1):

    if fit_params is not None:
        results_c_param, results_c_param_type, results_p_param, results_p_param_type = fit_params

    jsonpickle_numpy.register_handlers()

    with open(fname, 'r') as infile:
        data = json.load(infile)
    worlds = pickle.decode(data)

    w = worlds[0]

    T = w.T
    trials = w.trials

    Rho = w.environment.Rho

    repetitions = len(worlds)

    times = np.arange(0.+1,trials+1,1.)

    arm_cols = ['#007ecdff','#0000b7ff']

    for i in range(repetitions):
        print(i)

        w = worlds[i]

        if plot_policies:
            plt.figure(figsize=(10,5))
            for k in range(1,w.agent.nh):
                plt.plot(w.environment.Rho[:,k,k], label="lever "+str(k), c=arm_cols[k-1], linewidth=3)
            for t in range(w.agent.T-1):
                plt.plot((w.agent.posterior_policies[:,t,1]* w.agent.posterior_context[:,t]).sum(axis=1), ".", label="action", color='darkorange')
            if fit_params is not None:
                print(results_p_param[i])
                fct = fit_functions[results_p_param_type[i]]
                param = results_p_param[i,results_p_param_type[i]:]
                plt.plot(fct(times, *param))
            plt.ylim([-0.1,1.1])
            lgd = plt.legend(fontsize=16, bbox_to_anchor=(1.04,1), loc="upper left", ncol=1) #bbox_to_anchor=(0, 1.02, 1, 0.2), mode="expand"
            plt.yticks(fontsize=18)
            plt.xticks(fontsize=18)
            plt.xlabel("trial", fontsize=20)
            plt.ylabel("reward probabilities", fontsize=20)
            ax = plt.gca().twinx()
            ax.set_ylim([-0.1,1.1])
            ax.set_yticks([0,1])
            ax.set_yticklabels(["$a_{1}$","$a_{2}$"],fontsize=18)
            ax.yaxis.set_ticks_position('right')
            if save_num == i:
                    plt.savefig(os.path.join(folder,fname[:-5]+"_run"+str(i)+"_context.svg"), bbox_extra_artists=(lgd,), bbox_inches='tight')
                    plt.savefig(os.path.join(folder,fname[:-5]+"_run"+str(i)+"_context.png"), bbox_extra_artists=(lgd,), bbox_inches='tight')
            plt.show()

#        if plot_actions:
#            plt.figure(figsize=(10,5))
#            for k in range(1,w.agent.nh):
#                plt.plot(w.environment.Rho[:,k,k], label="lever "+str(k), c=arm_cols[k-1], linewidth=3)
#            for t in range(w.agent.T-1):
#                plt.plot((w.actions[:,t]-1), ".", label="action", color='darkorange')
#            plt.ylim([-0.1,1.1])
#            lgd = plt.legend(fontsize=16, bbox_to_anchor=(1.04,1), loc="upper left", ncol=1) #bbox_to_anchor=(0, 1.02, 1, 0.2), mode="expand"
#            plt.yticks(fontsize=18)
#            plt.xticks(fontsize=18)
#            plt.xlabel("trial", fontsize=20)
#            plt.ylabel("reward probabilities", fontsize=20)
#            ax = plt.gca().twinx()
#            ax.set_ylim([-0.1,1.1])
#            ax.set_yticks([0,1])
#            ax.set_yticklabels(["$a_{1}$","$a_{2}$"],fontsize=18)
#            ax.yaxis.set_ticks_position('right')
#            plt.show()
#
#        if plot_prior_pol:
#            plt.figure(figsize=(10,5))
#            for k in range(1,w.agent.nh):
#                plt.plot(w.environment.Rho[:,k,k], label="lever "+str(k), c=arm_cols[k-1], linewidth=3)
#            prior_policies = np.exp(scs.digamma(w.agent.posterior_dirichlet_pol) - scs.digamma(w.agent.posterior_dirichlet_pol.sum(axis=1))[:,np.newaxis,:])
#            prior_policies /= prior_policies.sum(axis=1)[:,np.newaxis,:]
#            plt.plot((prior_policies[:,2]), ".", label="action 2", color='darkorange')
#            plt.plot((prior_policies[:,1]), ".", label="action 1", color='red')
#            plt.ylim([-0.1,1.1])
#            lgd = plt.legend(fontsize=16, bbox_to_anchor=(1.04,1), loc="upper left", ncol=1) #bbox_to_anchor=(0, 1.02, 1, 0.2), mode="expand"
#            plt.yticks(fontsize=18)
#            plt.xticks(fontsize=18)
#            plt.xlabel("trial", fontsize=20)
#            plt.ylabel("reward probabilities", fontsize=20)
#            ax = plt.gca().twinx()
#            ax.set_ylim([-0.1,1.1])
#            ax.set_yticks([0,1])
#            ax.set_yticklabels(["$a_{1}$","$a_{2}$"],fontsize=18)
#            ax.yaxis.set_ticks_position('right')
#            plt.show()

        if plot_context:
            plt.figure(figsize=(10,5))
            for k in range(1,w.agent.nh):
                plt.plot(w.environment.Rho[:,k,k], label="lever "+str(k), c=arm_cols[k-1], linewidth=3)
            for t in range(1,w.agent.T):
                plt.plot(w.agent.posterior_context[:,t,1], ".", label="context", color='deeppink')
            if fit_params is not None:
                print(results_c_param[i])
                fct = fit_functions[results_c_param_type[i]]
                param = results_c_param[i,results_c_param_type[i]:]
                plt.plot(fct(times, *param))
            plt.ylim([-0.1,1.1])
            lgd = plt.legend(fontsize=16, bbox_to_anchor=(1.04,1), loc="upper left", ncol=1) #bbox_to_anchor=(0, 1.02, 1, 0.2), mode="expand"
            plt.yticks(fontsize=18)
            plt.xticks(fontsize=18)
            plt.xlabel("trial", fontsize=20)
            plt.ylabel("reward probabilities", fontsize=20)
            ax = plt.gca().twinx()
            ax.set_ylim([-0.1,1.1])
            ax.set_yticks([0,1])
            ax.set_yticklabels(["$c_{1}$","$c_{2}$"],fontsize=18)
            ax.yaxis.set_ticks_position('right')
            if save_num == i:
                    plt.savefig(os.path.join(folder,fname[:-5]+"_run"+str(i)+"_action.svg"), bbox_extra_artists=(lgd,), bbox_inches='tight')
                    plt.savefig(os.path.join(folder,fname[:-5]+"_run"+str(i)+"_action.png"), bbox_extra_artists=(lgd,), bbox_inches='tight')
            plt.show()

def save_run_plots(fnames, save_nums, tendencies, prefix="", check=False, plot_context=True, plot_policies=True, plot_prior=True, plot_likelihood=True, plot_actions=True, fit_params=None):

    w_runs = []
    w_checks = []
    for i,f in enumerate(fnames):
        jsonpickle_numpy.register_handlers()

        fname = os.path.join(folder, f)

        with open(fname, 'r') as infile:
            data = json.load(infile)
        worlds = pickle.decode(data)

        w_runs.append(worlds[save_nums[i]])

        if check:

            check_name = 'check_'+f
            fname = os.path.join(folder, check_name)

            with open(fname, 'r') as infile:
                data = json.load(infile)
            worlds = pickle.decode(data)

            w_checks.append(worlds[save_nums[i]])


    if check:
        name_prefix = 'check_'
    else:
        name_prefix = ''

    arm_cols = ['#007ecdff','#0000b7ff']

    action_cols = ['#cc6600']#['#993d00', '#ffa366']#['#993d00', '#ff6600', '#ffa366']

    context_cols = ['#ff1493']#['#99004d', '#ff66b3']#['#99004d', '#ff0080', '#ff66b3']

    for i in range(len(w_runs)):

        w = w_runs[i]

        trials = w.trials

        Rho = w.environment.Rho

        times = np.arange(0.+1,trials+1,1.)

        actions = w.actions[:,0]

        post_pol = (w.agent.posterior_policies[:,0,:,:]* w.agent.posterior_context[:,0,np.newaxis,:]).sum(axis=-1)

        prior_pol = np.exp(scs.digamma(w.agent.posterior_dirichlet_pol) - scs.digamma(w.agent.posterior_dirichlet_pol.sum(axis=1))[:,np.newaxis,:])
        prior_pol /= prior_pol.sum(axis=1)[:,np.newaxis,:]
        marginal_prior = (prior_pol[:,:,:] * w.agent.posterior_context[:,0,np.newaxis,:]).sum(axis=-1)

        likelihood = (w.agent.likelihood[:,0,:,:]* w.agent.posterior_context[:,0,np.newaxis,:]).sum(axis=-1)
        likelihood /= likelihood.sum(axis=1)[:,np.newaxis]

        posterior_context = w.agent.posterior_context[:,1,1]

        if check:

            w_check = w_checks[i]

            check_trials = w_check.trials

            Rho = np.append(Rho, w_check.environment.Rho, axis=0)

            times = np.arange(1,times[-1]+check_trials+1,1.)

            actions = np.append(actions, w_check.actions[:,0])

            post_pol_check = (w_check.agent.posterior_policies[:,0,:,:]* w_check.agent.posterior_context[:,0,np.newaxis,:]).sum(axis=-1)
            post_pol = np.append(post_pol, post_pol_check, axis=0)

            prior_pol = np.exp(scs.digamma(w_check.agent.posterior_dirichlet_pol) - scs.digamma(w_check.agent.posterior_dirichlet_pol.sum(axis=1))[:,np.newaxis,:])
            prior_pol /= prior_pol.sum(axis=1)[:,np.newaxis,:]
            marginal_prior_check = (prior_pol[:,:,:] * w_check.agent.posterior_context[:,0,np.newaxis,:]).sum(axis=-1)
            marginal_prior = np.append(marginal_prior, marginal_prior_check, axis=0)

            likelihood_check = (w_check.agent.likelihood[:,0,:,:]* w_check.agent.posterior_context[:,0,np.newaxis,:]).sum(axis=-1)
            likelihood_check /= likelihood_check.sum(axis=1)[:,np.newaxis]
            likelihood = np.append(likelihood, likelihood_check, axis=0)

            posterior_context = np.append(posterior_context, w_check.agent.posterior_context[:,1,1])


        with sns.axes_style("ticks"):
            if plot_actions:
                plt.figure(figsize=(10,5))
                for k in range(1,w.agent.nh):
                    plt.plot(times, Rho[:,k,k], label="lever "+str(k), c=arm_cols[k-1], linewidth=4, alpha=0.5)
                plt.plot(actions, ".", label="action", color=action_cols[0], ms=10)
                plt.ylim([-0.01,1.01])
                plt.xlim([0,times[-1]])
                plt.yticks([0.0,0.5,1.0])
                lgd = plt.legend(fontsize=16, bbox_to_anchor=(1.04,1), loc="upper left", ncol=1) #bbox_to_anchor=(0, 1.02, 1, 0.2), mode="expand"
                plt.yticks(fontsize=18)
                plt.xticks(fontsize=18)
                plt.xlabel("trial", fontsize=20)
                plt.ylabel("probability", fontsize=20)
                ax = plt.gca().twinx()
                ax.set_ylim([-0.01,1.01])
                ax.set_yticks([0,1])
                ax.set_yticklabels(["$a_{1}$","$a_{2}$"],fontsize=18)
                ax.yaxis.set_ticks_position('right')
                ax.set_ylabel("action", fontsize=22, rotation=270)
                plt.title("chosen actions", fontsize=22)
                plt.savefig(os.path.join(folder,name_prefix+prefix+"_actions.svg"), bbox_extra_artists=(lgd,), bbox_inches='tight')
                plt.savefig(os.path.join(folder,name_prefix+prefix+"_actions.png"), bbox_extra_artists=(lgd,), bbox_inches='tight')
                plt.show()

            if plot_policies:
                plt.figure(figsize=(10,5))
                for k in range(1,w.agent.nh):
                    plt.plot(times, Rho[:,k,k], label="lever "+str(k), c=arm_cols[k-1], linewidth=4, alpha=0.5)
                plt.plot(times, post_pol[:,1], ".", label="h="+tendencies[i], color=action_cols[0], ms=10)
                if fit_params is not None:
                    results_c_param, results_c_param_type, results_p_param, results_p_param_type = fit_params[i]
                    fct = fit_functions[results_p_param_type]
                    param = results_p_param[results_p_param_type:]
                    plt.plot(fct(times, *param), color=action_cols[0], linewidth=3)
                    print("action switch time", round(results_p_param[2]))
                plt.ylim([-0.01,1.01])
                plt.xlim([0,times[-1]])
                plt.yticks([0.0,0.5,1.0])
                lgd = plt.legend(fontsize=16, bbox_to_anchor=(1.04,1), loc="upper left", ncol=1) #bbox_to_anchor=(0, 1.02, 1, 0.2), mode="expand"
                plt.yticks(fontsize=18)
                plt.xticks(fontsize=18)
                plt.xlabel("trial", fontsize=20)
                plt.ylabel("probability", fontsize=20)
                ax = plt.gca().twinx()
                ax.set_ylim([-0.01,1.01])
                ax.set_yticks([0,1])
                ax.set_yticklabels(["$a_{1}$","$a_{2}$"],fontsize=18)
                ax.yaxis.set_ticks_position('right')
                ax.set_ylabel("action", fontsize=22, rotation=270)
                plt.title("posterior over actions", fontsize=22)
                plt.savefig(os.path.join(folder,name_prefix+prefix+"_posterior_actions.svg"), bbox_extra_artists=(lgd,), bbox_inches='tight')
                plt.savefig(os.path.join(folder,name_prefix+prefix+"_posterior_actions.png"), bbox_extra_artists=(lgd,), bbox_inches='tight')
                plt.show()

            if plot_prior:
                plt.figure(figsize=(10,5))
                for k in range(1,w.agent.nh):
                    plt.plot(times, Rho[:,k,k], label="lever "+str(k), c=arm_cols[k-1], linewidth=4, alpha=0.5)
                plt.plot(times, marginal_prior[:,1], ".", label="h="+tendencies[i], color=action_cols[0], ms=10)
#                if fit_params is not None:
#                    results_c_param, results_c_param_type, results_p_param, results_p_param_type = fit_params[i]
#                    fct = fit_functions[results_p_param_type]
#                    param = results_p_param[results_p_param_type:]
#                    plt.plot(fct(times, *param), color=action_cols[i])
                plt.ylim([-0.01,1.01])
                plt.xlim([0,times[-1]])
                plt.yticks([0.0,0.5,1.0])
                lgd = plt.legend(fontsize=16, bbox_to_anchor=(1.04,1), loc="upper left", ncol=1) #bbox_to_anchor=(0, 1.02, 1, 0.2), mode="expand"
                plt.yticks(fontsize=18)
                plt.xticks(fontsize=18)
                plt.xlabel("trial", fontsize=20)
                plt.ylabel("probability", fontsize=20)
                ax = plt.gca().twinx()
                ax.set_ylim([-0.01,1.01])
                ax.set_yticks([0,1])
                ax.set_yticklabels(["$a_{1}$","$a_{2}$"],fontsize=18)
                ax.yaxis.set_ticks_position('right')
                ax.set_ylabel("action", fontsize=22, rotation=270)
                plt.title("prior over actions", fontsize=22)
                plt.savefig(os.path.join(folder,name_prefix+prefix+"_prior_actions.svg"), bbox_extra_artists=(lgd,), bbox_inches='tight')
                plt.savefig(os.path.join(folder,name_prefix+prefix+"_prior_actions.png"), bbox_extra_artists=(lgd,), bbox_inches='tight')
                plt.show()

            if plot_likelihood:
                plt.figure(figsize=(10,5))
                for k in range(1,w.agent.nh):
                    plt.plot(times, Rho[:,k,k], label="lever "+str(k), c=arm_cols[k-1], linewidth=4, alpha=0.5)
                plt.plot(times, likelihood[:,1], ".", label="h="+tendencies[i], color=action_cols[0], ms=10)
#                if fit_params is not None:
#                    results_c_param, results_c_param_type, results_p_param, results_p_param_type = fit_params[i]
#                    fct = fit_functions[results_p_param_type]
#                    param = results_p_param[results_p_param_type:]
#                    plt.plot(fct(times, *param), color=action_cols[i])
                plt.ylim([-0.01,1.01])
                plt.xlim([0,times[-1]])
                plt.yticks([0.0,0.5,1.0])
                lgd = plt.legend(fontsize=16, bbox_to_anchor=(1.04,1), loc="upper left", ncol=1) #bbox_to_anchor=(0, 1.02, 1, 0.2), mode="expand"
                plt.yticks(fontsize=18)
                plt.xticks(fontsize=18)
                plt.xlabel("trial", fontsize=20)
                plt.ylabel("probability", fontsize=20)
                ax = plt.gca().twinx()
                ax.set_ylim([-0.01,1.01])
                ax.set_yticks([0,1])
                ax.set_yticklabels(["$a_{1}$","$a_{2}$"],fontsize=18)
                ax.yaxis.set_ticks_position('right')
                ax.set_ylabel("action", fontsize=22, rotation=270)
                plt.title("likelihood over actions", fontsize=22)
                plt.savefig(os.path.join(folder,name_prefix+prefix+"_likelihood_actions.svg"), bbox_extra_artists=(lgd,), bbox_inches='tight')
                plt.savefig(os.path.join(folder,name_prefix+prefix+"_likelihood_actions.png"), bbox_extra_artists=(lgd,), bbox_inches='tight')
                plt.show()

            if plot_context:
                plt.figure(figsize=(10,5))
                for k in range(1,w.agent.nh):
                    plt.plot(times, Rho[:,k,k], label="lever "+str(k), c=arm_cols[k-1], linewidth=4, alpha=0.5)
                plt.plot(times, posterior_context, ".", label="h="+tendencies[i], color=context_cols[0], ms=10)
                if fit_params is not None:
                    results_c_param, results_c_param_type, results_p_param, results_p_param_type = fit_params[i]
                    fct = fit_functions[results_c_param_type]
                    param = results_c_param[results_c_param_type:]
                    plt.plot(fct(times, *param), color=context_cols[0], linewidth=3)
                    print("context switch time", round(results_c_param[2]))
                plt.ylim([-0.01,1.01])
                plt.xlim([0,times[-1]])
                plt.yticks([0.0,0.5,1.0])
                lgd = plt.legend(fontsize=16, bbox_to_anchor=(1.04,1), loc="upper left", ncol=1) #bbox_to_anchor=(0, 1.02, 1, 0.2), mode="expand"
                plt.yticks(fontsize=18)
                plt.xticks(fontsize=18)
                plt.xlabel("trial", fontsize=20)
                plt.ylabel("probability", fontsize=20)
                ax = plt.gca().twinx()
                ax.set_ylim([-0.01,1.01])
                ax.set_yticks([0,1])
                ax.set_yticklabels(["$c_{1}$","$c_{2}$"],fontsize=18)
                ax.yaxis.set_ticks_position('right')
                ax.set_ylabel("context", fontsize=22, rotation=270)
                plt.title("posterior over contexts", fontsize=22)
                plt.savefig(os.path.join(folder,name_prefix+prefix+"_posterior_context.svg"), bbox_extra_artists=(lgd,), bbox_inches='tight')
                plt.savefig(os.path.join(folder,name_prefix+prefix+"_posterior_context.png"), bbox_extra_artists=(lgd,), bbox_inches='tight')
                plt.show()


def plot_run(check=False, naive=False):

    h = 1
    p = 99
    rew_p = 1
    prefix = "deval"#'sudden'

    if naive:
        test = "test_"
        post = "_check"
    elif check:
        test = "check_"
        post = "_check"
    else:
        test = "test_"
        post = ""

    run_name = test+prefix+"_h"+str(h)+"_t"+str(p)+"_r"+str(rew_p)+"_p90.json"

    analysis_name = run_name[:-5] + "_ananlysis"+post+".json"

    fname = os.path.join(folder, run_name)

    if analysis_name in os.listdir(folder):

        ana_fname = os.path.join(folder, analysis_name)

        jsonpickle_numpy.register_handlers()

        with open(ana_fname, 'r') as infile:
            data = json.load(infile)

        results = pickle.decode(data)

        results_won, results_chosen, results_context, \
        results_c_param, results_c_param_type, \
        results_p_param, results_p_param_type = results

        fit_params = [results_c_param, results_c_param_type, \
                      results_p_param, results_p_param_type ]

        print(np.nanmedian(results_chosen[:]))
        print(np.nanmedian(results_c_param[:,2]))

    else:
        fit_params = None

    plot_beliefs(fname, plot_context=True, plot_policies=True, plot_actions=False, plot_prior_pol=True, fit_params=fit_params)


def calculate_analyses(tendencies, p_values=None, reward_probs=None, trainings=None, check=False, naive=False, deval=False, recalc=False):

    if naive:
        test = ""
        post = "_check"
    elif check:
        test = "check_"
        post = "_check"
    elif deval:
        test = "deval_"
        post = ""
    else:
        test = ""
        post = ""

    if trainings is None:
        trainings = [100]

    if reward_probs is None:
        reward_probs = [90]

    if p_values is None:
        p_values = [99]

    for i,h in enumerate(tendencies):

        for j,p in enumerate(p_values):

            for m, r in enumerate(reward_probs):

                for k,t in enumerate(trainings):

                    print(h, p, r, t)

                    run_name = test+"h"+str(h)+"_t"+str(p)+"_p"+str(r)+"_train"+str(t)+".json"

                    analysis_name = run_name[:-5] + "_ananlysis"+post+".json"

                    if analysis_name in os.listdir(folder):
                        time_diff  = os.path.getmtime(os.path.join(folder, analysis_name)) - os.path.getmtime(os.path.join(folder, run_name))
                        if time_diff <= 0:
                            new = True
                        else:
                            new = False

                    if analysis_name not in os.listdir(folder) or recalc or new:

                        fname = os.path.join(folder, run_name)

                        if check:

                            run_name_training = "h"+str(h)+"_t"+str(p)+"_p"+str(r)+"_train"+str(t)+".json"

                            analysis_name_training = run_name_training[:-5] + "_ananlysis.json"
                            training_analysis = os.path.join(folder, analysis_name_training)

                            analyze_check(fname, training_analysis, check=check, naive=naive, save=True)
                        else:
                            analyze_run(fname, save=True)


def load_analyses(tendencies, p_values, reward_probs=None, trainings=None, check=False, naive=False, deval=False):

    if naive:
        test = ""
        post = "_check"
    elif check:
        test = "check_"
        post = "_check"
    elif deval:
        test = "deval_"
        post = ""
    else:
        test = ""
        post = ""

    if trainings is None:
        trainings = [100]

    if reward_probs is None:
        reward_probs = [90]

    run_name = test+"h"+str(tendencies[0])+"_t"+str(p_values[0]) \
               +"_p"+str(reward_probs[0])+"_train"+str(trainings[0])+".json"

    jsonpickle_numpy.register_handlers()

    fname = os.path.join(folder, run_name)

    with open(fname, 'r') as infile:
        data = json.load(infile)
    worlds = pickle.decode(data)
    repetitions = len(worlds)

    results_c = np.zeros((len(tendencies),len(p_values),len(reward_probs),len(trainings),repetitions ,4))
    results_p = np.zeros((len(tendencies),len(p_values),len(reward_probs),len(trainings),repetitions,4))

    entropy_c = np.zeros((len(tendencies),len(p_values),len(reward_probs),len(trainings),repetitions))
    entropy_p = np.zeros((len(tendencies),len(p_values),len(reward_probs),len(trainings),repetitions))
    entropy_l = np.zeros((len(tendencies),len(p_values),len(reward_probs),len(trainings),repetitions))

    if check:
        chosen = np.zeros((len(tendencies),len(p_values),len(reward_probs),len(trainings),repetitions))
    else:
        chosen = np.zeros((len(tendencies),len(p_values),len(reward_probs),len(trainings),repetitions,3))

    for i,h in enumerate(tendencies):

        for j,p in enumerate(p_values):

            for m,r in enumerate(reward_probs):

                for k,t in enumerate(trainings):

                    print(h, p, r, t)

                    run_name = test+"h"+str(h)+"_t"+str(p)+"_p"+str(r)+"_train"+str(t)+".json"

                    analysis_name = run_name[:-5]+"_ananlysis"+post+".json"

                    fname = os.path.join(folder, analysis_name)

                    jsonpickle_numpy.register_handlers()

                    try:
                        with open(fname, 'r') as infile:
                            data = json.load(infile)

                    except json.JSONDecodeError:
                        print("corrupt file ... doing recalculation")

                        calculate_analyses([h], [p], reward_probs=[r], check=check, naive=naive, recalc=True)

                        with open(fname, 'r') as infile:
                            data = json.load(infile)

                    results = pickle.decode(data)

                    results_won, results_chosen, results_context, \
                    results_c_param, results_c_param_type, entropy_c_r, \
                    results_p_param, results_p_param_type, entropy_p_r, entropy_l_r = results

                    results_c[i,j,m,k] = results_c_param
                    results_p[i,j,m,k] = results_p_param

                    chosen[i,j,m,k] = results_chosen

                    entropy_c[i,j,m,k] = entropy_c_r

                    entropy_p[i,j,m,k] = entropy_p_r

                    entropy_l[i,j,m,k] = entropy_l_r

                    data = 0
                    results = 0
                    results_won, results_chosen, results_context, \
                    results_c_param, results_c_param_type, entropy_c_r, \
                    results_p_param, results_p_param_type, entropy_p_r, entropy_l_r = [0]*10


    return results_c, results_p, entropy_c, entropy_p, entropy_l, chosen


def load_checks(tendencies, reward_priors, p_values, prefixes, check=False, naive=False):

    if naive:
        test = "test_"
        post = "_check"
    elif check:
        test = "check_"
        post = "_check"
    else:
        test = "test_"
        post = ""

    measures = np.zeros((len(tendencies),len(reward_priors),len(p_values), len(prefixes),10))

    results_c = np.zeros((len(tendencies),len(reward_priors),len(p_values), len(prefixes),100,4))
    results_p = np.zeros((len(tendencies),len(reward_priors),len(p_values), len(prefixes),100,4))

    entropy_c = np.zeros((len(tendencies),len(reward_priors),len(p_values), len(prefixes),100))
    entropy_p = np.zeros((len(tendencies),len(reward_priors),len(p_values), len(prefixes),100))

    chosen = np.zeros((len(tendencies),len(reward_priors),len(p_values), len(prefixes),100))

    for i,h in enumerate(tendencies):

        for k,rew_p in enumerate(reward_priors):

            for j,p in enumerate(p_values):

                for l,prefix in enumerate(prefixes):

                    #print(h,rew_p,p,prefix)

                    run_name = test+prefix+"_h"+str(h)+"_t"+str(p)+"_r"+str(rew_p)+".json"

                    analysis_name = run_name[:-5]+"_ananlysis"+post+".json"

                    fname = os.path.join(folder, analysis_name)

                    jsonpickle_numpy.register_handlers()

                    with open(fname, 'r') as infile:
                        data = json.load(infile)

                    results = pickle.decode(data)

                    results_won, results_chosen, results_context, \
                    results_c_param, results_c_param_type, \
                    results_p_param, results_p_param_type = results

                    results_c[i,k,j,l] = results_c_param
                    results_p[i,k,j,l] = results_p_param

                    chosen[i,k,j,l] = results_chosen

                    mask_c = results_c_param[:,0] > 0
                    mask_p = results_p_param[:,0] > 0

                    for n in range(100):
                        if mask_c[n]:
                            fct = fit_functions[results_c_param_type[n]]
                            param = results_c_param[n,results_c_param_type[n]:]
                            entropy_c[i,k,j,l,n] = -(fct([0,199],*param)*ln(fct([0,199],*param))).sum()/2. \
                                        - ((1.-fct([0,199],*param))*ln(1.-fct([0,199],*param))).sum()/2.
                        if mask_p[n]:
                            fct = fit_functions[results_p_param_type[n]]
                            param = results_p_param[n,results_p_param_type[n]:]
                            entropy_p[i,k,j,l,n] = -(fct([0,199],*param)*ln(fct([0,199],*param))).sum()/2. \
                                        - ((1.-fct([0,199],*param))*ln(1.-fct([0,199],*param))).sum()/2.

                    measures[i,k,j,l] = [results_won.sum(axis=1).mean()/2.,
                                results_won[:,0].mean(), results_won[:,1].mean(),
                                results_chosen.mean(),
                                np.nanmedian(results_c_param[:,1],axis=0),
                                np.nanmedian(results_c_param[:,2],axis=0),
                                entropy_c[i,k,j,l].mean(),
                                np.nanmedian(results_p_param[:,1],axis=0),
                                np.nanmedian(results_p_param[:,2],axis=0),
                                entropy_p[i,k,j,l].mean()]

    return measures, results_c, results_p, entropy_c, entropy_p, chosen




def plot_analyses(print_regression_results=False):

    tendencies = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]

    tendency_names = (1./np.array(tendencies)).astype(str)
    t_ind_short = [0, 9, 18]
    #t_short = [tendencies[i] for i in t_ind_short]
    t_names_short = [""] * len(tendencies)
    for i in t_ind_short:
        t_names_short[i] = tendency_names[i]


    transition_probs = [99]#[100,99,98,97,96,95,94]#,93,92,91,90]#

    reward_probs = [100,95,90,85,80,75,70,65,60]

    calculate_analyses(tendencies, transition_probs, reward_probs=reward_probs, recalc=False)

    results_c, results_p, entropy_c, entropy_p, entropy_l, results_chosen = load_analyses(tendencies, transition_probs, reward_probs)

    prefix = ""
    reward = 2
    r = reward_probs[reward]
    test = ""

    numbers = []
    chosen_tendencies = []

    fnames = []
    fit_params = []
    h = 0
    chosen_tendencies.append(h)
    tendency = tendencies[h]
    p = transition_probs[0]
    run_name = test+prefix+"h"+str(tendency)+"_t"+str(p)+"_p"+str(r)+"_train100.json"
    fnames.append(run_name)
    action_times = results_p[h,0,reward,0,:,2]
    median = np.nanmedian(action_times)
    number = np.nanargmin(np.abs(action_times-median))
    numbers.append(number)

    fit_params.append([results_c[h,0,reward,0,number,:], 0, \
              results_p[h,0,reward,0,number,:], 0 ])

    save_run_plots([fnames[-1]], [numbers[-1]], [tendency_names[h]], prefix="strong", check=False, fit_params=[fit_params[-1]])

    h = -1
    chosen_tendencies.append(h)
    tendency = tendencies[h]
    p = transition_probs[0]
    run_name = test+prefix+"h"+str(tendency)+"_t"+str(p)+"_p"+str(r)+"_train100.json"
    fnames.append(run_name)
    action_times = results_p[h,0,reward,0,:,2]
    median = np.nanmedian(action_times)
    number = np.nanargmin(np.abs(action_times-median))
    numbers.append(number)

    fit_params.append([results_c[h,0,reward,0,number,:], 0, \
              results_p[h,0,reward,0,number,:], 0 ])

    #save_run_plots(fnames, numbers, check=False, fit_params=fit_params)

    save_run_plots([fnames[-1]], [numbers[-1]], [tendency_names[h]], prefix="weak", check=False, fit_params=[fit_params[-1]])

    fname = os.path.join(folder, "best_and_average.json")

    jsonpickle_numpy.register_handlers()
    pickled = pickle.encode([numbers, chosen_tendencies])
    with open(fname, 'w') as outfile:
        json.dump(pickled, outfile)

    plt.figure(figsize=(10,5))
    plot_c = (results_c[:,0,2,0,:,2]-100.).T
    num_tend = plot_c.shape[1]
    num_runs = plot_c.shape[0]
    plot_c_data = plot_c.T.reshape(plot_c.size)
    labels = np.tile(np.arange(num_tend), (num_runs, 1)).reshape(-1, order='f')
    data = pd.DataFrame({'chosen': plot_c_data, 'tendencies': labels})
    sns.lineplot(x='tendencies', y='chosen', data=data, ci = 95, color='#ff1493', estimator=np.nanmedian, label="context", linewidth=3)
    plot_p = np.array([results_p[i,0,2,0,:,2]-100. for i in range(len(tendencies))]).T
    plot_p_data = plot_p.T.reshape(plot_p.size)
    data = pd.DataFrame({'chosen': plot_p_data, 'tendencies': labels})
    sns.lineplot(x='tendencies', y='chosen', data=data, ci = 95, color='#cc6600', estimator=np.nanmedian, label="action", linewidth=3)
    #plt.xticks(range(len(prefixes)), prefixes)
    plt.ylim([0.,20.])
    plt.yticks([0,5,10,15,20])
    plt.xlim([len(tendencies)-1,0])
    plt.xticks(range(len(tendencies)), t_names_short, fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel("habitual tendency $h$", fontsize=20)
    plt.ylabel("trial after switch / habit strength", fontsize=20)
    plt.title("action and context infer times", fontsize=20)
    lgd = plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", fontsize=20)
    plt.savefig(os.path.join(folder,"context_action_infer_times.svg"), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig(os.path.join(folder,"context_action_infer_times.png"), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()

    print(np.nanmedian(plot_c, axis=0))

    # regression if habitual tendency delays context inference
    y = np.nanmedian(plot_c, axis=0)#plot_p.flatten(order='F')
    names = list(1./np.array(tendencies))
    X = np.array([[1]*len(names), names])
    reg = sm.OLS(y,X.T)
    results = reg.fit()
    if print_regression_results:
        print("context")
        print(results.summary())

    # regression if habitual tendency delays action adaptation
    y = np.nanmedian(plot_p, axis=0)#plot_p.flatten(order='F')
    names = list(1./np.array(tendencies))
    X = np.array([[1]*len(names), names])
    reg = sm.OLS(y,X.T)
    results = reg.fit()
    if print_regression_results:
        print("actions")
        print(results.summary())


    plt.figure(figsize=(10,5))
    e_c = entropy_c[:,0,2,0,:]
    e_c_data = e_c.reshape(e_c.size)
    e_c_data = pd.DataFrame({'chosen': e_c_data, 'tendencies': labels})
    e_p = entropy_p[:,0,2,0,:]
    e_p_data = e_p.reshape(e_p.size)
    e_p_data = pd.DataFrame({'chosen': e_p_data, 'tendencies': labels})
    sns.lineplot(x='tendencies', y='chosen', data=e_c_data, color='#ff1493', ci = 95, estimator=np.nanmedian, label='context')
    sns.lineplot(x='tendencies', y='chosen', data=e_p_data, color='#cc6600', ci = 95, estimator=np.nanmedian, label='action')
    plt.xlim([len(tendencies)-1,0])
    plt.xticks(range(len(tendencies)), t_names_short, fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel("habitual tendency $h$", fontsize=20)
    plt.ylabel("entropy", fontsize=20)
    plt.title("entropies of the posteriors", fontsize=20)
    lgd = plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", fontsize=20)
    plt.savefig(os.path.join(folder,"entropies_in_sudden_condition.svg"), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig(os.path.join(folder,"entropies_in_sudden_condition.png"), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()


    plt.figure(figsize=(10,5))
    chosen_0 = results_chosen[:,0,2,0,:,0]
    chosen_data = chosen_0.reshape(chosen_0.size)
    chosen_data = pd.DataFrame({'chosen': chosen_data, 'tendencies': labels})
    sns.lineplot(x='tendencies', y='chosen', data=chosen_data, color='blue', ci = 95, estimator=np.nanmedian)#, condition="alpha_init=1")
    #plt.xticks(range(len(prefixes)), prefixes)
    plt.ylim([0.,1.])
    plt.xlim([len(tendencies)-1,0])
    plt.xticks(range(len(tendencies)), t_names_short, fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel("habitual tendency $h$", fontsize=20)
    plt.ylabel("percent", fontsize=20)
    plt.title("proportion of optimal action chosen trials 1-100", fontsize=20)
    plt.savefig(os.path.join(folder,"optimal_action_chosen_before_switch.svg"), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig(os.path.join(folder,"optimal_action_chosen_before_switch.png"), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()

    # regression if habitual tendency increases rewarding behavior in context 1
    y = np.nanmedian(chosen_0, axis=1)#plot_p.flatten(order='F')
    names = list(1./np.array(tendencies))
    X = np.array([[1]*len(names), names])
    reg = sm.OLS(y,X.T)
    results = reg.fit()
    if print_regression_results:
        print(results.summary())


    plt.figure(figsize=(10,5))
    chosen_1 = results_chosen[:,0,2,0,:,1]
    chosen_data = chosen_1.reshape(chosen_1.size)
    chosen_data = pd.DataFrame({'chosen': chosen_data, 'tendencies': labels})
    sns.lineplot(x='tendencies', y='chosen', data=chosen_data, color='blue', ci = 95, estimator=np.nanmedian)#, condition="alpha_init=1")
    #plt.xticks(range(len(prefixes)), prefixes)
    plt.ylim([0.,1.])
    plt.xlim([len(tendencies)-1,0])
    plt.xticks(range(len(tendencies)), t_names_short, fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel("habitual tendency $h$", fontsize=20)
    plt.ylabel("percent", fontsize=20)
    plt.title("proportion of optimal action chosen trials 101-115", fontsize=20)
    plt.savefig(os.path.join(folder,"optimal_action_chosen_after_switch.svg"), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig(os.path.join(folder,"optimal_action_chosen_after_switch.png"), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()

    # regression if habitual tendency decreases rewarding behavior directly after switch
    y = np.nanmedian(chosen_0, axis=1)#plot_p.flatten(order='F')
    names = list(1./np.array(tendencies))
    X = np.array([[1]*len(names), names])
    reg = sm.OLS(y,X.T)
    results = reg.fit()
    if print_regression_results:
        print(results.summary())

    plt.figure(figsize=(10,5))
    chosen_2 = results_chosen[:,0,2,0,:,2]
    chosen_data = chosen_2.reshape(chosen_2.size)
    chosen_data = pd.DataFrame({'chosen': chosen_data, 'tendencies': labels})
    sns.lineplot(x='tendencies', y='chosen', data=chosen_data, color='blue', ci = 95, estimator=np.nanmedian)#, condition="alpha_init=1")
    #plt.xticks(range(len(prefixes)), prefixes)
    plt.ylim([0.,1.])
    plt.xlim([len(tendencies)-1,0])
    plt.xticks(range(len(tendencies)), t_names_short, fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel("habitual tendency $h$", fontsize=20)
    plt.ylabel("percent", fontsize=20)
    plt.title("proportion of optimal action chosen trials 116-200", fontsize=20)
    plt.savefig(os.path.join(folder,"optimal_action_chosen_after_switch2.svg"), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig(os.path.join(folder,"optimal_action_chosen_after_switch2.png"), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()

    # regression if habitual tendency increases rewarding behavior in context 2
    y = np.nanmedian(chosen_0, axis=1)#plot_p.flatten(order='F')
    names = list(1./np.array(tendencies))
    X = np.array([[1]*len(names), names])
    reg = sm.OLS(y,X.T)
    results = reg.fit()
    if print_regression_results:
        print(results.summary())


    noises = [0,2,4]

    colors = ['purple','red','orange']

    plt.figure(figsize=(10,5))
    chosen = results_chosen[:,:,:,0,:,0]
    for k,j in enumerate(noises):
        plot_c = chosen[:,0,j,:]
        chosen_0 = plot_c.reshape(plot_c.size)
        labels = np.tile(np.arange(num_tend), (num_runs, 1)).reshape(-1, order='f')
        data = pd.DataFrame({'chosen': chosen_0, 'tendencies': labels})
        sns.lineplot(x='tendencies', y='chosen', data=data, ci = 95, color=colors[k], estimator=np.nanmedian, label=r'$\nu$='+str(reward_probs[j]/100))
    plt.ylim([0.,1.])
    plt.xlim([len(tendencies)-1,0])
    plt.xticks(range(len(tendencies)), t_names_short, fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel("habitual tendencies", fontsize=20)
    plt.ylabel("percent", fontsize=20)
    plt.title("proportion of optimal action chosen trials 1-100", fontsize=20)
    lgd = plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", fontsize=20)
    plt.savefig(os.path.join(folder,"stochastic_optimal_action_chosen_before_switch_some.svg"), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig(os.path.join(folder,"stochastic_optimal_action_chosen_before_switch_some.png"), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10,5))
    chosen = results_chosen[:,:,:,0,:,1]
    for k,j in enumerate(noises):
        plot_c = chosen[:,0,j,:]
        chosen_0 = plot_c.reshape(plot_c.size)
        labels = np.tile(np.arange(num_tend), (num_runs, 1)).reshape(-1, order='f')
        data = pd.DataFrame({'chosen': chosen_0, 'tendencies': labels})
        sns.lineplot(x='tendencies', y='chosen', data=data, ci = 95, color=colors[k], estimator=np.nanmedian, label=r'$\nu$='+str(reward_probs[j]/100))
    plt.ylim([0.,1.])
    plt.xlim([len(tendencies)-1,0])
    plt.xticks(range(len(tendencies)), t_names_short, fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel("habitual tendencies", fontsize=20)
    plt.ylabel("percent", fontsize=20)
    plt.title("proportion of optimal action chosen trials 101-115", fontsize=20)
    lgd = plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", fontsize=20)
    plt.savefig(os.path.join(folder,"stochastic_optimal_action_chosen_after_switch_some.svg"), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig(os.path.join(folder,"stochastic_optimal_action_chosen_after_switch_some.png"), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10,5))
    chosen = results_chosen[:,:,:,0,:,2]
    for k,j in enumerate(noises):
        plot_c = chosen[:,0,j,:]
        chosen_0 = plot_c.reshape(plot_c.size)
        labels = np.tile(np.arange(num_tend), (num_runs, 1)).reshape(-1, order='f')
        data = pd.DataFrame({'chosen': chosen_0, 'tendencies': labels})
        sns.lineplot(x='tendencies', y='chosen', data=data, ci = 95, color=colors[k], estimator=np.nanmedian, label=r'$\nu$='+str(reward_probs[j]/100))
    plt.ylim([0.,1.])
    plt.xlim([len(tendencies)-1,0])
    plt.xticks(range(len(tendencies)), t_names_short, fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel("habitual tendencies", fontsize=20)
    plt.ylabel("percent", fontsize=20)
    plt.title("proportion of optimal action chosen trials 116-200", fontsize=20)
    lgd = plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", fontsize=20)
    plt.savefig(os.path.join(folder,"stochastic_optimal_action_chosen_after_switch2_some.svg"), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig(os.path.join(folder,"stochastic_optimal_action_chosen_after_switch2_some.png"), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10,5))
    for k,j in enumerate(noises):
        plot_c = results_c[:,0,j,0,:,2]-100
        chosen_0 = plot_c.T.reshape(plot_c.size)
        labels = np.tile(np.arange(num_tend), (num_runs, 1)).reshape(-1, order='f')
        data = pd.DataFrame({'chosen': chosen_0, 'tendencies': labels})
        sns.lineplot(x='tendencies', y='chosen', data=data, ci = 95, color=colors[k], estimator=np.nanmedian, label=r'$\nu$='+str(reward_probs[j]/100))
        y = plot_c.flatten(order='F')
        names = list(1./np.array(tendencies))
        X = np.array([names*num_runs]).T
        mask = ~np.isnan(y)
        reg = lm.LinearRegression().fit(X[mask,:],y[mask])
        print("influence of habitual tendency on context inference", reward_probs[j], "beta & r^2", reg.coef_, reg.score(X[mask,:],y[mask]))

    plt.ylim([0.,100.])
    plt.xlim([len(tendencies)-1,0])
    plt.xticks(range(len(tendencies)), t_names_short, fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel("habitual tendencies", fontsize=20)
    plt.ylabel("trial", fontsize=20)
    plt.title("context infer times", fontsize=20)
    lgd = plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", fontsize=20)
    plt.savefig(os.path.join(folder,"stochastic_context_infer_times_some.svg"), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig(os.path.join(folder,"stochastic_context_infer_times_some.png"), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()


    plt.figure(figsize=(10,5))
    for k,j in enumerate(noises):
        plot_p = results_p[:,0,j,0,:,2]-100
        chosen_0 = plot_p.T.reshape(plot_c.size)
        labels = np.tile(np.arange(num_tend), (num_runs, 1)).reshape(-1, order='f')
        data = pd.DataFrame({'chosen': chosen_0, 'tendencies': labels})
        sns.lineplot(x='tendencies', y='chosen', data=data, ci = 95, color=colors[k], estimator=np.nanmedian, label=r'$\nu$='+str(reward_probs[j]/100))
        y = plot_p.flatten(order='F')
        names = list(1./np.array(tendencies))
        X = np.array([names*num_runs]).T
        mask = ~np.isnan(y)
        reg = lm.LinearRegression().fit(X[mask,:],y[mask])
        print("influence of habitual tendency on action inference", reward_probs[j], "beta & r^2", reg.coef_, reg.score(X[mask,:],y[mask]))
    plt.xticks(range(len(reward_probs)), reward_probs)
    plt.ylim([0.,100.])
    plt.xlim([len(tendencies)-1,0])
    plt.xticks(range(len(tendencies)), t_names_short, fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel("habitual tendencies", fontsize=20)
    plt.ylabel("trial", fontsize=20)
    plt.title("action infer times", fontsize=20)
    lgd = plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", fontsize=20)
    plt.savefig(os.path.join(folder,"stochastic_action_infer_times_some.svg"), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig(os.path.join(folder,"stochastic_action_infer_times_some.png"), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()

    styles = ['-', '--', ':']
    plot_c = []
    plt.figure(figsize=(10,5))
    num_probs = len(reward_probs)
    for k,h in enumerate([0,num_probs,-1]):
        plot_c.append((results_p[h,0,:,0,:,2]-results_c[h,0,:,0,:,2]).T)
        chosen_0 = plot_c[-1].T.reshape(plot_c[-1].size)
        labels = np.tile(np.arange(num_probs), (num_runs, 1)).reshape(-1, order='f')
        data = pd.DataFrame({'chosen': chosen_0, 'tendencies': labels})
        ax = sns.lineplot(x='tendencies', y='chosen', data=data, ci = 95, estimator=np.nanmedian, label="h="+str(tendency_names[h]), linewidth=3)
        ax.lines[-1].set_linestyle(styles[k])
    plt.ylim([0.,100.])
    #plt.xlim([len(tendencies)-1,0])
    plt.xticks(range(len(reward_probs)), reward_probs, fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel("reward probability", fontsize=20)
    plt.ylabel("action - context inference", fontsize=20)
    plt.title("difference between inference times", fontsize=20)
    lgd = plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", fontsize=20)
    plt.show()

    plot_c = np.array([np.nanmedian(plot_c[i], axis=0) for i in range(len(plot_c))])
    y = plot_c.flatten()
    names = (0.5-np.array(reward_probs)/100.)*(-2)
    stab = np.array(list(names) * 3)
    #tend = np.array([0.01]*len(reward_probs) + [0.1]*len(reward_probs) + [1.0]*len(reward_probs))
    #cross = stab*tend
    X = np.array([[1]*len(stab), stab])
    reg = sm.OLS(y,X.T)
    results = reg.fit()
    if print_regression_results:
        print(results.summary())

    styles = ['-', '--', ':']
    plot_c = []
    plt.figure(figsize=(10,5))
    for k,h in enumerate([0,num_probs,-1]):
        plot_c.append((results_c[h,0,:i,0,:,2]-100).T)
        chosen_0 = plot_c[-1].T.reshape(plot_c[-1].size)
        labels = np.tile(np.arange(num_probs), (num_runs, 1)).reshape(-1, order='f')
        data = pd.DataFrame({'chosen': chosen_0, 'tendencies': labels})
        ax = sns.lineplot(x='tendencies', y='chosen', data=data, ci = 95, color='#ff1493', estimator=np.nanmedian, label="h="+str(tendency_names[h]), linewidth=3)
        ax.lines[-1].set_linestyle(styles[k])
    plt.ylim([0.,100.])
    plt.xlim([0,num_probs-1])
    plt.xticks(range(len(reward_probs)), reward_probs, fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel("reward probability", fontsize=20)
    plt.ylabel("context inference", fontsize=20)
    plt.title("context infer times", fontsize=20)
    lgd = plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", fontsize=20)
    plt.show()

    plot_c = np.array([np.nanmedian(plot_c[i], axis=0) for i in range(len(plot_c))])
    y = plot_c.flatten()
    names = (0.5-np.array(reward_probs)/100.)*(-2)
    stab = np.array(list(names) * 3)
    #tend = np.array([0.01]*len(reward_probs) + [0.1]*len(reward_probs) + [1.0]*len(reward_probs))
    #cross = stab*tend
    X = np.array([[1]*len(stab), stab])
    reg = sm.OLS(y,X.T)
    results = reg.fit()
    if print_regression_results:
        print(results.summary())

    styles = ['-', '--', ':']
    plot_p = []
    plt.figure(figsize=(10,5))
    for k,h in enumerate([0,num_probs,-1]):
#        plot_c = np.array([results_c[h,0,i,0,:,2]-100 for i in range(len(reward_probs))]).T
#        sns.lineplot(plot_c, ci = 95, color='#ff1493', estimator=np.nanmedian, condition="context, h="+str(tendency_names[h]), linestyle=styles[k])
        plot_p.append((results_p[h,0,:,0,:,2]-100).T)
        chosen_0 = plot_p[-1].T.reshape(plot_p[-1].size)
        labels = np.tile(np.arange(num_probs), (num_runs, 1)).reshape(-1, order='f')
        data = pd.DataFrame({'chosen': chosen_0, 'tendencies': labels})
        ax = sns.lineplot(x='tendencies', y='chosen', data=data, ci = 95, color='#cc6600', estimator=np.nanmedian, label="h="+str(tendency_names[h]), linestyle=styles[k], linewidth=3)
        ax.lines[-1].set_linestyle(styles[k])
    plt.ylim([0.,100.])
    plt.xlim([0,num_probs-1])
    x = np.array(reward_probs)
    x = 100-x
    plt.xticks(range(len(reward_probs)), x/100., fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel(r'stochasticity $1-\nu$', fontsize=20)
    plt.ylabel(r'habit strength $H$', fontsize=20)
    plt.title("habit strength", fontsize=20)
    lgd = plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", fontsize=20)
    plt.savefig(os.path.join(folder,"stochastic_habit_strength.svg"), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig(os.path.join(folder,"stochastic_habit_strength.png"), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()

    fit_results = np.zeros((len(plot_p),num_runs//5,3))
    for i in range(len(plot_p)):
        for j in range(num_runs//10):
            y = np.nanmedian(plot_p[i][j*10:(j+1)*10], axis=0)
            first = y.argmax() + 1
            x = np.array(reward_probs)
            x = 150-x
            fit_results[i,j], pcov = sc.optimize.curve_fit(exponential, \
                   x[:first], y[:first], p0=[1.,20.,10.], \
                   bounds=[[0, 0, 0],[np.inf, np.inf, np.inf]])
#            plt.figure()
#            plt.plot(x, np.nanmedian(plot_p[i][j*10:(j+1)*10], axis=0),'x')
#            print(fit_results[i,j])
#            plt.plot(x, exponential(x,*fit_results[i,j]))
#            plt.ylim([0,101])
#            plt.show()

    print(np.nanmean(fit_results, axis=1))

    for i in range(3):
        print(sc.stats.f_oneway(*fit_results[:,:,i]))

#    plot_p = np.array([np.nanmedian(plot_p[i], axis=0) for i in range(len(plot_p))])
#    y = plot_p.flatten()
#
#    stab = np.array(list(range(len(reward_probs))) * 3)
#    tend = np.array([0.01]*len(reward_probs) + [0.1]*len(reward_probs) + [1.0]*len(reward_probs))
#    cross = stab*tend
#    X = np.array([[1]*len(stab), stab, tend, cross])
#    reg = sm.OLS(y,X.T)
#    results = reg.fit()
#    print(results.summary())

    #return results_c, results_p, entropy_c, entropy_p, results_chosen

def plot_analyses_training():

    tendencies = [1,10,100]
    tendency_names = (1./np.array(tendencies)).astype(str)

    transition_probs = [99]#[100,99,98,97,96,95,94]

    trainings = [56, 100, 177, 316, 562, 1000, 1778]#, 3162, 5623, 10000]
    training_names = [1.75, 2., 2.25, 2.5, 2.72, 3., 3.25, 3.5, 3.75, 4.0]
    tr_ind_short = [1,5]#,9]
    tr_names_short = [""] * len(trainings)
    for i in tr_ind_short:
        tr_names_short[i] = trainings[i]

    calculate_analyses(tendencies, transition_probs, trainings=trainings, recalc=False)

    results_c, results_p, entropy_c, entropy_p, entropy_l, results_chosen = load_analyses(tendencies, transition_probs, trainings=trainings)


    styles = ['-', '--', ':']
    plot_p = []
    plt.figure(figsize=(10,5))
    for h in range(len(tendencies)):
        plot_p.append(np.array([results_p[h,0,0,i,:,2]-trainings[i] for i in range(len(trainings))]).T)
        num_train = plot_p[-1].shape[1]
        num_runs = plot_p[-1].shape[0]
        chosen_0 = plot_p[-1].T.reshape(plot_p[-1].size)
        labels = np.tile(np.arange(num_train), (num_runs, 1)).reshape(-1, order='f')
        data = pd.DataFrame({'chosen': chosen_0, 'tendencies': labels})
        ax = sns.lineplot(x='tendencies', y='chosen', data=data, ci = 95, color='#cc6600', estimator=np.nanmedian, label="h="+str(tendency_names[h]), linewidth=3)
        ax.lines[-1].set_linestyle(styles[h])
    #plt.xticks(range(len(prefixes)), prefixes)
    plt.ylim([0.,100.])
    #plt.xlim([len(tendencies)-1,0])
    plt.xticks(range(len(trainings)), tr_names_short, fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel(r'trainnig duration $d_{training}$', fontsize=20)
    plt.ylabel(r'habit strength $H$', fontsize=20)
    #plt.title("context infer times")
    lgd = plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", fontsize=20)
    plt.savefig(os.path.join(folder,"training_habit_strength.svg"), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig(os.path.join(folder,"training_habit_strength.png"), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()

    fit_results = np.zeros((len(plot_p),20,3))
    for i in range(len(plot_p)):
        for j in range(num_runs//10):
            y = np.nanmedian(plot_p[i][j*10:(j+1)*10], axis=0)
            first = y.argmax() + 1
            x = training_names
            fit_results[i,j], pcov = sc.optimize.curve_fit(exponential, \
                       x[:first], y[:first], p0=[5.,2.5,10.])
#            plt.figure()
#            plt.plot(x, np.nanmedian(plot_p[i][j*10:(j+1)*10], axis=0),'x')
#            plt.plot(x, exponential(x,*fit_results[i,j]))
#            plt.ylim([0,101])
#            plt.show()

    print(np.mean(fit_results, axis=1))

    for i in range(3):
        print(sc.stats.f_oneway(*fit_results[:,:,i]))

    print(10**np.mean(fit_results[:,:,1], axis=1))

    print("anova for habit strength")

    #return results_c, results_p, entropy_c, entropy_p, results_chosen


def plot_renewal():

    tendencies = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]

    tendency_names = (1./np.array(tendencies)).astype(str)
    t_ind_short = [0, 9, 18]
    t_names_short = [""] * len(tendencies)
    for i in t_ind_short:
        t_names_short[i] = tendency_names[i]

    transition_probs = [99]#[100,99,98,97,96,95,94]

    check = True
    naive = True
    calculate_analyses(tendencies, transition_probs, check=check, naive=naive, recalc=False)
    n_results_c, n_results_p, n_entropy_c, n_entropy_p, n_entropy_l, n_results_chosen = load_analyses(tendencies, transition_probs, check=check, naive=naive)

    naive = False
    calculate_analyses(tendencies, transition_probs, check=check, naive=naive, recalc=False)
    c_results_c, c_results_p, c_entropy_c, c_entropy_p, n_entropy_l, c_results_chosen = load_analyses(tendencies, transition_probs, check=check, naive=naive)


    fname = os.path.join(folder, "best_and_average.json")

    with open(fname, 'r') as infile:
        data = json.load(infile)
    numbers, chosen_tendencies = pickle.decode(data)

    rew_prob = 90
    train = 100

    fnames = []
    fit_params = []
    new_numbers = []
    chosen_names = []

    for i, h in enumerate(chosen_tendencies):
        tendency = tendencies[h]
        chosen_names.append(tendency_names[h])
        p = transition_probs[0]
        run_name = "h"+str(tendency)+"_t"+str(p)+"_p"+str(rew_prob)+"_train"+str(train)+".json"
        fnames.append(run_name)

        context_times = c_results_c[i,0,0,0,:,2]
        median = np.nanmedian(context_times)
        number = np.nanargmin(np.abs(context_times-median))
        new_numbers.append(number)

#        fit_params.append([c_results_c[h,0,0,0,numbers[i]], 1, \
#                  c_results_p[h,0,0,0,numbers[i]], 1 ])

    save_run_plots(fnames, new_numbers, chosen_names, fit_params=None, check=True)

    plt.figure(figsize=(10,5))
    chosen_c = c_results_chosen[:,0,0,0,:]
    num_tend = chosen_c.shape[0]
    num_runs = chosen_c.shape[1]
    chosen_c_data = chosen_c.T.reshape(chosen_c.size)
    labels = np.tile(np.arange(num_tend), (num_runs, 1)).reshape(-1, order='f')
    data = pd.DataFrame({'chosen': chosen_c_data, 'tendencies': labels})
    sns.lineplot(x='tendencies', y='chosen', data=data, color='blue', ci = 95, estimator=np.nanmedian, label="experienced")
    chosen_n = n_results_chosen[:,0,0,0,:]
    chosen_n_data = chosen_n.T.reshape(chosen_n.size)
    data = pd.DataFrame({'chosen': chosen_n_data, 'tendencies': labels})
    sns.lineplot(x='tendencies', y='chosen', data=data, color='green', ci = 95, estimator=np.nanmedian, label="naive")
    #plt.xticks(range(len(prefixes)), prefixes)
    plt.ylim([0.,1.])
    plt.xlim([len(tendencies)-1,0])
    plt.xticks(range(len(tendencies)), t_names_short, fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel("habitual tendency $h$", fontsize=20)
    plt.ylabel("percent", fontsize=20)
    #plt.title("proportion of optimal action chosen")
    lgd = plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", fontsize=20)
    plt.savefig(os.path.join(folder,"check_optimal_action_chosen.svg"), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig(os.path.join(folder,"check_optimal_action_chosen.png"), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()

    stat, p = sc.stats.ttest_ind(chosen_c.flatten(), chosen_n.flatten())

    print("p value of better performance:", p, "mean experience:", np.nanmedian(chosen_c), "mean naive:", np.nanmedian(chosen_n))

    print("trained")
    y = np.nanmedian(chosen_c, axis=1)#plot_p.flatten(order='F')
    names = list(1./np.array(tendencies))
    X = np.array([[1]*len(names), names])
    reg = sm.OLS(y,X.T)
    results = reg.fit()
    print(results.summary())

    print("naive")
    y = np.nanmedian(chosen_n, axis=1)#plot_p.flatten(order='F')
    names = list(1./np.array(tendencies))
    X = np.array([[1]*len(names), names])
    reg = sm.OLS(y,X.T)
    results = reg.fit()
    print(results.summary())


    plt.figure(figsize=(10,5))
    plot_c = c_results_c[:,0,0,0,:,2].T
    plot_c_data = plot_c.T.reshape(plot_c.size)
    data = pd.DataFrame({'chosen': plot_c_data, 'tendencies': labels})
    sns.lineplot(x='tendencies', y='chosen', data=data, ci = 95, color='#ff1493', estimator=np.nanmedian, label="experienced")
    plot_n = n_results_c[:,0,0,0,:,2].T
    plot_n_data = plot_n.T.reshape(plot_n.size)
    data = pd.DataFrame({'chosen': plot_n_data, 'tendencies': labels})
    sns.lineplot(x='tendencies', y='chosen', data=data, ci = 95, color='#ff1493', estimator=np.nanmedian, label="naive", linestyle="-.")
    #plt.xticks(range(len(prefixes)), prefixes)
    plt.ylim([0.,50.])
    plt.xlim([len(tendencies)-1,0])
    plt.xticks(range(len(tendencies)), t_names_short, fontsize=18)
    plt.xlabel("habitual tendency $h$", fontsize=20)
    plt.ylabel("trial", fontsize=20)
    plt.title("context convergence times", fontsize=22)
    plt.yticks(fontsize=18)
    lgd = plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", fontsize=20)
    plt.savefig(os.path.join(folder,"check_context_infer_times.svg"), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig(os.path.join(folder,"check_context_infer_times.png"), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10,5))
    plot_p_c = c_results_p[:,0,0,0,:,2].T
    plot_c_data = plot_p_c.T.reshape(plot_p_c.size)
    data = pd.DataFrame({'chosen': plot_c_data, 'tendencies': labels})
    sns.lineplot(x='tendencies', y='chosen', data=data, ci = 95, color='#cc6600', estimator=np.nanmedian, label="experienced")
    plot_p_n = n_results_p[:,0,0,0,:,2].T
    plot_p_data = plot_p_n.T.reshape(plot_p_n.size)
    data = pd.DataFrame({'chosen': plot_p_data, 'tendencies': labels})
    sns.lineplot(x='tendencies', y='chosen', data=data, ci = 95, color='#cc6600', estimator=np.nanmedian, label="naive", linestyle="-.")
    #plt.xticks(range(len(prefixes)), prefixes)
    plt.ylim([0.,50.])
    plt.xlim([len(tendencies)-1,0])
    plt.xticks(range(len(tendencies)), t_names_short, fontsize=18)
    plt.xlabel("habitual tendency $h$", fontsize=20)
    plt.ylabel("trial", fontsize=20)
    plt.title("action convergence times", fontsize=22)
    plt.yticks(fontsize=18)
    lgd = plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", fontsize=20)
    plt.savefig(os.path.join(folder,"check_action_infer_times.svg"), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig(os.path.join(folder,"check_action_infer_times.png"), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10,5))
    plot_c_n = n_results_c[:,0,0,0,:,2].T
    plot_c_data = plot_c_n.T.reshape(plot_c_n.size)
    data = pd.DataFrame({'chosen': plot_c_data, 'tendencies': labels})
    sns.lineplot(x='tendencies', y='chosen', data=data, color='#ff1493', estimator=np.nanmedian, label="context", linewidth=3)
    plot_p_n = n_results_p[:,0,0,0,:,2].T
    plot_p_data = plot_p_n.T.reshape(plot_p_n.size)
    data = pd.DataFrame({'chosen': plot_p_data, 'tendencies': labels})
    sns.lineplot(x='tendencies', y='chosen', data=data, ci = 95, color='#cc6600', estimator=np.nanmedian, label="action", linewidth=3)
    #plt.xticks(range(len(prefixes)), prefixes)
    plt.ylim([0.,50.])
    plt.xlim([len(tendencies)-1,0])
    plt.xticks(range(len(tendencies)), t_names_short, fontsize=18)
    plt.xlabel("habitual tendency $h$", fontsize=20)
    plt.ylabel("trial", fontsize=20)
    plt.title("naive agent", fontsize=22)
    plt.yticks(fontsize=18)
    lgd = plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", fontsize=20)
    plt.savefig(os.path.join(folder,"check_naive_agent.svg"), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig(os.path.join(folder,"check_naive_agent.png"), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10,5))
    plot_c_c = c_results_c[:,0,0,0,:,2].T
    plot_c_data = plot_c_c.T.reshape(plot_c_c.size)
    data = pd.DataFrame({'chosen': plot_c_data, 'tendencies': labels})
    sns.lineplot(x='tendencies', y='chosen', data=data, color='#ff1493', estimator=np.nanmedian, label="context", linewidth=3)
    plot_p_c = c_results_p[:,0,0,0,:,2].T
    plot_p_data = plot_p_c.T.reshape(plot_p_c.size)
    data = pd.DataFrame({'chosen': plot_p_data, 'tendencies': labels})
    sns.lineplot(x='tendencies', y='chosen', data=data, ci = 95, color='#cc6600', estimator=np.nanmedian, label="action", linewidth=3)
    #plt.xticks(range(len(prefixes)), prefixes)
    plt.ylim([0.,50.])
    plt.xlim([len(tendencies)-1,0])
    plt.xticks(range(len(tendencies)), t_names_short, fontsize=18)
    plt.xlabel("habitual tendency $h$", fontsize=20)
    plt.ylabel("trial", fontsize=20)
    plt.title("experienced agent", fontsize=22)
    plt.yticks(fontsize=18)
    lgd = plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", fontsize=20)
    plt.savefig(os.path.join(folder,"check_experienced_agent.svg"), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig(os.path.join(folder,"check_experienced_agent.png"), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()


def plot_analyses_deval():

    tendencies = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]

    #tendency_names = ["1", "0.5", "0.33", "0.25", "0.2", "0.17", "0.14", "0.13", "0.11", "0.1", "0.09", "0.05", "0.033", "0.025", "0.02", "0.017", "0.014", "0.013", "0.011", "0.01"] #100./(np.array(tendencies)-1))
    tendency_names = (1./np.array(tendencies)).astype(str)
    t_ind_short = [0, 9, 18]
    t_short = [tendency_names[i] for i in t_ind_short]
    t_names_short = [""] * len(tendencies)
    for i in t_ind_short:
        t_names_short[i] = tendency_names[i]

    transition_probs = [99]#[100,99,98,97,96]#,95,94]# [100,99,98,97,96,95,94]#,93,92,91,90]#

    calculate_analyses(tendencies, transition_probs, deval=True, recalc=False)

    results_c, results_p, entropy_c, entropy_p, entropy_l, results_chosen = load_analyses(tendencies, transition_probs, deval=True)

    rew_p = 1
    pre = 0
    test = "deval_"

    numbers = []
    chosen_tendencies = []

    fnames = []
    fit_params = []
    hs = []
    train = 100
    rew_prob = 90
    h = 0
    hs.append(tendency_names[h])
    chosen_tendencies.append(h)
    tendency = tendencies[h]
    p = transition_probs[0]
    run_name = test+"h"+str(tendency)+"_t"+str(p)+"_p"+str(rew_prob)+"_train"+str(train)+".json"
    fnames.append(run_name)
    action_times = results_p[h,0,0,0,:,2]
    median = np.nanmedian(action_times)
    print(median)
    number = np.nanargmin(np.abs(action_times-median))

    numbers.append(number)

    fit_params.append([results_c[h,0,0,0,number,:], 0, \
          results_p[h,0,0,pre,0,:], 0 ])

    h = -1
    hs.append(tendency_names[h])
    chosen_tendencies.append(h)
    tendency = tendencies[h]
    p = transition_probs[0]
    run_name = test+"h"+str(tendency)+"_t"+str(p)+"_p"+str(rew_prob)+"_train"+str(train)+".json"
    fnames.append(run_name)
    action_times = results_p[h,0,0,0,:,2]
    median = np.nanmedian(action_times)
    print(median)
    number = np.nanargmin(np.abs(action_times-median))

    numbers.append(number)

    fit_params.append([results_c[h,0,0,0,number,:], 0, \
          results_p[h,0,0,0,number,:], 0 ])

    print(fnames, numbers, hs)
    save_run_plots(fnames, numbers, hs, prefix="weak", check=False, fit_params=fit_params)

    plt.figure(figsize=(10,5))
    plot_c = (results_c[:,0,0,0,:,2]-100.).T
    num_tend = plot_c.shape[1]
    num_runs = plot_c.shape[0]
    plot_c_data = plot_c.T.reshape(plot_c.size)
    labels = np.tile(np.arange(num_tend), (num_runs, 1)).reshape(-1, order='f')
    data = pd.DataFrame({'chosen': plot_c_data, 'tendencies': labels})
    sns.lineplot(x='tendencies', y='chosen', data=data, ci = 95, color='#ff1493', estimator=np.nanmedian, label="context", linewidth=3)
    plot_p = np.array([results_p[i,0,0,0,:,2]-100. for i in range(len(tendencies))]).T
    plot_p_data = plot_p.T.reshape(plot_p.size)
    data = pd.DataFrame({'chosen': plot_p_data, 'tendencies': labels})
    sns.lineplot(x='tendencies', y='chosen', data=data, ci = 95, color='#cc6600', estimator=np.nanmedian, label="action", linewidth=3)
    #plt.xticks(range(len(prefixes)), prefixes)
    plt.ylim([0.,20.])
    plt.yticks([0,5,10,15,20])
    plt.xlim([len(tendencies)-1,0])
    plt.xticks(range(len(tendencies)), t_names_short, fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel("habitual tendency $h$", fontsize=20)
    plt.ylabel("trial after switch / habit strength", fontsize=20)
    plt.title("action and context infer times", fontsize=20)
    lgd = plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", fontsize=20)
    plt.savefig(os.path.join(folder,"context_action_infer_times.svg"), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig(os.path.join(folder,"context_action_infer_times.png"), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()

    #return results_c, results_p, entropy_c, entropy_p, results_chosen


def plot_environments_prob():

    for i in [100,90,80,70,60]:
        prefix = ""
        run_name = ""+prefix+"h"+str(1)+"_t"+str(99)+"_p"+str(i)+"_train100.json"
        fname = os.path.join(folder, run_name)

        jsonpickle_numpy.register_handlers()

        with open(fname, 'r') as infile:
            data = json.load(infile)
        worlds = pickle.decode(data)

        w = worlds[0]

        arm_cols = ['#007ecdff','#0000b7ff']

        plt.figure(figsize=(10,5))
        for k in range(1,w.agent.nh):
            plt.plot(w.environment.Rho[:,k,k], label="lever "+str(k), c=arm_cols[k-1], linewidth=3)

        plt.ylim([-0.1,1.1])
        lgd = plt.legend(fontsize=18, bbox_to_anchor=(1.04,1), loc="upper left", ncol=1) #bbox_to_anchor=(0, 1.02, 1, 0.2), mode="expand"
        plt.yticks(fontsize=18)
        plt.xticks(fontsize=18)
        plt.xlabel("trial", fontsize=20)
        plt.ylabel("reward probabilities", fontsize=20)
        plt.savefig(os.path.join(folder,"prob_"+str(i)+".svg"), bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.savefig(os.path.join(folder,"prob_"+str(i)+".png"), bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.show()


def plot_environments():

    run_name = "h"+str(1)+"_t"+str(99)+"_p"+str(90)+"_train100.json"
    fname = os.path.join(folder, run_name)

    jsonpickle_numpy.register_handlers()

    with open(fname, 'r') as infile:
        data = json.load(infile)
    worlds = pickle.decode(data)

    w = worlds[0]

    times = np.arange(1,w.trials+1)

    arm_cols = ['#00b7b7ff','#0000b7ff']

    plt.figure(figsize=(10,5))
    for k in range(1,w.agent.nh):
        plt.plot(times,w.environment.Rho[:,k,k], label="lever "+str(k), c=arm_cols[k-1], linewidth=4)

    lgd = plt.legend(fontsize=18, bbox_to_anchor=(1.04,1), loc="upper left", ncol=1) #bbox_to_anchor=(0, 1.02, 1, 0.2), mode="expand"
    plt.yticks([0.0, 0.5, 1.0], fontsize=18)
    plt.xticks(fontsize=18)
    plt.xlabel("trial", fontsize=20)
    plt.ylabel("reward probabilities", fontsize=20)
    plt.ylim([-0.01,1.01])
    plt.xlim([0,times[-1]+1])
    plt.savefig(os.path.join(folder,"cont_degrad.svg"), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig(os.path.join(folder,"cont_degrad.svg"), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()


    Rho = np.zeros((w.trials+100, w.agent.nh, w.agent.nh))

    Rho[:w.trials] = w.environment.Rho
    Rho[w.trials:] = w.environment.Rho[:100]

    times_check = np.arange(1, w.trials+100+1)

    plt.figure(figsize=(15,5))
    for k in range(1,w.agent.nh):
        plt.plot(times_check,Rho[:,k,k], label="lever "+str(k), c=arm_cols[k-1], linewidth=4)

    lgd = plt.legend(fontsize=18, bbox_to_anchor=(1.04,1), loc="upper left", ncol=1) #bbox_to_anchor=(0, 1.02, 1, 0.2), mode="expand"
    plt.yticks([0.0, 0.5, 1.0], fontsize=18)
    plt.xticks(range(0,301,25), fontsize=18)
    plt.xlabel("trial", fontsize=20)
    plt.ylabel("reward probabilities", fontsize=20)
    plt.ylim([-0.01,1.01])
    plt.xlim([0,times_check[-1]+1])
    plt.savefig(os.path.join(folder,"recollection.svg"), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig(os.path.join(folder,"recollection.png"), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()


def main():
    pass
    #plot_analyses(print_regression_results=False)
    #plot_analyses_training()
    #plot_checks()
    #plot_analyses_deval()
    #plot_run(check=False, naive=False)
    #plot_environments()
    #plot_environments_prob()

if __name__ == "__main__":
    main()