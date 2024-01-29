#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 16:46:41 2024

@author: sarah
"""

import numpy as np
from misc import *
import world
import environment as env
import agent as agt
import perception as prc
import action_selection as asl
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
import scipy as sc
import scipy.signal as ss
import gc
np.set_printoptions(threshold = 100000, precision = 5)

"""
run function
"""

def run_agent(par_list, trials, T, ns, na, nr, nc, f, states, contingencies, \
              state_trans=None, correct_choice=None, pol_lambda=0, \
              r_lambda=0):
    #set parameters:
    #learn_pol: initial concentration paramter for policy prior
    #trans_prob: reward probability
    #avg: True for average action selection, False for maximum selection
    #Rho: Environment's reward generation probabilities as a function of time
    #utility: goal prior, preference p(o)
    learn_pol, trans_prob, Rho, utility, unc = par_list


    """
    create matrices
    """


    #generating probability of observations in each state
    A = np.eye(ns)


    #state transition generative probability (matrix)
    if state_trans is None:
        B = np.zeros((ns, ns, na))

        for i in range(0,na):
            B[i+1,:,i] += 1
    else:
        B = state_trans.copy()

    # agent's beliefs about reward generation

    C = contingencies[0]


    # context transition matrix

    if nc>1:
        p = trans_prob
        q = 1.-p
        transition_matrix_context = np.zeros((nc, nc))
        transition_matrix_context += q/(nc-1)
        for i in range(nc):
            transition_matrix_context[i,i] = p
    else:
        transition_matrix_context = np.array([[1]])

    # context observation matrix

    if nc > 1:
        D = np.zeros((nc,nc)) + unc
        for c in range(nc):
            D[c,c] = 1-(unc*(nc-1))
    else:
        D = np.array([[1]])

    """
    create environment (grid world)
    """

    environment = env.MultiArmedBandid(A, B, Rho, \
                                    trials = trials, T = T, \
                                        correct_choice=correct_choice)


    """
    create policies
    """

    pol = np.array(list(itertools.product(list(range(na)), repeat=T-1)))

    npi = pol.shape[0]

    # concentration parameters
    alphas = np.zeros((npi, nc)) + learn_pol

    prior_pi = alphas / alphas.sum(axis=0)


    """
    set state prior (where agent thinks it starts)
    """

    state_prior = np.zeros((ns))

    state_prior[0] = 1.

    """
    set action selection method
    """

    ac_sel = asl.DirichletSelector(trials=trials, T=T, number_of_actions=na, factor=f, calc_dkl=False, calc_entropy=False, draw_true_post=False)

    """
    set context prior
    """

    if nc > 1:
        prior_context = np.zeros((nc)) + 1./nc #0.1/(nc-1)
        #prior_context[0] = 0.9
    else:
        prior_context = np.array([1])

    """
    set up agent
    """

    # perception
    bayes_prc = prc.HierarchicalPerception(A, B, C, transition_matrix_context,
                                      state_prior, utility, prior_pi, 
                                      T=T, 
                                      pol_lambda=pol_lambda, r_lambda=r_lambda,
                                      non_decaying=1)
    bayes_prc.generative_model_states = bayes_prc.generative_model_states[:,:,:,None]

    # agent
    bayes_pln = agt.BayesianPlanner(bayes_prc, ac_sel, pol,
                      trials = trials, T = T,
                      prior_states = state_prior,
                      prior_policies = prior_pi,
                      number_of_states = ns,
                      prior_context = prior_context,
                      learn_habit = False,
                      learn_rew = False,
                      #save_everything = True,
                      number_of_policies = npi,
                      number_of_rewards = nr)


    """
    create world
    """

    w = world.World(environment, bayes_pln, trials = trials, T = T)

    """
    simulate experiment
    """
    for t in range(trials):
        w.agent.perception.generative_model_rewards = contingencies[t][:,:,None]
        w.simulate_experiment([t])

    return w



"""
set condition dependent up parameters
"""

def run_RDM_simulations(repetitions, folder, rerun_simulations=False):
    
    print("running RDM simulations...")

    trials = 100
    T = 2
    ns = 3
    na = 2
    nr = 2
    nc = 1
    u = 0.99
    utility = np.array([1-u,u])
    f = 3.5
    pol_lambda = 0.
    r_lambda = 0.
    trans = 0.
    unc = 0
    
    coherence_levels = [0.20, 0.15, 0.10, -0.10, -0.15, -0.20]

    contingencies, correct_choice, coherence_trials, choice_trials = RDM_timeseries(coherence_levels, trials)
    Rho = contingencies
    
    state_trans = np.zeros((ns, ns, na))
    state_trans[:,:,0] = np.array([[0, 0, 0],
                                   [1, 1, 0],
                                   [0, 0, 1]])
    
    state_trans[:,:,1] = np.array([[0, 0, 0],
                                   [0, 1, 0],
                                   [1, 0, 1]])
    
    states = np.zeros(trials)

    for tendency in [1000]:

        
        prefix = ""
        run_name = "RDM_"+prefix+"_test2.json"
        fname = os.path.join(folder, run_name)
        
        jsonpickle_numpy.register_handlers()

        if not rerun_simulations and (run_name in os.listdir(folder)):
            with open(fname, 'r') as infile:
                data = json.load(infile)

            worlds = pickle.decode(data)
            print(len(worlds))
            num_w_old = len(worlds)
        else:
            worlds = []
            num_w_old = 0

        learn_pol = tendency
        parameters = [learn_pol, trans/100., Rho, utility, unc/100.]

        for i in range(num_w_old, repetitions):
            worlds.append(run_agent(parameters, trials, T, ns, na, nr, nc,\
                                    f, states, contingencies, \
                                    state_trans=state_trans, \
                                    correct_choice=correct_choice, \
                                    pol_lambda = pol_lambda,\
                                    r_lambda = r_lambda))
            w = worlds[-1]
            print("============")
            choices = w.actions[:,0]
            correct = (choices == correct_choice)
            print("percent correct:", correct.sum()/trials)
            RTs = w.agent.action_selection.RT[:,0]
            print("RT:", np.mean(RTs))
            RT_corr = np.mean(RTs[correct==1])
            RT_incorr = np.mean(RTs[correct==0])
            print("correct RT:", RT_corr)
            print("incorrect RT:", RT_incorr)
            # plt.figure()
            # post_pol = np.einsum('tpc,tc->tp', w.agent.posterior_policies[:,0,:,:], w.agent.posterior_context[:,0,:])
            # like = np.einsum('tpc,tc->tp', w.agent.likelihood[:,0,:,:], w.agent.posterior_context[:,0,:])
            # plt.plot(post_pol[:,1], '.')
            # plt.plot(like[:,1], 'x')
            # plt.ylim([0,1])
            # plt.show()
            # plt.figure()
            # plt.plot(w.agent.action_selection.RT[:,0], '.')
            # #plt.plot(Rho[:,2,2])
            # #plt.plot(Rho[:,1,1])
            # #plt.ylim([ESS*10,2000])
            # plt.ylim([0,2000])
            # plt.savefig("Dir_h"+str(int(learn_pol))+"_RT_timecourse"+str(i)+".svg")#"ESS"+str(ESS)+"_h"+str(int(learn_pol))+"_RT_timecourse"+str(i)+".svg")#
            # plt.show()
            # plt.figure()
            # plt.hist(w.agent.action_selection.RT[:,0])
            # plt.savefig("uncertain_Dir_h"+str(int(learn_pol))+"_RT_hist"+str(i)+"_1000trials.svg")#"ESS"+str(ESS)+"_h"+str(int(learn_pol))+"_RT_hist"+str(i)+".svg")#
            # plt.show()
            # plt.figure()
            # plt.plot(w.agent.posterior_context[:,0,:], 'x')
            # plt.show()

        jsonpickle_numpy.register_handlers()
        pickled = pickle.encode(worlds)
        with open(fname, 'w') as outfile:
            json.dump(pickled, outfile)

        pickled = 0
        worlds = 0

        gc.collect()



def analyze_RDM_simulations(folder):
    
    print("preparing analyses and plots...")

    prefix = ""
    tendencies = [1000]
    run_name = "RDM_"+prefix+"_test2.json"
    fname = os.path.join(folder, run_name)

    jsonpickle_numpy.register_handlers()

    with open(fname, 'r') as infile:
        data = json.load(infile)

    worlds_old = pickle.decode(data)
    print("number of agents:", len(worlds_old))

    repetitions = len(worlds_old)
    trials = worlds_old[0].trials
    num_types = len(tendencies)
    correct = np.zeros(repetitions*trials*num_types)
    contingency = np.zeros(repetitions*trials*num_types)
    coherence = np.zeros(repetitions*trials*num_types)
    RT = np.zeros(repetitions*trials*num_types)
    agent = np.zeros(repetitions*trials*num_types)
    trial_num = np.zeros(repetitions*trials*num_types)
    binned_RT = np.zeros(repetitions*trials*num_types)
    non_dec_time = 100

    bin_size = 250
    t_s = 0.2

    sim_type = 0
    for tendency in tendencies:

        jsonpickle_numpy.register_handlers()

        with open(fname, 'r') as infile:
            data = json.load(infile)

        worlds_old = pickle.decode(data)

        repetitions = len(worlds_old)
        trials = worlds_old[0].trials

        offset = sim_type*repetitions*trials

        for i in range(repetitions):
            w = worlds_old[i]
            correct[offset+i*trials:offset+(i+1)*trials] = (w.actions[:,0] == w.environment.correct_choice).astype(int)
            contingency[offset+i*trials:offset+(i+1)*trials] = w.environment.Rho[:,0,1]
            coherence[offset+i*trials:offset+(i+1)*trials] = np.around(w.environment.Rho[:,0,1] - 0.5, decimals=2)
            np.testing.assert_allclose((contingency[offset+i*trials:offset+(i+1)*trials] > 0.5).astype(int), w.environment.correct_choice)
            np.testing.assert_allclose((coherence[offset+i*trials:offset+(i+1)*trials] > 0).astype(int), w.environment.correct_choice)
            RT[offset+i*trials:offset+(i+1)*trials] = t_s*w.agent.action_selection.RT[:,0] + non_dec_time
            agent[offset+i*trials:offset+(i+1)*trials] = i
            trial_num[offset+i*trials:offset+(i+1)*trials] = np.arange(0,trials)
            # binned_RT[offset+i*trials:offset+(i+1)*trials] = t_s*(bin_size//2 + bin_size*(w.agent.action_selection.RT[:,0]//bin_size)) +non_dec_time

        sim_type+=1

    data_dict = {"correct": correct, "RT": RT, "agent": agent,
                 "contingency": contingency, "coherence": coherence,
                 # "congruent": congruent, "binned_RT": binned_RT,
                 "trial_num": trial_num}
    data = pd.DataFrame(data_dict)

    print(data)

    cutoff = non_dec_time + 500#5000#2*500

    plt.figure()
    sns.lineplot(x='contingency', y='RT', data=data, errorbar=('ci', 95), estimator=np.nanmean, linewidth=3)#, style='congruent'
    #plt.plot([0+bin_size,cutoff-bin_size], [0.5,0.5], '--', color='grey', alpha=0.5)
    # plt.ylim([0,1.05])
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.xlabel("contingency", fontsize=16)
    plt.ylabel("RT", fontsize=16)
    # plt.savefig("cont_RT.svg")
    plt.show()
    
    plt.figure()
    sns.lineplot(x='contingency', y='RT', data=data, errorbar=('ci', 95), style='correct', estimator=np.nanmean, linewidth=3)#, style='congruent'
    #plt.plot([0+bin_size,cutoff-bin_size], [0.5,0.5], '--', color='grey', alpha=0.5)
    # plt.ylim([0,1.05])
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.xlabel("coherence", fontsize=16)
    plt.ylabel("RT", fontsize=16)
    # plt.savefig("cont_RT.svg")
    plt.show()
    
    # plt.figure()
    # sns.lineplot(x='contingency', y='RT', data=data, style='congruent', ci = 95, estimator=np.nanmean, linewidth=3)
    # #plt.plot([0+bin_size,cutoff-bin_size], [0.5,0.5], '--', color='grey', alpha=0.5)
    # plt.ylim([0,1.05])
    # plt.yticks(fontsize=16)
    # plt.xticks(fontsize=16)
    # plt.xlabel("contingency", fontsize=16)
    # plt.ylabel("RT", fontsize=16)
    # # plt.savefig("cont_RT.svg")
    # plt.show()
    
    # plt.figure()
    # plt.title("tendency "+str(tendency)+", trans "+str(trans)+", unc "+str(unc))
    # sns.lineplot(x='binned_RT', y='correct', data=, style='congruent', hue='trans_probs', ci = 95, estimator=np.nanmean, linewidth=3)
    # #plt.ylim([0,1])
    # plt.show()
    # plt.figure()
    # plt.title("tendency "+str(tendency)+", trans "+str(trans)+", unc "+str(unc))
    # sns.lineplot(x='binned_RT', y='correct', data=data.query('tendencies==@tendency and trans_probs==@trans and binned_RT<=@cutoff'), style='congruent', hue='uncertainty', ci = 95, estimator=np.nanmean, linewidth=3)
    # #plt.ylim([0,1])
    # plt.show()

    plt.figure()
    sns.histplot(x='RT', data=data)#, binwidth=t_s*bin_size)
    # plt.savefig("RT_histogram.svg")
    plt.show()
    
    plt.figure()
    sns.histplot(x='RT', data=data, hue='coherence')#, binwidth=t_s*bin_size)
    # plt.savefig("RT_histogram_coh.svg")
    plt.show()
    
    plt.figure()
    sns.histplot(x='RT', data=data, y='correct')#, binwidth=t_s*bin_size)
    # plt.savefig("RT_histogram.svg")
    plt.show()
    
    plt.figure()
    sns.histplot(x='RT', data=data, y='correct', hue='coherence')#, binwidth=t_s*bin_size)
    # plt.savefig("RT_histogram_coh.svg")
    plt.show()
    
    # sns.histplot(x='RT', data=data.query('tendencies==@tendency and trans_probs==@trans and uncertainty==@unc and binned_RT<=@cutoff'), hue='congruent', binwidth=t_s*bin_size)

    return data, bin_size


def main():

    """
    set parameters
    """

    folder = "data"
    if not os.path.isdir(folder):
        os.mkdir(folder)

    repetitions = 25

    """
    run simulations
    """
    rerun_simulations = True
    # runs simulations with varying habitual tendency and reward probability
    # results are stored in data folder
    run_RDM_simulations(repetitions, folder, rerun_simulations=rerun_simulations)

    data, bin_size = analyze_RDM_simulations(folder)
    return data, bin_size


if __name__ == "__main__":
    data, bin_size = main()