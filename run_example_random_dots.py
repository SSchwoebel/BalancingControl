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
                                    trials = trials, T = T)


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

    ac_sel = asl.DirichletSelector(trials=trials, T=T, number_of_actions=na, factor=f, calc_dkl=False, calc_entropy=False, draw_true_post=True)

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

def run_RDM_simulations(repetitions, folder):

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
    
    coherence_levels = [0.5, 0.25, 0.1, -0.1, -0.25, -0.5]

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
        run_name = "RDM_"+prefix+"_test.json"
        fname = os.path.join(folder, run_name)
        
        jsonpickle_numpy.register_handlers()

        if run_name in os.listdir(folder):
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
            print(w.agent.prior_policies[-1])
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



def analyze_flanker_simulations(folder):

    tendencies = [1,10,100, 250]#[1,10,25,50,75,100, 250,1000]#1,10,100,
    probs = [90,95,99]
    uncertainties = [0,0.1,0.2,0.3,0.5,0.7,1,5,10]#,15,20]
    run_name = "flanker_alpha_h"+str(int(tendencies[0]))+"_t"+str(probs[0])+"_u"+str(uncertainties[0])+"_f3.5_ut0.99_test.json"
    fname = os.path.join(folder, run_name)

    jsonpickle_numpy.register_handlers()

    with open(fname, 'r') as infile:
        data = json.load(infile)

    worlds_old = pickle.decode(data)
    print(len(worlds_old))

    repetitions = len(worlds_old)
    trials = worlds_old[0].trials
    num_types = len(tendencies)*len(probs)*len(uncertainties)
    correct = np.zeros(repetitions*trials*num_types)
    RT = np.zeros(repetitions*trials*num_types)
    agent = np.zeros(repetitions*trials*num_types)
    congruent = np.zeros(repetitions*trials*num_types)
    trial_num = np.zeros(repetitions*trials*num_types)
    epoch = np.zeros(repetitions*trials*num_types)
    tend_arr = np.zeros(repetitions*trials*num_types)
    prob_arr = np.zeros(repetitions*trials*num_types)
    unc_arr  = np.zeros(repetitions*trials*num_types)
    binned_RT = np.zeros(repetitions*trials*num_types)
    prev_congruent = np.zeros(repetitions*trials*num_types) - 1
    non_dec_time = 100

    bin_size = 250
    t_s = 0.2

    sim_type = 0
    for tendency in tendencies:#,3,5,10,30,50,100]: #1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]:
        for trans in probs:#[100,99,98,97,96,95,94]:
            for unc in uncertainties:
                print(tendency, trans, unc)

                run_name = "flanker_alpha_h"+str(int(tendency))+"_t"+str(trans)+"_u"+str(unc)+"_f3.5_ut0.99_test.json"
                fname = os.path.join(folder, run_name)

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
                    RT[offset+i*trials:offset+(i+1)*trials] = t_s*w.agent.action_selection.RT[:,0] + non_dec_time
                    agent[offset+i*trials:offset+(i+1)*trials] = i
                    congruent[offset+i*trials:offset+(i+1)*trials] = np.logical_not(w.environment.congruent)
                    trial_num[offset+i*trials:offset+(i+1)*trials] = np.arange(0,trials)
                    epoch[offset+i*trials:offset+(i+1)*trials] = [-1]*10 + [0]*20 + [1]*20 + [2]*20 + [3]*(trials-70)
                    tend_arr[offset+i*trials:offset+(i+1)*trials] = tendency
                    prob_arr[offset+i*trials:offset+(i+1)*trials] = trans
                    unc_arr[offset+i*trials:offset+(i+1)*trials] = unc#/100
                    binned_RT[offset+i*trials:offset+(i+1)*trials] = t_s*(bin_size//2 + bin_size*(w.agent.action_selection.RT[:,0]//bin_size)) +non_dec_time
                    prev_congruent[offset+i*trials:offset+(i+1)*trials][1:] = congruent[offset+i*trials:offset+(i+1)*trials][:-1]

                sim_type+=1

    data_dict = {"correct": correct, "RT": RT, "agent": agent,
                 "congruent": congruent, "binned_RT": binned_RT,
                 "trial_num": trial_num, "epoch": epoch,
                 "uncertainty": unc_arr, "tendencies": tend_arr,
                 "trans_probs": prob_arr, "prev_cong": prev_congruent}
    data = pd.DataFrame(data_dict)

    # plt.figure()
    # for i in range(0,3):
    #     sns.lineplot(x='num_in_run', y='RT', data=data.query('epoch == @i'), style='congruent', label=str(i), ci = 95, estimator=np.nanmean, linewidth=3)
    # plt.show()
    tendency=100
    trans=90
    unc=0.2
    cutoff = non_dec_time + 500#5000#2*500
    plt.figure()
    plt.title("tendency "+str(tendency)+", trans "+str(trans))
    sns.lineplot(x='uncertainty', y='RT', data=data.query('tendencies==@tendency and trans_probs==@trans and uncertainty<1.1 and binned_RT<=@cutoff'), style='congruent', ci = 95, estimator=np.nanmean, linewidth=3)
    #plt.ylim([200,1000])
    plt.gca().invert_xaxis()
    plt.show()

    plt.figure()
    plt.title("tendency "+str(tendency)+", trans "+str(trans))
    sns.lineplot(x='trans_probs', y='RT', data=data.query('tendencies==@tendency and uncertainty==@unc and binned_RT<=@cutoff'), style='congruent', ci = 95, estimator=np.nanmean, linewidth=3)
    #plt.ylim([200,1000])
    #plt.gca().invert_xaxis()
    plt.show()

    # accuracy
    plt.figure()
    #plt.title("tendency "+str(tendency)+", trans "+str(trans)+", unc "+str(unc))
    sns.lineplot(x='binned_RT', y='correct', data=data.query('tendencies==@tendency and trans_probs==@trans and uncertainty==@unc and binned_RT<=@cutoff'), style='congruent', ci = 95, estimator=np.nanmean, linewidth=3)
    #plt.plot([0+bin_size,cutoff-bin_size], [0.5,0.5], '--', color='grey', alpha=0.5)
    plt.ylim([0,1.05])
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.xlabel("RT", fontsize=16)
    plt.ylabel("Prop correct", fontsize=16)
    plt.savefig("accuracy.svg")
    plt.show()
    plt.figure()
    plt.title("tendency "+str(tendency)+", trans "+str(trans)+", unc "+str(unc))
    sns.lineplot(x='binned_RT', y='correct', data=data.query('tendencies==@tendency and uncertainty==@unc and binned_RT<=@cutoff'), style='congruent', hue='trans_probs', ci = 95, estimator=np.nanmean, linewidth=3)
    #plt.ylim([0,1])
    plt.show()
    plt.figure()
    plt.title("tendency "+str(tendency)+", trans "+str(trans)+", unc "+str(unc))
    sns.lineplot(x='binned_RT', y='correct', data=data.query('tendencies==@tendency and trans_probs==@trans and binned_RT<=@cutoff'), style='congruent', hue='uncertainty', ci = 95, estimator=np.nanmean, linewidth=3)
    #plt.ylim([0,1])
    plt.show()

    plt.figure()
    sns.histplot(x='RT', data=data.query('tendencies==@tendency and trans_probs==@trans and uncertainty==@unc and binned_RT<=@cutoff'), hue='congruent', binwidth=t_s*bin_size)
    plt.savefig("RT_histogram.svg")
    plt.show()

    # gratton
    plt.figure(figsize=(4,5))
    palette = [(0,0,0), (0,0,0)]
    #plt.title("tendency "+str(tendency)+", trans "+str(trans)+", unc "+str(unc))
    sns.lineplot(x='prev_cong', y='RT', data=data.query('tendencies==@tendency and trans_probs==@trans and uncertainty==@unc and trial_num>0'), style='congruent', hue='congruent', ci = 95, estimator=np.nanmean, linewidth=3, markers=True, markersize=12, palette=palette)
    #plt.ylim([200,1000])
    plt.xticks([0,1], labels=["CON", "INC"], fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim([-0.25,1.25])
    plt.ylim([200,700])
    plt.xlabel("Previous trial type", fontsize=16)
    plt.ylabel("RT", fontsize=16)
    plt.savefig("gratton.svg")
    plt.show()
    plt.figure()
    plt.title("tendency "+str(tendency)+", trans "+str(trans)+", unc "+str(unc))
    sns.lineplot(x='prev_cong', y='RT', data=data.query('tendencies==@tendency and uncertainty==@unc and trial_num>0 and binned_RT<=@cutoff'), style='congruent', hue='trans_probs', ci = 95, estimator=np.nanmean, linewidth=3)
    #plt.ylim([200,1000])
    plt.show()
    plt.figure()
    plt.title("tendency "+str(tendency)+", trans "+str(trans)+", unc "+str(unc))
    sns.lineplot(x='prev_cong', y='RT', data=data.query('tendencies==@tendency and trans_probs==@trans and trial_num>0 and binned_RT<=@cutoff'), style='congruent', hue='uncertainty', ci = 95, estimator=np.nanmean, linewidth=3)
    #plt.ylim([200,1000])
    plt.show()
    # plt.figure()
    # sns.lineplot(x='num_in_run', y='RT', data=data.query('congruent == 1 and trial_num > 50'), ci = 95, estimator=np.nanmedian, linewidth=3)
    # sns.lineplot(x='num_in_run', y='RT', data=data.query('congruent == 0 and trial_num > 50'), ci = 95, estimator=np.nanmedian, linewidth=3)
    # plt.show()

    return data, bin_size


def main():

    """
    set parameters
    """

    folder = "data"
    if not os.path.isdir(folder):
        os.mkdir(folder)

    repetitions = 15

    """
    run simulations
    """
    # runs simulations with varying habitual tendency and reward probability
    # results are stored in data folder
    run_RDM_simulations(repetitions, folder)

    #data, bin_size = analyze_flanker_simulations(folder)
    return data, bin_size


if __name__ == "__main__":
    data, bin_size = main()