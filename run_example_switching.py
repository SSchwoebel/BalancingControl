#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 15:33:22 2021

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
import bottleneck as bn
import gc
np.set_printoptions(threshold = 100000, precision = 5)

"""
run function
"""

def run_agent(par_list, trials, T, ns, na, nr, nc, f, contexts, states, state_trans=None):
    #set parameters:
    #learn_pol: initial concentration paramter for policy prior
    #trans_prob: reward probability
    #avg: True for average action selection, False for maximum selection
    #Rho: Environment's reward generation probabilities as a function of time
    #utility: goal prior, preference p(o)
    learn_pol, trans_prob, Rho, utility = par_list


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

    # concentration parameters
    C_alphas = np.ones((nr, ns, nc))
    # initialize state in front of levers so that agent knows it yields no reward
    C_alphas[:,:4,:] = np.array([100,1])[:,None,None]
    C_alphas[:,4:,0] = np.array([[1, 100],
                                  [100, 1]])
    C_alphas[:,4:,1] = np.array([[100, 1],
                                  [1, 100]])

    # agent's initial estimate of reward generation probability
    C_agent = np.zeros((nr, ns, nc))
    for c in range(nc):
        C_agent[:,:,c] = np.array([(C_alphas[:,i,c])/(C_alphas[:,i,c]).sum() for i in range(ns)]).T


    # context transition matrix

    p = trans_prob
    q = 1.-p
    transition_matrix_context = np.zeros((nc, nc))
    transition_matrix_context += q/(nc-1)
    for i in range(nc):
        transition_matrix_context[i,i] = p
        
    # context observation matrix
    D = np.eye(nc)

    """
    create environment (grid world)
    """

    environment = env.TaskSwitching(A, B, Rho, D, states, contexts, trials = trials, T = T)


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

    state_prior[:4] = 1./4

    """
    set action selection method
    """

    ac_sel = asl.DirichletSelector(trials=trials, T=T, number_of_actions=na, factor=f, calc_dkl=False, calc_entropy=False)

    """
    set context prior
    """

    prior_context = np.zeros((nc)) + 0.1/(nc-1)
    prior_context[0] = 0.9

    """
    set up agent
    """

    # perception
    bayes_prc = prc.HierarchicalPerception(A, B, C_agent, transition_matrix_context, state_prior, utility, prior_pi, alphas, C_alphas, T=T, generative_model_context=D)

    # agent
    bayes_pln = agt.BayesianPlanner(bayes_prc, ac_sel, pol,
                      trials = trials, T = T,
                      prior_states = state_prior,
                      prior_policies = prior_pi,
                      number_of_states = ns,
                      prior_context = prior_context,
                      learn_habit = True,
                      learn_rew = True,
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
    w.simulate_experiment(range(trials))

    return w



"""
set condition dependent up parameters
"""

def run_switsching_simulations(repetitions, folder):

    trials = 60
    T = 2
    ns = 6
    na = 2
    nr = 2
    nc = 2
    u = 0.99
    utility = np.array([1-u,u])
    f = 0.5

    Rho = np.zeros((trials, nr, ns))

    for tendency in [1000]:#,3,5,10,30,50,100]: #1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]:
        for trans in [95]:#[100,99,98,97,96,95,94]:


            Rho[:], contexts, states, state_trans, correct_choice = switching_timeseries(trials, nr=nr, ns=ns, na=na, nc=nc, stable_length=10)

            # plt.figure()
            # plt.plot(Rho[:,2,2])
            # plt.plot(Rho[:,1,1])
            # plt.show()

            worlds = []
            learn_pol = tendency
            parameters = [learn_pol, trans/100., Rho, utility]

            for i in range(repetitions):
                worlds.append(run_agent(parameters, trials, T, ns, na, nr, nc, f, contexts, states, state_trans=state_trans))
                w = worlds[-1]
                choices = w.actions[:,0]
                correct = (choices == correct_choice).sum()
                print("percent correct:", correct/trials)
                plt.figure()
                post_pol = np.einsum('tpc,tc->tp', w.agent.posterior_policies[:,0,:,:], w.agent.posterior_context[:,0,:])
                like = np.einsum('tpc,tc->tp', w.agent.likelihood[:,0,:,:], w.agent.posterior_context[:,0,:])
                plt.plot(post_pol[:,1], '.')
                plt.plot(like[:,1], 'x')
                plt.ylim([0,1])
                plt.show()
                plt.figure()
                plt.plot(w.agent.action_selection.RT[:,0], '.')
                #plt.plot(Rho[:,2,2])
                #plt.plot(Rho[:,1,1])
                #plt.ylim([ESS*10,2000])
                plt.ylim([0,2000])
                plt.savefig("Dir_h"+str(int(learn_pol))+"_RT_timecourse"+str(i)+".svg")#"ESS"+str(ESS)+"_h"+str(int(learn_pol))+"_RT_timecourse"+str(i)+".svg")#
                plt.show()
                plt.figure()
                plt.hist(w.agent.action_selection.RT[:,0])
                plt.savefig("uncertain_Dir_h"+str(int(learn_pol))+"_RT_hist"+str(i)+"_1000trials.svg")#"ESS"+str(ESS)+"_h"+str(int(learn_pol))+"_RT_hist"+str(i)+".svg")#
                plt.show()
                plt.figure()
                plt.plot(w.agent.posterior_context[:,0,:], 'x')
                plt.show()

            run_name = "switching_h"+str(int(learn_pol))+"_t"+str(trans)+".json"
            fname = os.path.join(folder, run_name)

            jsonpickle_numpy.register_handlers()
            pickled = pickle.encode(worlds)
            with open(fname, 'w') as outfile:
                json.dump(pickled, outfile)

            pickled = 0
            worlds = 0

            gc.collect()

def main():

    """
    set parameters
    """

    folder = "data"
    if not os.path.isdir(folder):
        os.mkdir(folder)

    repetitions = 2

    """
    run simulations
    """
    # runs simulations with varying habitual tendency and reward probability
    # results are stored in data folder
    run_switsching_simulations(repetitions, folder)


if __name__ == "__main__":
    main()