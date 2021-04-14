#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 17:56:24 2017

@author: sarah
"""

import numpy as np

from analysis import plot_analyses, plot_analyses_training, plot_analyses_deval
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
def run_agent(par_list, trials, T, ns, na, nr, nc, deval=False, ESS=None):

    #set parameters:
    #learn_pol: initial concentration paramter for policy prior
    #trans_prob: reward probability
    #avg: True for average action selection, False for maximum selection
    #Rho: Environment's reward generation probabilities as a function of time
    #utility: goal prior, preference p(o)
    learn_pol, trans_prob, avg, Rho, utility = par_list


    """
    create matrices
    """


    #generating probability of observations in each state
    A = np.eye(ns)


    #state transition generative probability (matrix)
    B = np.zeros((ns, ns, na))

    for i in range(0,na):
        B[i+1,:,i] += 1

    # agent's beliefs about reward generation

    # concentration parameters
    C_alphas = np.ones((nr, ns, nc))
    # initialize state in front of levers so that agent knows it yields no reward
    C_alphas[0,0,:] = 100
    for i in range(1,nr):
        C_alphas[i,0,:] = 1

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

    """
    create environment (grid world)
    """

    environment = env.MultiArmedBandid(A, B, Rho, trials = trials, T = T)


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

    if ESS is not None:

        ac_sel = asl.DirichletSelector(trials = trials, T = T,
                                      number_of_actions = na)

    elif avg:

        ac_sel = asl.AveragedSelector(trials = trials, T = T,
                                      number_of_actions = na)

    else:

        ac_sel = asl.MaxSelector(trials = trials, T = T,
                                      number_of_actions = na)

    """
    set context prior
    """

    prior_context = np.zeros((nc)) + 0.1/(nc-1)
    prior_context[0] = 0.9

    """
    set up agent
    """

    # perception
    bayes_prc = prc.HierarchicalPerception(A, B, C_agent, transition_matrix_context, state_prior, utility, prior_pi, alphas, C_alphas, T=T)

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
    if not deval:
        w.simulate_experiment(range(trials))

    else:
        w.simulate_experiment(range(trials//2))
        # reset utility to implement devaluation
        ut = utility[1:].sum()
        bayes_prc.prior_rewards[2:] = ut / (nr-2)
        bayes_prc.prior_rewards[:2] = (1-ut) / 2

        w.simulate_experiment(range(trials//2, trials))

    return w

"""
set condition dependent up parameters
"""

def run_rew_prob_simulations(repetitions, utility, avg, T, ns, na, nr, nc, folder):

    n_training = 1
    n_test = 100
    trials =  100+n_test#number of trials
    trials_training = trials - n_test

    Rho = np.zeros((trials, nr, ns))

    for tendency in [1000]:#,3,5,10,30,50,100]: #1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]:
        for trans in [99]:#[100,99,98,97,96,95,94]:
            for prob in [90]:#[100,95,90,85,80,75,70,65,60]:
                print(tendency, trans, prob)

                Rho[:] = generate_bandit_timeseries_habit(trials_training, nr, ns, n_test,p=prob/100.)

                plt.figure()
                plt.plot(Rho[:,2,2])
                plt.plot(Rho[:,1,1])
                plt.show()

                worlds = []
                learn_pol = tendency
                parameters = [learn_pol, trans/100., avg, Rho, utility]

                ESS = 30

                for i in range(repetitions):
                    worlds.append(run_agent(parameters, trials, T, ns, na, nr, nc, ESS=ESS))
                    w = worlds[-1]
                    plt.figure()
                    post_pol = np.einsum('tpc,tc->tp', w.agent.posterior_policies[:,0,:,:], w.agent.posterior_context[:,0,:])
                    like = np.einsum('tpc,tc->tp', w.agent.likelihood[:,0,:,:], w.agent.posterior_context[:,0,:])
                    plt.plot(post_pol[:,1], '.')
                    plt.plot(like[:,1], 'x')
                    plt.ylim([0,1])
                    plt.show()
                    plt.figure()
                    plt.plot(w.agent.action_selection.RT[:,0])
                    #plt.plot(Rho[:,2,2])
                    #plt.plot(Rho[:,1,1])
                    #plt.ylim([ESS*10,2000])
                    plt.ylim([0,2000])
                    plt.savefig("uncertain_Dir_2pol_RT_timecourse_1000trials.svg")#"ESS"+str(ESS)+"_h"+str(int(learn_pol))+"_RT_timecourse"+str(i)+".svg")#
                    plt.show()
                    plt.figure()
                    plt.hist(w.agent.action_selection.RT[:,0])
                    plt.savefig("uncertain_Dir_2pol_RT_hist_1000trials.svg")#"ESS"+str(ESS)+"_h"+str(int(learn_pol))+"_RT_hist"+str(i)+".svg")#
                    plt.show()

                run_name = "Dir_2pol_uncertain_1000trials.json"
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

    T = 2 #number of time steps in each trial
    nb = 2
    no = nb+1 #number of observations
    ns = nb+1 #number of states
    na = nb #number of actions
    nr = nb+1
    nc = nb

    folder = "data"
    if not os.path.isdir(folder):
        os.mkdir(folder)

    run_args = [T, ns, na, nr, nc]

    u = 0.99
    utility = np.zeros(nr)
    for i in range(1,nr):
        utility[i] = u/(nr-1)
    utility[0] = (1.-u)

    repetitions = 1

    avg = True

    """
    run simulations
    """
    # runs simulations with varying habitual tendency and reward probability
    # results are stored in data folder
    run_rew_prob_simulations(repetitions, utility, avg, *run_args, folder)
    # run habit task and reward probability analyses and plot results.
    # function analyzes data, plots average runs and habit strength
    # This function requires simulation data files
    # can be run independently from the simulation function
    #plot_analyses(print_regression_results=False)

    # run simulations with varying training duration
    # results are stored in data folder
    #run_training_duration_simulations(repetitions, utility, avg, *run_args, folder)
    # run training duration analyses and plot results.
    # function analyzes data, plots average runs and habit strength
    # This function requires simulation data files
    # can be run independently from the simulation function
    #plot_analyses_training()

    # run devaluation simulations
    # results are stored in data folder
    #run_deval_simulations(repetitions, utility, avg, *run_args, folder)
    # run devaluation analyses and plot results.
    # function analyzes data, plots average runs and habit strength
    # This function requires simulation data files
    # can be run independently from the simulation function
    #plot_analyses_deval()


if __name__ == "__main__":
    main()

"""
reload data
"""
#with open(fname, 'r') as infile:
#    data = json.load(infile)
#
#w_new = pickle.decode(data)

"""
parallelized calls for multiple runs
"""
#if len(par_list) < 8:
#    num_threads = len(par_list)
#else:
#    num_threads = 7
#
#pool = Pool(num_threads)
#
#pool.map(run_agent, par_list)
