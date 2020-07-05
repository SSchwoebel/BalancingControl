#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 17:56:24 2017

@author: sarah
"""

import numpy as np

from analysis import plot_renewal
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
import scipy.special as scs
np.set_printoptions(threshold = 100000, precision = 5)



"""
set parameters
"""
agent = 'bethe'
#agent = 'meanfield'

save_data = False

trials = 100 #number of trials
T = 2 #number of time steps in each trial
nb = 2
no = nb+1 #number of observations
ns = nb+1 #number of states
na = nb #number of actions
npi = na**(T-1)
nr = nb+1
nc = nb #1
n_parallel = 1
noise = 1e-9

proni = "/home/sarah/proni/sarah"

"""
run function
"""
def run_agent(par_list, w_old, trials=trials):
    
    #set parameters:
    #obs_unc: observation uncertainty condition
    #state_unc: state transition uncertainty condition
    #goal_pol: evaluate only policies that lead to the goal
    #utility: goal prior, preference p(o)
    trans_prob, avg, Rho = par_list
    
    
    """
    create matrices
    """
    ns = w_old.environment.Theta.shape[0]
    nr = w_old.environment.Rho.shape[1]
    na = w_old.environment.Theta.shape[2]
    T = w_old.T
    utility = w_old.agent.perception.prior_rewards.copy()
    
    #generating probability of observations in each state

    A = np.eye(ns)
        
    
    #state transition generative probability (matrix)
    B = np.zeros((ns, ns, na))
    
    for i in range(0,na):
        B[i+1,:,i] += 1
    
    # create reward generation
#            
#    C = np.zeros((utility.shape[0], ns))
#    
#    vals = np.array([0., 1./5., 0.95, 1./5., 1/5., 1./5.])
#    
#    for i in range(ns):
#        C[:,i] = [1-vals[i],vals[i]]
#    
#    changes = np.array([0.01, -0.01])
#    Rho = generate_bandit_timeseries(C, nb, trials, changes)
            
    # agent's beliefs about reward generation
    
    C_alphas = w_old.agent.perception.dirichlet_rew_params.copy()
    
    C_agent = w_old.agent.perception.generative_model_rewards.copy()
    #np.array([np.random.dirichlet(C_alphas[:,i]) for i in range(ns)]).T
    
    # context transition matrix
    
    transition_matrix_context = w_old.agent.perception.transition_matrix_context.copy()
                            
    """
    create environment (grid world)
    """
    
    environment = env.MultiArmedBandid(A, B, Rho, trials = trials, T = T)
    
    
    """
    create policies
    """
    
    pol = w_old.agent.policies
    
    #pol = pol[-2:]
    npi = pol.shape[0]
    
    # prior over policies

    #prior_pi[170] = 1. - 1e-3
    alphas = w_old.agent.perception.dirichlet_pol_params.copy()
#    for i in range(nb):
#        alphas[i+1,i] = 100
    #alphas[170] = 100
    prior_pi = np.exp(scs.digamma(alphas) - scs.digamma(alphas.sum(axis=0))[np.newaxis,:])
    prior_pi /= prior_pi.sum(axis=0)
    
    
    """
    set state prior (where agent thinks it starts)
    """
    
    state_prior = np.zeros((ns))
    
    state_prior[0] = 1.

    """
    set action selection method
    """

    if avg:
    
        ac_sel = asl.AveragedSelector(trials = trials, T = T, 
                                      number_of_actions = na)
    else:
        
        ac_sel = asl.MaxSelector(trials = trials, T = T, 
                                      number_of_actions = na)
    
#    ac_sel = asl.AveragedPolicySelector(trials = trials, T = T, 
#                                        number_of_policies = npi,
#                                        number_of_actions = na)
    
    prior_context = np.zeros((nc)) + 1./(nc)#np.dot(transition_matrix_context, w_old.agent.posterior_context[-1,-1])
        
#    prior_context[0] = 1.
    
    """
    set up agent
    """
        
    pol_par = alphas

    # perception
    bayes_prc = prc.HierarchicalPerception(A, B, C_agent, transition_matrix_context, state_prior, utility, prior_pi, pol_par, C_alphas, T=T)
    
    bayes_pln = agt.BayesianPlanner(bayes_prc, ac_sel, pol,
                      trials = trials, T = T,
                      prior_states = state_prior,
                      prior_policies = prior_pi,
                      number_of_states = ns, 
                      prior_context = prior_context,
                      learn_habit = True,
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
    
    
    """
    plot and evaluate results
    """
#    plt.figure()
#    
#    for i in range(ns):
#        plt.plot(w.environment.Rho[:,0,i], label=str(i))
#        
#    plt.legend()
#    plt.show()
#    
#    print("won:", int(w.rewards.sum()/trials*100), "%")
#    
#    stayed = np.array([((w.actions[i,0] - w.actions[i+1,0])==0) for i in range(trials-1)])
#    
#    print("stayed:", int(stayed.sum()/trials*100), "%")
    
    return w
    


def run_renewal_simulations(repetitions, utility, avg, T, ns, na, nr, nc, folder):

    n_training = 1
    
    Rho = np.zeros((trials, nr, ns))
    
    Rho[:] = generate_bandit_timeseries_training(trials*2, nr, ns, nb, n_training)[:trials]
    
    for tendency in [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]:
        for trans in [99]:
            print(tendency, trans)
            worlds = []
            par_list = [trans/100., avg, Rho]
                
            run_name ="h"+str(tendency)+"_t"+str(trans)+"_p90_train100.json"
            fname = os.path.join(folder, run_name)
            
            jsonpickle_numpy.register_handlers()
            
            with open(fname, 'r') as infile:
                data = json.load(infile)
                
            worlds_old = pickle.decode(data)
            
            repetitions = len(worlds_old)
            
            for i in range(repetitions):
                
                w_old = worlds_old[i]
                worlds.append(run_agent(par_list, w_old))
                
            check_name = "check_"+"h"+str(tendency)+"_t"+str(trans)+"_p90_train100.json"
            fname = os.path.join(folder, check_name)
            
            jsonpickle_numpy.register_handlers()
            pickled = pickle.encode(worlds)
            with open(fname, 'w') as outfile:
                json.dump(pickled, outfile)
                        


def main():

    """
    set parameters
    """
    
    T = 2 #number of time steps in each trial
    nb = 2
    no = nb+1 #number of observations
    ns = nb+1 #number of states
    na = nb #number of actions
    npi = na**(T-1)
    nr = nb+1
    nc = nb #1
    n_parallel = 1
    
    folder = "data"
    if not os.path.isdir(folder):
        raise Exception("run_rew_prob_simulations() needs to be run first")
        
    run_args = [T, ns, na, nr, nc]
    
    u = 0.99
    utility = np.zeros(nr)
    for i in range(1,nr):
        utility[i] = u/(nr-1)
    utility[0] = (1.-u)
    
    repetitions = 20
    
    avg = True
    
    run_renewal_simulations(repetitions, utility, avg, *run_args, folder)
    
    plot_renewal()
    
    
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
