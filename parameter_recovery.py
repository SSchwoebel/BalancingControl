#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 14:55:36 2020

@author: sarah
"""

import torch

from analysis import plot_analyses, plot_analyses_training, plot_analyses_deval
from misc import *

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
import scipy as sc
import scipy.signal as ss
import bottleneck as bn
import gc


"""
run function
"""
def run_agent(w_old):
    
    #set parameters:
    #obs_unc: observation uncertainty condition
    #state_unc: state transition uncertainty condition
    #goal_pol: evaluate only policies that lead to the goal
    #utility: goal prior, preference p(o)
    
    
    """
    create matrices
    """
    ns = w_old.environment.Theta.shape[0]
    nr = w_old.environment.Rho.shape[1]
    na = w_old.environment.Theta.shape[2]
    nc = w_old.agent.perception.transition_matrix_context.shape[0]
    T = w_old.T
    trials = w_old.trials
    utility = torch.tensor(w_old.agent.perception.prior_rewards)
    observations = torch.tensor(w_old.observations).clone()
    rewards = torch.tensor(w_old.rewards).clone()
    responses = torch.tensor(w_old.actions).clone()
    
    #generating probability of observations in each state

    A = torch.tensor(w_old.agent.perception.generative_model_observations).clone()
    B = torch.tensor(w_old.agent.perception.generative_model_states).clone()
    transition_matrix_context = torch.tensor(w_old.agent.perception.transition_matrix_context).clone()
    
    pol = w_old.agent.policies
    
    #pol = pol[-2:]
    npi = pol.shape[0]
    
    npi = pol.shape[0]
    
    # concentration parameters
    alphas = torch.ones((npi, nc))

    prior_pi = alphas / alphas.sum(axis=0)
    
    # prior over policies
    # concentration parameters    
    C_alphas = torch.ones((nr, ns, nc))
    # initialize state in front of levers so that agent knows it yields no reward
    C_alphas[0,0,:] = 100
    for i in range(1,nr):
        C_alphas[i,0,:] = 1
    
    # agent's initial estimate of reward generation probability
    C_agent = torch.zeros((nr, ns, nc))
    C_agent[:] = C_alphas / C_alphas.sum(dim=0)#[None,:,:]
    

    parameters = {'alpha': 1, 'dir_rew': C_alphas, 'rew_gen_mod': C_agent, 'prior_pol': prior_pi}
    
    """
    set state prior (where agent thinks it starts)
    """
    
    state_prior = torch.zeros((ns))
    
    state_prior[0] = 1.
    
    prior_context = torch.zeros((nc)) + 1./(nc)#torch.dot(transition_matrix_context, w_old.agent.posterior_context[-1,-1])
        
#    prior_context[0] = 1.
    
    """
    set up agent
    """
        
    pol_par = alphas

    # perception
    bayes_prc = prc.HierarchicalPerception(A, B, C_agent, transition_matrix_context, state_prior, utility, prior_pi, pol_par, C_alphas, T=T)
    
    bayes_pln = agt.BayesianPlanner(bayes_prc, None, pol,
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
    
    infer = inf.Inferrer(bayes_pln, observations, rewards, responses, trials, T, parameters)
    
    """
    simulate experiment
    """
    infer.infer_posterior()
    
    """
    plot and evaluate results
    """

    samples = infer.sample_posterior()
    plt.figure()
    sns.distplot(samples.numpy())
    plt.show()
    
    return infer
    


def run_inference(repetitions, folder):
    
    for tendency in [1]:#,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]:
        for trans in [99]:
            print(tendency, trans)
            inference = []
                
            run_name ="h"+str(tendency)+"_t"+str(trans)+"_p90_train100.json"
            fname = os.path.join(folder, run_name)
            
            jsonpickle_numpy.register_handlers()
            
            with open(fname, 'r') as infile:
                data = json.load(infile)
                
            worlds_old = pickle.decode(data)
            
            repetitions = len(worlds_old)
            
            for i in range(1):
                
                w_old = worlds_old[i]
                inference.append(run_agent(w_old))
                
        return inference
                        


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
    utility = torch.zeros(nr)
    for i in range(1,nr):
        utility[i] = u/(nr-1)
    utility[0] = (1.-u)
    
    repetitions = 20
    
    avg = True
    
    inference = run_inference(repetitions, folder)
    
    
if __name__ == "__main__":
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
    utility = torch.zeros(nr)
    for i in range(1,nr):
        utility[i] = u/(nr-1)
    utility[0] = (1.-u)
    
    repetitions = 20
    
    avg = True
    
    inference = run_inference(repetitions, folder)
    