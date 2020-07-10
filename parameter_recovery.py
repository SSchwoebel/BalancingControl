#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 17:44:57 2020

@author: sarah
"""

import numpy as np

import world 
import agent as agt
import perception as prc
import inference as infer
import matplotlib.pylab as plt
import jsonpickle as pickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import json
#import seaborn as sns
import pandas as pd
import os
np.set_printoptions(threshold = 100000, precision = 5)

import pymc3 as pm
import theano
import theano.tensor as tt
import gc

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
def run_agent(w_old, alpha_true, run_name):
    
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
    T = w_old.T
    trials = w_old.trials
    observations = w_old.observations.copy()
    rewards = w_old.rewards.copy()
    actions = w_old.actions.copy()
    utility = w_old.agent.perception.prior_rewards.copy()
    A = w_old.agent.perception.generative_model_observations.copy()
    B = w_old.agent.perception.generative_model_states.copy()
    
    transition_matrix_context = w_old.agent.perception.transition_matrix_context.copy()
    
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
    
    
    """
    create policies
    """
    
    pol = w_old.agent.policies.copy()
    
    #pol = pol[-2:]
    npi = pol.shape[0]
    
    # prior over policies
    
    alpha = 1
    alphas = np.zeros_like(w_old.agent.perception.dirichlet_pol_params.copy()) + alpha
    
    prior_pi = alphas.copy()
    prior_pi /= prior_pi.sum(axis=0)
    
    
    """
    set state prior (where agent thinks it starts)
    """
    
    state_prior = np.zeros((ns))
    
    state_prior[0] = 1.
    
    prior_context = np.zeros((nc)) + 1./(nc)#np.dot(transition_matrix_context, w_old.agent.posterior_context[-1,-1])
        
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
    
    w = world.FakeWorld(bayes_pln, observations, rewards, actions, trials = trials, T = T)
    
    """
    simulate experiment
    """
    fixed = {'rew_mod': C_agent, 'beta_rew': C_alphas}
    
    inferrer = infer.LogLike(w.fit_model, fixed)
    
    ndraws = 3000
    nburn = 1000
    
    obs_actions = actions[:,0]
    obs_rewards = rewards[:,1]
    
    # use PyMC3 to sampler from log-likelihood
    with pm.Model() as opmodel:
        # uniform priors on m and c
        hab_ten = pm.Uniform('h', 0., 2.)
    
        # convert m and c to a tensor vector
        alpha = tt.as_tensor_variable([hab_ten])
        probs_a, probs_r = inferrer(alpha)
    
        # use a DensityDist
        pm.Categorical('actions', probs_a, observed=obs_actions)
        pm.Categorical('rewards', probs_r, observed=obs_rewards)
        
        step = pm.Metropolis()#S=np.ones(1)*0.01)
    
        trace = pm.sample(ndraws, tune=nburn, discard_tuned_samples=True, step=step, cores=5)
    
        # plot the traces
        plt.figure()
        _ = pm.traceplot(trace)#, lines=('h', 1./alpha_true))
        plt.show()
#        plt.figure()
#        _ = pm.plot_posterior(trace, var_names=['h'], ref_val=(1./alpha_true))
#        plt.show()
        
        # save the traces
        fname = pm.save_trace(trace)
        fname_dict = {run_name[:-5]: fname}
    
#    for i in range(3):
#        alpha = 10**i
#        alphas = np.zeros_like(w_old.agent.perception.dirichlet_pol_params.copy()) + alpha
#        
#        prior_pi = np.exp(scs.digamma(alphas) - scs.digamma(alphas.sum(axis=0))[np.newaxis,:])
#        prior_pi /= prior_pi.sum(axis=0)
#        
#        parameters = {'rew_mod': C_agent, 'beta_rew': C_alphas, 'prior_pol': prior_pi, 'alpha_pol': alphas}
#        
#        evidence = w.estimate_par_evidence(parameters)
#        
#        print(evidence)
    
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
    
    return fname_dict
    


def run_fitting(repetitions, utility, avg, T, ns, na, nr, nc, folder):
    
    for tendency in [1]:
        for trans in [99]:
            print(tendency, trans)
            traces = []
                
            run_name ="h"+str(tendency)+"_t"+str(trans)+"_p90_train100.json"
            fname = os.path.join(folder, run_name)
            
            jsonpickle_numpy.register_handlers()
            
            with open(fname, 'r') as infile:
                data = json.load(infile)
                
            worlds_old = pickle.decode(data)
            
            repetitions = len(worlds_old)
            
            for i in [1]:
                
                w_old = worlds_old[i]
                traces.append(run_agent(w_old, tendency, run_name))
                
            fname = os.path.join(folder, run_name[:-5]+"_traces.json")
                    
            jsonpickle_numpy.register_handlers()
            pickled = pickle.encode(traces)
            with open(fname, 'w') as outfile:
                json.dump(pickled, outfile)
            
            pickled = 0
            traces = 0
            
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
    
    run_fitting(repetitions, utility, avg, *run_args, folder)
    
    
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
    utility = np.zeros(nr)
    for i in range(1,nr):
        utility[i] = u/(nr-1)
    utility[0] = (1.-u)
    
    repetitions = 20
    
    avg = True
    
    worlds = run_fitting(repetitions, utility, avg, *run_args, folder)
    
