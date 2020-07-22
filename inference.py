#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 18:31:45 2020

@author: sarah
"""

import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt

import world 
import agent as agt
import perception as prc

plt.style.use('seaborn-darkgrid')
print('Running on PyMC3 v{}'.format(pm.__version__))

# for reproducibility here's some version info for modules used in this notebook
import theano
import theano.tensor as tt



class LogLike(tt.Op):

    """
    Specify what type of object will be passed and returned to the Op when it is
    called. In our case we will be passing it a vector of values (the parameters
    that define our model) and returning a single "scalar" value (the
    log-likelihood)
    """
    itypes = [tt.lvector] # expects a vector of parameter values when called
    otypes = [tt.dmatrix, tt.dmatrix] # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike, fixed):

        # add inputs as class attributes
        self.likelihood = loglike
        self.fixed = fixed

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        parameters, = inputs  # this will contain my variables

        # call the log-likelihood function
        probs_a, probs_r = self.likelihood(parameters,self.fixed)

        outputs[0][0] = np.array(probs_a) # output the log-likelihood
        outputs[1][0] = np.array(probs_r) # output the log-likelihood
        
        
class Inferrer:
    def __init__(self, worlds):
        
        self.nruns = len(worlds)
        
        w = worlds[0]
        
        self.setup_agent(w)
        
        self.actions = np.array([w.actions[:,0] for w in worlds])
        
        self.rewards = np.array([w.rewards[:,1] for w in worlds])
        
        self.inferrer = LogLike(self.agent.fit_model, self.fixed)
    
    def setup_agent(self, w):
        
        ns = w.environment.Theta.shape[0]
        nr = w.environment.Rho.shape[1]
        na = w.environment.Theta.shape[2]
        nc = w.agent.perception.generative_model_rewards.shape[2]
        T = w.T
        trials = w.trials
        observations = w.observations.copy()
        rewards = w.rewards.copy()
        actions = w.actions.copy()
        utility = w.agent.perception.prior_rewards.copy()
        A = w.agent.perception.generative_model_observations.copy()
        B = w.agent.perception.generative_model_states.copy()
        
        transition_matrix_context = w.agent.perception.transition_matrix_context.copy()
        
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
        
        pol = w.agent.policies.copy()
        
        #pol = pol[-2:]
        npi = pol.shape[0]
        
        # prior over policies
        
        alpha = 1
        alphas = np.zeros_like(w.agent.perception.dirichlet_pol_params.copy()) + alpha
        
        prior_pi = alphas.copy()
        prior_pi /= prior_pi.sum(axis=0)
        
        state_prior = np.zeros((ns))
        
        state_prior[0] = 1.
        
        prior_context = np.zeros((nc)) + 1./(nc)#np.dot(transition_matrix_context, w.agent.posterior_context[-1,-1])
            
    #    prior_context[0] = 1.
            
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
        
        self.agent = world.FakeWorld(bayes_pln, observations, rewards, actions, trials = trials, T = T)
        
        self.fixed = {'rew_mod': C_agent, 'beta_rew': C_alphas}
        
    def run_single_inference(self, idx, ndraws=300, nburn=100, cores=4):
        
        self.idx = idx
        
        curr_model = self.single_model(idx)
        
        with curr_model:
            
            step = pm.Metropolis()#S=np.ones(1)*0.01)
        
            trace = pm.sample(ndraws, tune=nburn, discard_tuned_samples=True, step=step, cores=cores)
        
            # plot the traces
#            plt.figure()
#            _ = pm.traceplot(trace)#, lines=('h', 1./alpha_true))
#            plt.show()
#            plt.figure()
#            _ = pm.plot_posterior(trace, var_names=['h'], ref_val=(1./alpha_true))
#            plt.show()
            
            # save the traces
            fname = pm.save_trace(trace)
            
        return fname
        
    def run_group_inference(self, ndraws=300, nburn=100, cores=5):
        
        curr_model = self.group_model()
        
        with curr_model:
            
            step = pm.Metropolis()#S=np.ones(1)*0.01)
        
            trace = pm.sample(ndraws, tune=nburn, discard_tuned_samples=True, step=step, cores=cores)
        
            # plot the traces
#            plt.figure()
#            _ = pm.traceplot(trace)#, lines=('h', 1./alpha_true))
#            plt.show()
#            plt.figure()
#            _ = pm.plot_posterior(trace, var_names=['h'], ref_val=(1./alpha_true))
#            plt.show()
            
            # save the traces
            fname = pm.save_trace(trace)
            
        return fname
    
    def single_model(self, idx):
        
        with pm.Model() as smodel:
            # uniform priors on h
            hab_ten = pm.DiscreteUniform('h', 0., 20.)
            
            # convert to a tensor
            alpha = tt.as_tensor_variable([hab_ten*5+1])
            probs_a, probs_r = self.inferrer(alpha)
        
            # use a DensityDist
            pm.Categorical('actions', probs_a, observed=self.actions[idx])
            pm.Categorical('rewards', probs_r, observed=self.rewards[idx])
            
        return smodel
    
    def group_model(self):
        
        with pm.Model() as gmodel:
            # uniform priors on h
            m = pm.DiscreteUniform('h', 0., 20.)
            std = pm.InverseGamma('s', 3., 0.5)
            mean = 2*m+1
            alphas = np.arange(1., 101., 5.)
            p = self.discreteNormal(alphas, mean, std)
            
            for i in range(self.nruns):
                idx = pm.Categorical('h_{}'.format(i), p)
                
                hab_ten = alphas[idx]
                alpha = tt.as_tensor_variable([hab_ten])
                probs_a, probs_r = self.inferrer(alpha)
            
                # use a DensityDist
                pm.Categorical('actions_{}'.format(i), probs_a, observed=self.actions[i])
                pm.Categorical('rewards_{}'.format(i), probs_r, observed=self.rewards[i])
                
        return gmodel

    def discreteNormal(self, x, mean, std):
        
        p = np.exp(-(x - mean)/(2*std**2))
        p /= p.sum()
        return p
    
    def plot_inference(self, trace_name, model='single', idx=None):
        
        if model=='single':
            curr_model = self.single_model(idx)
        elif model=='group':
            curr_model = self.group_model()
        
        with curr_model:
            
            # save the traces
            trace = pm.load_trace(trace_name)
        
            # plot the traces
            plt.figure()
            _ = pm.traceplot(trace)#, lines=('h', 1./alpha_true))
            plt.show()
    #        plt.figure()
    #        _ = pm.plot_posterior(trace, var_names=['h'], ref_val=(1./alpha_true))
    #        plt.show()
        
        