#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 11:21:08 2021

@author: sarah
"""

import torch as ar
array = ar.tensor

from tqdm import tqdm
import pyro
import pyro.distributions as dist
import agent as agt
import perception as prc
import action_selection as asl


class SingleInference(object):
    
    def __init__(self, agent, data):
        
        self.agent = agent
        self.trials = agent.trials
        self.T = agent.T
        self.data = data
        
    def model(self):
        # generative model of behavior with Normally distributed params (within subject!!)
        
        # tell pyro about prior over parameters: alpha and beta of lambda which is between 0 and 1
        # alpha = beta = 1 equals uniform prior
        alpha_lamb_pi = ar.ones(1)
        beta_lamb_pi = ar.ones(1)
        # sample initial vaue of parameter from Beta distribution
        lamb_pi = pyro.sample('lamb_pi', dist.Beta(alpha_lamb_pi, beta_lamb_pi))
        
        # tell pyro about prior over parameters: alpha and beta of lambda which is between 0 and 1
        # alpha = beta = 1 equals uniform prior
        alpha_lamb_r = ar.ones(1)
        beta_lamb_r = ar.ones(1)
        # sample initial vaue of parameter from Beta distribution
        lamb_r = pyro.sample('lamb_r', dist.Beta(alpha_lamb_r, beta_lamb_r))
        
        # tell pyro about prior over parameters: decision temperature
        # uniform between 0 and 20??
        concentration_dec_temp = ar.tensor(2.)
        rate_dec_temp = ar.tensor(1.)
        # sample initial vaue of parameter from normal distribution
        dec_temp = pyro.sample('dec_temp', dist.Gamma(concentration_dec_temp, rate_dec_temp))
        
        self.agent.reset()
        self.agent.set_parameters(pol_lambda=lamb_pi, r_lambda=lamb_r, dec_temp=dec_temp)
        
        for tau in range(self.trials):
            for t in range(self.T):
                
                if t==0:
                    prev_response = None
                    context = None
                else:
                    prev_response = self.data["actions"][tau, t-1]
                    context = None
        
                observation = self.data["observations"][tau, t]
        
                reward = self.data["rewards"][tau, t]
        
                self.agent.update_beliefs(tau, t, observation, reward, prev_response, context)
        
                if t < self.T-1:
                
                    probs = self.agent.perception.posterior_actions[-1]
                    
                    if ar.isnan(probs[0]):
                        print(probs)
                        print(dec_temp, lamb_pi, lamb_r)
            
                    curr_response = self.data["actions"][tau, t]
                    
                    pyro.sample('res_{}_{}'.format(tau, t), dist.Categorical(probs), obs=curr_response)
                    

    def guide(self):
        # approximate posterior. assume MF: each param has his own univariate Normal.
        
        # tell pyro about posterior over parameters: alpha and beta of lambda which is between 0 and 1
        alpha_lamb_pi = pyro.param("alpha_lamb_pi", ar.ones(1), constraint=ar.distributions.constraints.greater_than_eq(1.))
        beta_lamb_pi = pyro.param("beta_lamb_pi", ar.ones(1), constraint=ar.distributions.constraints.greater_than_eq(1.))
        # sample vaue of parameter from Beta distribution
        lamb_pi = pyro.sample('lamb_pi', dist.Beta(alpha_lamb_pi, beta_lamb_pi))
        
        # tell pyro about posterior over parameters: alpha and beta of lambda which is between 0 and 1
        alpha_lamb_r = pyro.param("alpha_lamb_r", ar.ones(1), constraint=ar.distributions.constraints.greater_than_eq(1.))
        beta_lamb_r = pyro.param("beta_lamb_r", ar.ones(1), constraint=ar.distributions.constraints.greater_than_eq(1.))
        # sample initial vaue of parameter from Beta distribution
        lamb_r = pyro.sample('lamb_r', dist.Beta(alpha_lamb_r, beta_lamb_r))
        
        # tell pyro about posterior over parameters: mean and std of the decision temperature
        concentration_dec_temp = pyro.param("concentration_dec_temp", ar.ones(1)*2, constraint=ar.distributions.constraints.positive)
        rate_dec_temp = pyro.param("rate_dec_temp", ar.ones(1), constraint=ar.distributions.constraints.positive)
        # sample initial vaue of parameter from normal distribution
        dec_temp = pyro.sample('dec_temp', dist.Gamma(concentration_dec_temp, rate_dec_temp))
        
        param_dict = {"alpha_lamb_pi": alpha_lamb_pi, "beta_lamb_pi": beta_lamb_pi, "lamb_pi": lamb_pi,
                      "alpha_lamb_r": alpha_lamb_r, "beta_lamb_r": beta_lamb_r, "lamb_r": lamb_r,
                      "concentration_dec_temp": concentration_dec_temp, "rate_dec_temp": rate_dec_temp, "dec_temp": dec_temp}
        #print(param_dict)
        
        return param_dict
        
        
    def infer_posterior(self,
                        iter_steps=10000,
                        num_particles=100,
                        optim_kwargs={'lr': .01}):
        """Perform SVI over free model parameters.
        """

        pyro.clear_param_store()

        svi = pyro.infer.SVI(model=self.model,
                  guide=self.guide,
                  optim=pyro.optim.Adam(optim_kwargs),
                  loss=pyro.infer.Trace_ELBO(num_particles=num_particles,
                                  #set below to true once code is vectorized
                                  vectorize_particles=False))

        loss = []
        pbar = tqdm(range(iter_steps), position=0)
        for step in pbar:
            loss.append(ar.tensor(svi.step()))
            pbar.set_description("Mean ELBO %6.2f" % ar.tensor(loss[-20:]).mean())
            if ar.isnan(loss[-1]):
                break

        self.loss = loss
        
        return self.loss