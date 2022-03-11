#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 11:21:08 2021

@author: sarah
"""

import torch as ar
array = ar.tensor
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
import distributions as analytical_dists

from tqdm import tqdm
import pyro
import pyro.distributions as dist
import agent as agt
import perception as prc
import action_selection as asl

#device = ar.device("cuda") if ar.cuda.is_available() else ar.device("cpu")
#device = ar.device("cuda")
device = ar.device("cpu")
#torch.set_num_threads(4)
print("Running on device", device)


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
        alpha_lamb_pi = ar.ones(1).to(device)
        beta_lamb_pi = ar.ones(1).to(device)
        # sample initial vaue of parameter from Beta distribution
        lamb_pi = pyro.sample('lamb_pi', dist.Beta(alpha_lamb_pi, beta_lamb_pi)).to(device)
        
        # tell pyro about prior over parameters: alpha and beta of lambda which is between 0 and 1
        # alpha = beta = 1 equals uniform prior
        alpha_lamb_r = ar.ones(1).to(device)
        beta_lamb_r = ar.ones(1).to(device)
        # sample initial vaue of parameter from Beta distribution
        lamb_r = pyro.sample('lamb_r', dist.Beta(alpha_lamb_r, beta_lamb_r)).to(device)
        
        # tell pyro about prior over parameters: alpha and beta of h which is between 0 and 1
        # alpha = beta = 1 equals uniform prior
        alpha_h = ar.ones(1).to(device)
        beta_h = ar.ones(1).to(device)
        # sample initial vaue of parameter from Beta distribution
        h = pyro.sample('h', dist.Beta(alpha_h, beta_h)).to(device)
        
        # tell pyro about prior over parameters: decision temperature
        # uniform between 0 and 20??
        concentration_dec_temp = ar.tensor(1.).to(device)
        rate_dec_temp = ar.tensor(0.5).to(device)
        # sample initial vaue of parameter from normal distribution
        dec_temp = pyro.sample('dec_temp', dist.Gamma(concentration_dec_temp, rate_dec_temp)).to(device)
        param_dict = {"pol_lambda": lamb_pi, "r_lambda": lamb_r, "h": h, "dec_temp": dec_temp}
        
        self.agent.reset(param_dict)
        #self.agent.set_parameters(pol_lambda=lamb_pi, r_lambda=lamb_r, dec_temp=dec_temp)
        
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
                    #print(probs)
                    if ar.any(ar.isnan(probs)):
                        print(probs)
                        print(dec_temp, lamb_pi, lamb_r)
            
                    curr_response = self.data["actions"][tau, t]
                    #print(curr_response)
                    # print(tau, t, probs, curr_response)
                    #print(tau,t,param_dict)
                    
                    pyro.sample('res_{}_{}'.format(tau, t), dist.Categorical(probs.T), obs=curr_response)
                    

    def guide(self):
        # approximate posterior. assume MF: each param has his own univariate Normal.
        
        # tell pyro about posterior over parameters: alpha and beta of lambda which is between 0 and 1
        alpha_lamb_pi = pyro.param("alpha_lamb_pi", ar.ones(1), constraint=ar.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        beta_lamb_pi = pyro.param("beta_lamb_pi", ar.ones(1), constraint=ar.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        # sample vaue of parameter from Beta distribution
        # print()
        # print(alpha_lamb_pi, beta_lamb_pi)
        lamb_pi = pyro.sample('lamb_pi', dist.Beta(alpha_lamb_pi, beta_lamb_pi)).to(device)
        
        # tell pyro about posterior over parameters: alpha and beta of lambda which is between 0 and 1
        alpha_lamb_r = pyro.param("alpha_lamb_r", ar.ones(1), constraint=ar.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        beta_lamb_r = pyro.param("beta_lamb_r", ar.ones(1), constraint=ar.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        # sample initial vaue of parameter from Beta distribution
        lamb_r = pyro.sample('lamb_r', dist.Beta(alpha_lamb_r, beta_lamb_r)).to(device)
        
        # tell pyro about posterior over parameters: alpha and beta of lambda which is between 0 and 1
        alpha_h = pyro.param("alpha_h", ar.ones(1), constraint=ar.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        beta_h = pyro.param("beta_h", ar.ones(1), constraint=ar.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        # sample initial vaue of parameter from Beta distribution
        h = pyro.sample('h', dist.Beta(alpha_lamb_r, beta_lamb_r)).to(device)
        
        # tell pyro about posterior over parameters: mean and std of the decision temperature
        concentration_dec_temp = pyro.param("concentration_dec_temp", ar.ones(1)*3., constraint=ar.distributions.constraints.positive).to(device)#interval(0., 7.))
        rate_dec_temp = pyro.param("rate_dec_temp", ar.ones(1), constraint=ar.distributions.constraints.positive).to(device)
        # sample initial vaue of parameter from normal distribution
        dec_temp = pyro.sample('dec_temp', dist.Gamma(concentration_dec_temp, rate_dec_temp)).to(device)
        
        param_dict = {"alpha_lamb_pi": alpha_lamb_pi, "beta_lamb_pi": beta_lamb_pi, "lamb_pi": lamb_pi,
                      "alpha_lamb_r": alpha_lamb_r, "beta_lamb_r": beta_lamb_r, "lamb_r": lamb_r,
                      "alpha_h": alpha_h, "beta_h": beta_h, "h": h,
                      "concentration_dec_temp": concentration_dec_temp, "rate_dec_temp": rate_dec_temp, "dec_temp": dec_temp}
        #print(param_dict)
        
        return param_dict
        
        
    def infer_posterior(self,
                        iter_steps=1000,
                        num_particles=10,
                        optim_kwargs={'lr': .01}):
        """Perform SVI over free model parameters.
        """

        pyro.clear_param_store()

        svi = pyro.infer.SVI(model=self.model,
                  guide=self.guide,
                  optim=pyro.optim.Adam(optim_kwargs),
                  loss=pyro.infer.Trace_ELBO(num_particles=num_particles,
                                  #set below to true once code is vectorized
                                  vectorize_particles=True))

        loss = []
        #pbar = tqdm(range(iter_steps), position=0)
        for step in range(iter_steps):#pbar:
            loss.append(ar.tensor(svi.step()).to(device))
            #pbar.set_description("Mean ELBO %6.2f" % ar.tensor(loss[-20:]).mean())
            if ar.isnan(loss[-1]):
                break

        self.loss = [l.cpu() for l in loss]
        
        alpha_lamb_pi = pyro.param("alpha_lamb_pi").data.numpy()
        beta_lamb_pi = pyro.param("beta_lamb_pi").data.numpy()
        alpha_lamb_r = pyro.param("alpha_lamb_r").data.numpy()
        beta_lamb_r = pyro.param("beta_lamb_r").data.numpy()
        alpha_h = pyro.param("alpha_lamb_r").data.numpy()
        beta_h = pyro.param("beta_lamb_r").data.numpy()
        concentration_dec_temp = pyro.param("concentration_dec_temp").data.numpy()
        rate_dec_temp = pyro.param("rate_dec_temp").data.numpy()
        
        param_dict = {"alpha_lamb_pi": alpha_lamb_pi, "beta_lamb_pi": beta_lamb_pi,
                      "alpha_lamb_r": alpha_lamb_r, "beta_lamb_r": beta_lamb_r,
                      "alpha_h": alpha_h, "beta_h": beta_h,
                      "concentration_dec_temp": concentration_dec_temp, "rate_dec_temp": rate_dec_temp}
        
        return self.loss, param_dict
    
    def sample_posteriors(self, num_samples=1000):
        
        # tell pyro about posterior over parameters: alpha and beta of lambda which is between 0 and 1
        alpha_lamb_pi = pyro.param("alpha_lamb_pi").data
        beta_lamb_pi = pyro.param("beta_lamb_pi").data
        # sample vaue of parameter from Beta distribution
        lamb_pis = []
        for i in range(num_samples):
            lamb_pi = pyro.sample('lamb_pi', dist.Beta(alpha_lamb_pi, beta_lamb_pi)).numpy()
            lamb_pis.append(lamb_pi)
            
        lamb_pis = np.array(lamb_pis).squeeze()
        
        # tell pyro about posterior over parameters: alpha and beta of lambda which is between 0 and 1
        alpha_lamb_r = pyro.param("alpha_lamb_r").data
        beta_lamb_r = pyro.param("beta_lamb_r").data
        # sample initial vaue of parameter from Beta distribution
        lamb_rs = []
        for i in range(num_samples):
            lamb_r = pyro.sample('lamb_r', dist.Beta(alpha_lamb_r, beta_lamb_r)).numpy()
            lamb_rs.append(lamb_r)
            
        lamb_rs = np.array(lamb_rs).squeeze()
        
        # tell pyro about posterior over parameters: mean and std of the decision temperature
        concentration_dec_temp = pyro.param("concentration_dec_temp").data
        rate_dec_temp = pyro.param("rate_dec_temp").data
        # sample initial vaue of parameter from normal distribution
        dec_temps = []
        for i in range(num_samples):        
            dec_temp = pyro.sample('dec_temp', dist.Gamma(concentration_dec_temp, rate_dec_temp)).numpy()
            dec_temps.append(dec_temp)
            
        dec_temps = np.array(dec_temps).squeeze()
            
        data = {"lamb_pi": lamb_pis, "lamb_r": lamb_rs, "dec_temp": dec_temps}
        
        df = pd.DataFrame(data)
        
        param_dict = {"alpha_lamb_pi": alpha_lamb_pi, "beta_lamb_pi": beta_lamb_pi,
                      "alpha_lamb_r": alpha_lamb_r, "beta_lamb_r": beta_lamb_r,
                      "concentration_dec_temp": concentration_dec_temp, "rate_dec_temp": rate_dec_temp}
        
        return df, param_dict
    
    def analytical_posteriors(self):
        
        alpha_lamb_pi = pyro.param("alpha_lamb_pi").data.cpu().numpy()
        beta_lamb_pi = pyro.param("beta_lamb_pi").data.cpu().numpy()
        alpha_lamb_r = pyro.param("alpha_lamb_r").data.cpu().numpy()
        beta_lamb_r = pyro.param("beta_lamb_r").data.cpu().numpy()
        alpha_h = pyro.param("alpha_lamb_r").data.cpu().numpy()
        beta_h = pyro.param("beta_lamb_r").data.cpu().numpy()
        concentration_dec_temp = pyro.param("concentration_dec_temp").data.cpu().numpy()
        rate_dec_temp = pyro.param("rate_dec_temp").data.cpu().numpy()
        
        param_dict = {"alpha_lamb_pi": alpha_lamb_pi, "beta_lamb_pi": beta_lamb_pi,
                      "alpha_lamb_r": alpha_lamb_r, "beta_lamb_r": beta_lamb_r,
                      "alpha_h": alpha_h, "beta_h": beta_h,
                      "concentration_dec_temp": concentration_dec_temp, "rate_dec_temp": rate_dec_temp}
        
        x_lamb = np.arange(0.01,1.,0.01)
        
        y_lamb_pi = analytical_dists.Beta(x_lamb, alpha_lamb_pi, beta_lamb_pi)
        y_lamb_r = analytical_dists.Beta(x_lamb, alpha_lamb_r, beta_lamb_r)
        y_h = analytical_dists.Beta(x_lamb, alpha_h, beta_h)
        
        x_dec_temp = np.arange(0.01,10.,0.01)
        
        y_dec_temp = analytical_dists.Gamma(x_dec_temp, concentration=concentration_dec_temp, rate=rate_dec_temp)
        
        xs = [x_lamb, x_lamb, x_lamb, x_dec_temp]
        ys = [y_lamb_pi, y_lamb_r, y_h, y_dec_temp]
        
        return xs, ys, param_dict
    
    
    def plot_posteriors(self):
        
        #df, param_dict = self.sample_posteriors()
        
        xs, ys, param_dict = self.analytical_posteriors()
        
        lamb_pi_name = "$\\lambda_{\\pi}$ as Beta($\\alpha$="+str(param_dict["alpha_lamb_pi"][0])+", $\\beta$="+str(param_dict["beta_lamb_pi"][0])+")"
        lamb_r_name = "$\\lambda_{r}$ as Beta($\\alpha$="+str(param_dict["alpha_lamb_r"][0])+", $\\beta$="+str(param_dict["beta_lamb_r"][0])+")"
        h_name = "h"
        dec_temp_name = "$\\gamma$ as Gamma(conc="+str(param_dict["concentration_dec_temp"][0])+", rate="+str(param_dict["rate_dec_temp"][0])+")"
        names = [lamb_pi_name, lamb_r_name, h_name, dec_temp_name]
        xlabels = ["forgetting rate prior policies: $\\lambda_{\pi}$",
                   "forgetting rate reward probabilities: $\\lambda_{r}$",
                   "h",
                   "decision temperature: $\\gamma$"]
        #xlims = {"lamb_pi": [0,1], "lamb_r": [0,1], "dec_temp": [0,10]}
        
        for i in range(len(xs)):
            plt.figure()
            plt.title(names[i])
            plt.plot(xs[i],ys[i])
            plt.xlim([xs[i][0]-0.01,xs[i][-1]+0.01])
            plt.xlabel(xlabels[i])
            plt.show()
            
        print(param_dict)
        
        