#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 11:21:08 2021

@author: sarah
"""

import torch as ar
array = ar.tensor
from torch.distributions import constraints, biject_to
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
        self.nsum = len(data)
        
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
        
        for tau in pyro.markov(range(self.trials)):
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
        h = pyro.sample('h', dist.Beta(alpha_h, beta_h)).to(device)
        
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
        
        #return param_dict
        
        
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
        pbar = tqdm(range(iter_steps), position=0)
        for step in pbar:#range(iter_steps):
            loss.append(ar.tensor(svi.step()).to(device))
            pbar.set_description("Mean ELBO %6.2f" % ar.tensor(loss[-20:]).mean())
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
        alpha_h = pyro.param("alpha_h").data.cpu().numpy()
        beta_h = pyro.param("beta_h").data.cpu().numpy()
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
        
        
class GroupInference(object):
    
    def __init__(self, agent, data):
        
        self.agent = agent
        self.trials = agent.trials
        self.T = agent.T
        self.data = data
        self.nsubs = len(data['rewards'][0,0])
        
    def model(self):
        # generative model of behavior with Normally distributed params (within subject!!)
        
        mu_lamb_pi_alpha = pyro.param('mu_lamb_pi_alpha', ar.ones(1), constraint=ar.distributions.constraints.positive).to(device)
        sig_lamb_pi_alpha = pyro.param('sig_lamb_pi_alpha', ar.ones(1), constraint=ar.distributions.constraints.positive).to(device)
        mu_lamb_pi_beta = pyro.param('mu_lamb_pi_beta', ar.ones(1), constraint=ar.distributions.constraints.positive).to(device)
        sig_lamb_pi_beta = pyro.param('sig_lamb_pi_beta', ar.ones(1), constraint=ar.distributions.constraints.positive).to(device)
        # alpha_lamb_pi = ar.ones(1).to(device)
        # beta_lamb_pi = ar.ones(1).to(device)
        
        # tell pyro about prior over parameters: alpha and beta of lambda which is between 0 and 1
        # alpha = beta = 1 equals uniform prior
        mu_lamb_r_alpha = pyro.param('mu_lamb_r_alpha', ar.ones(1), constraint=ar.distributions.constraints.positive).to(device)
        sig_lamb_r_alpha = pyro.param('sig_lamb_r_alpha', ar.ones(1), constraint=ar.distributions.constraints.positive).to(device)
        mu_lamb_r_beta = pyro.param('mu_lamb_r_beta', ar.ones(1), constraint=ar.distributions.constraints.positive).to(device)
        sig_lamb_r_beta = pyro.param('sig_lamb_r_beta', ar.ones(1), constraint=ar.distributions.constraints.positive).to(device)
        # alpha_lamb_r = ar.ones(1).to(device)
        # beta_lamb_r = ar.ones(1).to(device)
        
        # tell pyro about prior over parameters: alpha and beta of h which is between 0 and 1
        # alpha = beta = 1 equals uniform prior
        mu_h_alpha = pyro.param('mu_h_alpha', ar.ones(1), constraint=ar.distributions.constraints.positive).to(device)
        sig_h_alpha = pyro.param('sig_h_alpha', ar.ones(1), constraint=ar.distributions.constraints.positive).to(device)
        mu_h_beta = pyro.param('mu_h_beta', ar.ones(1), constraint=ar.distributions.constraints.positive).to(device)
        sig_h_beta = pyro.param('sig_h_beta', ar.ones(1), constraint=ar.distributions.constraints.positive).to(device)
        
        # tell pyro about prior over parameters: decision temperature
        # uniform between 0 and 20??
        concentration_dec_temp = pyro.param("concentration_dec_temp", ar.ones(1)*3., constraint=ar.distributions.constraints.positive).to(device)#interval(0., 7.))
        rate_dec_temp = pyro.param("rate_dec_temp", ar.ones(1), constraint=ar.distributions.constraints.positive).to(device)
        
        #for ind in range(self.nsubs):#pyro.plate("subject", len(self.data)):
        with pyro.plate('subject', self.nsubs) as ind:
            # print(ind)
            
            alpha_lamb_pi = pyro.sample('alpha_lamb_pi', dist.LogNormal(mu_lamb_pi_alpha, sig_lamb_pi_alpha)).to(device)
            beta_lamb_pi = pyro.sample('beta_lamb_pi', dist.LogNormal(mu_lamb_pi_beta, sig_lamb_pi_beta)).to(device)
            
            alpha_lamb_r = pyro.sample('alpha_lamb_r', dist.LogNormal(mu_lamb_r_alpha, sig_lamb_r_alpha)).to(device)
            beta_lamb_r = pyro.sample('beta_lamb_r', dist.LogNormal(mu_lamb_r_beta, sig_lamb_r_beta)).to(device)
            
            alpha_h = pyro.sample('alpha_lamb_h', dist.LogNormal(mu_h_alpha, sig_h_alpha)).to(device)
            beta_h = pyro.sample('beta_lamb_h', dist.LogNormal(mu_h_beta, sig_h_beta)).to(device)
        
            lamb_pi = pyro.sample('lamb_pi', dist.Beta(alpha_lamb_pi, beta_lamb_pi)).to(device)
            lamb_r = pyro.sample('lamb_r', dist.Beta(alpha_lamb_r, beta_lamb_r)).to(device)
            h = pyro.sample('h', dist.Beta(alpha_h, beta_h)).to(device)
            dec_temp = pyro.sample('dec_temp', dist.Gamma(concentration_dec_temp, rate_dec_temp)).to(device)
        
            param_dict = {"pol_lambda": lamb_pi, "r_lambda": lamb_r, "h": h, "dec_temp": dec_temp}
            # print(param_dict)
            
            self.agent.reset(param_dict)
            #self.agent.set_parameters(pol_lambda=lamb_pi, r_lambda=lamb_r, dec_temp=dec_temp)
            
            
            for tau in pyro.markov(range(self.trials)):
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
                        # print(curr_response.shape)
                        # print(probs.shape)
                        
                        pyro.sample('res_{}_{}'.format(tau, t), dist.Categorical(probs.permute(1,2,0)), obs=curr_response)
            
            # for tau in pyro.markov(range(self.trials)):
            #     for t in range(self.T):
            
            #         self.agent.update_beliefs(tau, t)
            
            #         if t < self.T-1:
                    
            #             probs = self.agent.perception.posterior_actions[-1]
            #             #print(probs)
            #             if ar.any(ar.isnan(probs)):
            #                 print(probs)
            #                 print(dec_temp, lamb_pi, lamb_r)
                
            #             curr_response = self.agent.perception.responses[:,tau,t]
            #             #print(curr_response)
            #             # print(tau, t, probs, curr_response)
            #             #print(tau,t,param_dict)
            #             draw_probs = probs.permute(2,0,1)
                        
            #             pyro.sample('res_{}_{}'.format(tau, t), dist.Categorical(draw_probs), obs=curr_response)
                    

    def guide(self):
        # approximate posterior. assume MF: each param has his own univariate Normal.
        
        npars = 2*3
        
        scale_tril = pyro.param("scale_tril", ar.tril(ar.eye(npars)), constraint=ar.distributions.constraints.lower_cholesky)
        mu = pyro.param("mu", ar.zeros(npars), constraint=ar.distributions.constraints.real_vector)
        
        # mu_lamb_pi_alpha = pyro.param('mu_lamb_pi_alpha', ar.ones(1), constraint=ar.distributions.constraints.positive).to(device)
        # sig_lamb_pi_alpha = pyro.param('sig_lamb_pi_alpha', ar.ones(1), constraint=ar.distributions.constraints.positive).to(device)
        # mu_lamb_pi_beta = pyro.param('mu_lamb_pi_beta', ar.ones(1), constraint=ar.distributions.constraints.positive).to(device)
        # sig_lamb_pi_beta = pyro.param('sig_lamb_pi_beta', ar.ones(1), constraint=ar.distributions.constraints.positive).to(device)
        # # alpha_lamb_pi = ar.ones(1).to(device)
        # # beta_lamb_pi = ar.ones(1).to(device)
        
        # # tell pyro about prior over parameters: alpha and beta of lambda which is between 0 and 1
        # # alpha = beta = 1 equals uniform prior
        # mu_lamb_r_alpha = pyro.param('mu_lamb_r_alpha', ar.ones(1), constraint=ar.distributions.constraints.positive).to(device)
        # sig_lamb_r_alpha = pyro.param('sig_lamb_r_alpha', ar.ones(1), constraint=ar.distributions.constraints.positive).to(device)
        # mu_lamb_r_beta = pyro.param('mu_lamb_r_beta', ar.ones(1), constraint=ar.distributions.constraints.positive).to(device)
        # sig_lamb_r_beta = pyro.param('sig_lamb_r_beta', ar.ones(1), constraint=ar.distributions.constraints.positive).to(device)
        # # alpha_lamb_r = ar.ones(1).to(device)
        # # beta_lamb_r = ar.ones(1).to(device)
        
        # # tell pyro about prior over parameters: alpha and beta of h which is between 0 and 1
        # # alpha = beta = 1 equals uniform prior
        # mu_h_alpha = pyro.param('mu_h_alpha', ar.ones(1), constraint=ar.distributions.constraints.positive).to(device)
        # sig_h_alpha = pyro.param('sig_h_alpha', ar.ones(1), constraint=ar.distributions.constraints.positive).to(device)
        # mu_h_beta = pyro.param('mu_h_beta', ar.ones(1), constraint=ar.distributions.constraints.positive).to(device)
        # sig_h_beta = pyro.param('sig_h_beta', ar.ones(1), constraint=ar.distributions.constraints.positive).to(device)
        # alpha_h = ar.ones(1).to(device)
        # beta_h = ar.ones(1).to(device)
        
        # # tell pyro about posterior over parameters: alpha and beta of lambda which is between 0 and 1
        # alpha_lamb_pi = pyro.param("alpha_lamb_pi", ar.ones(1)*10, constraint=ar.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        # beta_lamb_pi = pyro.param("beta_lamb_pi", ar.ones(1)*10, constraint=ar.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        
        # # tell pyro about posterior over parameters: alpha and beta of lambda which is between 0 and 1
        # alpha_lamb_r = pyro.param("alpha_lamb_r", ar.ones(1)*10, constraint=ar.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        # beta_lamb_r = pyro.param("beta_lamb_r", ar.ones(1)*10, constraint=ar.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        
        # # tell pyro about posterior over parameters: alpha and beta of lambda which is between 0 and 1
        # alpha_h = pyro.param("alpha_h", ar.ones(1)*10, constraint=ar.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        # beta_h = pyro.param("beta_h", ar.ones(1)*10, constraint=ar.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        
        # tell pyro about posterior over parameters: mean and std of the decision temperature
        concentration_dec_temp = pyro.param("concentration_dec_temp", ar.ones(1)*3., constraint=ar.distributions.constraints.positive).to(device)#interval(0., 7.))
        rate_dec_temp = pyro.param("rate_dec_temp", ar.ones(1), constraint=ar.distributions.constraints.positive).to(device)
        
                
        #with pyro.plate("subject") as ind:
        #for ind in range(self.nsubs):
        with pyro.plate('subject', self.nsubs) as ind:
            
            # alpha_lamb_pi = pyro.sample('alpha_lamb_pi', dist.LogNormal(lamb_pi_conc[0], sig_lamb_pi_alpha)).to(device)
            # beta_lamb_pi = pyro.sample('beta_lamb_pi', dist.LogNormal(mu_lamb_pi_beta, sig_lamb_pi_beta)).to(device)
            
            # alpha_lamb_r = pyro.sample('alpha_lamb_r', dist.LogNormal(mu_lamb_r_alpha, sig_lamb_r_alpha)).to(device)
            # beta_lamb_r = pyro.sample('beta_lamb_r', dist.LogNormal(mu_lamb_r_beta, sig_lamb_r_beta)).to(device)
            
            # alpha_h = pyro.sample('alpha_lamb_h', dist.LogNormal(mu_h_alpha, sig_h_alpha)).to(device)
            # beta_h = pyro.sample('beta_lamb_h', dist.LogNormal(mu_h_beta, sig_h_beta)).to(device)
            
            hyper_params = pyro.sample("hyper_params", dist.MultivariateNormal(mu, scale_tril=scale_tril)).to(device)
            # print(hyper_params.shape)
            
            lamb_pi_conc = ar.exp(hyper_params[...,:2])
            lamb_r_conc = ar.exp(hyper_params[...,2:4])
            h_conc = ar.exp(hyper_params[...,4:6])
            
            lamb_pi = pyro.sample('lamb_pi', dist.Beta(lamb_pi_conc[...,0], lamb_pi_conc[...,1])).to(device)
            lamb_r = pyro.sample('lamb_r', dist.Beta(lamb_r_conc[...,0], lamb_r_conc[...,1])).to(device)
            h = pyro.sample('h', dist.Beta(h_conc[...,0], h_conc[...,1])).to(device)
            dec_temp = pyro.sample('dec_temp', dist.Gamma(concentration_dec_temp, rate_dec_temp)).to(device)

        
        param_dict = {"lamb_pi": lamb_pi, "lamb_r": lamb_r, "h": h,
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
        pbar = tqdm(range(iter_steps), position=0)
        for step in pbar:#range(iter_steps):
            loss.append(ar.tensor(svi.step()).to(device))
            pbar.set_description("Mean ELBO %6.2f" % ar.tensor(loss[-20:]).mean())
            if ar.isnan(loss[-1]):
                break

        self.loss = [l.cpu() for l in loss]
        
        # alpha_lamb_pi = pyro.param("alpha_lamb_pi").data.numpy()
        # beta_lamb_pi = pyro.param("beta_lamb_pi").data.numpy()
        # alpha_lamb_r = pyro.param("alpha_lamb_r").data.numpy()
        # beta_lamb_r = pyro.param("beta_lamb_r").data.numpy()
        # alpha_h = pyro.param("alpha_lamb_r").data.numpy()
        # beta_h = pyro.param("beta_lamb_r").data.numpy()
        # concentration_dec_temp = pyro.param("concentration_dec_temp").data.numpy()
        # rate_dec_temp = pyro.param("rate_dec_temp").data.numpy()
        
        # param_dict = {"alpha_lamb_pi": alpha_lamb_pi, "beta_lamb_pi": beta_lamb_pi,
        #               "alpha_lamb_r": alpha_lamb_r, "beta_lamb_r": beta_lamb_r,
        #               "alpha_h": alpha_h, "beta_h": beta_h,
        #               "concentration_dec_temp": concentration_dec_temp, "rate_dec_temp": rate_dec_temp}
        
        return self.loss#, param_dict
    
    def sample_posterior_predictive(self, n_samples=5):

        elbo = pyro.infer.Trace_ELBO()
        post_sample_dict = {}
        
        predictive = pyro.infer.Predictive(model=self.model, guide=self.guide, num_samples=n_samples)
        samples = predictive.get_samples()

        #pbar = tqdm(range(n_samples), position=0)

        # for n in pbar:
        #     pbar.set_description("Sample posterior depth")
        #     # get marginal posterior over planning depths
        #     post_samples = elbo.compute_marginals(self.model, config_enumerate(self.guide))
        #     print(post_samples)
        #     for name in post_samples.keys():
        #         post_sample_dict.setdefault(name, [])
        #         post_sample_dict[name].append(post_samples[name].probs.detach().clone())

        # for name in post_sample_dict.keys():
        #     post_sample_dict[name] = ar.stack(post_sample_dict[name]).numpy()
            
        # post_sample_df = pd.DataFrame(post_sample_dict)
        
        
        reordered_sample_dict = {}
        all_keys = []
        for key in samples.keys():
            if key[:3] != 'res':
                reordered_sample_dict[key] = np.array([])
                all_keys.append(key)
                
        reordered_sample_dict['subject'] = np.array([])
        
        #nsubs = len(self.data)
        for sub in range(self.nsubs):
            for key in set(all_keys):
                reordered_sample_dict[key] = np.append(reordered_sample_dict[key], samples[key][:,sub].detach().numpy())#.squeeze()
            reordered_sample_dict['subject'] = np.append(reordered_sample_dict['subject'], [sub]*n_samples).squeeze()

        # for key in samples.keys():
        #     if key[:3] != 'res':
        #         sub = int(key[-1])
        #         reordered_sample_dict[key[:-2]] = np.append(reordered_sample_dict[key[:-2]], samples[key].detach().numpy()).squeeze()
        #         reordered_sample_dict['subject'] = np.append(reordered_sample_dict['subject'], [sub]).squeeze()
                
        sample_df = pd.DataFrame(reordered_sample_dict)

        return sample_df
    
    def sample_posterior(self, n_samples=5):
        # keys = ["lamb_pi", "lamb_r", "h", "dec_temp"]
        
        lamb_pi_global = np.zeros((n_samples, self.nsubs))
        lamb_r_global = np.zeros((n_samples, self.nsubs))
        h_global = np.zeros((n_samples, self.nsubs))
        dec_temp_global = np.zeros((n_samples, self.nsubs))
        
        for i in range(n_samples):
            sample = self.guide()
            for key in sample.keys():
                sample.setdefault(key, ar.ones(1))
            lamb_pi = sample["lamb_pi"]
            lamb_r = sample["lamb_r"]
            h = sample["h"]
            dec_temp = sample["dec_temp"]
            
            lamb_pi_global[i] = lamb_pi.detach().numpy()
            lamb_r_global[i] = lamb_r.detach().numpy()
            h_global[i] = h.detach().numpy()
            dec_temp_global[i] = dec_temp.detach().numpy()
        
        lamb_pi_flat = np.array([lamb_pi_global[i,n] for i in range(n_samples) for n in range(self.nsubs)])
        lamb_r_flat = np.array([lamb_r_global[i,n] for i in range(n_samples) for n in range(self.nsubs)])
        h_flat = np.array([h_global[i,n] for i in range(n_samples) for n in range(self.nsubs)])
        dec_temp_flat = np.array([dec_temp_global[i,n] for i in range(n_samples) for n in range(self.nsubs)])
        
        subs_flat = np.array([n for i in range(n_samples) for n in range(self.nsubs)])
        
        
        sample_dict = {"lamb_pi": lamb_pi_flat, "lamb_r": lamb_r_flat,
                       "h": h_flat, "dec_temp": dec_temp_flat, "subject": subs_flat}
        
        sample_df = pd.DataFrame(sample_dict)
        
        return sample_df
    
    def analytical_posteriors(self):
        
        alpha_lamb_pi = pyro.param("alpha_lamb_pi").data.cpu().numpy()
        beta_lamb_pi = pyro.param("beta_lamb_pi").data.cpu().numpy()
        alpha_lamb_r = pyro.param("alpha_lamb_r").data.cpu().numpy()
        beta_lamb_r = pyro.param("beta_lamb_r").data.cpu().numpy()
        alpha_h = pyro.param("alpha_h").data.cpu().numpy()
        beta_h = pyro.param("beta_h").data.cpu().numpy()
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
    
    
    def plot_posteriors(self, n_samples=5):
        
        #df, param_dict = self.sample_posteriors()
        
        #sample_df = self.sample_posterior_marginals(n_samples=n_samples)
        
        sample_df = self.sample_posterior(n_samples=n_samples)
        
        plt.figure()
        sns.displot(data=sample_df, x='h', hue='subject')
        plt.show()
        
        plt.figure()
        sns.displot(data=sample_df, x='lamb_r', hue='subject')
        plt.show()
        
        plt.figure()
        sns.displot(data=sample_df, x='lamb_pi', hue='subject')
        plt.show()
        
        plt.figure()
        sns.displot(data=sample_df, x='dec_temp', hue='subject')
        plt.show()
        
        # plt.figure()
        # sns.histplot(marginal_df["h_1"])
        # plt.show()
        
        return sample_df
    
class Group2Inference(object):
    
    def __init__(self, agent, data):
        
        self.agent = agent
        self.trials = agent.trials
        self.T = agent.T
        self.data = data
        self.nsubs = len(data['rewards'][0,0])
        

    def model(self):
        """
        Generative model of behavior with a NormalGamma
        prior over free model parameters.
        """
        npar = 4  # number of parameters

        # define hyper priors over model parameters
        a = pyro.param('a', ar.ones(npar), constraint=constraints.positive)
        lam = pyro.param('lam', ar.ones(npar), constraint=constraints.positive)
        tau = pyro.sample('tau', dist.Gamma(a, a/lam).to_event(1))

        sig = 1/ar.sqrt(tau)

        # each model parameter has a hyperprior defining group level mean
        m = pyro.param('m', ar.zeros(npar))
        s = pyro.param('s', ar.ones(npar), constraint=constraints.positive)
        mu = pyro.sample('mu', dist.Normal(m, s*sig).to_event(1))

        
        #for ind in range(self.nsubs):#pyro.plate("subject", len(self.data)):
        with pyro.plate('subject', self.nsubs) as ind:
            # print(ind)
            
            
            base_dist = dist.Normal(0., 1.).expand_by([npar]).to_event(1)
            transform = dist.transforms.AffineTransform(mu, sig)
            locs = pyro.sample('locs', dist.TransformedDistribution(base_dist, [transform]))
        
            param_dict = {"pol_lambda": ar.sigmoid(locs[...,0]), 
                          "r_lambda": ar.sigmoid(locs[...,1]), "dec_temp": ar.exp(locs[...,2]),
                          "h": ar.sigmoid(locs[...,3])}
            # print(param_dict)
            
            self.agent.reset(param_dict)
            #self.agent.set_parameters(pol_lambda=lamb_pi, r_lambda=lamb_r, dec_temp=dec_temp)
            
            
            for tau in pyro.markov(range(self.trials)):
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
                            print(param_dict)
                
                        curr_response = self.data["actions"][tau, t]
                        #print(curr_response)
                        # print(tau, t, probs, curr_response)
                        #print(tau,t,param_dict)
                        # print(curr_response.shape)
                        # print(probs.shape)
                        
                        pyro.sample('res_{}_{}'.format(tau, t), dist.Categorical(probs.permute(1,2,0)), obs=curr_response)
            
            # for tau in pyro.markov(range(self.trials)):
            #     for t in range(self.T):
            
            #         self.agent.update_beliefs(tau, t)
            
            #         if t < self.T-1:
                    
            #             probs = self.agent.perception.posterior_actions[-1]
            #             #print(probs)
            #             if ar.any(ar.isnan(probs)):
            #                 print(probs)
            #                 print(dec_temp, lamb_pi, lamb_r)
                
            #             curr_response = self.agent.perception.responses[:,tau,t]
            #             #print(curr_response)
            #             # print(tau, t, probs, curr_response)
            #             #print(tau,t,param_dict)
            #             draw_probs = probs.permute(2,0,1)
                        
            #             pyro.sample('res_{}_{}'.format(tau, t), dist.Categorical(draw_probs), obs=curr_response)
                    

    def guide(self):
        # approximate posterior. assume MF: each param has his own univariate Normal.
        
        npar = 4
        trns = biject_to(constraints.positive)

        m_hyp = pyro.param('m_hyp', ar.zeros(2*npar))
        st_hyp = pyro.param('scale_tril_hyp',
                       ar.eye(2*npar),
                       constraint=constraints.lower_cholesky)

        hyp = pyro.sample('hyp',
                     dist.MultivariateNormal(m_hyp, scale_tril=st_hyp),
                     infer={'is_auxiliary': True})

        unc_mu = hyp[..., :npar]
        unc_tau = hyp[..., npar:]

        c_tau = trns(unc_tau)

        ld_tau = trns.inv.log_abs_det_jacobian(c_tau, unc_tau)
        ld_tau = dist.util.sum_rightmost(ld_tau, ld_tau.dim() - c_tau.dim() + 1)

        mu = pyro.sample("mu", dist.Delta(unc_mu, event_dim=1))
        tau = pyro.sample("tau", dist.Delta(c_tau, log_density=ld_tau, event_dim=1))

        m_locs = pyro.param('m_locs', ar.zeros(self.nsubs, npar))
        st_locs = pyro.param('scale_tril_locs',
                        ar.eye(npar).repeat(self.nsubs, 1, 1),
                        constraint=constraints.lower_cholesky)

        with pyro.plate('runs', self.nsubs):
            locs = pyro.sample("locs", dist.MultivariateNormal(m_locs, scale_tril=st_locs))

        return {'tau': tau, 'mu': mu, 'locs': locs, "pol_lambda": ar.sigmoid(locs[...,0]), 
                "r_lambda": ar.sigmoid(locs[...,1]), "dec_temp": ar.exp(locs[...,2]), 
                "h": ar.sigmoid(locs[...,3])}

        
        
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
        pbar = tqdm(range(iter_steps), position=0)
        for step in pbar:#range(iter_steps):
            loss.append(ar.tensor(svi.step()).to(device))
            pbar.set_description("Mean ELBO %6.2f" % ar.tensor(loss[-20:]).mean())
            if ar.isnan(loss[-1]):
                break

        self.loss = [l.cpu() for l in loss]
        
        # alpha_lamb_pi = pyro.param("alpha_lamb_pi").data.numpy()
        # beta_lamb_pi = pyro.param("beta_lamb_pi").data.numpy()
        # alpha_lamb_r = pyro.param("alpha_lamb_r").data.numpy()
        # beta_lamb_r = pyro.param("beta_lamb_r").data.numpy()
        # alpha_h = pyro.param("alpha_lamb_r").data.numpy()
        # beta_h = pyro.param("beta_lamb_r").data.numpy()
        # concentration_dec_temp = pyro.param("concentration_dec_temp").data.numpy()
        # rate_dec_temp = pyro.param("rate_dec_temp").data.numpy()
        
        # param_dict = {"alpha_lamb_pi": alpha_lamb_pi, "beta_lamb_pi": beta_lamb_pi,
        #               "alpha_lamb_r": alpha_lamb_r, "beta_lamb_r": beta_lamb_r,
        #               "alpha_h": alpha_h, "beta_h": beta_h,
        #               "concentration_dec_temp": concentration_dec_temp, "rate_dec_temp": rate_dec_temp}
        
        return self.loss#, param_dict
    
    def sample_posterior_predictive(self, n_samples=5):

        elbo = pyro.infer.Trace_ELBO()
        post_sample_dict = {}
        
        predictive = pyro.infer.Predictive(model=self.model, guide=self.guide, num_samples=n_samples)
        samples = predictive.get_samples()

        #pbar = tqdm(range(n_samples), position=0)

        # for n in pbar:
        #     pbar.set_description("Sample posterior depth")
        #     # get marginal posterior over planning depths
        #     post_samples = elbo.compute_marginals(self.model, config_enumerate(self.guide))
        #     print(post_samples)
        #     for name in post_samples.keys():
        #         post_sample_dict.setdefault(name, [])
        #         post_sample_dict[name].append(post_samples[name].probs.detach().clone())

        # for name in post_sample_dict.keys():
        #     post_sample_dict[name] = ar.stack(post_sample_dict[name]).numpy()
            
        # post_sample_df = pd.DataFrame(post_sample_dict)
        
        
        reordered_sample_dict = {}
        all_keys = []
        for key in samples.keys():
            if key[:3] != 'res':
                reordered_sample_dict[key] = np.array([])
                all_keys.append(key)
                
        reordered_sample_dict['subject'] = np.array([])
        
        #nsubs = len(self.data)
        for sub in range(self.nsubs):
            for key in set(all_keys):
                reordered_sample_dict[key] = np.append(reordered_sample_dict[key], samples[key][:,sub].detach().numpy())#.squeeze()
            reordered_sample_dict['subject'] = np.append(reordered_sample_dict['subject'], [sub]*n_samples).squeeze()

        # for key in samples.keys():
        #     if key[:3] != 'res':
        #         sub = int(key[-1])
        #         reordered_sample_dict[key[:-2]] = np.append(reordered_sample_dict[key[:-2]], samples[key].detach().numpy()).squeeze()
        #         reordered_sample_dict['subject'] = np.append(reordered_sample_dict['subject'], [sub]).squeeze()
                
        sample_df = pd.DataFrame(reordered_sample_dict)

        return sample_df
    
    def sample_posterior(self, n_samples=5):
        # keys = ["lamb_pi", "lamb_r", "h", "dec_temp"]
        
        lamb_pi_global = np.zeros((n_samples, self.nsubs))
        lamb_r_global = np.zeros((n_samples, self.nsubs))
        alpha_0_global = np.zeros((n_samples, self.nsubs))
        dec_temp_global = np.zeros((n_samples, self.nsubs))
        
        for i in range(n_samples):
            sample = self.guide()
            for key in sample.keys():
                sample.setdefault(key, ar.ones(1))
            lamb_pi = sample["r_lambda"]
            lamb_r = sample["pol_lambda"]
            alpha_0 = sample["h"]
            dec_temp = sample["dec_temp"]
            
            lamb_pi_global[i] = lamb_pi.detach().numpy()
            lamb_r_global[i] = lamb_r.detach().numpy()
            alpha_0_global[i] = alpha_0.detach().numpy()
            dec_temp_global[i] = dec_temp.detach().numpy()
        
        lamb_pi_flat = np.array([lamb_pi_global[i,n] for i in range(n_samples) for n in range(self.nsubs)])
        lamb_r_flat = np.array([lamb_r_global[i,n] for i in range(n_samples) for n in range(self.nsubs)])
        alpha_0_flat = np.array([alpha_0_global[i,n] for i in range(n_samples) for n in range(self.nsubs)])
        dec_temp_flat = np.array([dec_temp_global[i,n] for i in range(n_samples) for n in range(self.nsubs)])
        
        subs_flat = np.array([n for i in range(n_samples) for n in range(self.nsubs)])
        
        
        sample_dict = {"lamb_pi": lamb_pi_flat, "lamb_r": lamb_r_flat,
                       "h": alpha_0_flat, 
                       "dec_temp": dec_temp_flat, "subject": subs_flat}
        
        sample_df = pd.DataFrame(sample_dict)
        
        return sample_df
    
    def analytical_posteriors(self):
        
        alpha_lamb_pi = pyro.param("alpha_lamb_pi").data.cpu().numpy()
        beta_lamb_pi = pyro.param("beta_lamb_pi").data.cpu().numpy()
        alpha_lamb_r = pyro.param("alpha_lamb_r").data.cpu().numpy()
        beta_lamb_r = pyro.param("beta_lamb_r").data.cpu().numpy()
        alpha_h = pyro.param("alpha_h").data.cpu().numpy()
        beta_h = pyro.param("beta_h").data.cpu().numpy()
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
    
    
    def plot_posteriors(self, n_samples=5):
        
        #df, param_dict = self.sample_posteriors()
        
        #sample_df = self.sample_posterior_marginals(n_samples=n_samples)
        
        sample_df = self.sample_posterior(n_samples=n_samples)
        
        plt.figure()
        sns.displot(data=sample_df, x='h', hue='subject')
        plt.xlim([0,1])
        plt.show()
        
        plt.figure()
        sns.displot(data=sample_df, x='lamb_r', hue='subject')
        plt.xlim([0,1])
        plt.show()
        
        plt.figure()
        sns.displot(data=sample_df, x='lamb_pi', hue='subject')
        plt.xlim([0,1])
        plt.show()
        
        plt.figure()
        sns.displot(data=sample_df, x='dec_temp', hue='subject')
        plt.xlim([0,10])
        plt.show()
        
        # plt.figure()
        # sns.histplot(marginal_df["h_1"])
        # plt.show()
        
        return sample_df