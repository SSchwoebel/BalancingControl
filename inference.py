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
import json

from tqdm import tqdm
import pyro
import pyro.distributions as dist
import agent as agt
import perception as prc
import action_selection as asl

pyro.clear_param_store()

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
        print("init", self.data["observations"].shape)

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

        # # tell pyro about prior over parameters: alpha and beta of h which is between 0 and 1
        # # alpha = beta = 1 equals uniform prior
        # alpha_h = ar.ones(1).to(device)
        # beta_h = ar.ones(1).to(device)
        # # sample initial vaue of parameter from Beta distribution
        # h = pyro.sample('h', dist.Beta(alpha_h, beta_h)).to(device)

        # tell pyro about prior over parameters: decision temperature
        # uniform between 0 and 20??
        concentration_dec_temp = ar.tensor(1.).to(device)
        rate_dec_temp = ar.tensor(0.5).to(device)
        # sample initial vaue of parameter from normal distribution
        dec_temp = pyro.sample('dec_temp', dist.Gamma(concentration_dec_temp, rate_dec_temp)).to(device)
        param_dict = {"pol_lambda": lamb_pi[:,None], "r_lambda": lamb_r[:,None], "dec_temp": dec_temp[:,None]}#, "h": h

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

        # # tell pyro about posterior over parameters: alpha and beta of lambda which is between 0 and 1
        # alpha_h = pyro.param("alpha_h", ar.ones(1), constraint=ar.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        # beta_h = pyro.param("beta_h", ar.ones(1), constraint=ar.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        # # sample initial vaue of parameter from Beta distribution
        # h = pyro.sample('h', dist.Beta(alpha_h, beta_h)).to(device)

        # tell pyro about posterior over parameters: mean and std of the decision temperature
        concentration_dec_temp = pyro.param("concentration_dec_temp", ar.ones(1)*3., constraint=ar.distributions.constraints.positive).to(device)#interval(0., 7.))
        rate_dec_temp = pyro.param("rate_dec_temp", ar.ones(1), constraint=ar.distributions.constraints.positive).to(device)
        # sample initial vaue of parameter from normal distribution
        dec_temp = pyro.sample('dec_temp', dist.Gamma(concentration_dec_temp, rate_dec_temp)).to(device)

        param_dict = {"alpha_lamb_pi": alpha_lamb_pi, "beta_lamb_pi": beta_lamb_pi, "lamb_pi": lamb_pi,
                      "alpha_lamb_r": alpha_lamb_r, "beta_lamb_r": beta_lamb_r, "lamb_r": lamb_r,
                      #"alpha_h": alpha_h, "beta_h": beta_h, "h": h,
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

    def sample_posterior(self, n_samples=5):
        # keys = ["lamb_pi", "lamb_r", "h", "dec_temp"]

        lamb_pi_global = np.zeros((n_samples, 1))
        lamb_r_global = np.zeros((n_samples, 1))
        # alpha_0_global = np.zeros((n_samples, self.nsubs))
        dec_temp_global = np.zeros((n_samples, 1))

        for i in range(n_samples):
            sample = self.guide()
            for key in sample.keys():
                sample.setdefault(key, ar.ones(1))
            lamb_pi = sample["r_lambda"]
            lamb_r = sample["pol_lambda"]
            #alpha_0 = sample["h"]
            dec_temp = sample["dec_temp"]

            lamb_pi_global[i] = lamb_pi.detach().numpy()
            lamb_r_global[i] = lamb_r.detach().numpy()
            #alpha_0_global[i] = alpha_0.detach().numpy()
            dec_temp_global[i] = dec_temp.detach().numpy()

        lamb_pi_flat = np.array([lamb_pi_global[i,n] for i in range(n_samples) for n in range(1)])
        lamb_r_flat = np.array([lamb_r_global[i,n] for i in range(n_samples) for n in range(1)])
        #alpha_0_flat = np.array([alpha_0_global[i,n] for i in range(n_samples) for n in range(self.nsubs)])
        dec_temp_flat = np.array([dec_temp_global[i,n] for i in range(n_samples) for n in range(1)])

        subs_flat = np.array([n for i in range(n_samples) for n in range(1)])


        sample_dict = {"lamb_pi": lamb_pi_flat, "lamb_r": lamb_r_flat,
                       #h": alpha_0_flat,
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


    def save_parameters(self, fname):

        pyro.get_param_store().save(fname)

    def load_parameters(self, fname):

        pyro.get_param_store().load(fname)



class GeneralGroupInference(object):

    def __init__(self, agent, data):

        pyro.clear_param_store()

        self.agent = agent
        self.trials = agent.trials
        self.T = agent.T
        self.data = data
        self.nsubs = len(data['rewards'][0,0])
        self.svi = None
        self.loss = []
        self.npars = self.agent.perception.npars
        self.mask = agent.perception.mask

    def model(self):
        """
        Generative model of behavior with a NormalGamma
        prior over free model parameters.
        """

        # define hyper priors over model parameters
        a = pyro.param('a', ar.ones(self.npars), constraint=constraints.positive)
        lam = pyro.param('lam', ar.ones(self.npars), constraint=constraints.positive)
        tau = pyro.sample('tau', dist.Gamma(a, a/lam).to_event(1))

        sig = 1/ar.sqrt(tau)

        # each model parameter has a hyperprior defining group level mean
        m = pyro.param('m', ar.zeros(self.npars))
        s = pyro.param('s', ar.ones(self.npars), constraint=constraints.positive)
        mu = pyro.sample('mu', dist.Normal(m, s*sig).to_event(1))


        with pyro.plate('subject', self.nsubs) as ind:


            base_dist = dist.Normal(0., 1.).expand_by([self.npars]).to_event(1)
            transform = dist.transforms.AffineTransform(mu, sig)
            locs = pyro.sample('locs', dist.TransformedDistribution(base_dist, [transform]))

            self.agent.reset(locs)

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
                        if ar.any(ar.isnan(probs)):
                            print(probs)
                            #print(param_dict)
                            print(tau,t)

                        curr_response = self.data["actions"][tau, t]*self.mask[tau].long()

                        pyro.sample('res_{}_{}'.format(tau, t), dist.Categorical(probs.permute(1,2,0)).mask(self.mask[tau]), obs=curr_response)



    def guide(self):

        trns = biject_to(constraints.positive)

        m_hyp = pyro.param('m_hyp', ar.zeros(2*self.npars))
        st_hyp = pyro.param('scale_tril_hyp',
                       ar.eye(2*self.npars),
                       constraint=constraints.lower_cholesky)

        hyp = pyro.sample('hyp',
                     dist.MultivariateNormal(m_hyp, scale_tril=st_hyp),
                     infer={'is_auxiliary': True})

        unc_mu = hyp[..., :self.npars]
        unc_tau = hyp[..., self.npars:]

        c_tau = trns(unc_tau)

        ld_tau = trns.inv.log_abs_det_jacobian(c_tau, unc_tau)
        ld_tau = dist.util.sum_rightmost(ld_tau, ld_tau.dim() - c_tau.dim() + 1)

        mu = pyro.sample("mu", dist.Delta(unc_mu, event_dim=1))
        tau = pyro.sample("tau", dist.Delta(c_tau, log_density=ld_tau, event_dim=1))

        m_locs = pyro.param('m_locs', ar.zeros(self.nsubs, self.npars))
        st_locs = pyro.param('scale_tril_locs',
                        ar.eye(self.npars).repeat(self.nsubs, 1, 1),
                        constraint=constraints.lower_cholesky)

        with pyro.plate('subject', self.nsubs):
            locs = pyro.sample("locs", dist.MultivariateNormal(m_locs, scale_tril=st_locs))

        return {'tau': tau, 'mu': mu, 'locs': locs}

    def init_svi(self, optim_kwargs={'lr': .01},
                 num_particles=10):

        #pyro.clear_param_store()

        self.svi = pyro.infer.SVI(model=self.model,
                  guide=self.guide,
                  optim=pyro.optim.Adam(optim_kwargs),
                  loss=pyro.infer.Trace_ELBO(num_particles=num_particles,
                                  #set below to true once code is vectorized
                                  vectorize_particles=True))


    def infer_posterior(self,
                        iter_steps=1000, optim_kwargs={'lr': .01},
                                     num_particles=10):
        """Perform SVI over free model parameters.
        """

        #pyro.clear_param_store()
        if self.svi is None:
            self.init_svi(optim_kwargs, num_particles)

        loss = []
        pbar = tqdm(range(iter_steps), position=0)
        for step in pbar:#range(iter_steps):
            loss.append(ar.tensor(self.svi.step()).to(device))
            pbar.set_description("Mean ELBO %6.2f" % ar.tensor(loss[-20:]).mean())
            if ar.isnan(loss[-1]):
                break

        self.loss += [l.cpu() for l in loss]

    def sample_posterior_predictive(self, n_samples=5):

        pass

    def sample_posterior(self, n_samples=5):
        # keys = ["lamb_pi", "lamb_r", "h", "dec_temp"]

        param_names = self.agent.perception.param_names
        sample_dict = {param: [] for param in param_names}
        sample_dict["subject"] = []

        for i in range(n_samples):
            sample = self.guide()
            for key in sample.keys():
                sample.setdefault(key, ar.ones(1))

            par_sample = self.agent.locs_to_pars(sample["locs"])

            for param in param_names:
                sample_dict[param].extend(list(par_sample[param].detach().numpy()))

            sample_dict["subject"].extend(list(range(self.nsubs)))

        sample_df = pd.DataFrame(sample_dict)

        return sample_df

    def analytical_posteriors(self):

        pass


    def save_parameters(self, fname):

        pyro.get_param_store().save(fname)

    def load_parameters(self, fname):
        
        pyro.clear_param_store()
        pyro.get_param_store().load(fname)

    def parameters(self):

        params = pyro.get_param_store()
        return {key: params[key] for key in params.keys()}

    def save_elbo(self, fname):

        with open(fname, 'w') as outfile:
            json.dump([float(l) for l in self.loss], outfile)

    def load_elbo(self, fname):

        with open(fname, 'r') as infile:
            loss = json.load(infile)

        self.loss = [ar.tensor(l) for l in loss]