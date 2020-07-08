#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 10:31:33 2020

@author: sarah
"""


from tqdm import tqdm
import pandas as pd
import numpy as np

import torch

from pyro import clear_param_store, get_param_store
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.enum import get_importance_trace
from pyro.optim import Adam, SGD, ASGD
import pyro.distributions as dist
from pyro import sample, param, plate
from torch.distributions import constraints
import gc


class Inferrer(object):
    
    def __init__(self, agent, observations, rewards, responses, trials, T, parameters, fixed_params=None):
        
        self.agent = agent # agent used for computing response probabilities
        self.observations = observations # stimulus and action outcomes presented to each participant
        self.rewards = rewards
        self.responses = responses # measured behavioral data accross all subjects
        self.trials = trials
        self.T = T
        self.parameters = parameters

    def model(self):
        """
        Full generative model of behavior.
        """
        # define hyper priors over model parameters.
        # each model parameter has a hyperpriors defining group level mean
        m = param('m', torch.ones(1))#, constraint=constraints.interval(1.,100.))
        s = param('s', torch.ones(1), constraint=constraints.positive)
        print('m', m)
        print('s', s)
        
        alpha = sample('loc', dist.Normal(m, s).to_event(1))
        
        self.parameters['alpha'] = torch.zeros((self.agent.npi, self.agent.nc), requires_grad=True) + alpha

        self.agent.reset(self.parameters)
        
        #gc.collect()
        
        for tau in range(self.trials):
            for t in range(self.T):              

                #update single trial
                
                if t==0:
                    response = None
                else:
                    response = self.responses[tau, t-1]
                
                reward = self.rewards[tau, t]
                    
                observation = self.observations[tau, t]
                
                self.agent.update_beliefs(tau, t, observation, reward, response)
                
                if t>0:
                
                    probs = self.agent.posterior_actions
                        
                    with plate('responses_{}_{}'.format(tau, t), 1):
                        sample('obs_{}_{}'.format(tau, t), dist.Categorical(probs=probs), obs=response)
                        
            #gc.collect()
            
    def guide(self):
        """Approximate posterior over model parameters.
        """
        npar = 1 #number of parameters
        
        m_locs = param('m_locs', torch.ones(npar))#, constraint=constraints.interval(1.,100.))
        st_locs = param('scale_tril_locs', torch.ones(npar), 
                   constraint=constraints.positive)


        locs = sample("locs", dist.Normal(m_locs, st_locs))
        
        return {'locs': locs}

    def infer_posterior(self, 
                        iter_steps = 500,
                        num_particles = 10,
                        optim_kwargs = {'lr':.01}):
        """Perform SVI over free model parameters.
        """

        clear_param_store()

        svi = SVI(model=self.model,
                  guide=self.guide,
                  optim=Adam(optim_kwargs),
                  loss=Trace_ELBO(num_particles=num_particles, 
                                  vectorize_particles=False))

        loss = []
        pbar = tqdm(range(iter_steps), position=0)
        for step in pbar:
            loss.append(svi.step())
            pbar.set_description("Mean ELBO %6.2f" % torch.tensor(loss[-20:]).mean())
            if np.isnan(loss[-1]):
                break
                
        self.loss = loss

    def sample_posterior(self, n_samples=1000):
        """Sample from the posterior distribution over model parameters.
        """
        
        trans_pars = torch.zeros(n_samples)
        
        for i in range(n_samples):
            trans_pars[i] = self.guide()['locs'].detach()
        
        return trans_pars

    
    def _get_quantiles(self, quantiles):
        """
        Returns posterior quantiles each latent variable. Example::
            print(agent.get_quantiles([0.05, 0.5, 0.95]))
        :param quantiles: A list of requested quantiles between 0 and 1.
        :type quantiles: torch.tensor or list
        :return: A dict mapping sample site name to a list of quantile values.
        :rtype: dict
        """
        
        raise NotImplementedError
    
    def formated_results(self, par_names, labels=None):
        """Returns median, 5th and 95th percentile for each parameter and subject. 
        """
        nsub = self.runs
        npar = self.npar
        
        if labels is None:
            labels = par_names
        
        
        quantiles = self._get_quantiles([.05, .5, .95])
            
        locs = quantiles['locs'].transpose(dim0=0, dim1=-1).transpose(dim0=1, dim1=-1)

        if self.fixed_values:
            x = torch.zeros(3, nsub, npar)
            x[..., self.locs['fixed']] = self.values
            x[..., self.locs['free']] = locs.detach()
        else:
            x = locs.detach()
        
        self.agent.set_parameters(x, set_variables=False)
        
        par_values = {}
        for name in par_names:
            values = getattr(self.agent, name)
            if values.dim() < 3:
                values = values.unsqueeze(dim=-1)
            par_values[name] = values
        
        count = {}
        percentiles = {}
        for name in par_names:
            count.setdefault(name, 0)
            for lbl in labels:
                if lbl.startswith(name):
                    percentiles[lbl] = par_values[name][..., count[name]].numpy().reshape(-1)
                    count[name] += 1
        
        df_percentiles = pd.DataFrame(percentiles)
        
        subjects = torch.arange(1, nsub+1).repeat(3, 1).reshape(-1)
        df_percentiles['subjects'] = subjects.numpy()
        
        from numpy import tile, array
        variables = tile(array(['5th', 'median', '95th']), [nsub, 1]).T.reshape(-1)
        df_percentiles['variables'] = variables
        
        return df_percentiles.melt(id_vars=['subjects', 'variables'], var_name='parameter')

    def get_log_evidence_per_subject(self, num_particles = 100, max_plate_nesting=1):
        """Return subject specific log model evidence"""
        
        model = self.model
        guide = self.guide
        notnans = self.notnans
        
        elbo = torch.zeros(self.runs)
        for i in range(num_particles):
            model_trace, guide_trace = get_importance_trace('flat', max_plate_nesting, model, guide)
            obs_log_probs = torch.zeros(notnans.shape)
            for site in model_trace.nodes.values():
                if site['name'].startswith('obs'):
                    obs_log_probs[notnans] = site['log_prob'].detach()
                elif site['name'] == 'locs':
                    elbo += site['log_prob'].detach()
            
            elbo += torch.einsum('ijk->k', obs_log_probs)

            for site in guide_trace.nodes.values():
                if site['name'] == 'locs':
                    elbo -= site['log_prob'].detach()
        
        return elbo/num_particles