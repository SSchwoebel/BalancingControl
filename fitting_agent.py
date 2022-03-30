#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 16:10:45 2022

@author: sarah
"""

import jax.numpy as jnp
from jax.lax import scan
import jax.scipy.special as scs


class Agent(object):
    def __init__(self, big_trans_matrix, obs_matrix, opt_params, fix_rew_counts, preference, prior_states, policies, na, possible_policies, T):
        
        self.big_trans_matrix = big_trans_matrix
        self.obs_matrix = obs_matrix
        self.fix_rew_counts = fix_rew_counts
        self.preference = preference
        self.policies = policies
        self.na = na
        self.possible_policies = possible_policies
        self.T = T
        
        self.lambda_r = opt_params["lamb_r"]
        self.lambda_pi = opt_params["lamb_pi"]
        self.alpha = 1./opt_params["h"]
        self.dec_temp = opt_params["dec_temp"]
        
        npart = self.lambda_r.shape[1]
        self.prior_states = jnp.repeat(prior_states[:,None], npart, axis=1)
        self.bwd_init = jnp.repeat((jnp.ones_like(prior_states)/prior_states.shape[0])[:,None], npart, axis=1)
    
    def step(self, carry, curr_observed):
        
        rew_counts, pol_counts = carry
        
        curr_obs = curr_observed["obs"]
        curr_rew = curr_observed["rew"]
        prev_policy = curr_observed["policy"]
        t = curr_observed["t"]
        tau = curr_observed["tau"]
        
        total_counts = jnp.append(self.fix_rew_counts, rew_counts)
        rew_matrix = total_counts / total_counts.sum(axis=0)
        prior_policies = pol_counts / pol_counts(axis=0)
        
        rew_messages = self.make_rew_messages(rew_matrix, curr_rew)
        obs_messages = self.make_obs_messages(curr_obs)
        fwd_messages, fwd_norms = self.make_fwd_messages(rew_messages, obs_messages)
        bwd_messages = self.make_bwd_messages(rew_messages, obs_messages)
        
        posterior_states = fwd_messages*bwd_messages*obs_messages[...,None]*rew_messages
        norm = posterior_states.sum(axis=0)
        fwd_norms = jnp.concatenate([fwd_norms, norm[-1][None,:]], axis=0)
        
        posterior_policies = self.eval_posterior_policies(fwd_norms, prior_policies)
        
        marginal_posterior_states = jnp.einsum('stpn,pn->stn', posterior_states, posterior_policies)
        
        posterior_actions = self.post_actions_from_policies(posterior_policies, t)
        
        if t==self.T-1:
            rew_counts = self.update_rew_counts(rew_counts, curr_rew, marginal_posterior_states, t=t)
            pol_counts = self.update_pol_counts(pol_counts, prev_policy, posterior_policies)
        
        return (rew_counts, pol_counts), posterior_actions
    
    def make_rew_messages(self, rew_matrix, curr_rew):
        
        rew_messages = []
        for i, rew in enumerate(curr_rew):
            if rew is not None:
                rew_messages.append(rew_matrix[rew])
            else:
                rew_messages.append(jnp.einsum('r,rsn->sn', self.preference, rew_matrix))
                
        return jnp.stack(rew_messages).transpose((1,0,2))
    
    def make_obs_messages(self, curr_obs):
        
        obs_messages = []
        for i, obs in enumerate(curr_obs):
            if obs is not None:
                obs_messages.append(self.obs_matrix[obs])
            else:
                no = self.obs_matrix.shape[0]
                obs_messages.append(jnp.dot(jnp.ones(no)/no, self.obs_matrix))
                
        return jnp.stack(obs_messages).transpose((1,0))
    
    def make_fwd_messages(self, rew_messages, obs_messages):
        
        input_messages = [(rew_messages[:,i], obs_messages[:,i], self.big_trans_matrix[...,i]) for i in range(self.T-1)]
        init = self.prior_states
        fwd = scan(self.scan_messages, init, input_messages)
        
        fwd_messages = fwd[:,0,...].transpose((1,0,2,3))
        fwd_norms = fwd[:,1,...]
        
        return fwd_messages, fwd_norms
    
    def make_bwd_messages(self, rew_messages, obs_messages):
        
        input_messages = [(rew_messages[:,i+1], obs_messages[:,i+1], self.big_trans_matrix[...,i].transpose((1,0,2))) for i in range(self.T-1)]
        init = self.bwd_init
        bwd = scan(self.scan_messages, init, input_messages, reverse=True)
        
        bwd_messages = jnp.flip(bwd[:,0,...], axis=0).transpose((1,0,2,3))
        
        return bwd_messages
        
    def scan_messages(self, carry, input_message):
        
        old_message = carry
        obs_message, rew_message, trans_matrix = input_message
        
        tmp_message = jnp.einsum('spn,shp,sn,sn->hpn', old_message, trans_matrix, obs_message, obs_message)
        
        norm = tmp_message.sum(axis=0)
        message = jnp.where(norm > 0, tmp_message/norm, tmp_message)
        norm = jnp.where(self.possible_policies[:,None], norm, 0)
        
        return message, (message, norm)
        
    def eval_posterior_policies(self, fwd_norms, prior_policies):
        
        likelihood = (fwd_norms[-1]+1e-10).prod(axis=0)
        norm = likelihood.sum(axis=0)
        post = jnp.power(likelihood/norm,self.dec_temp[None,:]) * prior_policies
        posterior_policies = post / post.sum(axis=0)
        
        return posterior_policies
    
    def post_actions_from_policies(self, posterior_policies, t):
            
        post_actions = jnp.stack([posterior_policies[self.policies[t]==a].sum() for a in range(self.na)])
        
        return post_actions
            
    def contract_posterior_policies(self, posterior_policies, a, t):
        
        return posterior_policies[self.policies[t]==a].sum()
        
    def update_rew_counts(self, prev_rew_counts, curr_rew, post_states, t=None):
        
        #note to self: try implemementing with binary mask multiplication instead of separated matrices
        # maybe using jnp.where ?
        if t is None:
            for i, rew in enumerate(curr_rew):
                if rew is not None:
                    rew_counts = (1-self.lambda_r)[None,None,...]*prev_rew_counts + self.lambda_r[None,None,...] \
                        + jnp.eye(self.learn_rew_mask.shape[0])[rew][:,None,...]*post_states[-prev_rew_counts.shape[1]:,i,...][None,:,...]
                    prev_rew_counts = rew_counts
        else:
            rew_counts = (1-self.lambda_r)[None,None,...]*prev_rew_counts + self.lambda_r[None,None,...] \
                        + jnp.eye(self.learn_rew_mask.shape[0])[rew][:,None,...]*post_states[-prev_rew_counts.shape[1]:,t,...][None,:,...]
            
        return rew_counts
    
    def update_pol_counts(self, prev_pol_counts, prev_policy, posterior_policies):
        
        pol_counts = (1-self.lambda_pi)[None,...]*prev_pol_counts + (self.lambda_pi*self.alpha)[None,...] + posterior_policies
        
        return pol_counts