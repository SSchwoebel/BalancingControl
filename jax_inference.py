#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 17:21:47 2022

@author: sarah
"""

import jax.numpy as jnp
from jax.lax import scan, cond
from jax import random, jit
from functools import partial
import numpyro as pyro
import numpyro.distributions as dist
import itertools
import jsonpickle as pickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import json
import os
import copy
import matplotlib.pylab as plt
import seaborn as sns

from jax.config import config
config.update("jax_enable_x64", True)

def load_data(fname):

    jsonpickle_numpy.register_handlers()

    with open(fname, 'r') as infile:
        loaded = json.load(infile)

    data = pickle.decode(loaded)
    
    return data

trials =  201#number of trials
T = 3 #number of time steps in each trial
nb = 4
ns = 3+nb #number of states
no = ns #number of observations
na = 2 #number of actions
npi = na**(T-1)
nr = 2

#generating probability of observations in each state
A = jnp.eye(no)#.to(device)
obs_matrix = A

#state transition generative probability (matrix)
B = jnp.zeros((ns, ns, na))
b1 = 0.7
nb1 = 1.-b1
b2 = 0.7
nb2 = 1.-b2

state_trans_matrix = jnp.array([[[  0,  0,  0,  0,  0,  0,  0,],
                     [ b1,  0,  0,  0,  0,  0,  0,],
                     [nb1,  0,  0,  0,  0,  0,  0,],
                     [  0,  1,  0,  1,  0,  0,  0,],
                     [  0,  0,  1,  0,  1,  0,  0,],
                     [  0,  0,  0,  0,  0,  1,  0,],
                     [  0,  0,  0,  0,  0,  0,  1,],],

                    [[  0,  0,  0,  0,  0,  0,  0,],
                     [nb2,  0,  0,  0,  0,  0,  0,],
                     [ b2,  0,  0,  0,  0,  0,  0,],
                     [  0,  0,  0,  1,  0,  0,  0,],
                     [  0,  0,  0,  0,  1,  0,  0,],
                     [  0,  1,  0,  0,  0,  1,  0,],
                     [  0,  0,  1,  0,  0,  0,  1,],]]).transpose((1,2,0))


u = 0.999
utility = jnp.array([1-u, u])

preference = utility

fix_rew_counts = jnp.array([[100]*3, [1]*3])[:,:,None]

policies = jnp.array(list(itertools.product(list(range(na)), repeat=T-1)))

def calc_big_trans_matrix(state_trans_matrix, policies):

    npi = policies.shape[0]
    big_trans_matrix = jnp.stack([jnp.stack([state_trans_matrix[:,:,policies[pi,t]] for pi in range(npi)]) for t in range(T-1)]).transpose((2,3,1,0))
    
    return big_trans_matrix

def calc_possible_policies(data, policies):
    
    npi = policies.shape[0]
    all_possible_policies = [[[True]*npi]*T]*trials
    
    for curr_observed in data:
        
        curr_obs = curr_observed["obs"]
        curr_rew = curr_observed["rew"]
        response = curr_observed["response"]
        t = curr_observed["t"]
        tau = curr_observed["tau"]
        
        if t == 0:
            all_possible_policies[tau][t] = jnp.stack(all_possible_policies[tau][t])
        if t>0:# and t < self.T - 1:
            curr_possible_policies = policies[:,t-1]==response[t-1]
            curr_possible_policies = jnp.logical_and(all_possible_policies[tau][t-1], curr_possible_policies)
            all_possible_policies[tau][t] = jnp.stack(curr_possible_policies)
            
        if t == T-1:
            all_possible_policies[tau] = jnp.stack(all_possible_policies[tau])
            
    return jnp.stack(all_possible_policies)


def transform_data(data):

    shaped_data = []
    
    for curr_observed in data:

        curr_obs = curr_observed["obs"]
        curr_rew = curr_observed["rew"]
        response = curr_observed["response"]
        t = curr_observed["t"]
        tau = curr_observed["tau"]

        shaped_data.append(jnp.stack(
            [jnp.stack(curr_obs), 
             jnp.stack(curr_rew), 
             jnp.stack(response), 
             jnp.stack([tau]*T), 
             jnp.stack([t]*T)]))
        
    return jnp.stack(shaped_data)

big_trans_matrix = calc_big_trans_matrix(state_trans_matrix, policies)

state_prior = jnp.eye(ns)[0]

npart = 1
npi = policies.shape[0]
prior_states = jnp.repeat(jnp.repeat(state_prior[:,None], npi, axis=1)[:,:,None], npart, axis=1) #+ 1e-20
bwd_init = jnp.repeat(jnp.repeat((jnp.ones_like(state_prior)/prior_states.shape[0])[:,None], npi, axis=1)[:,:,None], npart, axis=1)
print(bwd_init.shape)

run = 0
lp = 0.3
lr = 0.3
dt = 5.
tend = 1./1000

folder = "data"

run_name = "twostage_results"+str(run)+"_pl"+str(lp)+"_rl"+str(lr)+"_dt"+str(dt)+"_tend"+str(int(1./tend))+".json"
fname = os.path.join(folder, run_name)

data = load_data(fname=fname)

all_possible_policies = calc_possible_policies(data, policies)
print(all_possible_policies.shape)

nicely_shaped_data = transform_data(data)
print(nicely_shaped_data.shape)

def Bayesian_habit_model():
    
    # generative model of behavior with Normally distributed params (within subject!!)

    # tell pyro about prior over parameters: alpha and beta of lambda which is between 0 and 1
    # alpha = beta = 1 equals uniform prior
    alpha_lambda_pi = jnp.ones(1)#.to(device)
    beta_lambda_pi = jnp.ones(1)#.to(device)
    # sample initial vaue of parameter from Beta distribution
    lambda_pi = pyro.sample('lambda_pi', dist.Beta(alpha_lambda_pi, beta_lambda_pi))#.to(device)

    # tell pyro about prior over parameters: alpha and beta of lambda which is between 0 and 1
    # alpha = beta = 1 equals uniform prior
    alpha_lambda_r = jnp.ones(1)#.to(device)
    beta_lambda_r = jnp.ones(1)#.to(device)
    # sample initial vaue of parameter from Beta distribution
    lambda_r = pyro.sample('lambda_r', dist.Beta(alpha_lambda_r, beta_lambda_r))#.to(device)

    # tell pyro about prior over parameters: alpha and beta of h which is between 0 and 1
    # alpha = beta = 1 equals uniform prior
    alpha_h = jnp.ones(1)#.to(device)
    beta_h = jnp.ones(1)#.to(device)
    # sample initial vaue of parameter from Beta distribution
    h = pyro.sample('h', dist.Beta(alpha_h, beta_h))#.to(device)
    alpha = 1./h
    
    init_pol_counts = jnp.zeros((npi, 1)) + alpha
    init_rew_counts = jnp.ones((nr, ns-3, 1))

    # tell pyro about prior over parameters: decision temperature
    # uniform between 0 and 20??
    concentration_dec_temp = jnp.array(1.)#.to(device)
    rate_dec_temp = jnp.array(0.5)#.to(device)
    # sample initial vaue of parameter from normal distribution
    dec_temp = pyro.sample('dec_temp', dist.Gamma(concentration_dec_temp, rate_dec_temp))#.to(device)
    
    
    @jit
    def step(carry, curr_data):
        
        rew_counts, pol_counts = carry
        
        curr_obs, curr_rew, curr_response, curr_tau, curr_t = curr_data
        
        #curr_obs = curr_observed["obs"]
        #curr_rew = curr_observed["rew"]
        t = curr_t[0]
        tau = curr_tau[0]
        
        possible_policies = all_possible_policies[tau,t]
        
        total_counts = jnp.concatenate([fix_rew_counts, rew_counts], axis=1)
        rew_matrix = total_counts / total_counts.sum(axis=0)
        prior_policies = pol_counts / pol_counts.sum(axis=0)
    
        @jit
        def make_rew_messages(rew_matrix, curr_rew):

            def future_func(rew):
                return jnp.einsum('r,rsn->sn', preference, rew_matrix)
                
            def past_func(rew):
                return rew_matrix[rew]
            
            messages = []
            for i, rew in enumerate(curr_rew):
                messages.append(cond(rew != -1, past_func, future_func, rew))
                #if rew != -1:
                #    rew_messages.append(rew_matrix[rew])
                #else:
                #    rew_messages.append(jnp.einsum('r,rsn->sn', preference, rew_matrix))

            return jnp.stack(messages).transpose((1,0,2))

        @jit
        def make_obs_messages(curr_obs):

            def future_func(obs):
                return jnp.dot(jnp.ones(no)/no, obs_matrix)
                
            def past_func(obs):
                return obs_matrix[obs]

            messages = []
            for i, obs in enumerate(curr_obs):
                messages.append(cond(obs != -1, past_func, future_func, obs))
                #if obs != -1:
                #    obs_messages.append(obs_matrix[obs])
                #else:
                #    no = obs_matrix.shape[0]
                #    obs_messages.append(jnp.dot(jnp.ones(no)/no, obs_matrix))

            return jnp.stack(messages).transpose((1,0))
            i += 1

        @jit    
        def scan_fwd_messages(carry, input_message):

            i, old_message = carry
            rew_message, obs_message = input_message
            tmp_message = jnp.einsum('hpn,shp,hn,hn->spn', old_message, big_trans_matrix[...,i], obs_message, rew_message)

            norm = tmp_message.sum(axis=0)
            message = jnp.where(norm > 0, tmp_message/norm[None,...], tmp_message)
            #norms = jnp.where(possible_policies[:,None], norm, 0)
            i += 1

            return (i, message), (message, norm)

        @jit
        def make_fwd_messages(rew_messages, obs_messages):

            input_messages = jnp.stack([jnp.stack([rew_messages[:,i], obs_messages[:,i][:,None]]) for i in range(T-1)])
            init = (0, prior_states)

            carry, fwd = scan(scan_fwd_messages, init, input_messages)
            fwd_messages, fwd_norms = fwd
            fwd_messages = jnp.concatenate([init[1][None,...], fwd_messages], axis=0).transpose((1,0,2,3))

            return fwd_messages, fwd_norms

        @jit    
        def scan_bwd_messages(carry, input_message):

            i, old_message = carry
            rew_message, obs_message = input_message

            #print("old", old_message)
            tmp_message = jnp.einsum('hpn,shp,hn,hn->spn', old_message, big_trans_matrix[...,i].transpose((1,0,2)), obs_message, rew_message)
            # print(tmp_message)

            norm = tmp_message.sum(axis=0)
            message = tmp_message#jnp.where(norm > 0, tmp_message/norm[None,...], tmp_message)

            return (i, message), (message, norm)

        @jit
        def make_bwd_messages(rew_messages, obs_messages):

            input_messages = jnp.stack([jnp.stack([rew_messages[:,i+1], obs_messages[:,i+1][:,None]]) for i in range(T-1)])
            init = (0, bwd_init)
            carry = init
            carry, bwd = scan(scan_bwd_messages, init, input_messages, reverse=True)
            bwd_messages, bwd_norms = bwd
            bwd_messages = jnp.concatenate([bwd_messages, init[1][None,...]], axis=0).transpose((1,0,2,3))

            return bwd_messages

        @jit    
        def eval_posterior_policies(fwd_norms, prior_policies):

            likelihood = (fwd_norms).prod(axis=0)#+1e-10
            norm_const = likelihood.sum(axis=0)
            post = jnp.power(likelihood/norm_const, dec_temp[None,:]) * prior_policies
            posterior_policies = post / post.sum(axis=0)

            return posterior_policies

        def post_actions_from_policies(posterior_policies, t):

            post_actions = jnp.array([jnp.where(policies[:,t]==a, posterior_policies, 0).sum() for a in range(na)])

            return post_actions

        @jit        
        def contract_posterior_policies(posterior_policies, a, t):

            return posterior_policies[policies[t]==a].sum()

        @jit    
        def update_rew_counts(prev_rew_counts, curr_rew, post_states, t):

            #note to self: try implemementing with binary mask multiplication instead of separated matrices
            # maybe using jnp.where ?
            no = prev_rew_counts.shape[0]
            #if t == -1:
            #    for i, rew in enumerate(curr_rew):
            #        if rew != -1:
            #            rew_counts = (1-lambda_r)[None,None,...]*prev_rew_counts + lambda_r[None,None,...] \
            #                + jnp.eye(no)[rew][:,None,...]*post_states[-prev_rew_counts.shape[1]:,i,...][None,:,...]
            #            prev_rew_counts = rew_counts
            rew = curr_rew[t]
            new_rew_counts = (1-lambda_r)[None,None,...]*prev_rew_counts + lambda_r[None,None,...] \
                        + jnp.eye(no)[rew][:,None,None]*post_states[-prev_rew_counts.shape[1]:,t,...][None,:,:]

            return new_rew_counts

        @jit
        def update_pol_counts(prev_pol_counts, posterior_policies):

            new_pol_counts = (1-lambda_pi)[None,...]*prev_pol_counts + (lambda_pi)[None,...]*alpha + posterior_policies

            return new_pol_counts
        
        @jit
        def update_counts(rew_counts, curr_rew, marginal_posterior_states, t, pol_counts, posterior_policies):
            new_rew_counts = update_rew_counts(rew_counts, curr_rew, marginal_posterior_states, t)
            new_pol_counts = update_pol_counts(pol_counts, posterior_policies)
            
            return new_rew_counts, new_pol_counts
        
        @jit
        def do_not_update_counts(rew_counts, curr_rew, marginal_posterior_states, t, pol_counts, posterior_policies):
            new_rew_counts = rew_counts
            new_pol_counts = pol_counts
            return new_rew_counts, new_pol_counts
        
        rew_messages = make_rew_messages(rew_matrix, curr_rew)
        obs_messages = make_obs_messages(curr_obs)
        fwd_messages, fwd_norms = make_fwd_messages(rew_messages, obs_messages)
        bwd_messages = make_bwd_messages(rew_messages, obs_messages)
        
        post_states = fwd_messages*bwd_messages*obs_messages[...,None,None]*rew_messages[...,None]
        norm = post_states.sum(axis=0)
        
        posterior_states = jnp.where(norm[None,...] > 0, post_states/norm[None,...],  post_states)
        fwd_norms = jnp.concatenate([fwd_norms, norm[-1][None,:]], axis=0)
        fwd_norms = jnp.where(possible_policies[:,None], fwd_norms, 0)
        
        posterior_policies = eval_posterior_policies(fwd_norms, prior_policies)
        
        marginal_posterior_states = jnp.einsum('stpn,pn->stn', posterior_states, posterior_policies)
        
        posterior_actions = post_actions_from_policies(posterior_policies, t)
        
        new_rew_counts, new_pol_counts = cond(t==T-1, update_counts, do_not_update_counts, 
             rew_counts, curr_rew, marginal_posterior_states, t, pol_counts, posterior_policies)
        #if t==T-1:
        #    rew_counts = update_rew_counts(rew_counts, curr_rew, marginal_posterior_states, t)
        #    pol_counts = update_pol_counts(pol_counts, posterior_policies)
        
        pyro.sample('res_{}_{}'.format(tau, t), dist.Categorical(posterior_actions.T), obs=curr_response)
        
        return (new_rew_counts, new_pol_counts), posterior_actions
    
    
    carry = (init_rew_counts, init_pol_counts)
    
    _, all_posterior_actions = scan(step, carry, nicely_shaped_data)
    
    
    
def guide():
    # approximate posterior. assume MF: each param has his own univariate Normal.

    # tell pyro about posterior over parameters: alpha and beta of lambda which is between 0 and 1
    alpha_lambda_pi = pyro.param("alpha_lambda_pi", jnp.ones(1), constraint=pyro.distributions.constraints.positive)#.to(device)#greater_than_eq(1.))
    beta_lambda_pi = pyro.param("beta_lambda_pi", jnp.ones(1), constraint=pyro.distributions.constraints.positive)#.to(device)#greater_than_eq(1.))
    # sample vaue of parameter from Beta distribution
    # print()
    # print(alpha_lamb_pi, beta_lamb_pi)
    lambda_pi = pyro.sample('lambda_pi', dist.Beta(alpha_lambda_pi, beta_lambda_pi))#.to(device)

    # tell pyro about posterior over parameters: alpha and beta of lambda which is between 0 and 1
    alpha_lambda_r = pyro.param("alpha_lambda_r", jnp.ones(1), constraint=pyro.distributions.constraints.positive)#.to(device)#greater_than_eq(1.))
    beta_lambda_r = pyro.param("beta_lambda_r", jnp.ones(1), constraint=pyro.distributions.constraints.positive)#.to(device)#greater_than_eq(1.))
    # sample initial vaue of parameter from Beta distribution
    lambda_r = pyro.sample('lambda_r', dist.Beta(alpha_lambda_r, beta_lambda_r))#.to(device)

    # tell pyro about posterior over parameters: alpha and beta of lambda which is between 0 and 1
    alpha_h = pyro.param("alpha_h", jnp.ones(1), constraint=pyro.distributions.constraints.positive)#.to(device)#greater_than_eq(1.))
    beta_h = pyro.param("beta_h", jnp.ones(1), constraint=pyro.distributions.constraints.positive)#.to(device)#greater_than_eq(1.))
    # sample initial vaue of parameter from Beta distribution
    h = pyro.sample('h', dist.Beta(alpha_h, beta_h))#.to(device)

    # tell pyro about posterior over parameters: mean and std of the decision temperature
    concentration_dec_temp = pyro.param("concentration_dec_temp", jnp.ones(1)*3., constraint=pyro.distributions.constraints.positive)#.to(device)#interval(0., 7.))
    rate_dec_temp = pyro.param("rate_dec_temp", jnp.ones(1), constraint=pyro.distributions.constraints.positive)#.to(device)
    # sample initial vaue of parameter from normal distribution
    dec_temp = pyro.sample('dec_temp', dist.Gamma(concentration_dec_temp, rate_dec_temp))#.to(device)
    
def infer_posterior(iter_steps=1000,
                    #num_particles=10,
                    optim_kwargs={'lr': .01}):
    """Perform SVI over free model parameters.
    """

    #pyro.clear_param_store()

    svi = pyro.infer.SVI(model=Bayesian_habit_model,
              guide=guide,
              optim=pyro.optim.Adam(step_size=0.01),# from numpyro documentation #optim_kwargs
              loss=pyro.infer.Trace_ELBO(num_particles=1))#num_particles=num_particles
    #                          #set below to true once code is vectorized
    #                          vectorize_particles=True))

    rng_key = random.PRNGKey(100)
    loss = []
    svi_result = svi.run(rng_key, iter_steps)#, stable_update=True)
    params = svi_result.params
    loss = svi_result.losses
    print(svi_result)
    # pbar = tqdm(range(iter_steps), position=0)
    # for step in pbar:
    #     loss.append(jnp.array(svi.step()))#.to(device))
    #     pbar.set_description("Mean ELBO %6.2f" % jnp.array(loss[-20:]).mean())
    #     if jnp.isnan(loss[-1]):
    #         break

    # self.loss = [l.cpu() for l in loss]

    # final_elbo = -pyro.infer.Trace_ELBO(num_particles=1000).loss(rng_key, self.params, self.model, self.guide)
    # print(final_elbo)

    # with pyro.handlers.seed(rng_seed=0):
    #     trace = pyro.handlers.trace(self.model).get_trace()
    # print(pyro.util.format_shapes(trace))

    # alpha_lamb_pi = self.params("alpha_lamb_pi")#.data.numpy()
    # beta_lamb_pi = self.params("beta_lamb_pi")#.data.numpy()
    # alpha_lamb_r = self.params("alpha_lamb_r")#.data.numpy()
    # beta_lamb_r = self.params("beta_lamb_r")#.data.numpy()
    # alpha_h = self.params("alpha_lamb_r")#.data.numpy()
    # beta_h = self.params("beta_lamb_r")#.data.numpy()
    # concentration_dec_temp = self.params("concentration_dec_temp")#.data.numpy()
    # rate_dec_temp = self.params("rate_dec_temp")#.data.numpy()

    param_dict = params #{"alpha_lamb_pi": alpha_lamb_pi, "beta_lamb_pi": beta_lamb_pi,
                  #"alpha_lamb_r": alpha_lamb_r, "beta_lamb_r": beta_lamb_r,
                  #"alpha_h": alpha_h, "beta_h": beta_h,
                  #"concentration_dec_temp": concentration_dec_temp, "rate_dec_temp": rate_dec_temp}
    print(param_dict)

    return loss, param_dict

infer_posterior()

