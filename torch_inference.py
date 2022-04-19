#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 17:21:47 2022

@author: sarah
"""
import torch
from torch import random
import pyro
import pyro.distributions as dist
from tqdm import tqdm

#from jax.lax import scan, cond
import itertools
import jsonpickle as pickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import json
import os
import copy
import matplotlib.pylab as plt
import seaborn as sns


def cond(query, true_func, false_func, *operands):
    if query:
        true_func(operands)
    else:
        false_func(operands)
        
def scan(func, init, xs, reverse=False):
    carry = init
    ys = []
    if reverse:
        ins = reversed(xs)
    else:
        ins = xs
    for x in ins:
        carry, y = func(carry, x)
        ys.append(y)
        
    if type(y) is tuple:
        l = len(y)
        out = []
        for j in range(l):
            results = [y[j] for y in ys]
            if reverse:
                outs = reversed(results)
            else:
                outs = results
            out.append(torch.stack([o for o in outs]))
    elif y is not None:
        if reverse:
            outs = reversed(ys)
        else:
            outs = ys
        out = torch.stack([o for o in outs])
    else: out = None
        
    return carry, out
        

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
A = torch.eye(no)#.to(device)
obs_matrix = A

#state transition generative probability (matrix)
B = torch.zeros((ns, ns, na))
b1 = 0.7
nb1 = 1.-b1
b2 = 0.7
nb2 = 1.-b2

state_trans_matrix = torch.tensor([[[  0,  0,  0,  0,  0,  0,  0,],
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
                     [  0,  0,  1,  0,  0,  0,  1,],]]).permute((1,2,0))


u = 0.999
utility = torch.tensor([1-u, u])

preference = utility

fix_rew_counts = torch.tensor([[100]*3, [1]*3])[:,:,None]

policies = torch.tensor(list(itertools.product(list(range(na)), repeat=T-1)))

def calc_big_trans_matrix(state_trans_matrix, policies):

    npi = policies.shape[0]
    big_trans_matrix = torch.stack([torch.stack([state_trans_matrix[:,:,policies[pi,t]] for pi in range(npi)]) for t in range(T-1)]).permute((2,3,1,0))
    
    return big_trans_matrix

def calc_possible_policies(data, policies):
    
    npi = policies.shape[0]
    all_possible_policies = torch.tensor([[[True]*npi]*T]*trials, dtype=torch.bool)
    
    for curr_observed in data:
        
        c_obs = curr_observed["obs"]
        c_rew = curr_observed["rew"]
        c_response = curr_observed["response"]
        t = curr_observed["t"]
        tau = curr_observed["tau"]
        
        # if t == 0:
        #     all_possible_policies[tau][t] = torch.tensor(all_possible_policies[tau][t], dtype=torch.bool)
        if t>0:# and t < self.T - 1:
            curr_possible_policies = policies[:,t-1]==c_response[t-1]
            curr_possible_policies = torch.logical_and(all_possible_policies[tau][t-1], curr_possible_policies)
            all_possible_policies[tau][t] = curr_possible_policies#torch.stack(curr_possible_policies)
            
        if t == T-1:
            all_possible_policies[tau] = all_possible_policies[tau]
            
    return all_possible_policies


def transform_data(data):

    shaped_data = []
    
    for curr_observed in data:

        c_obs = curr_observed["obs"]
        c_rew = curr_observed["rew"]
        c_response = curr_observed["response"]
        t = curr_observed["t"]
        tau = curr_observed["tau"]

        shaped_data.append(torch.stack(
            [torch.tensor(c_obs), 
             torch.tensor(c_rew), 
             torch.tensor(c_response), 
             torch.tensor([tau]*T), 
             torch.tensor([t]*T)]))
        
    return torch.stack(shaped_data)

def create_flat_response_array(data):
    
    flat_responses = []
    
    for curr_observed in data:

        c_obs = curr_observed["obs"]
        c_rew = curr_observed["rew"]
        c_response = curr_observed["response"]
        t = curr_observed["t"]
        tau = curr_observed["tau"]

        if t < T-1:
            flat_responses.append(c_response[t])
        
    return torch.tensor(flat_responses)

big_trans_matrix = calc_big_trans_matrix(state_trans_matrix, policies)

state_prior = torch.eye(ns)[0]

npart = 1
npi = policies.shape[0]
prior_states = torch.repeat_interleave(torch.repeat_interleave(state_prior[:,None], npi, dim=1)[:,:,None], npart, dim=1) #+ 1e-20
bwd_init = torch.repeat_interleave(torch.repeat_interleave((torch.ones_like(state_prior)/prior_states.shape[0])[:,None], npi, dim=1)[:,:,None], npart, dim=1)
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

flat_responses = create_flat_response_array(data)
print(flat_responses.shape)



def make_rew_messages(rew_matrix, curr_rew):
    
    messages = []
    for i, rew in enumerate(curr_rew):
        if rew != -1:
            messages.append(rew_matrix[rew])
        else:
            messages.append(torch.einsum('r,rsn->sn', preference, rew_matrix))

    return torch.stack(messages).permute((1,0,2))


def make_obs_messages(curr_obs):

    messages = []
    for i, obs in enumerate(curr_obs):
        if obs != -1:
            messages.append(obs_matrix[obs])
        else:
            no = obs_matrix.shape[0]
            messages.append(torch.matmul(torch.ones(no)/no, obs_matrix))

    return torch.stack(messages).permute((1,0))

    
def scan_fwd_messages(carry, input_message):

    i, old_message = carry
    rew_message, obs_message = input_message
    tmp_message = torch.einsum('hpn,shp,hn,hn->spn', old_message, big_trans_matrix[...,i], obs_message, rew_message)

    norm = tmp_message.sum(axis=0)
    message = torch.where(norm > 0, tmp_message/norm[None,...], tmp_message)
    #norms = torch.where(possible_policies[:,None], norm, 0)
    i += 1

    return (i, message), (message, norm)


def make_fwd_messages(rew_messages, obs_messages):

    input_messages = torch.stack([torch.stack([rew_messages[:,i], obs_messages[:,i][:,None]]) for i in range(T-1)])
    init = (0, prior_states)

    carry, fwd = scan(scan_fwd_messages, init, input_messages)
    fwd_messages, fwd_norms = fwd
    fwd_messages = torch.cat([init[1][None,...], fwd_messages], axis=0).permute((1,0,2,3))

    return fwd_messages, fwd_norms

    
def scan_bwd_messages(carry, input_message):

    i, old_message = carry
    rew_message, obs_message = input_message

    #print("old", old_message)
    tmp_message = torch.einsum('hpn,shp,hn,hn->spn', old_message, big_trans_matrix[...,i].permute((1,0,2)), obs_message, rew_message)#
    # print(tmp_message)

    norm = tmp_message.sum(axis=0)
    message = tmp_message#torch.where(norm > 0, tmp_message/norm[None,...], tmp_message)
    i += 1

    return (i, message), (message, norm)


def make_bwd_messages(rew_messages, obs_messages):

    input_messages = torch.stack([torch.stack([rew_messages[:,i+1], obs_messages[:,i+1][:,None]]) for i in range(T-1)])
    init = (0, bwd_init)
    carry = init
    carry, bwd = scan(scan_bwd_messages, init, input_messages, reverse=True)
    bwd_messages, bwd_norms = bwd
    bwd_messages = torch.cat([bwd_messages, init[1][None,...]], axis=0).permute((1,0,2,3))

    return bwd_messages

    
def eval_posterior_policies(fwd_norms, prior_policies, dec_temp):

    likelihood = (fwd_norms).prod(axis=0)#+1e-10
    norm_const = likelihood.sum(axis=0)
    post = torch.pow(likelihood/norm_const, dec_temp[None,:]) * prior_policies
    posterior_policies = post / post.sum(axis=0)

    return posterior_policies


def post_actions_from_policies(posterior_policies, t):
    
    print(posterior_policies.shape)
    post_actions = torch.stack([torch.where((policies[:,t]==a)[...,None], posterior_policies, torch.zeros_like(posterior_policies)).sum(axis=0) for a in range(na)])

    return post_actions


def update_rew_counts(prev_rew_counts, curr_rew, post_states, t, lambda_r):

    #note to self: try implemementing with binary mask multiplication instead of separated matrices
    # maybe using torch.where ?
    no = prev_rew_counts.shape[0]
    #if t == -1:
    #    for i, rew in enumerate(curr_rew):
    #        if rew != -1:
    #            rew_counts = (1-lambda_r)[None,None,...]*prev_rew_counts + lambda_r[None,None,...] \
    #                + torch.eye(no)[rew][:,None,...]*post_states[-prev_rew_counts.shape[1]:,i,...][None,:,...]
    #            prev_rew_counts = rew_counts
    rew = curr_rew[t]
    new_rew_counts = (1-lambda_r)[None,None,...]*prev_rew_counts + lambda_r[None,None,...] \
                + torch.eye(no)[rew][:,None,None]*post_states[-prev_rew_counts.shape[1]:,t,...][None,:,:]

    return new_rew_counts


def update_pol_counts(prev_pol_counts, posterior_policies, alpha, lambda_pi):

    new_pol_counts = (1-lambda_pi)[None,...]*prev_pol_counts + (lambda_pi)[None,...]*alpha + posterior_policies

    return new_pol_counts


def update_counts(rew_counts, curr_rew, marginal_posterior_states, t, pol_counts, 
                  posterior_policies, lambda_r, alpha, lambda_pi):
    new_rew_counts = update_rew_counts(rew_counts, curr_rew, marginal_posterior_states, t, lambda_r)
    new_pol_counts = update_pol_counts(pol_counts, posterior_policies, alpha, lambda_pi)
    
    return new_rew_counts, new_pol_counts


def do_not_update_counts(rew_counts, curr_rew, marginal_posterior_states, t, 
                         pol_counts, posterior_policies, lambda_r, alpha, lambda_pi):
    new_rew_counts = rew_counts
    new_pol_counts = pol_counts
    return new_rew_counts, new_pol_counts


def Bayesian_habit_model():
    
    # generative model of behavior with Normally distributed params (within subject!!)

    # tell pyro about prior over parameters: alpha and beta of lambda which is between 0 and 1
    # alpha = beta = 1 equals uniform prior
    alpha_lambda_pi = torch.ones(1)#.to(device)
    beta_lambda_pi = torch.ones(1)#.to(device)
    # sample initial vaue of parameter from Beta distribution
    lambda_pi = pyro.sample('lambda_pi', dist.Beta(alpha_lambda_pi, beta_lambda_pi))#.to(device)

    # tell pyro about prior over parameters: alpha and beta of lambda which is between 0 and 1
    # alpha = beta = 1 equals uniform prior
    alpha_lambda_r = torch.ones(1)#.to(device)
    beta_lambda_r = torch.ones(1)#.to(device)
    # sample initial vaue of parameter from Beta distribution
    lambda_r = pyro.sample('lambda_r', dist.Beta(alpha_lambda_r, beta_lambda_r))#.to(device)

    # tell pyro about prior over parameters: alpha and beta of h which is between 0 and 1
    # alpha = beta = 1 equals uniform prior
    alpha_h = torch.ones(1)#.to(device)
    beta_h = torch.ones(1)#.to(device)
    # sample initial vaue of parameter from Beta distribution
    h = pyro.sample('h', dist.Beta(alpha_h, beta_h))#.to(device)
    alpha = 1./h
    
    init_pol_counts = torch.zeros((npi, 1)) + alpha
    init_rew_counts = torch.ones((nr, ns-3, 1))

    # tell pyro about prior over parameters: decision temperature
    # uniform between 0 and 20??
    concentration_dec_temp = torch.tensor(1.)#.to(device)
    rate_dec_temp = torch.tensor(0.5)#.to(device)
    # sample initial vaue of parameter from normal distribution
    dec_temp = pyro.sample('dec_temp', dist.Gamma(concentration_dec_temp, rate_dec_temp))#.to(device)
    
    
    
    def step(carry, curr_data):
        
        rew_counts, pol_counts = carry
        
        # print(curr_data)
        
        # curr_obs = curr_data[0]
        # curr_rew = curr_data[1]
        # curr_responses = curr_data[2]
        # curr_tau = curr_data[3]
        # curr_t = curr_data[4]
        # curr_action = curr_data[5][0]
        
        curr_obs, curr_rew, curr_response, curr_tau, curr_t = curr_data
        
        #curr_obs = curr_observed["obs"]
        #curr_rew = curr_observed["rew"]
        t = curr_t[0]
        tau = curr_tau[0]
        
        possible_policies = all_possible_policies[tau,t]
        
        total_counts = torch.cat([fix_rew_counts, rew_counts], dim=1)
        rew_matrix = total_counts / total_counts.sum(axis=0)
        prior_policies = pol_counts / pol_counts.sum(axis=0)

        
        
        def sample_action(posterior_policies, curr_action):
            
            posterior_actions = post_actions_from_policies(posterior_policies, t)
            print('res_{}_{}'.format(tau, t), posterior_actions)
            res = pyro.sample('res_{}_{}'.format(tau, t), dist.Categorical(probs=posterior_actions), obs=curr_action)
            print(res)
        
        def dont_sample(posterior_actions, curr_action):
            pass
        
        rew_messages = make_rew_messages(rew_matrix, curr_rew)
        obs_messages = make_obs_messages(curr_obs)
        fwd_messages, fwd_norms = make_fwd_messages(rew_messages, obs_messages)
        bwd_messages = make_bwd_messages(rew_messages, obs_messages)
        
        post_states = fwd_messages*bwd_messages*obs_messages[...,None,None]*rew_messages[...,None]
        norm = post_states.sum(axis=0)
        
        posterior_states = torch.where(norm[None,...] > 0, post_states/norm[None,...],  post_states)
        fwd_norms = torch.cat([fwd_norms, norm[-1][None,:]], axis=0)
        fwd_norms = torch.where(possible_policies[:,None], fwd_norms, torch.zeros_like(fwd_norms))
        
        posterior_policies = eval_posterior_policies(fwd_norms, prior_policies, dec_temp)
        
        marginal_posterior_states = torch.einsum('stpn,pn->stn', posterior_states, posterior_policies)
        
        # new_rew_counts, new_pol_counts = cond(t==T-1, update_counts, do_not_update_counts, 
        #      rew_counts, curr_rew, marginal_posterior_states, t, pol_counts, 
        #      posterior_policies, lambda_r, alpha, lambda_pi)
        if t==T-1:
            new_rew_counts = update_rew_counts(rew_counts, curr_rew, marginal_posterior_states, t, lambda_r)
            new_pol_counts = update_pol_counts(pol_counts, posterior_policies, alpha, lambda_pi)
        else:
            new_rew_counts = rew_counts
            new_pol_counts = pol_counts
            
            sample_action(posterior_policies, curr_response[t])
        
        #cond(t<T-2, sample_action, dont_sample, posterior_actions, curr_action)
        
        return (new_rew_counts, new_pol_counts), None
    
    
    carry = (init_rew_counts, init_pol_counts)
    
    _, _ = scan(step, carry, nicely_shaped_data)
    
    #posterior_responses = torch.stack([pa for k, pa in enumerate(posterior_actions) if k%T != T-1])
    
    #print(posterior_responses.shape)
    
    # with pyro.plate("N", flat_responses.shape[0]) as ind:
    #     pyro.sample('responses', dist.Categorical(posterior_responses[ind,:,0]), obs=flat_responses[ind])#, sample_shape=(trials*(T-1), npart))
    
    
def guide():
    # approximate posterior. assume MF: each param has his own univariate Normal.

    # tell pyro about posterior over parameters: alpha and beta of lambda which is between 0 and 1
    alpha_lambda_pi = pyro.param("alpha_lambda_pi", torch.ones(1), constraint=pyro.distributions.constraints.positive)#.to(device)#greater_than_eq(1.))
    beta_lambda_pi = pyro.param("beta_lambda_pi", torch.ones(1), constraint=pyro.distributions.constraints.positive)#.to(device)#greater_than_eq(1.))
    # sample vaue of parameter from Beta distribution
    # print()
    # print(alpha_lamb_pi, beta_lamb_pi)
    pyro.sample('lambda_pi', dist.Beta(alpha_lambda_pi, beta_lambda_pi))#.to(device)

    # tell pyro about posterior over parameters: alpha and beta of lambda which is between 0 and 1
    alpha_lambda_r = pyro.param("alpha_lambda_r", torch.ones(1), constraint=pyro.distributions.constraints.positive)#.to(device)#greater_than_eq(1.))
    beta_lambda_r = pyro.param("beta_lambda_r", torch.ones(1), constraint=pyro.distributions.constraints.positive)#.to(device)#greater_than_eq(1.))
    # sample initial vaue of parameter from Beta distribution
    pyro.sample('lambda_r', dist.Beta(alpha_lambda_r, beta_lambda_r))#.to(device)

    # tell pyro about posterior over parameters: alpha and beta of lambda which is between 0 and 1
    alpha_h = pyro.param("alpha_h", torch.ones(1), constraint=pyro.distributions.constraints.positive)#.to(device)#greater_than_eq(1.))
    beta_h = pyro.param("beta_h", torch.ones(1), constraint=pyro.distributions.constraints.positive)#.to(device)#greater_than_eq(1.))
    # sample initial vaue of parameter from Beta distribution
    pyro.sample('h', dist.Beta(alpha_h, beta_h))#.to(device)

    # tell pyro about posterior over parameters: mean and std of the decision temperature
    concentration_dec_temp = pyro.param("concentration_dec_temp", torch.ones(1)*3., constraint=pyro.distributions.constraints.positive)#.to(device)#interval(0., 7.))
    rate_dec_temp = pyro.param("rate_dec_temp", torch.ones(1), constraint=pyro.distributions.constraints.positive)#.to(device)
    # sample initial vaue of parameter from normal distribution
    pyro.sample('dec_temp', dist.Gamma(concentration_dec_temp, rate_dec_temp))#.to(device)
    
def infer_posterior(iter_steps=1000,
                    #num_particles=10,
                    optim_kwargs={'lr': .01}):
    """Perform SVI over free model parameters.
    """

    #pyro.clear_param_store()

    svi = pyro.infer.SVI(model=Bayesian_habit_model,
              guide=guide,
              optim=pyro.optim.Adam(optim_kwargs),# from numpyro documentation #optim_kwargs
              loss=pyro.infer.Trace_ELBO(num_particles=1))#num_particles=num_particles
    #                          #set below to true once code is vectorized
    #                          vectorize_particles=True))

    #rng_key = random.PRNGKey(100)
    loss = []
    # svi_result = svi.run(rng_key, iter_steps, stable_update=False)
    # params = svi_result.params
    # loss = svi_result.losses
    # print(svi_result)
    pbar = tqdm(range(iter_steps), position=0)
    for step in pbar:
        loss.append(torch.tensor(svi.step()))#.to(device))
        pbar.set_description("Mean ELBO %6.2f" % torch.tensor(loss[-20:]).mean())
        if torch.isnan(loss[-1]):
            break

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
