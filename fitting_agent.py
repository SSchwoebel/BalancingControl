#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 16:10:45 2022

@author: sarah
"""

import jax.numpy as jnp
from jax.lax import scan
from jax import random
import itertools
import jsonpickle as pickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import json
import os
import copy


class Perception(object):
    def __init__(self, big_trans_matrix, obs_matrix, opt_params, fix_rew_counts, preference, prior_states, policies, na, possible_policies, T):
        
        # has shape ns x ns x npi x T-1
        self.big_trans_matrix = big_trans_matrix
        self.obs_matrix = obs_matrix
        self.fix_rew_counts = fix_rew_counts
        self.preference = preference
        self.policies = policies
        self.na = na
        self.all_possible_policies = possible_policies
        self.T = T
        
        self.lambda_r = opt_params["lambda_r"]
        self.lambda_pi = opt_params["lambda_pi"]
        self.alpha = 1./opt_params["h"]
        self.dec_temp = opt_params["dec_temp"]
        
        npart = self.lambda_r.shape[0]
        npi = policies.shape[0]
        self.prior_states = jnp.repeat(jnp.repeat(prior_states[:,None], npi, axis=1)[:,:,None], npart, axis=1)
        self.bwd_init = jnp.repeat(jnp.repeat((jnp.ones_like(prior_states)/prior_states.shape[0])[:,None], npi, axis=1)[:,:,None], npart, axis=1)
    
    def step(self, carry, curr_observed):
        
        rew_counts, pol_counts = carry
        
        curr_obs = curr_observed["obs"]
        curr_rew = curr_observed["rew"]
        t = curr_observed["t"]
        tau = curr_observed["tau"]
        
        self.possible_policies = jnp.stack(self.all_possible_policies[tau][t])
        
        total_counts = jnp.concatenate([self.fix_rew_counts, rew_counts], axis=1)
        rew_matrix = total_counts / total_counts.sum(axis=0)
        prior_policies = pol_counts / pol_counts.sum(axis=0)
        
        rew_messages = self.make_rew_messages(rew_matrix, curr_rew)
        obs_messages = self.make_obs_messages(curr_obs)
        fwd_messages, fwd_norms = self.make_fwd_messages(rew_messages, obs_messages)
        bwd_messages = self.make_bwd_messages(rew_messages, obs_messages)
        
        posterior_states = fwd_messages*bwd_messages*obs_messages[...,None,None]*rew_messages[...,None]
        norm = posterior_states.sum(axis=0)
        fwd_norms = jnp.concatenate([fwd_norms, norm[-1][None,:]], axis=0)
        
        posterior_policies = self.eval_posterior_policies(fwd_norms, prior_policies)
        
        marginal_posterior_states = jnp.einsum('stpn,pn->stn', posterior_states, posterior_policies)
        
        posterior_actions = self.post_actions_from_policies(posterior_policies, t)
        
        if t==self.T-1:
            rew_counts = self.update_rew_counts(rew_counts, curr_rew, marginal_posterior_states, t=t)
            pol_counts = self.update_pol_counts(pol_counts, posterior_policies)
        
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
        carry = init
        fwd_messages = []
        fwd_norms = []
        for input_message in input_messages:
            carry, output = self.scan_messages(carry, input_message)
            fwd_messages.append(output[0])
            fwd_norms.append(output[1])
        #fwd = scan(self.scan_messages, init, input_messages)
        fwd_messages = jnp.stack([init]+fwd_messages).transpose((1,0,2,3))
        fwd_norms = jnp.stack(fwd_norms)
        
        # fwd_messages = fwd[:,0,...].transpose((1,0,2,3))
        # fwd_norms = fwd[:,1,...]
        
        return fwd_messages, fwd_norms
    
    def make_bwd_messages(self, rew_messages, obs_messages):
        
        input_messages = [(rew_messages[:,i+1], obs_messages[:,i+1], self.big_trans_matrix[...,i].transpose((1,0,2))) for i in range(self.T-1)]
        init = self.bwd_init
        carry = init
        bwd_messages = []
        bwd_norms = []
        for input_message in input_messages:
            carry, output = self.scan_messages(carry, input_message)
            bwd_messages.append(output[0])
        #fwd = scan(self.scan_messages, init, input_messages)
        bwd_messages = jnp.flip(jnp.stack([init]+bwd_messages), axis=0).transpose((1,0,2,3))
        # bwd = scan(self.scan_messages, init, input_messages, reverse=True)
        
        # bwd_messages = jnp.flip(bwd[:,0,...], axis=0).transpose((1,0,2,3))
        
        return bwd_messages
        
    def scan_messages(self, carry, input_message):
        
        old_message = carry
        #print(input_message)
        obs_message, rew_message, trans_matrix = input_message
        
        #print(old_message)
        tmp_message = jnp.einsum('spn,shp,sn,sn->hpn', old_message, trans_matrix, obs_message, obs_message)
        
        norm = tmp_message.sum(axis=0)
        message = jnp.where(norm > 0, tmp_message/norm, tmp_message)
        norm = jnp.where(self.possible_policies[:,None], norm, 0)
        
        return message, (message, norm)
        
    def eval_posterior_policies(self, fwd_norms, prior_policies):
        
        likelihood = (fwd_norms+1e-10).prod(axis=0)
        norm = likelihood.sum(axis=0)
        post = jnp.power(likelihood/norm,self.dec_temp[None,:]) * prior_policies
        posterior_policies = post / post.sum(axis=0)
        
        return posterior_policies
    
    def post_actions_from_policies(self, posterior_policies, t):
            
        post_actions = jnp.stack([posterior_policies[self.policies[:,t]==a].sum() for a in range(self.na)])
        
        return post_actions
            
    def contract_posterior_policies(self, posterior_policies, a, t):
        
        return posterior_policies[self.policies[t]==a].sum()
        
    def update_rew_counts(self, prev_rew_counts, curr_rew, post_states, t=None):
        
        #note to self: try implemementing with binary mask multiplication instead of separated matrices
        # maybe using jnp.where ?
        ns = prev_rew_counts.shape[1]
        if t is None:
            for i, rew in enumerate(curr_rew):
                if rew is not None:
                    rew_counts = (1-self.lambda_r)[None,None,...]*prev_rew_counts + self.lambda_r[None,None,...] \
                        + jnp.eye(ns)[rew][:,None,...]*post_states[-prev_rew_counts.shape[1]:,i,...][None,:,...]
                    prev_rew_counts = rew_counts
        else:
            rew = curr_rew[t]
            rew_counts = (1-self.lambda_r)[None,None,...]*prev_rew_counts + self.lambda_r[None,None,...] \
                        + jnp.eye(ns)[rew][:,None,...]*post_states[-prev_rew_counts.shape[1]:,t,...][None,:,...]
            
        return rew_counts
    
    def update_pol_counts(self, prev_pol_counts, posterior_policies):
        
        pol_counts = (1-self.lambda_pi)[None,...]*prev_pol_counts + (self.lambda_pi*self.alpha)[None,...] + posterior_policies
        
        return pol_counts

    
class Agent(object):
    def __init__(self, state_trans_matrix, obs_matrix, opt_params, fix_rew_counts, preference, prior_states, policies, na, T, trials, rng_key):
        
        self.T = T
        self.npi = policies.shape[0]
        self.policies = policies
        big_trans_matrix = self.calc_big_trans_matrix(state_trans_matrix, policies)
        possible_policies = [[[True]*self.npi]*T]*trials
        self.perception = Perception(big_trans_matrix, obs_matrix, opt_params, fix_rew_counts, preference, prior_states, policies, na, possible_policies, T)
        self.state_trans_matrix = state_trans_matrix
        self.obs_matrix = obs_matrix
        self.opt_params = opt_params
        self.na = na
        self.rng_key = rng_key
        updated_states = prior_states.shape[0] - fix_rew_counts.shape[1]
        nr = fix_rew_counts.shape[0]
        init_rew_counts = jnp.ones((nr, updated_states, 1))
        alpha = 1./opt_params["h"]
        init_pol_counts = jnp.zeros((self.npi, 1)) + alpha
        self.carry = (init_rew_counts, init_pol_counts)
        
    def calc_big_trans_matrix(self, state_trans_matrix, policies):
        
        npi = policies.shape[0]
        return jnp.stack([jnp.stack([state_trans_matrix[:,:,policies[pi,t]] for pi in range(self.npi)]) for t in range(self.T-1)]).T
    
    # def init_counts(self):
    #     pass
    
    def step(self, curr_observed):
        
        curr_obs = curr_observed["obs"]
        curr_rew = curr_observed["rew"]
        response = curr_observed["response"]
        t = curr_observed["t"]
        tau = curr_observed["tau"]
        
        if t == 0:
            self.perception.all_possible_policies[tau][t] = jnp.stack(self.perception.all_possible_policies[tau][t])
        if t>0 and t < self.T - 1:
            possible_policies = self.policies[:,t-1]==response[t-1]
            possible_policies = jnp.logical_and(self.perception.all_possible_policies[tau][t-1], possible_policies)
            self.perception.all_possible_policies[tau][t] = jnp.stack(possible_policies)
            
        
        new_carry, posterior_actions = self.perception.step(self.carry, curr_observed)
        self.carry = new_carry
        
        if t < self.T-1:
            response = self.generate_response(posterior_actions)
        else:
            response = None
        
        return response
        
    def generate_response(self, posterior_actions):
        
        a = random.choice(self.rng_key, jnp.arange(self.na), p=posterior_actions)
        
        return a
    
class Environment(object):
    def __init__(self, state_trans_matrix, reward_matrix, observation_matrix, rng_key, initial_state=0):
        self.curr_state = initial_state
        self.initial_state = initial_state
        self.rng_key = rng_key
        self.state_trans_matrix = state_trans_matrix
        self.ns = state_trans_matrix.shape[0]
        self.reward_matrix = reward_matrix
        self.nr = reward_matrix.shape[1]
        self.observation_matrix = observation_matrix
        self.no = observation_matrix.shape[0]
    
    def init_trial(self, tau):
        self.curr_state = self.initial_state
        rew = self.generate_reward(tau)
        obs = self.generate_observation()
        
        return rew, obs
    
    def generate_reward(self, tau):
        
        rew = random.choice(self.rng_key, jnp.arange(self.nr), p=self.reward_matrix[tau,:,self.curr_state])
        
        return rew
    
    def generate_observation(self):
        
        obs = random.choice(self.rng_key, jnp.arange(self.no), p=self.observation_matrix[:,self.curr_state])
        
        return obs
    
    def generate_state_transition(self, response):
        
        state = random.choice(self.rng_key, jnp.arange(self.ns), p=self.state_trans_matrix[:,self.curr_state,response])
        
        return state
    
    def step(self, response, tau):
        self.curr_state = self.generate_state_transition(response)
        rew = self.generate_reward(tau)
        obs = self.generate_observation()
        print(tau)
        print(self.curr_state, rew, obs, response)
        
        return obs, rew
    
    
class World(object):
    def __init__(self, state_trans_matrix, reward_matrix, observation_matrix, opt_params, fix_rew_counts, preference, prior_states, policies, na, T, trials, rng_key):
        
        self.environment = Environment(state_trans_matrix, reward_matrix, observation_matrix, rng_key)
        self.agent = Agent(state_trans_matrix, observation_matrix, opt_params, fix_rew_counts, preference, prior_states, policies, na, T, trials, rng_key)
        self.T = T
        
        self.observations = []
        self.rewards = []
        self.responses = []
        self.trials = []
        self.time_steps = []
        self.results = []
        self.curr_observed = {"obs": [None]*T, "rew": [None]*T, "tau": None, "t": None, "respone": [None]*(T-1)}
        
    def step(self, tau, t):
        
        #print(tau, t)
        if t==0:
            self.curr_observed = {"obs": [None]*self.T, "rew": [None]*self.T, "tau": None, "t": None, "response": [None]*(self.T-1)}
            obs, rew = self.environment.init_trial(tau)
        else:
            obs, rew = self.environment.step(self.curr_observed["response"][t-1], tau)  
            
        #print(self.curr_observed)
        
        self.curr_observed["tau"] = tau
        self.curr_observed["t"] = t
        self.curr_observed["obs"][t] = int(obs)
        self.curr_observed["rew"][t] = int(rew)      
        
        response = self.agent.step(self.curr_observed)
        
        self.observations.append(int(obs))
        self.rewards.append(int(rew))
        if response is not None:
            self.responses.append(int(response))
        else:
            self.responses.append(None)
        self.trials.append(tau)
        self.time_steps.append(t)
        
        if t < self.T-1:
            self.curr_observed["response"][t] = int(response)
        
        self.results.append(self.curr_observed.copy())
        
        if t==self.T-1:
            self.curr_observed = {}
            
        if tau==200:
            self.test = self.results.copy()
            print("test")
            print(self.test[-1])
        
    def run(self, trials):
        
        for tau in range(trials):
            for t in range(self.T):
                self.step(tau, t)
                if tau == trials-1 and t==0:
                    print("inside loop")
                    print(self.results[-1])
                    
        print("after loop")
        print(self.results[-3])
                
        return self.results.copy()
                
                
class SimulateTwoStageTask(object):
    def __init__(self):
        pass
    
    def read_twostage_contingencies(self, fname = 'data/twostep_rho.json', trials=-1):
        
        jsonpickle_numpy.register_handlers()
        with open(fname, 'r') as infile:
            data = json.load(infile)

        Rho = pickle.decode(data)[:trials]
        
        import matplotlib.pylab as plt
        
        # plt.figure()
        # plt.plot(Rho[:,1,3:])
        # plt.show()
        # print(Rho)
        # print(Rho.shape)
        
        return jnp.array(Rho)
    
    def run(self, lambda_r, lambda_pi, dec_temp, h=1000):
        
        self.opt_params = {"lambda_r": jnp.array([lambda_r]), "lambda_pi": jnp.array([lambda_pi]), "dec_temp": jnp.array([dec_temp]), "h": jnp.array([h])}
                
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
        observation_matrix = A

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
        utility = jnp.array([u, 1-u])
            
        preference = utility
        
        fix_rew_counts = jnp.array([[100]*3, [1]*3])[:,:,None]
        
        policies = jnp.array(list(itertools.product(list(range(na)), repeat=T-1)))
        
        prior_states = jnp.zeros(ns)
        prior_states.at[0].set(1)
        
        reward_matrix = self.read_twostage_contingencies(trials=trials)
        
        rng_key = random.PRNGKey(1)
        
        w = World(state_trans_matrix, reward_matrix, observation_matrix,
                  self.opt_params, fix_rew_counts, preference, prior_states,
                  policies, na, T, trials, rng_key)
        
        results =  w.run(trials)
        print("outside of run")
        print(results[-3])
        
        i=0
        folder = 'data'
        run_name = "twostage_results"+str(i)+"_pl"+str(lambda_pi)+"_rl"+str(lambda_r)+"_dt"+str(dec_temp)+"_tend"+str(int(1./h))+".json"
        fname = os.path.join(folder, run_name)
        data = w.results

        pickled = pickle.encode(data)
        with open(fname, 'w') as outfile:
            json.dump(pickled, outfile)
            
        return w, results
        
        
if __name__ == '__main__':
    
    simulation = SimulateTwoStageTask()
    w, results = simulation.run(0.3, 0.7, 5., 0.01)
    
    first_actions = jnp.array([r['response'] for r in w.results if r['t']==2])
    
    #rewarded = 