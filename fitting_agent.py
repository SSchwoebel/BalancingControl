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
import matplotlib.pylab as plt
import seaborn as sns


class Perception(object):
    def __init__(self, big_trans_matrix, obs_matrix, opt_params, fix_rew_counts, preference, prior_states, policies, na, possible_policies, T):
        
        # has shape ns x ns x npi x T-1
        self.big_trans_matrix = big_trans_matrix + 1e-20
        self.obs_matrix = obs_matrix + 1e-20
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
        self.prior_states = jnp.repeat(jnp.repeat(prior_states[:,None], npi, axis=1)[:,:,None], npart, axis=1) #+ 1e-20
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
        # print(bwd_messages.shape)
        # print("0", bwd_messages[:,0,0,0])
        # print("1", bwd_messages[:,1,0,0])
        # print("2", bwd_messages[:,2,0,0])
        
        posterior_states = fwd_messages*bwd_messages*obs_messages[...,None,None]*rew_messages[...,None]
        norm = posterior_states.sum(axis=0)
        posterior_states = jnp.where(norm[None,...] > 0, posterior_states/norm[None,...],  posterior_states)
        fwd_norms = jnp.concatenate([fwd_norms, norm[-1][None,:]], axis=0)
        
        posterior_policies = self.eval_posterior_policies(fwd_norms, prior_policies)
        
        marginal_posterior_states = jnp.einsum('stpn,pn->stn', posterior_states, posterior_policies)
        
        posterior_actions = self.post_actions_from_policies(posterior_policies, t)
        print(posterior_actions)
        
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
            # print("carry", carry)
            # print("input messages: rew, obs trans")
            # print(input_message[0])
            # print(input_message[1])
            # print(input_message[2][:,:,0])
            carry, output = self.scan_messages(carry, input_message)
            # print("message")
            # print(output[0][:,0])
            bwd_messages.append(output[0])
        #fwd = scan(self.scan_messages, init, input_messages)
        bwd_messages = jnp.flip(jnp.stack([init]+bwd_messages), axis=0).transpose((1,0,2,3))
        # bwd = scan(self.scan_messages, init, input_messages, reverse=True)
        
        # bwd_messages = jnp.flip(bwd[:,0,...], axis=0).transpose((1,0,2,3))
        
        return bwd_messages
        
    def scan_messages(self, carry, input_message):
        
        old_message = carry
        #print("input", input_message)
        rew_message, obs_message, trans_matrix = input_message
        
        #print("old", old_message)
        tmp_message = jnp.einsum('hpn,shp,h,hn->spn', old_message, trans_matrix, obs_message, rew_message)
        # print(tmp_message)
        
        norm = tmp_message.sum(axis=0)
        message = jnp.where(norm > 0, tmp_message/norm[None,...], tmp_message)
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
        no = prev_rew_counts.shape[0]
        if t is None:
            for i, rew in enumerate(curr_rew):
                if rew is not None:
                    rew_counts = (1-self.lambda_r)[None,None,...]*prev_rew_counts + self.lambda_r[None,None,...] \
                        + jnp.eye(no)[rew][:,None,...]*post_states[-prev_rew_counts.shape[1]:,i,...][None,:,...]
                    prev_rew_counts = rew_counts
        else:
            rew = curr_rew[t]
            rew_counts = (1-self.lambda_r)[None,None,...]*prev_rew_counts + self.lambda_r[None,None,...] \
                        + jnp.eye(no)[rew][:,None,None]*post_states[-prev_rew_counts.shape[1]:,t,...][None,:,:]
            
        return rew_counts
    
    def update_pol_counts(self, prev_pol_counts, posterior_policies):
        
        pol_counts = (1-self.lambda_pi)[None,...]*prev_pol_counts + (self.lambda_pi)[None,...] + posterior_policies#*self.alpha
                
        # print("counts")
        # print(pol_counts)
        
        return pol_counts

    
class Agent(object):
    def __init__(self, state_trans_matrix, obs_matrix, opt_params, fix_rew_counts, preference, prior_states, policies, na, T, trials, rng_seed):
        
        self.T = T
        self.npi = policies.shape[0]
        self.policies = policies
        big_trans_matrix = self.calc_big_trans_matrix(state_trans_matrix, policies)
        print(big_trans_matrix.shape)
        possible_policies = [[[True]*self.npi]*T]*trials
        self.perception = Perception(big_trans_matrix, obs_matrix, opt_params, fix_rew_counts, preference, prior_states, policies, na, possible_policies, T)
        self.state_trans_matrix = state_trans_matrix
        self.obs_matrix = obs_matrix
        self.opt_params = opt_params
        self.na = na
        self.rng_key = random.PRNGKey(rng_seed)
        updated_states = prior_states.shape[0] - fix_rew_counts.shape[1]
        nr = fix_rew_counts.shape[0]
        init_rew_counts = jnp.ones((nr, updated_states, 1))
        alpha = 1./opt_params["h"]
        init_pol_counts = jnp.zeros((self.npi, 1)) + alpha
        self.carry = (init_rew_counts, init_pol_counts)
        
    def calc_big_trans_matrix(self, state_trans_matrix, policies):
        
        npi = policies.shape[0]
        big_trans_matrix = jnp.stack([jnp.stack([state_trans_matrix[:,:,policies[pi,t]] for pi in range(self.npi)]) for t in range(self.T-1)]).transpose((2,3,1,0))
        return big_trans_matrix
    
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
        if t>0:# and t < self.T - 1:
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
        
        key, subkey = random.split(self.rng_key)
        a = random.choice(subkey, jnp.arange(self.na), p=posterior_actions)
        self.rng_key = key
        
        return a
    
class Environment(object):
    def __init__(self, state_trans_matrix, reward_matrix, observation_matrix, rng_seed, initial_state=0):
        self.curr_state = initial_state
        self.initial_state = initial_state
        self.rng_key = random.PRNGKey(rng_seed)
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
        
        return rew, obs, self.curr_state
    
    def generate_reward(self, tau):
        
        key, subkey = random.split(self.rng_key)
        rew = random.choice(subkey, jnp.arange(self.nr), p=self.reward_matrix[tau,:,self.curr_state])
        self.rng_key = key
        
        return rew
    
    def generate_observation(self):
        
        key, subkey = random.split(self.rng_key)
        obs = random.choice(subkey, jnp.arange(self.no), p=self.observation_matrix[:,self.curr_state])
        self.rng_key = key
        
        return obs
    
    def generate_state_transition(self, response):
        
        key, subkey = random.split(self.rng_key)
        state = random.choice(subkey, jnp.arange(self.ns), p=self.state_trans_matrix[:,self.curr_state,response])
        self.rng_key = key
        
        return state
    
    def step(self, response, tau):
        self.curr_state = self.generate_state_transition(response)
        rew = self.generate_reward(tau)
        obs = self.generate_observation()
        
        return obs, rew, self.curr_state
    
    
class World(object):
    def __init__(self, state_trans_matrix, reward_matrix, observation_matrix, opt_params, fix_rew_counts, preference, prior_states, policies, na, T, trials, rng_seed):
        
        self.environment = Environment(state_trans_matrix, reward_matrix, observation_matrix, rng_seed)
        self.agent = Agent(state_trans_matrix, observation_matrix, opt_params, fix_rew_counts, preference, prior_states, policies, na, T, trials, rng_seed)
        self.T = T
        self.trials = 0
        
        self.results = []
        self.curr_observed = {"obs": [None]*T, "rew": [None]*T, "state": [None]*T, "tau": None, "t": None, "response": [None]*(T-1)}
        
    def step(self, tau, t):
        
        print(tau, t)
        if t==0:
            self.curr_observed = {"obs": [None]*self.T, "rew": [None]*self.T, "state": [None]*self.T, "tau": None, "t": None, "response": [None]*(self.T-1)}
            obs, rew, state = self.environment.init_trial(tau)
        else:
            obs, rew, state = self.environment.step(self.curr_observed["response"][t-1], tau)  
            
        #print(self.curr_observed)
        
        self.curr_observed["tau"] = tau
        self.curr_observed["t"] = t
        self.curr_observed["obs"][t] = int(obs)
        self.curr_observed["rew"][t] = int(rew)   
        self.curr_observed["state"][t] = int(state)
        
        response = self.agent.step(self.curr_observed)
        
        if t < self.T-1:
            self.curr_observed["response"][t] = int(response)
        
        self.results.append(copy.deepcopy(self.curr_observed))
        
        if t==self.T-1:
            self.curr_observed = {}
        
    def run(self, trials):
        
        self.trials += trials
        for tau in range(trials):
            for t in range(self.T):
                self.step(tau, t)
                
        return self.results
                
                
class SimulateTwoStageTask(object):
    def __init__(self, rng_seed):
        self.rng_seed = rng_seed
    
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
    
    def run(self, lambda_r, lambda_pi, dec_temp, h=0.001):
        
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
        utility = jnp.array([1-u, u])
            
        preference = utility
        
        fix_rew_counts = jnp.array([[100]*3, [1]*3])[:,:,None]
        
        policies = jnp.array(list(itertools.product(list(range(na)), repeat=T-1)))
        
        prior_states = jnp.eye(ns)[0]
        
        reward_matrix = self.read_twostage_contingencies(trials=trials)
        
        w = World(state_trans_matrix, reward_matrix, observation_matrix,
                  self.opt_params, fix_rew_counts, preference, prior_states,
                  policies, na, T, trials, self.rng_seed)
        
        results =  w.run(trials)
        
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
    
    stayed = []
    
    for s in [3, 27]:#range(10):
        
        simulation = SimulateTwoStageTask(rng_seed=s)
        w, results = simulation.run(0.6, 0.3, 1., 1./1000)
        
        for r in w.results:
            if r['t']==2:
                print(r)
        
        first_actions = jnp.array([r['response'][0] for r in w.results if r['t']==2])
        
        rewarded = jnp.array([r['rew'][-1]==1 for r in w.results if r['t']==2 and r['tau'] < w.trials-1])
        unrewarded = jnp.logical_not(rewarded)
        
        # common = jnp.array([r['state'][1]==(r['response'][0]+1) for i, r in enumerate(w.results) if r['t']==2 and r['tau'] < w.trials-1])
        
        # rare = jnp.logical_not(common)
        
        rare1 = jnp.logical_and(jnp.array([r['state'][1]==2 for r in w.results if r['t']==2 and r['tau'] < w.trials-1]), 
                                first_actions[:w.trials-1] == 0) 
        rare2 = jnp.logical_and(jnp.array([r['state'][1]==1 for r in w.results if r['t']==2 and r['tau'] < w.trials-1]), 
                                first_actions[:w.trials-1] == 1) 
        rare = jnp.logical_or(rare1, rare2)
        
        common = jnp.logical_not(rare)
        
        rewarded_common = jnp.where(jnp.logical_and(rewarded,common) == True)[0]
        rewarded_rare = jnp.where(jnp.logical_and(rewarded,rare) == True)[0]
        unrewarded_common = jnp.where(jnp.logical_and(unrewarded,common) == True)[0]
        unrewarded_rare = jnp.where(jnp.logical_and(unrewarded,rare) == True)[0]
    
        names = ["rewarded common", "rewarded rare", "unrewarded common", "unrewarded rare"]
        
        index_list = [rewarded_common, rewarded_rare,
                      unrewarded_common, unrewarded_rare]
        
        stayed_list = [(first_actions[index_list[i]] == first_actions[index_list[i]+1]).sum()/float(len(index_list[i])) for i in range(4)]
        stayed.append(stayed_list)
        
    stayed_arr = jnp.array(stayed)
    
    plt.figure()
    g = sns.barplot(data=stayed_arr)
    g.set_xticklabels(names, rotation=45, horizontalalignment='right', fontsize=16)
    plt.ylim([0,1])
    plt.yticks(jnp.arange(0,1.1,0.2),fontsize=16)
    plt.title("habit and goal-directed", fontsize=18)
    plt.savefig("habit_and_goal.svg",dpi=300)
    plt.ylabel("stay probability",fontsize=16)
    plt.show()
    
    
    
    
    