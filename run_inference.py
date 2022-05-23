#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 14:09:11 2021

@author: sarah
"""


import torch as ar
array = ar.tensor

ar.set_num_threads(6)
print("torch threads", ar.get_num_threads())

import pyro
import pyro.distributions as dist
import agent as agt
import perception as prc
import action_selection as asl
import inference_twostage as inf

import itertools
import matplotlib.pylab as plt
from matplotlib.animation import FuncAnimation
from multiprocessing import Pool
from matplotlib.colors import LinearSegmentedColormap
import jsonpickle as pickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import json
import seaborn as sns
import pandas as pd
import os
import scipy as sc
import scipy.signal as ss
import bottleneck as bn
import gc
import sys

#device = ar.device("cuda") if ar.cuda.is_available() else ar.device("cpu")
#device = ar.device("cuda")
#device = ar.device("cpu")

from inference_twostage import device

#ar.autograd.set_detect_anomaly(True)
###################################
"""load data"""


folder = "data"

true_vals = []

data = []

# pl = 0.7
# rl = 0.3
# dt = 5.
# tend = 1

# # 1, 2
# for i in [1, 2]:#, 1, 2, 3, 4
#     run_name = "twostage_agent"+str(i)+"_pl"+str(pl)+"_rl"+str(rl)+"_dt"+str(dt)+"_tend"+str(tend)+".json"
#     fname = os.path.join(folder, run_name)
    
#     jsonpickle_numpy.register_handlers()
        
#     with open(fname, 'r') as infile:
#         loaded = json.load(infile)
    
#     data_load = pickle.decode(loaded)

#     data.append({})
#     data[-1]["actions"] = ar.tensor(data_load["actions"]).to(device)
#     data[-1]["rewards"] = ar.tensor(data_load["rewards"]).to(device)
#     data[-1]["observations"] = ar.tensor(data_load["observations"]).to(device)
#     data[-1]["states"] = ar.tensor(data_load["states"]).to(device)
    
#     rewarded = data[-1]["rewards"][:-1,-1] == 1

#     unrewarded = rewarded==False
    
#     rare = ar.logical_or(ar.logical_and(data[-1]["states"][:-1,1]==2, data[-1]["actions"][:-1,0] == 0),
#                     ar.logical_and(data[-1]["states"][:-1,1]==1, data[-1]["actions"][:-1,0] == 1))

#     common = rare==False

#     names = ["rewarded common", "rewarded rare", "unrewarded common", "unrewarded rare"]
    
#     rewarded_common = ar.where(ar.logical_and(rewarded,common) == True)[0]
#     rewarded_rare = ar.where(ar.logical_and(rewarded,rare) == True)[0]
#     unrewarded_common = ar.where(ar.logical_and(unrewarded,common) == True)[0]
#     unrewarded_rare = ar.where(ar.logical_and(unrewarded,rare) == True)[0]
    
#     index_list = [rewarded_common, rewarded_rare,
#                   unrewarded_common, unrewarded_rare]

#     stayed = [(data[-1]["actions"][index_list[i],0] == data[-1]["actions"][index_list[i]+1,0]).sum()/float(len(index_list[i])) for i in range(4)]
    
#     plt.figure()
#     g = sns.barplot(data=stayed)
#     g.set_xticklabels(names, rotation=45, horizontalalignment='right', fontsize=16)
#     plt.ylim([0,1])
#     plt.yticks(ar.arange(0,1.1,0.2),fontsize=16)
#     plt.title("habit and goal-directed", fontsize=18)
#     plt.savefig("habit_and_goal.svg",dpi=300)
#     plt.ylabel("stay probability")
#     plt.show()
    
#     true_vals.append({"lamb_pi": pl, "lamb_r": rl, "dec_temp": dt, "h": 1./tend})
    

for i in range(1):    
    for pl in [0.1,0.3,0.5,0.7,0.9]:
        for rl in [0.1,0.3,0.5,0.7,0.9]:
            # TODO: wht does dt=9 not work?? gives control prob of nan
            for dt in [2.,5.]:
                for tend in [1]:#, 2, 10]:
                    
                    run_name = "twostage_agent"+str(i)+"_pl"+str(pl)+"_rl"+str(rl)+"_dt"+str(dt)+"_tend"+str(tend)+".json"
                    fname = os.path.join(folder, run_name)
                    
                    jsonpickle_numpy.register_handlers()
                        
                    with open(fname, 'r') as infile:
                        loaded = json.load(infile)
                    
                    data_load = pickle.decode(loaded)
    
                    data.append({})
                    data[-1]["actions"] = ar.tensor(data_load["actions"]).to(device)
                    data[-1]["rewards"] = ar.tensor(data_load["rewards"]).to(device)
                    data[-1]["observations"] = ar.tensor(data_load["observations"]).to(device)
                    data[-1]["states"] = ar.tensor(data_load["states"]).to(device)
                    
                    # rewarded = data[-1]["rewards"][:-1,-1] == 1
    
                    # unrewarded = rewarded==False
                    
                    # rare = ar.logical_or(ar.logical_and(data[-1]["states"][:-1,1]==2, data[-1]["actions"][:-1,0] == 0),
                    #                 ar.logical_and(data[-1]["states"][:-1,1]==1, data[-1]["actions"][:-1,0] == 1))
    
                    # common = rare==False
    
                    # names = ["rewarded common", "rewarded rare", "unrewarded common", "unrewarded rare"]
                    
                    # rewarded_common = ar.where(ar.logical_and(rewarded,common) == True)[0]
                    # rewarded_rare = ar.where(ar.logical_and(rewarded,rare) == True)[0]
                    # unrewarded_common = ar.where(ar.logical_and(unrewarded,common) == True)[0]
                    # unrewarded_rare = ar.where(ar.logical_and(unrewarded,rare) == True)[0]
                    
                    # index_list = [rewarded_common, rewarded_rare,
                    #               unrewarded_common, unrewarded_rare]
    
                    # stayed = [(data[-1]["actions"][index_list[i],0] == data[-1]["actions"][index_list[i]+1,0]).sum()/float(len(index_list[i])) for i in range(4)]
                    
                    # plt.figure()
                    # g = sns.barplot(data=stayed)
                    # g.set_xticklabels(names, rotation=45, horizontalalignment='right', fontsize=16)
                    # plt.ylim([0,1])
                    # plt.yticks(ar.arange(0,1.1,0.2),fontsize=16)
                    # plt.title("habit and goal-directed", fontsize=18)
                    # plt.savefig("habit_and_goal.svg",dpi=300)
                    # plt.ylabel("stay probability")
                    # plt.show()
                    
                    true_vals.append({"lamb_pi": pl, "lamb_r": rl, "dec_temp": dt,})# "h": 1./tend})
                    
print('analyzing '+str(len(true_vals))+' data sets')


###################################
"""experiment parameters"""

trials =  201#number of trials
T = 3 #number of time steps in each trial
nb = 4
ns = 3+nb #number of states
no = ns #number of observations
na = 2 #number of actions
npi = na**(T-1)
nr = 2

learn_pol=tend
learn_habit=True

learn_rew = 1

utility = []

#ut = [0.5, 0.6, 0.7, 0.8, 0.9, 1. - 1e-5]
#ut = [0.95, 0.96, 0.98, 0.99]
#ut = [0.985]
ut = [0.999]
for u in ut:
    utility.append(ar.zeros(nr).to(device))
    for i in range(1,nr):
        utility[-1][i] = u/(nr-1)#u/nr*i
    utility[-1][0] = (1.-u)
    
utility = utility[-1]

"""
create matrices
"""


#generating probability of observations in each state
A = ar.eye(no).to(device)


#state transition generative probability (matrix)
B = ar.zeros((ns, ns, na)).to(device)
b1 = 0.7
nb1 = 1.-b1
b2 = 0.7
nb2 = 1.-b2

B[:,:,0] = array([[  0,  0,  0,  0,  0,  0,  0,],
                     [ b1,  0,  0,  0,  0,  0,  0,],
                     [nb1,  0,  0,  0,  0,  0,  0,],
                     [  0,  1,  0,  1,  0,  0,  0,],
                     [  0,  0,  0,  0,  1,  0,  0,],
                     [  0,  0,  1,  0,  0,  1,  0,],
                     [  0,  0,  0,  0,  0,  0,  1,],])

B[:,:,1] = array([[  0,  0,  0,  0,  0,  0,  0,],
                     [nb2,  0,  0,  0,  0,  0,  0,],
                     [ b2,  0,  0,  0,  0,  0,  0,],
                     [  0,  0,  0,  1,  0,  0,  0,],
                     [  0,  1,  0,  0,  1,  0,  0,],
                     [  0,  0,  0,  0,  0,  1,  0,],
                     [  0,  0,  1,  0,  0,  0,  1,],])

# create reward generation
#
#    C = ar.zeros((utility.shape[0], ns))
#
#    vals = array([0., 1./5., 0.95, 1./5., 1/5., 1./5.])
#
#    for i in range(ns):
#        C[:,i] = [1-vals[i],vals[i]]
#
#    changes = array([0.01, -0.01])
#    Rho = generate_bandit_timeseries(C, nb, trials, changes)

# agent's beliefs about reward generation

C_alphas = ar.zeros((nr, ns)).to(device) 
C_alphas += learn_rew
C_alphas[0,:3] = 100
for i in range(1,nr):
    C_alphas[i,0] = 1
#    C_alphas[0,1:,:] = 100
#    for c in range(nb):
#        C_alphas[1,c+1,c] = 100
#        C_alphas[0,c+1,c] = 1
#C_alphas[:,13] = [100, 1]

#C_agent = ar.zeros((nr, ns, nc))
# for c in range(nc):
#     C_agent[:,:,c] = array([(C_alphas[:,i,c])/(C_alphas[:,i,c]).sum() for i in range(ns)]).T
C_agent = C_alphas[:,:] / C_alphas[:,:].sum(axis=0)[None,:]
#array([ar.random.dirichlet(C_alphas[:,i]) for i in range(ns)]).T

# context transition matrix

transition_matrix_context = ar.ones(1).to(device)


"""
create policies
"""

pol = array(list(itertools.product(list(range(na)), repeat=T-1))).to(device)

#pol = pol[-2:]
npi = pol.shape[0]

# prior over policies

prior_pi = ar.ones(npi).to(device)
prior_pi /= npi #ar.zeros(npi) + 1e-3/(npi-1)
#prior_pi[170] = 1. - 1e-3
alphas = ar.zeros((npi)).to(device) 
alphas += learn_pol
alpha_0 = array([learn_pol]).to(device)
#    for i in range(nb):
#        alphas[i+1,i] = 100
#alphas[170] = 100
prior_pi = alphas / alphas.sum()


"""
set state prior (where agent thinks it starts)
"""

state_prior = ar.zeros((ns)).to(device)

state_prior[0] = 1.

prior_context = array([1.]).to(device)

#    prior_context[0] = 1.

"""
set up agent
"""
#bethe agent

pol_par = alphas

data_obs = ar.stack([d["observations"] for d in data], dim=-1)
data_rew = ar.stack([d["rewards"] for d in data], dim=-1)
data_act = ar.stack([d["actions"] for d in data], dim=-1)

structured_data = {"observations": data_obs, "rewards": data_rew, "actions": data_act}

# perception
bayes_prc = prc.Group2Perception(A, B, C_agent, transition_matrix_context, 
                                       state_prior, utility, prior_pi, pol,
                                       #data_obs, data_rew, data_act,
                                       alpha_0, C_alphas, T=T, trials=trials,
                                       pol_lambda=0, r_lambda=0,
                                       non_decaying=3, dec_temp=1,)
                                       #nsubs = len(data))

agent = agt.FittingAgent(bayes_prc, [], pol,
                  trials = trials, T = T,
                  prior_states = state_prior,
                  prior_policies = prior_pi,
                  number_of_states = ns,
                  prior_context = prior_context,
                  learn_habit = learn_habit,
                  learn_rew = True,
                  #save_everything = True,
                  number_of_policies = npi,
                  number_of_rewards = nr)


###################################
"""run inference"""

# inferrer = inf.SingleInference(agent, data[0])

inferrer = inf.Group2Inference(agent, structured_data)

num_steps = 500
size_chunk = num_steps

for i in range(num_steps//size_chunk):
    print('taking steps '+str(i*(size_chunk)+1)+' to '+str((i+1)*(size_chunk))+' out of total '+str(num_steps))
    inferrer.infer_posterior(iter_steps=size_chunk, num_particles=15)#, param_dict
    
    total_num_iter_so_far = i*size_chunk
    storage_name = 'recovered_'+str(total_num_iter_so_far)+'.save'#h_recovered
    storage_name = os.path.join(folder, storage_name)
    inferrer.save_parameters(storage_name)
    # inferrer.load_parameters(storage_name)
    
    loss = inferrer.loss
    plt.figure()
    plt.title("ELBO")
    plt.plot(loss)
    plt.ylabel("ELBO")
    plt.xlabel("iteration")
    plt.savefig('recovered_ELBO')
    plt.show()

n_samples=1000
sample_df = inferrer.sample_posterior(n_samples=n_samples) #inferrer.plot_posteriors(n_samples=1000)

inferred_values = []

for i in range(len(data)):
    mean_pl = sample_df[sample_df['subject']==i]['lamb_pi'].mean()
    mean_rl = sample_df[sample_df['subject']==i]['lamb_r'].mean()
    mean_dt = sample_df[sample_df['subject']==i]['dec_temp'].mean()
    # mean_h = sample_df[sample_df['subject']==i]['h'].mean()
    
    inferred_values.append({"lamb_pi": mean_pl, "lamb_r": mean_rl, "dec_temp": mean_dt})#, "h": mean_h})
    
true_pl = [val['lamb_pi'] for val in true_vals]
true_rl = [val['lamb_r'] for val in true_vals]
true_dt = [val['dec_temp'] for val in true_vals]
# true_h = [val['h'] for val in true_vals]

inferred_pl = [val['lamb_pi'] for val in inferred_values]
inferred_rl = [val['lamb_r'] for val in inferred_values]
inferred_dt = [val['dec_temp'] for val in inferred_values]
# inferred_h = [val['h'] for val in inferred_values]

total_df = sample_df.copy()
total_df['true_lamb_pi'] = ar.tensor(true_pl).repeat(n_samples)
total_df['true_lamb_r'] = ar.tensor(true_rl).repeat(n_samples)
total_df['true_dec_temp'] = ar.tensor(true_dt).repeat(n_samples)
# total_df['true_h'] = ar.tensor(true_h).repeat(n_samples)

sample_file = 'recovered_samples.csv'
fname = os.path.join(folder, sample_file)
total_df.to_csv(fname)

plt.figure()
sns.violinplot(data=total_df, x='true_lamb_pi', y='lamb_pi')
plt.show()

plt.figure()
sns.violinplot(data=total_df, x='true_lamb_r', y='lamb_r')
plt.show()

plt.figure()
sns.violinplot(data=total_df, x='true_dec_temp', y='dec_temp')
plt.show()

# plt.figure()
# sns.violinplot(data=total_df, x='true_h', y='h')
# plt.show()

plt.figure()
sns.scatterplot(x=true_pl, y=inferred_pl)
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel("true lamb_pi")
plt.ylabel("inferred lamb_pi")
plt.show()

plt.figure()
sns.scatterplot(x=true_rl, y=inferred_rl)
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel("true lamb_r")
plt.ylabel("inferred lamb_r")
plt.show()

plt.figure()
sns.scatterplot(x=true_dt, y=inferred_dt)
plt.xlim([0,10])
plt.ylim([0,10])
plt.xlabel("true dec_temp")
plt.ylabel("inferred dec_temp")
plt.show()

# plt.figure()
# sns.scatterplot(x=true_h, y=inferred_h)
# plt.xlim([0,1])
# plt.ylim([0,1])
# plt.xlabel("true h")
# plt.ylabel("inferred h")
# plt.show()

#print("this is inference for pl =", pl, "rl =", rl, "dt =", dt, "tend=", tend)
# print(param_dict)