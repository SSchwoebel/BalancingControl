#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 14:09:11 2021

@author: sarah
"""


import torch as ar
array = ar.tensor

ar.set_num_threads(1)
print("torch threads", ar.get_num_threads())


import pyro
import pyro.distributions as dist
import world
import environment as env
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
import glob
import scipy as sc
import scipy.signal as ss
from scipy.stats import pearsonr
import gc
import sys
from numpy import eye
from statsmodels.stats.multitest import multipletests
from scipy.io import loadmat
#device = ar.device("cuda") if ar.cuda.is_available() else ar.device("cpu")
#device = ar.device("cuda")
#device = ar.device("cpu")

from inference_twostage import device

#ar.autograd.set_detect_anomaly(True)
###################################
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

"""
run function
"""
def run_agent(par_list, trials=trials, T=T, ns=ns, na=na):

    #set parameters:
    #obs_unc: observation uncertainty condition
    #state_unc: state transition uncertainty condition
    #goal_pol: evaluate only policies that lead to the goal
    #utility: goal prior, preference p(o)
    avg, Rho, lamb, alpha, beta, w, p, utility, use_p, mask, restrict_alpha = par_list

    """
    create matrices
    """


    #generating probability of observations in each state
    A = ar.eye(no)


    #state transition generative probability (matrix)
    B = ar.zeros((ns, ns, na))
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

    # B[:,:,0] = array([[  0,  0,  0,  0,  0,  0,  0,],
    #                      [ b1,  0,  0,  0,  0,  0,  0,],
    #                      [nb1,  0,  0,  0,  0,  0,  0,],
    #                      [  0,  1,  0,  1,  0,  0,  0,],
    #                      [  0,  0,  1,  0,  1,  0,  0,],
    #                      [  0,  0,  0,  0,  0,  1,  0,],
    #                      [  0,  0,  0,  0,  0,  0,  1,],])

    # B[:,:,1] = array([[  0,  0,  0,  0,  0,  0,  0,],
    #                      [nb2,  0,  0,  0,  0,  0,  0,],
    #                      [ b2,  0,  0,  0,  0,  0,  0,],
    #                      [  0,  0,  0,  1,  0,  0,  0,],
    #                      [  0,  0,  0,  0,  1,  0,  0,],
    #                      [  0,  1,  0,  0,  0,  1,  0,],
    #                      [  0,  0,  1,  0,  0,  0,  1,],])



    transition_matrix_context = ar.ones(1)

    """
    create environment (grid world)
    """

    environment = env.MultiArmedBandid(A, B, Rho, trials = trials, T = T)


    """
    create policies
    """

    pol = array(list(itertools.product(list(range(na)), repeat=T-1)))

    #pol = pol[-2:]
    npi = pol.shape[0]


    """
    set state prior (where agent thinks it starts)
    """

    state_prior = ar.zeros((ns))

    state_prior[0] = 1.

    """
    set action selection method
    """

    if avg:

        sel = 'avg'

        ac_sel = asl.AveragedSelector(trials = trials, T = T,
                                      number_of_actions = na)
    else:

        sel = 'max'

        ac_sel = asl.MaxSelector(trials = trials, T = T,
                                      number_of_actions = na)

#    ac_sel = asl.AveragedPolicySelector(trials = trials, T = T,
#                                        number_of_policies = npi,
#                                        number_of_actions = na)

    prior_context = array([1.])

#    prior_context[0] = 1.

    """
    set up agent
    """


    Q_mf_init = [ar.zeros((3,na)), ar.zeros((3,na))]
    Q_mb_init = [ar.zeros((3,na)), ar.zeros((3,na))]

    # perception
    mbmf_prc = prc.mfmbOrigPerception(B, pol, Q_mf_init, Q_mb_init, utility,
                                    lamb, alpha, beta, w,
                                    p, nsubs=1, use_p=use_p, mask=mask[:,None],
                                    restrict_alpha=restrict_alpha)
    mbmf_prc.reset()

    planner = agt.FittingAgent(mbmf_prc, ac_sel, pol,
                      trials = trials, T = T,
                      number_of_states = ns,
                      #save_everything = True,
                      number_of_policies = npi,
                      number_of_rewards = nr)



    """
    create world
    """

    wrld = world.GroupWorld(environment, planner, trials = trials, T = T)

    """
    simulate experiment
    """

    wrld.simulate_experiment(range(trials))


    return wrld


"""create data"""


folder = "data"

true_vals = []

data = []

utility = []

#ut = [0.5, 0.6, 0.7, 0.8, 0.9, 1. - 1e-5]
#ut = [0.95, 0.96, 0.98, 0.99]
#ut = [0.985]
ut = [0.999]
for u in ut:
    utility.append(ar.tensor([-1.,1.,0.]))

utility = utility[-1]

Rho_data_fname = 'dawrandomwalks.mat'

fname = os.path.join(folder, Rho_data_fname)

rew_probs = loadmat(fname)['dawrandomwalks']
assert trials==rew_probs.shape[-1]

never_reward = ns-nb

Rho = ar.zeros((trials, nr, ns))

Rho[:,1,:never_reward] = 0.
Rho[:,0,:never_reward] = 1.

Rho[:,1,never_reward:never_reward+2] = ar.from_numpy(rew_probs[0,:,:]).permute((1,0))
Rho[:,0,never_reward:never_reward+2] = ar.from_numpy(1-rew_probs[0,:,:]).permute((1,0))

Rho[:,1,never_reward+2:] = ar.from_numpy(rew_probs[1,:,:]).permute((1,0))
Rho[:,0,never_reward+2:] = ar.from_numpy(1-rew_probs[1,:,:]).permute((1,0))

plt.figure(figsize=(10,5))
for i in range(4):
    plt.plot(Rho[:,1,3+i], label="$p_{}$".format(i+1), linewidth=4)
plt.ylim([0,1])
plt.yticks(ar.arange(0,1.1,0.2),fontsize=18)
plt.ylabel("reward probability", fontsize=20)
plt.xlim([-0.1, trials+0.1])
plt.xticks(range(0,trials+1,50),fontsize=18)
plt.xlabel("trials", fontsize=20)
plt.legend(fontsize=18, bbox_to_anchor=(1.04,1))
plt.savefig("twostep_prob.svg",dpi=300)
plt.show()

# make param combinations:
    
prefix = 'mbmfOrig_'

use_p = True

if use_p:
    n_pars = 5
else:
    n_pars = 4

restrict_alpha = True

if use_p:
    p_str = "usep_"
else:
    p_str = ""
    
if restrict_alpha:
    restr_str = "resticted_"
else:
    restr_str = ""

# prepare for savin results
# make base filename and folder string
fname_base = prefix+"recovered_"+p_str+restr_str
print(fname_base)
# define folder where we want to save data
base_dir = os.path.join(folder,fname_base[:-1])

remove_old = True

# make directory if it doesnt exist
if fname_base[:-1] not in os.listdir('data'):
    os.mkdir(base_dir)
# if it does exist, empty previous results, if we want that (remove_old==True)
elif remove_old:
    svgs = glob.glob(os.path.join(base_dir,"*.svg"))
    for file in svgs:
        os.remove(file)
    csvs = glob.glob(os.path.join(base_dir,"*.csv"))
    for file in csvs:
        os.remove(file)
    saves = glob.glob(os.path.join(base_dir,"*.save"))
    for file in saves:
        os.remove(file)
    agents = glob.glob(os.path.join(base_dir,"twostage_agent*"))
    for file in agents:
        os.remove(file)

nsubs = 188
true_values_tensor = ar.rand((nsubs,n_pars,1))

# prob for invalid answer (e.g. no reply)
p_invalid = 1.-1./201.


stayed = []
indices = []

for pars in true_values_tensor:

    if use_p:
        discount, norm_lr, norm_dt, weight, perserv = pars
    else:
        discount, norm_lr, norm_dt, weight = pars
        perserv = ar.tensor([0])

    dt = 8*norm_dt
    if restrict_alpha:
        lr = 0.1 + norm_lr*0.9
    else:
        lr = norm_lr

    print(discount, lr, dt, weight, perserv)

    # init = array([0.6, 0.4, 0.6, 0.4])

    # Rho_fname = 'twostep_rho.json'

    # jsonpickle_numpy.register_handlers()

    # fname = os.path.join(folder, Rho_fname)

    # if Rho_fname not in os.listdir(folder) or recalc_rho==True:
    #     Rho[:] = generate_randomwalk(trials, nr, ns, nb, sigma, init)
    #     pickled = pickle.encode(Rho)
    #     with open(fname, 'w') as outfile:
    #         json.dump(pickled, outfile)
    # else:
    #     with open(fname, 'r') as infile:
    #         data = json.load(infile)
    #     if arr_type == "numpy":
    #         Rho[:] = pickle.decode(data)[:trials]
    #     else:
    #         Rho[:] = ar.from_numpy(pickle.decode(data))[:trials]

    worlds = []
    l = []
    avg = True
    lamb = discount#0.3
    alpha = lr#0.6
    beta = dt#4.
    w = weight#4.
    p = perserv
    
    valid_prob_matrix = ar.zeros((trials)) + p_invalid
    valid = ar.bernoulli(valid_prob_matrix).bool()
    
    pars = [avg, Rho, lamb, alpha, beta, w, p, utility, use_p, valid, restrict_alpha]

    worlds.append(run_agent(pars))

    wrld = worlds[-1]

    # rewarded = ar.where(w.rewards[:trials-1,-1] == 1)[0]

    # unrewarded = ar.where(w.rewards[:trials-1,-1] == 0)[0]

    rewarded = wrld.rewards[:trials-1,-1] == 1

    unrewarded = rewarded==False#wrld.rewards[:trials-1,-1] == 0

    # rare = ar.cat((ar.where(own_logical_and(wrld.environment.hidden_states[:,1]==2, wrld.actions[:,0] == 0) == True)[0],
    #                  ar.where(own_logical_and(wrld.environment.hidden_states[:,1]==1, wrld.actions[:,0] == 1) == True)[0]))
    # rare.sort()

    # common = ar.cat((ar.where(own_logical_and(wrld.environment.hidden_states[:,1]==2, wrld.actions[:,0] == 1) == True)[0],
    #                    ar.where(own_logical_and(wrld.environment.hidden_states[:,1]==1, wrld.actions[:,0] == 0) == True)[0]))
    # common.sort()

    rare = ar.logical_or(ar.logical_and(wrld.environment.hidden_states[:trials-1,1]==2, wrld.actions[:trials-1,0] == 0),
                   ar.logical_and(wrld.environment.hidden_states[:trials-1,1]==1, wrld.actions[:trials-1,0] == 1))

    common = rare==False#own_logical_or(own_logical_and(wrld.environment.hidden_states[:trials-1,1]==2, wrld.actions[:trials-1,0] == 1),
             #        own_logical_and(wrld.environment.hidden_states[:trials-1,1]==1, wrld.actions[:trials-1,0] == 0))

    names = ["rewarded common", "rewarded rare", "unrewarded common", "unrewarded rare"]

    # index_list = [ar.intersect1d(rewarded, common), ar.intersect1d(rewarded, rare),
    #              ar.intersect1d(unrewarded, common), ar.intersect1d(unrewarded, rare)]

    rewarded_common = ar.where(ar.logical_and(rewarded,common) == True)[0]
    rewarded_rare = ar.where(ar.logical_and(rewarded,rare) == True)[0]
    unrewarded_common = ar.where(ar.logical_and(unrewarded,common) == True)[0]
    unrewarded_rare = ar.where(ar.logical_and(unrewarded,rare) == True)[0]

    index_list = [rewarded_common, rewarded_rare,
                 unrewarded_common, unrewarded_rare]

    stayed_list = [(wrld.actions[index_list[i],0] == wrld.actions[index_list[i]+1,0]).sum()/float(len(index_list[i])) for i in range(4)]

    stayed.append(stayed_list)

    run_name = "twostage_agent_daw_mbmfOrig"+str(i)+"_disc"+str(discount)+"_lr"+str(lr)+"_dt"+str(dt)+"weight"+str(weight)+"_perserv"+str(perserv)+".json"
    fname = os.path.join(base_dir, run_name)

    # actions = wrld.actions.numpy()
    # observations = wrld.observations.numpy()
    # rewards = wrld.rewards.numpy()
    # states = wrld.environment.hidden_states.numpy()
    data.append({"actions": wrld.actions, "observations": wrld.observations, "rewards": wrld.rewards, "states": wrld.environment.hidden_states, "mask": valid})

    jsonpickle_numpy.register_handlers()
    pickled = pickle.encode(data[-1])
    with open(fname, 'w') as outfile:
        json.dump(pickled, outfile)

    pickled = 0

    gc.collect()

    #print(gc.get_count())

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

    true_vals.append({"lamb": lamb, "alpha": alpha, "beta": beta, "w": w, "p": p})

stayed_arr = array(stayed)

plt.figure()
g = sns.barplot(data=stayed_arr)
g.set_xticklabels(names, rotation=45, horizontalalignment='right', fontsize=16)
plt.ylim([0,1])
plt.yticks(ar.arange(0,1.1,0.2),fontsize=16)
plt.title("habit and goal-directed", fontsize=18)
plt.savefig("habit_and_goal_mbmf.svg",dpi=300)
plt.ylabel("stay probability")
plt.show()

print('analyzing '+str(len(true_vals))+' data sets')


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

# B[:,:,0] = array([[  0,  0,  0,  0,  0,  0,  0,],
#                      [ b1,  0,  0,  0,  0,  0,  0,],
#                      [nb1,  0,  0,  0,  0,  0,  0,],
#                      [  0,  1,  0,  1,  0,  0,  0,],
#                      [  0,  0,  1,  0,  1,  0,  0,],
#                      [  0,  0,  0,  0,  0,  1,  0,],
#                      [  0,  0,  0,  0,  0,  0,  1,],])

# B[:,:,1] = array([[  0,  0,  0,  0,  0,  0,  0,],
#                      [nb2,  0,  0,  0,  0,  0,  0,],
#                      [ b2,  0,  0,  0,  0,  0,  0,],
#                      [  0,  0,  0,  1,  0,  0,  0,],
#                      [  0,  0,  0,  0,  1,  0,  0,],
#                      [  0,  1,  0,  0,  0,  1,  0,],
#                      [  0,  0,  1,  0,  0,  0,  1,],])


transition_matrix_context = ar.ones(1).to(device)


"""
create policies
"""

pol = array(list(itertools.product(list(range(na)), repeat=T-1))).to(device)

#pol = pol[-2:]
npi = pol.shape[0]


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


data_obs = ar.stack([d["observations"] for d in data], dim=-1)
data_rew = ar.stack([d["rewards"] for d in data], dim=-1)
data_act = ar.stack([d["actions"] for d in data], dim=-1)
data_mask = ar.stack([d["mask"] for d in data], dim=-1)

structured_data = {"observations": data_obs, "rewards": data_rew, "actions": data_act}

Q_mf_init = [ar.zeros((3,na)), ar.zeros((3,na))]
Q_mb_init = [ar.zeros((3,na)), ar.zeros((3,na))]

# perception
perception = prc.mfmbOrigPerception(B, pol, Q_mf_init, Q_mb_init, utility,
                                    lamb, alpha, beta, w,
                                    p, nsubs=1, use_p=use_p, mask=data_mask)

agent = agt.FittingAgent(perception, [], pol,
                      trials = trials, T = T,
                      number_of_states = ns,
                      #save_everything = True,
                      number_of_policies = npi,
                      number_of_rewards = nr)


###################################
"""inference convenience functions"""

def infer(inferrer, iter_steps, fname_str):

    inferrer.infer_posterior(iter_steps=iter_steps, num_particles=15, optim_kwargs={'lr': .01})#, param_dict

    storage_name = os.path.join(base_dir, fname_str+'.save')#h_recovered
    inferrer.save_parameters(storage_name)
    # inferrer.load_parameters(storage_name)

    loss = inferrer.loss
    plt.figure()
    plt.title("ELBO")
    plt.plot(loss)
    plt.ylabel("ELBO")
    plt.xlabel("iteration")
    plt.savefig(os.path.join(base_dir, fname_str+'_ELBO.svg'))
    plt.show()

def sample_posterior(inferrer, fname_str, n_samples=500):

    sample_df = inferrer.sample_posterior(n_samples=n_samples) #inferrer.plot_posteriors(n_samples=1000)
    # inferrer.plot_posteriors(n_samples=n_samples)

    inferred_values = []

    for i in range(len(data)):
        mean_lamb = sample_df[sample_df['subject']==i]['lamb'].mean()
        mean_alpha = sample_df[sample_df['subject']==i]['alpha'].mean()
        mean_beta = sample_df[sample_df['subject']==i]['beta'].mean()
        mean_w = sample_df[sample_df['subject']==i]['w'].mean()
        if use_p:
            mean_p = sample_df[sample_df['subject']==i]['p'].mean()

        if use_p:
            inferred_values.append({"lamb": mean_lamb, "alpha": mean_alpha, "beta": mean_beta, "w": mean_w, "p": mean_p})
        else:
            inferred_values.append({"lamb": mean_lamb, "alpha": mean_alpha, "beta": mean_beta, "w": mean_w})

    true_lamb = [val['lamb'] for val in true_vals]
    true_alpha = [val['alpha'] for val in true_vals]
    true_beta = [val['beta'] for val in true_vals]
    true_w = [val['w'] for val in true_vals]
    if use_p:
        true_p = [val['p'] for val in true_vals]

    inferred_lamb = [val['lamb'] for val in inferred_values]
    inferred_alpha = [val['alpha'] for val in inferred_values]
    inferred_beta = [val['beta'] for val in inferred_values]
    inferred_w = [val['w'] for val in inferred_values]
    if use_p:
        inferred_p = [val['p'] for val in inferred_values]

    total_df = sample_df.copy()
    total_df['true_lamb'] = ar.tensor(true_lamb).repeat(n_samples)
    total_df['true_alpha'] = ar.tensor(true_alpha).repeat(n_samples)
    total_df['true_beta'] = ar.tensor(true_beta).repeat(n_samples)
    total_df['true_w'] = ar.tensor(true_w).repeat(n_samples)
    if use_p:
        total_df['true_p'] = ar.tensor(true_p).repeat(n_samples)

    total_df['inferred_lamb'] = ar.tensor(inferred_lamb).repeat(n_samples)
    total_df['inferred_alpha'] = ar.tensor(inferred_alpha).repeat(n_samples)
    total_df['inferred_beta'] = ar.tensor(inferred_beta).repeat(n_samples)
    total_df['inferred_w'] = ar.tensor(inferred_w).repeat(n_samples)
    if use_p:
        total_df['inferred_p'] = ar.tensor(inferred_p).repeat(n_samples)

    sample_file = os.path.join(base_dir, fname_str+'.csv')
    total_df.to_csv(sample_file)

    return total_df


def plot_posterior(total_df, fname_str):

    # new_df = sample_df.copy()
    # new_df['true_pol_lambda'] = ar.zeros(len(data)*n_samples) - 1
    # new_df['true_r_lambda'] = ar.zeros(len(data)*n_samples) - 1
    # new_df['true_dec_temp'] = ar.zeros(len(data)*n_samples) - 1

    # for i in range(len(data)):
    #     new_df.loc[new_df['subject']==i,'true_pol_lambda'] = true_vals[i]['pol_lambda']
    #     new_df.loc[new_df['subject']==i,'true_r_lambda']= true_vals[i]['r_lambda']
    #     new_df.loc[new_df['subject']==i,'true_dec_temp'] = true_vals[i]['dec_temp']

    # import numpy
    # print(numpy.allclose(total_df['true_pol_lambda'], new_df['true_pol_lambda']))

    # plt.figure()
    # sns.violinplot(data=total_df, x='true_pol_lambda', y='pol_lambda', alpha=0.5)
    # sns.stripplot(data=total_df, x='true_pol_lambda', y='pol_lambda', hue='subject')
    # g = plt.gca()
    # g.set_xlim(left=-0.1, right=1.1)
    # g.set_ylim(bottom=-0.1, top=1.1)
    # plt.show()

    # plt.figure()
    # sns.violinplot(data=total_df, x='true_r_lambda', y='r_lambda', alpha=0.5)
    # sns.stripplot(data=total_df, x='true_r_lambda', y='r_lambda', hue='subject')
    # plt.xlim([-0.1, 1.1])
    # plt.ylim([-0.1, 1.1])
    # plt.show()

    # plt.figure()
    # sns.violinplot(data=total_df, x='true_dec_temp', y='dec_temp', alpha=0.5)
    # sns.stripplot(data=total_df, x='true_dec_temp', y='dec_temp', hue='subject')
    # plt.xlim([0, 10])
    # plt.ylim([0, 10])
    # plt.show()

    # if infer_h:
    #     plt.figure()
    #     sns.violinplot(data=total_df, x='true_h', y='h', alpha=0.5)
    #     sns.stripplot(data=total_df, x='true_h', y='h', hue='subject')
    #     plt.xlim([-0.1, 1.1])
    #     plt.ylim([-0.1, 1.1])
    #     plt.show()

    plt.figure()
    sns.scatterplot(data=total_df, x="true_lamb", y="inferred_lamb")
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel("true lamb")
    plt.ylabel("inferred lamb")
    plt.savefig(os.path.join(base_dir, fname_str+"_lamb.svg"))
    plt.show()

    plt.figure()
    sns.scatterplot(data=total_df, x="true_alpha", y="inferred_alpha")
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel("true alphaa")
    plt.ylabel("inferred alpha")
    plt.savefig(os.path.join(base_dir, fname_str+"_alpha.svg"))
    plt.show()

    plt.figure()
    sns.scatterplot(data=total_df, x="true_beta", y="inferred_beta")
    plt.xlim([0,10])
    plt.ylim([0,10])
    plt.xlabel("true beta")
    plt.ylabel("inferred beta")
    plt.savefig(os.path.join(base_dir, fname_str+"_beta.svg"))
    plt.show()

    plt.figure()
    sns.scatterplot(data=total_df, x="true_w", y="inferred_w")
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel("true w")
    plt.ylabel("inferred w")
    plt.savefig(os.path.join(base_dir, fname_str+"_w.svg"))
    plt.show()

    if use_p:
        plt.figure()
        sns.scatterplot(data=total_df, x="true_p", y="inferred_p")
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        plt.xlabel("true p")
        plt.ylabel("inferred p")
        plt.savefig(os.path.join(base_dir, fname_str+"_p.svg"))
        plt.show()


def plot_correlations(total_df, fname_str):

    smaller_df = pd.DataFrame()
    smaller_df['mean discounting parameter'] = total_df['inferred_lamb']
    smaller_df['mean learing rate'] = total_df['inferred_alpha']
    smaller_df['mean decision temperature mf'] = total_df['inferred_beta']
    smaller_df['mean decision temperature mb'] = total_df['inferred_w']
    if use_p:
        smaller_df['mean repetition bias'] = total_df['inferred_p']

    smaller_df['true discounting parameter'] = total_df['true_lamb']
    smaller_df['true learing rate'] = total_df['true_alpha']
    smaller_df['true decision temperature mf'] = total_df['true_beta']
    smaller_df['true decision temperature mb'] = total_df['true_w']
    if use_p:
        smaller_df['true repetition bias'] = total_df['true_p']

    rho = smaller_df.corr()
    pval = smaller_df.corr(method=lambda x, y: pearsonr(x, y)[1]) - eye(*rho.shape)
    reject, pval_corrected, alphaS, alphaB = multipletests(pval, method='bonferroni')

    plt.figure()
    sns.heatmap(smaller_df.corr(), annot=True, fmt='.2f')#[pval_corrected<alphaB]
    plt.savefig(os.path.join(base_dir, fname_str+"_mean_corr.svg"))
    plt.show()

    sample_df = pd.DataFrame()
    sample_df['sampled discounting parameter'] = total_df['lamb']
    sample_df['sampled learing rate'] = total_df['alpha']
    sample_df['sampled decision temperature mf'] = total_df['beta']
    sample_df['sampled decision temperature mb'] = total_df['w']
    if use_p:
        sample_df['sampled repetition bias'] = total_df['p']

    sample_df['true discounting parameter'] = total_df['true_lamb']
    sample_df['true learing rate'] = total_df['true_alpha']
    sample_df['true decision temperature mf'] = total_df['true_beta']
    sample_df['true decision temperature mb'] = total_df['true_w']
    if use_p:
        sample_df['true repetition bias'] = total_df['true_p']

    rho = sample_df.corr()
    pval = sample_df.corr(method=lambda x, y: pearsonr(x, y)[1]) - eye(*rho.shape)
    reject, pval_corrected, alphaS, alphaB = multipletests(pval, method='bonferroni')

    plt.figure()
    sns.heatmap(sample_df.corr(), annot=True, fmt='.2f')#[pval_corrected<alphaB]
    plt.savefig(os.path.join(base_dir, fname_str+"_sample_corr.svg"))
    plt.show()


"""run inference"""

# inferrer = inf.SingleInference(agent, structured_data)#data[0])

inferrer = inf.GeneralGroupInference(agent, structured_data)

print("this is inference using", type(inferrer))

num_steps = 600
size_chunk = 50
total_num_iter_so_far = 0

for i in range(total_num_iter_so_far, num_steps, size_chunk):
    print('taking steps '+str(i+1)+' to '+str(i+size_chunk)+' out of total '+str(num_steps))

    fname_str = fname_base + str(total_num_iter_so_far+size_chunk)+'_'+str(nsubs)+'agents'

    infer(inferrer, size_chunk, fname_str)
    total_num_iter_so_far += size_chunk
    full_df = sample_posterior(inferrer, fname_str)
    plot_posterior(full_df, fname_str)

    plot_correlations(full_df, fname_str)
    
    print("This is recovery for the twostage task using the original w and beta model.")
    print("The settings are: use p", use_p, "restrict alpha", restrict_alpha)

#print("this is inference for pl =", pl, "rl =", rl, "dt =", dt, "tend=", tend)
# print(param_dict)

print(full_df.corr())

print("This is recovery for the twostage task using the original w and beta model.")
print("The settings are: use p", use_p, "restrict alpha", restrict_alpha)