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
import scipy as sc
import scipy.signal as ss
from scipy.stats import pearsonr
import gc
import sys
from numpy import eye
from statsmodels.stats.multitest import multipletests
from scipy.io import loadmat
import numpy as np
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


"""read data"""

def load_dataset(fname):
    dataset = loadmat(fname)
    np.testing.assert_allclose(dataset['dawrandomwalks'], rew_probs, err_msg='This person has different reward probabilities')

    nan_actions = ar.from_numpy(dataset['A'] - 1).permute(1,0)
    nan_mask = ar.isnan(nan_actions)
    actions = ar.where(ar.logical_or(nan_mask,nan_actions==-1), ar.zeros_like(nan_actions), nan_actions).long()
    # have true if there is no NaN action in this miniblock
    mask = ar.logical_not(ar.any(nan_mask, dim=1))

    first_stage_rewards = ar.zeros(trials) + 2
    second_stage_nan_rewards = ar.from_numpy(dataset['R'][0])
    second_stage_rewards = ar.where(mask, second_stage_nan_rewards, ar.zeros_like(second_stage_nan_rewards)+2)
    rewards = ar.stack([first_stage_rewards, first_stage_rewards, second_stage_rewards], dim=-1).long()

    first_stage_nan_states = ar.from_numpy(dataset['S'][0] - 1)
    first_stage_states = ar.where(mask, first_stage_nan_states, ar.zeros_like(first_stage_nan_states))
    second_stage_nan_states = ar.from_numpy(dataset['S'][1] - 1)
    second_stage_states = ar.where(mask, second_stage_nan_states, ar.zeros_like(second_stage_nan_states))
    third_stage_nan_states = second_stage_states+2 + actions[:,1]*2
    third_stage_states = ar.where(mask, third_stage_nan_states, ar.zeros_like(third_stage_nan_states))
    states = ar.stack([first_stage_states, second_stage_states, third_stage_states], dim=-1).long()

    observations = states

    common_trans = ar.from_numpy(dataset['trans'][0]).long()

    data_dict = {"actions": actions, "observations": observations, "rewards": rewards, "states": states, 'mask': mask}
    return data_dict, common_trans, dataset

base_folder = "data"
data_folder = os.path.join(base_folder,"twostage_data", "raw_data_188")

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

fname = os.path.join(base_folder, Rho_data_fname)

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

use_p = False

if use_p:
    n_pars = 5
else:
    n_pars = 4

stayed = []
indices = []

dataset_names = os.listdir(data_folder)
nsubs = len(dataset_names)

for i in range(nsubs):

    fname = os.path.join(data_folder, dataset_names[i])
    data_dict, common_trans, dataset = load_dataset(fname)

    data.append(data_dict)
    # print(data_dict['mask'])

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

    rewarded = data_dict['rewards'][:trials-1,2][data_dict['mask'][:trials-1]] > 0

    unrewarded = rewarded==False#w.rewards[:trials-1,-1] == 0

    common = common_trans[:trials-1][data_dict['mask'][:trials-1]] > 0

    rare = common == False

    names = ["rewarded common", "rewarded rare", "unrewarded common", "unrewarded rare"]

    # index_list = [ar.intersect1d(rewarded, common), ar.intersect1d(rewarded, rare),
    #              ar.intersect1d(unrewarded, common), ar.intersect1d(unrewarded, rare)]

    rewarded_common = ar.where(ar.logical_and(rewarded,common) == True)[0]
    rewarded_rare = ar.where(ar.logical_and(rewarded,rare) == True)[0]
    unrewarded_common = ar.where(ar.logical_and(unrewarded,common) == True)[0]
    unrewarded_rare = ar.where(ar.logical_and(unrewarded,rare) == True)[0]

    index_list = [rewarded_common, rewarded_rare,
                 unrewarded_common, unrewarded_rare]

    stayed_list = [(data_dict['actions'][index_list[i],0] == data_dict['actions'][index_list[i]+1,0]).sum()/float(len(index_list[i])) for i in range(4)]

    stayed.append(stayed_list)


    gc.collect()


stayed_arr = array(stayed)

plt.figure()
g = sns.barplot(data=stayed_arr)
g.set_xticklabels(names, rotation=45, horizontalalignment='right', fontsize=16)
plt.ylim([0.5,1])
# plt.yticks(ar.arange(0,1.1,0.2),fontsize=16)
plt.title("habit and goal-directed", fontsize=18)
plt.savefig("habit_and_goal_mbmf.svg",dpi=300)
plt.ylabel("stay probability")
plt.show()

print('analyzing '+str(nsubs)+' data sets')


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

# B[:,:,0] = array([[  0,  0,  0,  0,  0,  0,  0,],
#                       [ b1,  0,  0,  0,  0,  0,  0,],
#                       [nb1,  0,  0,  0,  0,  0,  0,],
#                       [  0,  1,  0,  1,  0,  0,  0,],
#                       [  0,  0,  0,  0,  1,  0,  0,],
#                       [  0,  0,  1,  0,  0,  1,  0,],
#                       [  0,  0,  0,  0,  0,  0,  1,],])

# B[:,:,1] = array([[  0,  0,  0,  0,  0,  0,  0,],
#                       [nb2,  0,  0,  0,  0,  0,  0,],
#                       [ b2,  0,  0,  0,  0,  0,  0,],
#                       [  0,  0,  0,  1,  0,  0,  0,],
#                       [  0,  1,  0,  0,  1,  0,  0,],
#                       [  0,  0,  0,  0,  0,  1,  0,],
#                       [  0,  0,  1,  0,  0,  0,  1,],])

B[:,:,0] = array([[  0,  0,  0,  0,  0,  0,  0,],
                      [ b1,  0,  0,  0,  0,  0,  0,],
                      [nb1,  0,  0,  0,  0,  0,  0,],
                      [  0,  1,  0,  1,  0,  0,  0,],
                      [  0,  0,  1,  0,  1,  0,  0,],
                      [  0,  0,  0,  0,  0,  1,  0,],
                      [  0,  0,  0,  0,  0,  0,  1,],])

B[:,:,1] = array([[  0,  0,  0,  0,  0,  0,  0,],
                      [nb2,  0,  0,  0,  0,  0,  0,],
                      [ b2,  0,  0,  0,  0,  0,  0,],
                      [  0,  0,  0,  1,  0,  0,  0,],
                      [  0,  0,  0,  0,  1,  0,  0,],
                      [  0,  1,  0,  0,  0,  1,  0,],
                      [  0,  0,  1,  0,  0,  0,  1,],])


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

# init values, they dont matter, they will be overwritten as soon as inference starts anyways
lamb = ar.tensor([0.9])
alpha = ar.tensor([0.1])
beta = ar.tensor([4.])
w = ar.tensor([0.5])
p = ar.tensor([2.])
Q_mf_init = [ar.zeros((3,na)), ar.zeros((3,na))]
Q_mb_init = [ar.zeros((3,na)), ar.zeros((3,na))]

# perception
perception = prc.mfmb2Perception(B, pol, Q_mf_init, Q_mb_init, utility,
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

def infer(inferrer, iter_steps, prefix, total_num_iter_so_far):

    inferrer.infer_posterior(iter_steps=iter_steps, num_particles=15, optim_kwargs={'lr': .01})#, param_dict

    storage_name = prefix+'recovered_'+str(total_num_iter_so_far+iter_steps)+'_'+str(nsubs)+'subjectsOrig.save'#h_recovered
    storage_name = os.path.join(base_folder, storage_name)
    inferrer.save_parameters(storage_name)
    # inferrer.load_parameters(storage_name)

    loss = inferrer.loss
    plt.figure()
    plt.title("ELBO")
    plt.plot(loss)
    plt.ylabel("ELBO")
    plt.xlabel("iteration")
    plt.savefig('recovered_ELBO')
    plt.savefig(prefix+'recovered_'+str(total_num_iter_so_far+iter_steps)+'_'+str(nsubs)+'subjectsOrig_ELBO.svg')
    plt.show()

def sample_posterior(inferrer, prefix, total_num_iter_so_far, n_samples=500):

    sample_df = inferrer.sample_posterior(n_samples=n_samples) #inferrer.plot_posteriors(n_samples=1000)
    # inferrer.plot_posteriors(n_samples=n_samples)

    inferred_values = []

    for i in range(len(data)):
        mean_lamb = sample_df[sample_df['subject']==i]['lamb'].mean()
        mean_alpha = sample_df[sample_df['subject']==i]['alpha'].mean()
        mean_beta_mf = sample_df[sample_df['subject']==i]['beta_mf'].mean()
        mean_beta_mb = sample_df[sample_df['subject']==i]['beta_mb'].mean()
        if use_p:
            mean_p = sample_df[sample_df['subject']==i]['p'].mean()

        if use_p:
            inferred_values.append({"lamb": mean_lamb, "alpha": mean_alpha, "beta_mf": mean_beta_mf, "beta_mb": mean_beta_mb, "p": mean_p})
        else:
            inferred_values.append({"lamb": mean_lamb, "alpha": mean_alpha, "beta_mf": mean_beta_mf, "beta_mb": mean_beta_mb})

    sample_file = prefix+'inferred_samples_'+str(total_num_iter_so_far)+'_'+str(nsubs)+'subjectsOrig.csv'
    fname = os.path.join(base_folder, sample_file)
    sample_df.to_csv(fname)

    return sample_df


def plot_posterior(total_df, total_num_iter_so_far, prefix):

    plt.figure()
    ax = sns.histplot(data=total_df, x="lamb", hue='subject')
    # plt.xlim([-0.1, 1.1])
    # plt.ylim([-0.1, 1.1])
    plt.savefig(prefix+"inferred_"+str(total_num_iter_so_far)+"_"+str(nsubs)+"subjectsOrig_lamb.svg")
    ax.get_legend().remove()
    plt.show()

    plt.figure()
    ax = sns.histplot(data=total_df, x="alpha", hue='subject')
    # plt.xlim([-0.1, 1.1])
    # plt.ylim([-0.1, 1.1])
    plt.savefig(prefix+"inferred_"+str(total_num_iter_so_far)+"_"+str(nsubs)+"subjectsOrig_alpha.svg")
    ax.get_legend().remove()
    plt.show()

    plt.figure()
    ax = sns.histplot(data=total_df, x="beta_mf", hue='subject')
    # plt.xlim([-0.1, 1.1])
    # plt.ylim([-0.1, 1.1])
    plt.savefig(prefix+"inferred_"+str(total_num_iter_so_far)+"_"+str(nsubs)+"subjectsOrig_beta_mf.svg")
    ax.get_legend().remove()
    plt.show()

    plt.figure()
    ax = sns.histplot(data=total_df, x="beta_mb", hue='subject')
    # plt.xlim([-0.1, 1.1])
    # plt.ylim([-0.1, 1.1])
    plt.savefig(prefix+"inferred_"+str(total_num_iter_so_far)+"_"+str(nsubs)+"subjectsOrig_beta_mb.svg")
    ax.get_legend().remove()
    plt.show()

    if use_p:
        plt.figure()
        ax = sns.histplot(data=total_df, x="p", hue='subject')
        # plt.xlim([-0.1, 1.1])
        # plt.ylim([-0.1, 1.1])
        plt.savefig(prefix+"inferred_"+str(total_num_iter_so_far)+"_"+str(nsubs)+"subjectsOrig_p.svg")
        ax.get_legend().remove()
        plt.show()


def plot_correlations(total_df, total_num_iter_so_far,prefix):

    sample_df = pd.DataFrame()
    sample_df['sampled discounting parameter'] = total_df['lamb']
    sample_df['sampled learing rate'] = total_df['alpha']
    sample_df['sampled decision temperature mf'] = total_df['beta_mf']
    sample_df['sampled decision temperature mb'] = total_df['beta_mb']
    if use_p:
        sample_df['sampled repetition bias'] = total_df['p']

    rho = sample_df.corr()
    pval = sample_df.corr(method=lambda x, y: pearsonr(x, y)[1]) - eye(*rho.shape)
    reject, pval_corrected, alphaS, alphaB = multipletests(pval, method='bonferroni')

    plt.figure()
    sns.heatmap(sample_df.corr(), annot=True, fmt='.2f')#[pval_corrected<alphaB]
    plt.savefig(prefix+"inferred_"+str(total_num_iter_so_far)+"_"+str(nsubs)+"subjectsOrig_sample_corr.svg")
    plt.show()


"""run inference"""

# inferrer = inf.SingleInference(agent, structured_data)#data[0])

inferrer = inf.GeneralGroupInference(agent, structured_data)

prefix = 'mbmf_'

print("this is inference using", type(inferrer))

num_steps = 500
size_chunk = 50
total_num_iter_so_far = 0

for i in range(total_num_iter_so_far, num_steps, size_chunk):
    print('taking steps '+str(i+1)+' to '+str(i+size_chunk)+' out of total '+str(num_steps))

    infer(inferrer, size_chunk, prefix, total_num_iter_so_far)
    total_num_iter_so_far += size_chunk
    full_df = sample_posterior(inferrer, prefix, total_num_iter_so_far)
    plot_posterior(full_df, total_num_iter_so_far, prefix)

    plot_correlations(full_df, total_num_iter_so_far, prefix)

#print("this is inference for pl =", pl, "rl =", rl, "dt =", dt, "tend=", tend)
# print(param_dict)

print(full_df.corr())