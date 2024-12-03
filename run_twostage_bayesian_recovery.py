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
import inference as inf

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

from inference import device

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
    avg, Rho, perception_args, utility, infer_h, valid = par_list
    learn_rew = 1

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

    C_alphas = ar.zeros((nr, ns)) + learn_rew
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
    
    pol_lambda = perception_args["policy rate"]
    r_lambda = perception_args["reward rate"]
    dec_temp = perception_args["dec temp"]    
    alpha_0 = perception_args["habitual tendency"]
    
    alphas = ar.zeros((npi)) + alpha_0
    prior_pi = alphas / alphas.sum(axis=0)
    learn_habit = True

    # perception
    bayes_prc = prc.Group2Perception(A, B, C_agent, transition_matrix_context,
                                           state_prior, utility, prior_pi, pol,
                                           alpha_0, C_alphas,
                                           learn_habit = learn_habit, mask=valid[:,None],
                                           learn_rew = True, T=T, trials=trials,
                                           pol_lambda=pol_lambda, r_lambda=r_lambda,
                                           non_decaying=(ns-nb), dec_temp=dec_temp, 
                                           nsubs=1, infer_alpha_0=infer_h, use_h=True)
    
    bayes_prc.set_parameters(par_dict=perception_args)
    bayes_prc.reset()

    bayes_pln = agt.FittingAgent(bayes_prc, ac_sel, pol,
                      trials = trials, T = T,
                      prior_states = state_prior,
                      prior_policies = prior_pi,
                      number_of_states = ns,
                      prior_context = prior_context,
                      #save_everything = True,
                      number_of_policies = npi,
                      number_of_rewards = nr)



    """
    create world
    """

    w = world.GroupWorld(environment, bayes_pln, trials = trials, T = T)

    """
    simulate experiment
    """

    w.simulate_experiment(range(trials))


    return w


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
    utility.append(ar.zeros(nr).to(device))
    for i in range(1,nr):
        utility[-1][i] = u/(nr-1)#u/nr*i
    utility[-1][0] = (1.-u)

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

infer_h = True

prefix = "BCC_"
param_names = ["policy rate", "reward rate", "dec temp", "habitual tendency"]
model_name = "Bayesian prior-based contextual control model"

if infer_h:
    n_pars = 4
    h_str = "inferh_"
else:
    n_pars = 3
    h_str = ""
    param_names = param_names[:-1]
    
max_dt = 6

# prepare for savin results
# make base filename and folder string
fname_base = prefix+"recovered_"+h_str
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
    

nsubs = 10
true_values_tensor = ar.rand((nsubs,n_pars,1))

# prob for invalid answer (e.g. no reply)
p_invalid = 1.-1./201.

stayed = []
indices = []

for pars in true_values_tensor:

    if infer_h:
        pl, rl, norm_dt, h = pars
        tend = h
    else:
        pl, rl, norm_dt = pars
        tend = ar.tensor([1])

    dt = max_dt*norm_dt+1

    print(pl, rl, dt, tend)
    
    perception_args = {"policy rate": pl, "reward rate": rl, "dec temp": dt, "habitual tendency": tend}

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
    prob_matrix = ar.zeros((trials)) + p_invalid
    valid = ar.bernoulli(prob_matrix).bool()
    pars = [avg, Rho,perception_args, utility, infer_h, valid]

    worlds.append(run_agent(pars))

    w = worlds[-1]

    # rewarded = ar.where(w.rewards[:trials-1,-1] == 1)[0]

    # unrewarded = ar.where(w.rewards[:trials-1,-1] == 0)[0]

    rewarded = w.rewards[:trials-1,-1] == 1

    unrewarded = rewarded==False#w.rewards[:trials-1,-1] == 0

    # rare = ar.cat((ar.where(own_logical_and(w.environment.hidden_states[:,1]==2, w.actions[:,0] == 0) == True)[0],
    #                  ar.where(own_logical_and(w.environment.hidden_states[:,1]==1, w.actions[:,0] == 1) == True)[0]))
    # rare.sort()

    # common = ar.cat((ar.where(own_logical_and(w.environment.hidden_states[:,1]==2, w.actions[:,0] == 1) == True)[0],
    #                    ar.where(own_logical_and(w.environment.hidden_states[:,1]==1, w.actions[:,0] == 0) == True)[0]))
    # common.sort()

    rare = ar.logical_or(ar.logical_and(w.environment.hidden_states[:trials-1,1]==2, w.actions[:trials-1,0] == 0),
                   ar.logical_and(w.environment.hidden_states[:trials-1,1]==1, w.actions[:trials-1,0] == 1))

    common = rare==False#own_logical_or(own_logical_and(w.environment.hidden_states[:trials-1,1]==2, w.actions[:trials-1,0] == 1),
             #        own_logical_and(w.environment.hidden_states[:trials-1,1]==1, w.actions[:trials-1,0] == 0))

    names = ["rewarded common", "rewarded rare", "unrewarded common", "unrewarded rare"]

    # index_list = [ar.intersect1d(rewarded, common), ar.intersect1d(rewarded, rare),
    #              ar.intersect1d(unrewarded, common), ar.intersect1d(unrewarded, rare)]

    rewarded_common = ar.where(ar.logical_and(rewarded,common) == True)[0]
    rewarded_rare = ar.where(ar.logical_and(rewarded,rare) == True)[0]
    unrewarded_common = ar.where(ar.logical_and(unrewarded,common) == True)[0]
    unrewarded_rare = ar.where(ar.logical_and(unrewarded,rare) == True)[0]

    index_list = [rewarded_common, rewarded_rare,
                 unrewarded_common, unrewarded_rare]

    stayed_list = [(w.actions[index_list[i],0] == w.actions[index_list[i]+1,0]).sum()/float(len(index_list[i])) for i in range(4)]

    stayed.append(stayed_list)

    run_name = "twostage_agent_daw_pl"+str(pl)+"_rl"+str(rl)+"_dt"+str(dt)+"_tend"+str(tend)+".json"
    fname = os.path.join(folder, run_name)

    # actions = w.actions.numpy()
    # observations = w.observations.numpy()
    # rewards = w.rewards.numpy()
    # states = w.environment.hidden_states.numpy()
    data.append({"actions": w.actions, "observations": w.observations, "rewards": w.rewards, "states": w.environment.hidden_states, 'mask': valid})

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

    true_vals.append(perception_args)

stayed_arr = array(stayed)

learn_habit = True
plt.figure()
g = sns.barplot(data=stayed_arr)
g.set_xticklabels(names, rotation=45, horizontalalignment='right', fontsize=16)
plt.ylim([0,1])
plt.yticks(ar.arange(0,1.1,0.2),fontsize=16)
if learn_habit:
    plt.title("habit and goal-directed", fontsize=18)
    plt.savefig("habit_and_goal.svg",dpi=300)
else:
    plt.title("purely goal-drected", fontsize=18)
    plt.savefig("pure_goal.svg",dpi=300)
plt.ylabel("stay probability")
plt.show()

print('analyzing '+str(len(true_vals))+' data sets')


learn_rew = 1

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


"""
set state prior (where agent thinks it starts)
"""

state_prior = ar.zeros((ns)).to(device)

state_prior[0] = 1.

prior_context = array([1.]).to(device)

#    prior_context[0] = 1.
prior_pi = ar.zeros(npi) / ar.zeros(npi).sum()
"""
set up agent
"""
#bethe agent

data_obs = ar.stack([d["observations"] for d in data], dim=-1)
data_rew = ar.stack([d["rewards"] for d in data], dim=-1)
data_act = ar.stack([d["actions"] for d in data], dim=-1)

structured_data = {"observations": data_obs, "rewards": data_rew, "actions": data_act}

# perception
pol_lambda = ar.tensor([1])
r_lambda = ar.tensor([0.5])
dec_temp = ar.tensor([2])   
alpha_0 = ar.tensor([1])

alphas = ar.zeros((npi)) + alpha_0
prior_pi = alphas / alphas.sum(axis=0)
learn_habit = True

npart = 15

# obs_message_list = [[]]
# for tau in range(trials):
#     obs_message_tau = []
#     rew_message_tau = []
#     for t in range(T):
#         observations = ar.stack(data_obs[tau][-t-1:])
#         obs_messages = []
#         for n in range(nsubs):
#             prev_obs = [A[o] for o in observations[-t-1:,n]]
#             obs = prev_obs + [ar.zeros((ns)).to(device)+1./ns]*(T-t-1)
#             obs = [ar.stack(obs).T.to(device)]*npart
#             obs_messages.append(ar.stack(obs, dim=-1))
#         obs_messages = ar.stack(obs_messages, dim=-1).to(device)
#         obs_message_tau.append(obs_messages)
    
#         rewards = ar.stack(data_rew[tau][-t-1:])
#         rew_messages = []
#         for n in range(nsubs):
#             rew_messages.append(ar.stack([ar.stack([generative_model_rewards[r,:,i,n].to(device) for r in rewards[-t-1:,n]]  \
#                                                    + [utility.matmul(generative_model_rewards[:,:,i,n].to(device)).to(device)]*(self.T-t-1)).T.to(device) for i in range(self.npart)], dim=-1).to(device))
#         rew_messages = ar.stack(rew_messages, dim=-1).to(device)
#         rew_message_tau.append(rew_messages)
        
#     obs_message_list.append(obs_message_tau)
    

# perception
bayes_prc = prc.Group2Perception(A, B, C_agent, transition_matrix_context,
                                       state_prior, utility, prior_pi, pol,
                                       alpha_0, C_alphas,
                                       learn_habit = learn_habit, mask=valid[:,None],
                                       learn_rew = True, T=T, trials=trials,
                                       pol_lambda=pol_lambda, r_lambda=r_lambda,
                                       non_decaying=(ns-nb), dec_temp=dec_temp, 
                                       nsubs=1, infer_alpha_0=infer_h, use_h=True)

agent = agt.FittingAgent(bayes_prc, [], pol,
                  trials = trials, T = T,
                  prior_states = state_prior,
                  prior_policies = prior_pi,
                  number_of_states = ns,
                  prior_context = prior_context,
                  #save_everything = True,
                  number_of_policies = npi,
                  number_of_rewards = nr,
                  nsubs = nsubs)


###################################
"""inference convenience functions"""

def infer(inferrer, iter_steps, fname_str):

    inferrer.infer_posterior(iter_steps=iter_steps, num_particles=npart, optim_kwargs={'lr': .01})#, param_dict

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
    
    sample_file = os.path.join(base_dir, fname_str+'_sample_df.csv')
    sample_df.to_csv(sample_file)
    
    smaller_df = pd.DataFrame()

    for name in param_names:
        means = []
        trues = []
        subs = []
        for i in range(len(data)):
            means.append(sample_df[sample_df['subject']==i][name].mean())
            trues.append(true_vals[i][name])
            subs.append(i)

        smaller_df["inferred "+name] = ar.tensor(means)
        smaller_df["true "+name] = ar.tensor(trues)
        smaller_df["subject"] = ar.tensor(subs)
        
    smaller_file = os.path.join(base_dir, fname_str+'_smaller_df.csv')
    smaller_df.to_csv(smaller_file)

    total_df = sample_df.copy()
    for name in param_names:
        total_df["true "+name] = ar.tensor(smaller_df["true "+name]).repeat(n_samples)
        total_df["inferred "+name] = ar.tensor(smaller_df["inferred "+name]).repeat(n_samples)

    total_file = os.path.join(base_dir, fname_str+'_total_df.csv')
    total_df.to_csv(total_file)

    return total_df, smaller_df, sample_df


def plot_inferred(smaller_df, fname_str, reg_fit=False):
    
    plot_df = smaller_df.drop('subject', axis=1)
                        
    axes_names = ["policy forgetting rate lambda_pi", "reward forgetting rate lambda_r", "decision temp gamma", "habitual tendency h"]
    ranges = [[0,1], [0,1], [1, max_dt], [0,1], [0, 1]]

    
    for i, name in enumerate(param_names):

        plt.figure()
        plt.plot(ranges[i],ranges[i], linestyle='-', color="grey", alpha=0.6)
        # sns.scatterplot(data=plot_df, x="true "+name, y="inferred "+name, ax=ax)
        sns.regplot(data=plot_df, x="true "+name, y="inferred "+name,
                   line_kws = {'color': 'green', 'alpha': 0.3}, fit_reg=reg_fit)
        plt.xlim(ranges[i])
        plt.ylim(ranges[i])
        plt.xlabel("true "+axes_names[i])
        plt.ylabel("inferred "+axes_names[i])
        plt.annotate(axes_names[i], (0.+0.1*ranges[i][1], ranges[i][1]-0.1*ranges[i][1]))
        plt.show()
        
        
def big_custom_plot(plot_df, fname_str, ELBO, fit_reg=False, annot=False):
    
    axes_names = ["policy forgetting rate lambda_pi", "reward forgetting rate lambda_r", "decision temp gamma", "habitual tendency h"]
    ranges = [[0,1], [0,1], [1, max_dt], [0,1], [0, 1]]
    positions = [[0,0], [0,1], [1,0], [1,1]]

    fig = plt.figure(layout='constrained', figsize=(14,12))
    axes = fig.subplots(3, 3)
    
    for i, name in enumerate(param_names):
    
        ax = axes[positions[i][0], positions[i][1]]
        ax.plot(ranges[i],ranges[i], linestyle='-', color="grey", alpha=0.6)
        # sns.scatterplot(data=plot_df, x="true "+name, y="inferred "+name, ax=ax)
        sns.regplot(data=plot_df, x="true "+name, y="inferred "+name, ax=ax,
                   line_kws = {'color': 'green', 'alpha': 0.3}, fit_reg=fit_reg)
        ax.set_xlim(ranges[i])
        ax.set_ylim(ranges[i])
        ax.set_xlabel("true "+axes_names[i])
        ax.set_ylabel("inferred "+axes_names[i])
        ax.annotate(axes_names[i], (0.+0.1*ranges[i][1], ranges[i][1]-0.1*ranges[i][1]), fontsize=16)
        
        if annot:
            (r, p) = pearsonr(plot_df["true "+name], plot_df["inferred "+name])
            ax.annotate("r = {:.2f} ".format(r)+"p = {:.3f}".format(p), 
                        (0.4*ranges[i][1], 0.05*ranges[i][1]), fontsize=16)
            # ax.annotate("p = {:.3f}".format(p),
            #             (0.7*ranges[i][1], 0.05*ranges[i][1]))
        
    ax = axes[2,0]
    # plt.title("ELBO")
    ax.plot(ELBO)
    ax.set_ylabel("ELBO", fontsize=16)
    ax.set_xlabel("iteration", fontsize=16)

    rho = plot_df.corr()
    pval = plot_df.corr(method=lambda x, y: pearsonr(x, y)[1]) - eye(*rho.shape)
    reject, pval_corrected, alphaS, alphaB = multipletests(pval, method='bonferroni')
    
    gs = axes[2, 1].get_gridspec()
    # remove the underlying axes
    for ax in axes[2, 1:]:
        ax.remove()
    axbig = fig.add_subplot(gs[2, 2])
    ax = axbig
    
    p_opacity = pval_corrected*0.5 +0.5

    sns.heatmap(plot_df.corr(), annot=True, fmt='.2f', alpha=p_opacity, 
                cmap='vlag', vmin=-1, vmax=1, ax=ax)
    
    # sns.heatmap(smaller_df.corr(), annot=True, fmt='.2f', ax=ax)#[pval_corrected<alphaB]
        
    try:
        plt.tight_layout()
    except:
        pass
    
    if fit_reg:
        name_str = "_regression"
    else:
        name_str = ""
    if annot:
        name_str += "_annot"
    
    plt.savefig(os.path.join(base_dir, fname_str+"_big_plot"+name_str+".svg"))
    plt.show()
    
    
def plot_results(sample_df, fname_str, ELBO, smaller_df):
    
    plot_df = smaller_df.drop('subject', axis=1)\
                        .reindex(["inferred "+name for name in param_names]\
                                 +["true "+name for name in param_names], axis=1)
        
    def annot_corrfunc(x, y, **kws):
        (r, p) = pearsonr(x, y)
        ax = plt.gca()
        ax.annotate("r = {:.2f} ".format(r),
                    xy=(.1, .9), xycoords=ax.transAxes)
        ax.annotate("p = {:.3f}".format(p),
                    xy=(.4, .9), xycoords=ax.transAxes)
        
    big_custom_plot(plot_df, fname_str, ELBO, fit_reg=True, annot=True)
    big_custom_plot(plot_df, fname_str, ELBO, fit_reg=True, annot=False)
    big_custom_plot(plot_df, fname_str, ELBO, fit_reg=False, annot=True)
    big_custom_plot(plot_df, fname_str, ELBO, fit_reg=False, annot=False)
    
    # plt.figure()
    # sns.pairplot(sample_df, kind='reg')
    # plt.savefig(os.path.join(base_dir, fname_str+"_pairplot_sample.svg"))
    # plt.show()
    
    plt.figure()
    f = sns.pairplot(data=plot_df, kind='reg', 
                     diag_kind="kde", corner=True,
                     plot_kws={'line_kws': {'color': 'green', 'alpha': 0.6}})
    f.map(annot_corrfunc)
    plt.savefig(os.path.join(base_dir, fname_str+"_pairplot_means_all.svg"))
    plt.show()
    
    plt.figure()
    xvars_of_interest = ["true "+name for name in param_names]
    yvars_of_interest = ["inferred "+name for name in param_names]
    f = sns.pairplot(data=plot_df, kind='reg', diag_kind="kde", corner=True,
                     plot_kws={'line_kws': {'color': 'green', 'alpha': 0.6}},
                     x_vars=xvars_of_interest, y_vars=yvars_of_interest)
    f.map(annot_corrfunc)
    plt.savefig(os.path.join(base_dir, fname_str+"_pairplot_means.svg"))
    plt.show()
    
    plt.figure()
    vars_of_interest = ["inferred "+name for name in param_names]
    f = sns.pairplot(data=plot_df, kind='reg', diag_kind="kde", corner=True,
                     plot_kws={'line_kws': {'color': 'green', 'alpha': 0.6}},
                     x_vars=vars_of_interest, y_vars=vars_of_interest)
    f.map(annot_corrfunc)
    plt.savefig(os.path.join(base_dir, fname_str+"_pairplot_means_inferred_corr.svg"))
    plt.show()
    
    # p_opacity = pval_corrected*0.5 +0.5
    
    # plt.figure()
    # sns.heatmap(plot_df.corr(), annot=True, fmt='.2f', alpha=p_opacity, 
    #             cmap='vlag', vmin=-1, vmax=1)
    # plt.show()


"""run inference"""

# inferrer = inf.SingleInference(agent, structured_data)#data[0])

inferrer = inf.GeneralGroupInference(agent, structured_data)

if infer_h:
    prefix = "h_"
else:
    prefix = ""

print("this is inference using", type(inferrer))

num_steps = 600
size_chunk = 50
total_num_iter_so_far = 0

for i in range(total_num_iter_so_far, num_steps, size_chunk):
    print('taking steps '+str(i+1)+' to '+str(i+size_chunk)+' out of total '+str(num_steps))

    fname_str = fname_base + str(total_num_iter_so_far+size_chunk)+'_'+str(nsubs)+'agents'

    infer(inferrer, size_chunk, fname_str)
    total_num_iter_so_far += size_chunk
    full_df, smaller_df, sample_df = sample_posterior(inferrer, fname_str) 
    
    # plot_posterior(full_df, fname_str)
    # plot_correlations(full_df, fname_str)
    
    plot_results(sample_df, fname_str, inferrer.loss, smaller_df)
    
    print("This is recovery for the twostage task using the "+model_name+"with "+str(nsubs)+" agents.")
    print("The settings are: infer h", infer_h)

#print("this is inference for pl =", pl, "rl =", rl, "dt =", dt, "tend=", tend)
# print(param_dict)

print("This is recovery for the twostage task using the "+model_name+"with "+str(nsubs)+" agents.")
print("The settings are: infer h", infer_h)