#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 17:56:24 2017

@author: sarah
"""

arr_type = "torch"
if arr_type == "numpy":
    import numpy as ar
    array = ar.array
else:
    import torch as ar
    array = ar.tensor
    
import numpy as np

#from plotting import *
from misc import *

import world
import environment as env
import agent as agt
import perception as prc
import action_selection as asl
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
from scipy.io import loadmat
import scipy.signal as ss
import bottleneck as bn
import gc
#ar.set_printoptions(threshold = 100000, precision = 5)

from inference_twostage import device

"""
set parameters
"""
agent = 'bethe'
#agent = 'meanfield'

save_data = False
#equidistant numbers in log space
numbers = [10, 17, 31, 56, 100, 177, 316, 562, 1000, 1778, 3162, 5623, 10000, 17782, 31622, 56234]
trials =  201#number of trials
T = 3 #number of time steps in each trial
nb = 4
ns = 3+nb #number of states
no = ns #number of observations
na = 2 #number of actions
npi = na**(T-1)
nr = 2

proni = "/home/sarah/proni/sarah/"

"""
run function
"""
def run_agent(par_list, trials=trials, T=T, ns=ns, na=na):

    #set parameters:
    #obs_unc: observation uncertainty condition
    #state_unc: state transition uncertainty condition
    #goal_pol: evaluate only policies that lead to the goal
    #utility: goal prior, preference p(o)
    learn_pol, avg, Rho, learn_habit, pol_lambda, r_lambda, dec_temp, utility = par_list
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

    # prior over policies

    prior_pi = ar.ones(npi)/npi #ar.zeros(npi) + 1e-3/(npi-1)
    #prior_pi[170] = 1. - 1e-3
    alphas = ar.zeros((npi)) + learn_pol
    alpha_0 = learn_pol
#    for i in range(nb):
#        alphas[i+1,i] = 100
    #alphas[170] = 100
    prior_pi = alphas / alphas.sum(axis=0)


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
    #bethe agent
    if agent == 'bethe':

        agnt = 'bethe'

        #pol_par = alphas

        # perception
        bayes_prc = prc.FittingPerception(A, B, C_agent, transition_matrix_context, 
                                               state_prior, utility, prior_pi, pol,
                                               alpha_0, C_alphas, T=T, trials=trials,
                                               pol_lambda=pol_lambda, r_lambda=r_lambda,
                                               non_decaying=(ns-nb), dec_temp=dec_temp)

        bayes_pln = agt.FittingAgent(bayes_prc, ac_sel, pol,
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
    #MF agent
    else:

        agnt = 'mf'

        bayes_prc = prc.MFPerception(A, B, utility, state_prior, T = T)



        bayes_pln = agt.BayesianMFPlanner(bayes_prc, [], ac_sel,
                                  trials = trials, T = T,
                                  prior_states = state_prior,
                                  policies = pol,
                                  number_of_states = ns,
                                  number_of_policies = npi)


    """
    create world
    """

    w = world.World(environment, bayes_pln, trials = trials, T = T)

    """
    simulate experiment
    """

#    w.simulate_experiment(range(trials-100))
#    new_ut = utility.copy()
#    new_ut[1] = utility[0]
#    new_ut /= new_ut.sum()
#    w.agent.perception.reset_preferences(0,new_ut, pol)
#    w.simulate_experiment(range(trials-100, trials))

    w.simulate_experiment(range(trials))


    """
    plot and evaluate results
    """
#    plt.figure()
#
#    for i in range(3,ns):
#        plt.plot(w.environment.Rho[:,1,i], label=str(i))
#
#    plt.ylim([0,1])
#    plt.legend()
#    plt.show()
#
#
#    rewarded = ar.where(w.rewards[:trials-1,-1] == 1)[0]
#    unrewarded = ar.where(w.rewards[:trials-1,-1] == 0)[0]
#
#    rare = ar.append(ar.where(w.environment.hidden_states[ar.where(w.actions[:,0] == 0)[0]] == 2)[0],
#                     ar.where(w.environment.hidden_states[ar.where(w.actions[:,0] == 1)[0]] == 1)[0])
#
#    common = ar.append(ar.where(w.environment.hidden_states[ar.where(w.actions[:,0] == 0)[0]] == 1)[0],
#                     ar.where(w.environment.hidden_states[ar.where(w.actions[:,0] == 1)[0]] == 2)[0])
#
#    names = ["rewarded common", "rewarded rare", "unrewarded common", "unrewarded rare"]
#
#    index_list = [ar.intersect1d(rewarded, common), ar.intersect1d(rewarded, rare),
#                 ar.intersect1d(unrewarded, common), ar.intersect1d(unrewarded, rare)]
#
#    stayed_list = [((w.actions[index_list[i],0] - w.actions[index_list[i]+1,0])==0).sum()/len(index_list[i]) for i in range(4)]
#
##    stayed_rew = ((w.actions[rewarded,0] - w.actions[rewarded+1,0]) == 0).sum()/len(rewarded)
##
##    stayed_unrew = ((w.actions[unrewarded,0] - w.actions[unrewarded+1,0]) == 0).sum()/len(unrewarded)
#
#    plt.figure()
#    plt.bar(x=names,height=stayed_list)
#    plt.show()


    return w

"""
set condition dependent up parameters
"""
utility = []

#ut = [0.5, 0.6, 0.7, 0.8, 0.9, 1. - 1e-5]
#ut = [0.95, 0.96, 0.98, 0.99]
#ut = [0.985]
ut = [0.999]
for u in ut:
    utility.append(ar.zeros(nr))
    for i in range(1,nr):
        utility[-1][i] = u/(nr-1)#u/nr*i
    utility[-1][0] = (1.-u)

changes = []

C = ar.zeros((nr, ns))

Rho = ar.zeros((trials, C.shape[0], C.shape[1]))
n_training = 1
#for i in range(4):
#    Rho[trials*i//4:trials*(i+1)//4] = generate_bandit_timeseries_training(trials//4, nr, ns, nb//2, n_training, (i%2)*2)#generate_bandit_timeseries_change(C, nb, trials, changes)
#Rho[:] = generate_bandit_timeseries_slowchange(trials, nr, ns, nb)
#prefix = "superslow"
#Rho[:] = generate_bandit_timeseries_training(trials, nr, ns, nb, n_training)
#Rho[trials//8:7*trials//8] = generate_bandit_timeseries_slowchange(3*trials//4, nr, ns, nb)
#prefix = "mediumslow"
#Rho[:] = generate_bandit_timeseries_training(trials, nr, ns, nb, n_training)
#Rho[trials//4:3*trials//4] = generate_bandit_timeseries_slowchange(trials//2, nr, ns, nb)
#prefix = "slow"
#Rho[:] = generate_bandit_timeseries_training(trials, nr, ns, nb, n_training)
#Rho[trials//3:2*trials//3-1] = generate_bandit_timeseries_slowchange(trials//3, nr, ns, nb)
#prefix = "notsoslow"
#Rho[:] = generate_bandit_timeseries_training(trials, nr, ns, nb, n_training)
#Rho[4*trials//10:6*trials//10] = generate_bandit_timeseries_slowchange(trials//5, nr, ns, nb)
#prefix = "notsosudden"
#Rho[:] = generate_bandit_timeseries_training(trials, nr, ns, nb, n_training)
#Rho[9*trials//20:11*trials//20] = generate_bandit_timeseries_slowchange(trials//10, nr, ns, nb)
#prefix = "almostsudden"
#Rho[:] = generate_bandit_timeseries_training(trials, nr, ns, nb, n_training)
#prefix = "sudden"
#Rho[:trials//2] = generate_bandit_timeseries_training(trials//2, nr, ns, nb, n_training)
#Rho[trials//2:] = generate_bandit_timeseries_slowchange(trials//2, nr, ns, nb)

repetitions = 5

#learn_rew = 21

avg = True

n_training = 1

sigma = 0.001

folder = 'data'

recalc_rho = False

for pl in [0.1,0.3,0.5,0.7,0.9]:
    for rl in [0.1,0.3,0.5,0.7,0.9]:
        # TODO: wht does dt=9 not work?? gives control prob of nan
        for dt in [2., 6.]:#[1.,3.,5.,7.]:
            
            stayed = []
            indices = []
            for tendency in [1, 2, 10, 100]:#[1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]:
                print(pl, rl, dt, tendency)
                tend = array([tendency])
            
                init = array([0.6, 0.4, 0.6, 0.4])
            
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
            
                worlds = []
                l = []
                learn_pol = tend
                learn_habit = True
                pol_lambda = ar.tensor([pl])#0.3
                r_lambda = ar.tensor([rl])#0.6
                dec_temp = ar.tensor([dt])#4.
                l.append([learn_pol, avg, Rho, learn_habit, pol_lambda, r_lambda, dec_temp])
            
                par_list = []
            
                for p in itertools.product(l, utility):
                    par_list.append(p[0]+[p[1]])
            
                par_list = par_list*repetitions
            
                for i, pars in enumerate(par_list):
                    worlds.append(run_agent(pars))
            
                    w = worlds[-1]
            
                    # rewarded = ar.where(w.rewards[:trials-1,-1] == 1)[0]
            
                    # unrewarded = ar.where(w.rewards[:trials-1,-1] == 0)[0]
                    
                    rewarded = w.rewards[:trials-1,-1] == 1
            
                    unrewarded = rewarded==False#w.rewards[:trials-1,-1] == 0
            
                    # TODO: go back to ar.logical_and when on pytorch version 1.5
                    # rare = ar.cat((ar.where(own_logical_and(w.environment.hidden_states[:,1]==2, w.actions[:,0] == 0) == True)[0],
                    #                  ar.where(own_logical_and(w.environment.hidden_states[:,1]==1, w.actions[:,0] == 1) == True)[0]))
                    # rare.sort()
            
                    # common = ar.cat((ar.where(own_logical_and(w.environment.hidden_states[:,1]==2, w.actions[:,0] == 1) == True)[0],
                    #                    ar.where(own_logical_and(w.environment.hidden_states[:,1]==1, w.actions[:,0] == 0) == True)[0]))
                    # common.sort()
                    
                    rare = own_logical_or(own_logical_and(w.environment.hidden_states[:trials-1,1]==2, w.actions[:trials-1,0] == 0),
                                   own_logical_and(w.environment.hidden_states[:trials-1,1]==1, w.actions[:trials-1,0] == 1))
            
                    common = rare==False#own_logical_or(own_logical_and(w.environment.hidden_states[:trials-1,1]==2, w.actions[:trials-1,0] == 1),
                             #        own_logical_and(w.environment.hidden_states[:trials-1,1]==1, w.actions[:trials-1,0] == 0))
            
                    names = ["rewarded common", "rewarded rare", "unrewarded common", "unrewarded rare"]
            
                    # index_list = [ar.intersect1d(rewarded, common), ar.intersect1d(rewarded, rare),
                    #              ar.intersect1d(unrewarded, common), ar.intersect1d(unrewarded, rare)]
                    
                    rewarded_common = ar.where(own_logical_and(rewarded,common) == True)[0]
                    rewarded_rare = ar.where(own_logical_and(rewarded,rare) == True)[0]
                    unrewarded_common = ar.where(own_logical_and(unrewarded,common) == True)[0]
                    unrewarded_rare = ar.where(own_logical_and(unrewarded,rare) == True)[0]
                    
                    index_list = [rewarded_common, rewarded_rare,
                                 unrewarded_common, unrewarded_rare]
            
                    stayed_list = [(w.actions[index_list[i],0] == w.actions[index_list[i]+1,0]).sum()/float(len(index_list[i])) for i in range(4)]
            
                    stayed.append(stayed_list)

                    run_name = "twostage_agent"+str(i)+"_pl"+str(pl)+"_rl"+str(rl)+"_dt"+str(dt)+"_tend"+str(tendency)+".json"
                    fname = os.path.join(folder, run_name)
                    
                    actions = w.actions.numpy()
                    observations = w.observations.numpy()
                    rewards = w.rewards.numpy()
                    states = w.environment.hidden_states.numpy()
                    data = {"actions": actions, "observations": observations, "rewards": rewards, "states": states}
    
                    jsonpickle_numpy.register_handlers()
                    pickled = pickle.encode(data)
                    with open(fname, 'w') as outfile:
                        json.dump(pickled, outfile)
            
                stayed_arr = array(stayed)
                
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

#    stayed_rew = ((w.actions[rewarded,0] - w.actions[rewarded+1,0]) == 0).sum()/len(rewarded)
#
#    stayed_unrew = ((w.actions[unrewarded,0] - w.actions[unrewarded+1,0]) == 0).sum()/len(unrewarded)

                #print(gc.get_count())
            
                pickled = 0
                #worlds = 0
            
                #print(gc.get_count())
            
                gc.collect()
            
                #print(gc.get_count())




# plt.figure()
# plt.plot(w.agent.action_selection.RT[:,0])
# plt.show()
"""
reload data
"""
#with open(fname, 'r') as infile:
#    data = json.load(infile)
#
#w_new = pickle.decode(data)

"""
parallelized calls for multiple runs
"""
#if len(par_list) < 8:
#    num_threads = len(par_list)
#else:
#    num_threads = 7
#
#pool = Pool(num_threads)
#
#pool.map(run_agent, par_list)
