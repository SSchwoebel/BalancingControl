#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 14:09:11 2021

@author: sarah
"""


import torch as ar
array = ar.tensor

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


###################################
"""load data"""

i = 0
pl = 0.1
rl = 0.9
dt = 1.

folder = "data"

run_name = "twostage_agent"+str(i)+"_pl"+str(pl)+"_rl"+str(rl)+"_dt"+str(dt)+"_tend1.json"
fname = os.path.join(folder, run_name)

jsonpickle_numpy.register_handlers()
    
with open(fname, 'r') as infile:
    loaded = json.load(infile)

data = pickle.decode(loaded)


###################################
"""experiment parameters"""

trials =  300#number of trials
T = 3 #number of time steps in each trial
nb = 4
ns = 3+nb #number of states
no = ns #number of observations
na = 2 #number of actions
npi = na**(T-1)
nr = 2
nc = 1

learn_pol=1
learn_habit=True

learn_rew = 1

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

C_alphas = ar.zeros((nr, ns, nc)) + learn_rew
C_alphas[0,:3,:] = 100
for i in range(1,nr):
    C_alphas[i,0,:] = 1
#    C_alphas[0,1:,:] = 100
#    for c in range(nb):
#        C_alphas[1,c+1,c] = 100
#        C_alphas[0,c+1,c] = 1
#C_alphas[:,13] = [100, 1]

#C_agent = ar.zeros((nr, ns, nc))
# for c in range(nc):
#     C_agent[:,:,c] = array([(C_alphas[:,i,c])/(C_alphas[:,i,c]).sum() for i in range(ns)]).T
C_agent = C_alphas[:,:,:] / C_alphas[:,:,:].sum(axis=0)[None,:,:]
#array([ar.random.dirichlet(C_alphas[:,i]) for i in range(ns)]).T

# context transition matrix

transition_matrix_context = ar.ones(1)


"""
create policies
"""

pol = array(list(itertools.product(list(range(na)), repeat=T-1)))

#pol = pol[-2:]
npi = pol.shape[0]

# prior over policies

prior_pi = ar.ones(npi)/npi #ar.zeros(npi) + 1e-3/(npi-1)
#prior_pi[170] = 1. - 1e-3
alphas = ar.zeros((npi, nc)) + learn_pol
#    for i in range(nb):
#        alphas[i+1,i] = 100
#alphas[170] = 100
prior_pi = alphas / alphas.sum(axis=0)


"""
set state prior (where agent thinks it starts)
"""

state_prior = ar.zeros((ns))

state_prior[0] = 1.

prior_context = array([1.])

#    prior_context[0] = 1.

"""
set up agent
"""
#bethe agent

pol_par = alphas

# perception
bayes_prc = prc.HierarchicalPerception(A, B, C_agent, transition_matrix_context, 
                                       state_prior, utility, prior_pi, 
                                       pol_par, C_alphas, T=T,
                                       pol_lambda=0, r_lambda=0,
                                       non_decaying=3, dec_temp=1)

bayes_pln = agt.BayesianPlanner(bayes_prc, [], pol,
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