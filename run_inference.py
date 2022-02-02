#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 14:09:11 2021

@author: sarah
"""
import jax.numpy as jnp
import jax.scipy.special as scs
import numpyro as pyro
import numpyro.distributions as dist
import jax.scipy as sc
import jax.scipy.signal as ss

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
import bottleneck as bn
import gc

#device = jnp.device("cuda") if jnp.cuda.is_available() else jnp.device("cpu")
#device = jnp.device("cuda")
#device = jnp.device("cpu")

#from inference_twostage import device

#jnp.autograd.set_detect_anomaly(True)
###################################
"""load data"""

i = 0
pl = 0.3
rl = 0.7
dt = 5.
tend = 1

folder = "data"

run_name = "twostage_agent"+str(i)+"_pl"+str(pl)+"_rl"+str(rl)+"_dt"+str(dt)+"_tend"+str(tend)+".json"
fname = os.path.join(folder, run_name)

jsonpickle_numpy.register_handlers()
    
with open(fname, 'r') as infile:
    loaded = json.load(infile)

data_load = pickle.decode(loaded)

data = {}
data["actions"] = jnp.array(data_load["actions"])#.to(device)
data["rewards"] = jnp.array(data_load["rewards"])#.to(device)
data["observations"] = jnp.array(data_load["observations"])#.to(device)


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

learn_pol=1000
learn_habit=True

learn_rew = 1

utility = []

#ut = [0.5, 0.6, 0.7, 0.8, 0.9, 1. - 1e-5]
#ut = [0.95, 0.96, 0.98, 0.99]
#ut = [0.985]
ut = [0.999]
for u in ut:
    utility.append(jnp.zeros(nr))#.to(device))
    for i in range(1,nr):
        utility[-1].at[i].set(u/(nr-1))#u/nr*i
    utility[-1].at[0].set(1.-u)
    
utility = utility[-1]

"""
create matrices
"""


#generating probability of observations in each state
A = jnp.eye(no)#.to(device)


#state transition generative probability (matrix)
B = jnp.zeros((ns, ns, na))#.to(device)
b1 = 0.7
nb1 = 1.-b1
b2 = 0.7
nb2 = 1.-b2

B.at[:,:,0].set(jnp.array([[  0,  0,  0,  0,  0,  0,  0,],
                     [ b1,  0,  0,  0,  0,  0,  0,],
                     [nb1,  0,  0,  0,  0,  0,  0,],
                     [  0,  1,  0,  1,  0,  0,  0,],
                     [  0,  0,  1,  0,  1,  0,  0,],
                     [  0,  0,  0,  0,  0,  1,  0,],
                     [  0,  0,  0,  0,  0,  0,  1,],]))

B.at[:,:,1].set(jnp.array([[  0,  0,  0,  0,  0,  0,  0,],
                     [nb2,  0,  0,  0,  0,  0,  0,],
                     [ b2,  0,  0,  0,  0,  0,  0,],
                     [  0,  0,  0,  1,  0,  0,  0,],
                     [  0,  0,  0,  0,  1,  0,  0,],
                     [  0,  1,  0,  0,  0,  1,  0,],
                     [  0,  0,  1,  0,  0,  0,  1,],]))

# create reward generation
#
#    C = jnp.zeros((utility.shape[0], ns))
#
#    vals = jnp.array([0., 1./5., 0.95, 1./5., 1/5., 1./5.])
#
#    for i in range(ns):
#        C[:,i] = [1-vals[i],vals[i]]
#
#    changes = jnp.array([0.01, -0.01])
#    Rho = generate_bandit_timeseries(C, nb, trials, changes)

# agent's beliefs about reward generation

C_alphas = jnp.zeros((nr, ns))#.to(device) 
C_alphas += learn_rew
C_alphas.at[0,:3].set(100)
for i in range(1,nr):
    C_alphas.at[i,0].set(1)
#    C_alphas[0,1:,:] = 100
#    for c in range(nb):
#        C_alphas[1,c+1,c] = 100
#        C_alphas[0,c+1,c] = 1
#C_alphas[:,13] = [100, 1]

#C_agent = jnp.zeros((nr, ns, nc))
# for c in range(nc):
#     C_agent[:,:,c] = jnp.array([(C_alphas[:,i,c])/(C_alphas[:,i,c]).sum() for i in range(ns)]).T
C_agent = C_alphas[:,:] / C_alphas[:,:].sum(axis=0)[None,:]
#jnp.array([jnp.random.dirichlet(C_alphas[:,i]) for i in range(ns)]).T

# context transition matrix

transition_matrix_context = jnp.ones(1)#.to(device)


"""
create policies
"""

pol = jnp.array(list(itertools.product(list(range(na)), repeat=T-1)))#.to(device)

#pol = pol[-2:]
npi = pol.shape[0]

# prior over policies

prior_pi = jnp.ones(npi)#.to(device)
prior_pi /= npi #jnp.zeros(npi) + 1e-3/(npi-1)
#prior_pi[170] = 1. - 1e-3
alphas = jnp.zeros((npi))#.to(device) 
alphas += learn_pol
alpha_0 = jnp.array([learn_pol])#.to(device)
#    for i in range(nb):
#        alphas[i+1,i] = 100
#alphas[170] = 100
prior_pi = alphas / alphas.sum()


"""
set state prior (where agent thinks it starts)
"""

state_prior = jnp.zeros((ns))#.to(device)

state_prior.at[0].set(1.)

prior_context = jnp.array([1.])#.to(device)

#    prior_context[0] = 1.

"""
set up agent
"""
#bethe agent

pol_par = alphas

# perception
bayes_prc = prc.FittingPerception(A, B, C_agent, transition_matrix_context, 
                                       state_prior, utility, prior_pi, pol,
                                       alpha_0, C_alphas, T=T, trials=trials,
                                       pol_lambda=0, r_lambda=0,
                                       non_decaying=3, dec_temp=1)

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

inferrer = inf.SingleInference(agent, data)

loss, param_dict = inferrer.infer_posterior(iter_steps=200, num_particles=200)

plt.figure()
plt.title("ELBO")
plt.plot(loss)
plt.ylabel("ELBO")
plt.xlabel("iteration")
plt.show()

inferrer.plot_posteriors()

print("this is inference for pl =", pl, "rl =", rl, "dt =", dt, "tend=", tend)
# print(param_dict)