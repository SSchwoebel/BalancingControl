#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 11:13:34 2021

@author: sarah
"""


import numpy as np
import matplotlib.pylab as plt
import action_selection as asl
import seaborn as sns
import pandas as pd
from misc import *
#sns.set_style("whitegrid", {"axes.edgecolor": "0.15"})#, "axes.spines.top": "False", "axes.spines.right": "False"})
#plt.style.use('seaborn-darkgrid')
plt.style.use('seaborn-whitegrid')

# test function
def run_action_selection(post, prior, like, trials = 100, crit_factor = 0.5):

    ac_sel = asl.DirichletSelector(trials, 2, npi, factor=crit_factor)
    samples = []
    for t in range(trials):
        ac_sel.select_desired_action(t, 0, post, list(range(npi)), like, prior)
        RT = int(ac_sel.RT[t,0])
        samples.append(list(ac_sel.accepted_pis[:RT]))

    return ac_sel.RT.squeeze().astype(int), samples


def show_sampling_trajectory(samples, RT):
    
    Dir_counts = np.zeros((RT+1,npi)) + 1
    estimated_q = np.zeros((RT+1,npi)) + 1./npi
    for i,s in enumerate(samples):
        Dir_counts[i+1,:] = Dir_counts[i,:]
        Dir_counts[i+1,s] += 1
        
        estimated_q[i+1] = Dir_counts[i+1] / Dir_counts[i+1].sum()
        
    plt.figure()
    plt.plot(estimated_q[:,0])
    plt.plot([post[0]]*RT, '--', color='gray')
    #plt.plot([post[1]]*RT, '--', color='gray')
    plt.ylim([0,1])
    plt.ylabel('q(a)',fontsize=16)
    plt.savefig('sampling_run.svg')
    plt.savefig('sampling_run.png')
    plt.show()
    
    plt.figure()
    logit = ln(estimated_q[:,0]/(1-estimated_q[:,0]))
    logit_true = ln(post[0] / (1-post[0]))
    plt.plot(logit)
    plt.plot([logit_true]*RT, '--', color='gray')
    #plt.ylim([0,1])
    plt.ylabel('logit $q(a_1)$',fontsize=16)
    plt.savefig('logit_sampling_run.svg')
    plt.savefig('logit_sampling_run.png')
    plt.show()
    
    plt.figure()
    plt.plot(ln(estimated_q))
    #plt.ylim([0,1])
    plt.show()


npi = 81
l_val = 0.16
p_val = 0.1

prior = np.zeros(npi)
prior[:] = (1-p_val) / (npi-1)
prior[1] = p_val
like = np.zeros(npi)
like[0] = l_val
like[1:] = (1-l_val) / (npi-1)

post = prior*like
post /= post.sum()

trials = 2

RTs, all_samples = run_action_selection(post, prior, like, trials)

ind = 0

samples = all_samples[ind]
RT = RTs[ind]

show_sampling_trajectory(samples, RT)