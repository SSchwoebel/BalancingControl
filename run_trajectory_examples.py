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
import os
#sns.set_style("whitegrid", {"axes.edgecolor": "0.15"})#, "axes.spines.top": "False", "axes.spines.right": "False"})
#plt.style.use('seaborn-darkgrid')
plt.style.use('seaborn-whitegrid')

folder = '/home/sarah/Nextcloud/RT_paper/RT_sampling_figs/'

# test function
def run_action_selection(post, prior, like, trials = 100, crit_factor = 0.5):

    ac_sel = asl.DirichletSelector(trials, 2, npi, factor=crit_factor)
    samples = []
    for t in range(trials):
        ac_sel.select_desired_action(t, 0, post, list(range(npi)), like, prior)
        RT = int(ac_sel.RT[t,0])
        samples.append(list(ac_sel.accepted_pis[:RT]))

    return ac_sel.RT.squeeze().astype(int), samples


def show_sampling_trajectory(samples, RT, npi, j=0, crit_factor = 0.5):
    print(RT)
    Dir_counts = np.zeros((RT+1,npi)) + 1
    estimated_q = np.zeros((RT+1,npi)) + 1./npi
    for i,s in enumerate(samples):
        Dir_counts[i+1,:] = Dir_counts[i,:]
        Dir_counts[i+1,s] += 1

        estimated_q[i+1] = Dir_counts[i+1] / Dir_counts[i+1].sum()

    plt.figure()
    plt.plot([post[0]]*(6000), '--', color='gray', label='true posterior', linewidth=3)
    plt.plot(estimated_q[:,0], label='sampled posterior', linewidth=3)
    #plt.plot([post[1]]*RT, '--', color='gray')
    plt.xlim([0,6000])
    plt.ylim([0,1])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel('q(a)',fontsize=16)
    plt.xlabel('sample number',fontsize=16)
    plt.legend(fontsize=14, loc=4)
    plt.savefig(os.path.join(folder,'sampling_run_npi'+str(npi)+'_f'+str(f)+'_'+str(j)+'.svg'))
    plt.savefig(os.path.join(folder,'sampling_run_npi'+str(npi)+'_f'+str(f)+'_'+str(j)+'.png'))
    plt.show()
    
    plt.figure()
    plt.plot([post[0]]*(6000), '--', color='gray', label='true posterior action 1', linewidth=3)
    plt.plot(estimated_q[:,0], label='sampled posterior action 1', linewidth=3)
    plt.plot(estimated_q[:,1:], linewidth=3)#, label='sampled posterior other actions'
    #plt.plot([post[1]]*RT, '--', color='gray')
    plt.xlim([0,6000])
    plt.ylim([0,1])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel('q(a)',fontsize=16)
    plt.xlabel('sample number',fontsize=16)
    plt.legend(fontsize=14, loc=4)
    plt.savefig(os.path.join(folder,'sampling_run_all_npi'+str(npi)+'_f'+str(f)+'_'+str(j)+'.svg'))
    plt.savefig(os.path.join(folder,'sampling_run_all_npi'+str(npi)+'_f'+str(f)+'_'+str(j)+'.png'))
    plt.show()

    plt.figure()
    logit_true = ln(post[0] / (1-post[0]))
    logit = ln(estimated_q[:,0]/(1-estimated_q[:,0]))
    plt.plot(logit, linewidth=3)
    plt.plot([logit_true]*(6000), '--', color='gray', linewidth=3)
    #plt.ylim([0,1])
    plt.xlim([0,6000])#RT+1])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14, loc=4)
    plt.ylabel('logit $q(a_1)$',fontsize=16)
    plt.xlabel('sample number',fontsize=16)
    plt.savefig(os.path.join(folder,'logit_sampling_run_npi'+str(npi)+'_f'+str(f)+'_'+str(j)+'.svg'))
    plt.savefig(os.path.join(folder,'logit_sampling_run_npi'+str(npi)+'_f'+str(f)+'_'+str(j)+'.png'))
    plt.show()

    plt.figure()
    plt.plot(ln(estimated_q))
    #plt.ylim([0,1])
    plt.xlim([0,RT+1])
    plt.ylabel('logit all $q$',fontsize=16)
    plt.show()


npi = 81
l_val = 0.525
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

f = 0.5

for i in range(20):

    RTs, all_samples = run_action_selection(post, prior, like, trials, crit_factor=f)

    ind = 0

    samples = all_samples[ind]
    RT = RTs[ind]

    show_sampling_trajectory(samples, RT, npi, i, crit_factor=f)


# npi = 2
# l_val = 0.8
# p_val = 0.6

# prior = np.zeros(npi)
# prior[:] = (1-p_val) / (npi-1)
# prior[1] = p_val
# like = np.zeros(npi)
# like[0] = l_val
# like[1:] = (1-l_val) / (npi-1)

# post = prior*like
# post /= post.sum()

# trials = 2

# f = 1.5

# for i in range(20):

#     RTs, all_samples = run_action_selection(post, prior, like, trials, crit_factor=f)

#     ind = 0

#     samples = all_samples[ind]
#     RT = RTs[ind]

#     show_sampling_trajectory(samples, RT, npi, i, crit_factor=f)