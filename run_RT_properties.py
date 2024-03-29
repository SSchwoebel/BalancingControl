#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 15:10:10 2020

@author: sarah
"""

import numpy as np
import matplotlib.pylab as plt
import action_selection as asl
import seaborn as sns
import pandas as pd
from scipy.stats import entropy
from scipy.stats.mstats import normaltest
#sns.set_style("whitegrid", {"axes.edgecolor": "0.15"})#, "axes.spines.top": "False", "axes.spines.right": "False"})
#plt.style.use('seaborn-darkgrid')
plt.style.use('seaborn-whitegrid')

# set up test
tests = ["conflict", "agreement", "goal", "habit"]#, "uncertainty"]
num_tests = len(tests)
test_vals = []

# test values
"""
vals = [0.8, 0.2]
conflict = vals.copy()
conflict.reverse()
"""
gp = 6
n = 81
val = 0.16
l = [val]*gp+[(1-6*val)/(n-gp)]*(n-gp)
v = 0.0055
p = [(1-(v*(n-gp)))/6]*gp+[v]*(n-gp)
conflict = [v]*gp+[(1-(v*(n-gp)))/6]*gp+[v]*(n-2*gp)
npi = n
flat = [1./npi]*npi

plt.figure()
plt.plot(range(1,npi+1), l, label='likelihood, habit, goal', linewidth=3)
plt.plot(range(1,npi+1), p, label='prior agreement', linewidth=3)
plt.plot(range(1,npi+1), conflict, label='prior conflict', linewidth=3)
plt.plot(range(1,npi+1), flat, label='flat', linewidth=3)
plt.ylim([0,0.25])
plt.legend()
plt.xlim([1,npi])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('policy', fontsize=16)
plt.ylabel('probability', fontsize=16)
plt.savefig('underlying_prior_like_for_distributions.svg')
plt.show()

# test function
def run_action_selection(post, prior, like, trials = 100, crit_factor = 0.5, calc_dkl = False):

    ac_sel = asl.DirichletSelector(trials, 2, npi, factor=crit_factor, calc_dkl=calc_dkl)
    for t in range(trials):
        ac_sel.select_desired_action(t, 0, post, list(range(npi)), like, prior)

    if calc_dkl:
        return ac_sel.RT.squeeze(), ac_sel.DKL_post.squeeze(), ac_sel.DKL_prior.squeeze()
    else:
        return ac_sel.RT.squeeze()

# set up number of trials
trials = 100

# conflict
prior = np.array(conflict)
like = np.array(l)
post = prior*like
post /= post.sum()
test_vals.append([post,prior,like])

# agreement
prior = np.array(p)
like = np.array(l)
post = prior*like
post /= post.sum()
test_vals.append([post,prior,like])

# goal
prior = np.array(flat)
like = np.array(l)
post = prior*like
post /= post.sum()
test_vals.append([post,prior,like])

# habit
prior = np.array(l)
like = np.array(flat)
post = prior*like
post /= post.sum()
test_vals.append([post,prior,like])

# uncertainty
# prior = np.array(flat)
# like = np.array(flat)
# post = prior*like
# post /= post.sum()
# test_vals.append([post,prior,like])


def plot_RT_distributions(num_tests, trials, test_vals, crit_factor=0.4):

    RT = np.zeros((num_tests, trials))
    for i, test in enumerate(tests):
        post, prior, like = test_vals[i]
        RT[i] = run_action_selection(post, prior, like, trials, crit_factor=crit_factor)
        is_normal = normaltest(RT[i])
        is_lognormal = normaltest(np.log(RT[i]))
        print(crit_factor, test)
        print("is normal?", is_normal)
        print("is log normal?", is_lognormal)
    RT = pd.DataFrame(data=RT.T, columns=tests)

    #plt.figure()
    # for i, test in enumerate(tests):
    #     plt.figure()
    #     plt.hist(RT[test],label=test)
    #     plt.show()
    #plt.legend()
    #lt.show()

    max_RT = np.amax(RT.values)
    min_RT = np.amin(RT.values)

    plt.figure()
    sns.histplot(RT[['conflict', 'agreement']], alpha=0.5, bins=50, binrange=[min_RT,max_RT],edgecolor='black')#, common_bins=False)#
    plt.savefig('RT_tests_histogram_conflict_agreement_'+str(crit_factor)+'factor_'+str(npi)+'npi_'+str(trials)+'trials.svg',dpi=600)
    plt.xlim(min_RT,max_RT)
    plt.ylim([0,trials*0.9])
    plt.xlabel('RT (#samples)')
    plt.show()

    plt.figure()
    sns.histplot(RT[["goal", "habit"]], alpha=0.5, bins=50, binrange=[min_RT,max_RT],edgecolor='black')#, common_bins=False)#
    plt.savefig('RT_tests_histogram_goal_habit_'+str(crit_factor)+'factor_'+str(npi)+'npi_'+str(trials)+'trials.svg',dpi=600)
    plt.xlim(min_RT,max_RT)
    plt.ylim([0,trials*0.9])
    plt.xlabel('RT (#samples)')
    plt.show()

    plt.figure()
    sns.histplot(RT, alpha=0.6, bins=50, binrange=[min_RT,max_RT],edgecolor='black')#, common_bins=False)#
    plt.savefig('RT_tests_histogram_all_'+str(crit_factor)+'factor_'+str(npi)+'npi_'+str(trials)+'trials.svg',dpi=600)
    plt.xlim(min_RT,max_RT)
    plt.ylim([0,trials*0.9])
    plt.xlabel('RT (#samples)')
    plt.show()

    if True:#max_RT < 0.9*trials:
        RT.to_pickle('RT_'+str(crit_factor)+'factor_'+str(npi)+'npi_'+str(trials)+'trials.pkl')

def plot_common_histogram(factors, trials):

    frames = [pd.read_pickle('RT_'+str(f)+'factor_'+str(npi)+'npi_'+str(trials)+'trials.pkl') for f in factors]
    frame = pd.concat(frames)

    max_RT = np.amax(frame.values)
    min_RT = np.amin(frame.values)

    frame['factor'] = np.repeat(factors, trials)

    bins = 50

    plt.figure()
    sns.histplot(frame[['conflict', 'agreement']], alpha=0.5, bins=bins, binrange=[min_RT,max_RT],edgecolor='black')#, common_bins=False)#
    # for f in frames:
    #     sns.histplot(f[['conflict', 'agreement']], alpha=0.5, bins=100, binrange=[min_RT,max_RT],edgecolor='black', common_bins=True)#
    plt.xlim(0,1000)
    plt.ylim([0,trials+100])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('RT (#samples)', fontsize=16)
    plt.ylabel('Count', fontsize=16)
    plt.savefig('RT_tests_histogram_conflict_agreement_all_'+str(npi)+'npi_'+str(trials)+'trials.svg',dpi=600)
    plt.show()

    plt.figure()
    sns.histplot(frame[["goal", "habit"]], alpha=0.5, bins=bins, binrange=[min_RT,max_RT],edgecolor='black')#, common_bins=False)#
    plt.xlim(0,1000)
    plt.ylim([0,trials+100])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('RT (#samples)', fontsize=16)
    plt.ylabel('Count', fontsize=16)
    plt.savefig('RT_tests_histogram_goal_habit_all_'+str(npi)+'npi_'+str(trials)+'trials.svg',dpi=600)
    plt.show()

    plt.figure()
    sns.histplot(frame, alpha=0.6, bins=bins, binrange=[min_RT,max_RT],edgecolor='black')#, common_bins=False)#
    plt.xlim(0,1000)
    plt.ylim([0,trials+100])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('RT (#samples)', fontsize=16)
    plt.ylabel('Count', fontsize=16)
    plt.savefig('RT_tests_histogram_all_all_'+str(npi)+'npi_'+str(trials)+'trials.svg',dpi=600)
    plt.show()

def evaluate_DKL(num_tests, trials, conflict):
    factors = np.arange(0.05,0.75,0.05)
    num_factors = len(factors)
    RT = np.zeros((num_factors,trials))
    DKL = np.zeros((3,num_factors*trials*2))
    DKL[1,:num_factors*trials] = 0
    DKL[1,num_factors*trials:] = 1
    post, prior, like = conflict

    for i,f in enumerate(factors):
        RT[i], DKL[2,i*trials:(i+1)*trials], \
        DKL[2,i*trials+num_factors*trials:(i+1)*trials+num_factors*trials] = \
            run_action_selection(post, prior, like, trials, crit_factor=f, calc_dkl=True)
        DKL[0,i*trials:(i+1)*trials] = f
        DKL[0,i*trials+num_factors*trials:(i+1)*trials+num_factors*trials] = f

    DKL_df = pd.DataFrame(data=DKL.T, columns=['factor', 'type', 'DKL'])

    plt.figure()
    sns.lineplot(data=DKL_df, x='factor', y='DKL', style='type', ci=95, linewidth=2)
    plt.xlim([factors[0], factors[-1]])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('factor $f$', fontsize=16)
    plt.ylabel('$D_{KL}$', fontsize=16)
    #plt.ylim([0,1.7])
    plt.savefig('dkl_threshold_factor_'+str(npi)+'npi_'+str(trials)+'trials.svg', dpi=600)
    plt.show()


def RT_of_like_entropy(trials):
    pass

# l_values = np.arange(1./npi, 0.16334205362982337, 0.01)
# num_tests = len(l_values)
# likes = np.array([[l]*gp+[(1-l*gp)/(n-gp)]*(n-gp) for l in l_values])
# prior = flat
# crit_factor = 0.5
# entropies = [entropy(l) for l in likes ]

# RT = np.zeros((2,num_tests*trials))
# for i, l in enumerate(likes):
#     post = l
#     like = l
#     RT[0, i*trials:(i+1)*trials] = run_action_selection(post, prior, like, trials, crit_factor=crit_factor)
#     RT[1, i*trials:(i+1)*trials] = entropies[i]
# RT_df = pd.DataFrame(data=RT.T, columns=['RT','entropy'])

# plt.figure()
# sns.lineplot(data=RT_df, x='entropy', y='RT')
# plt.title('mean RT as a function of likelihood entropy')
# plt.savefig('RT_like_entropy.svg')
# plt.show()

# RT = np.zeros((num_tests, trials))
# for i, l in enumerate(likes):
#     post = l
#     like = l
#     RT[i] = run_action_selection(post, prior, like, trials, crit_factor=crit_factor)
# RT = pd.DataFrame(data=RT.T, columns=entropies)

# plt.figure()
# sns.histplot(RT, bins=50)
# plt.show()

# means = [RT[e].mean() for e in entropies]
# means.reverse()
# e_r  = entropies.copy()
# e_r.reverse()
# plt.figure()
# plt.plot(means)
# plt.xticks(ticks=range(len(e_r)), labels=np.around(e_r, decimals=2))
# plt.title('mean RT as a function of likelihood entropy')
# plt.show()

#evaluate_DKL(num_tests, trials, test_vals[0])

factors = [0.1,0.3,0.5]
for f in factors:
    plot_RT_distributions(num_tests, trials, test_vals, f)
plot_common_histogram(factors, trials)




