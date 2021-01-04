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
n = 128
l = [0.16334205362982337]*gp+[0.00016350555918901252]*(n-gp)
p = [(1-(0.00327*(n-gp)))/6]*gp+[0.00327]*(n-gp)
conflict = [.00327]*gp+[(1-(0.00327*(n-gp)))/6]*gp+[.00327]*(n-2*gp)
npi = n
flat = [1./npi]*npi

plt.figure()
plt.plot(l, label='likelihood, habit, goal')
plt.plot(p, label='prior agreement')
plt.plot(conflict, label='prior conflict')
plt.ylim([0,0.2])
plt.legend()
plt.xlim([0,npi])
plt.xlabel('policy')
plt.ylabel('probability')
plt.savefig('underlying_prior_like_for_distributions.svg')
plt.show()

# test function
def run_action_selection(post, prior, like, trials = 100, crit_factor = 0.4, calc_dkl = False):

    ac_sel = asl.DirichletSelector(trials, 2, npi, calc_dkl=calc_dkl)
    for t in range(trials):
        ac_sel.select_desired_action(t, 0, post, list(range(npi)), like, prior, factor=crit_factor)

    if calc_dkl:
        return ac_sel.RT.squeeze(), ac_sel.DKL_post.squeeze(), ac_sel.DKL_prior.squeeze()
    else:
        return ac_sel.RT.squeeze()

# set up number of trials
trials = 5000

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

    if max_RT < 0.9*trials:
        RT.to_pickle('RT_'+str(crit_factor)+'factor_'+str(npi)+'npi_'+str(trials)+'trials.pkl')

def plot_common_histogram(factors, trials):

    frames = [pd.read_pickle('RT_'+str(f)+'factor_'+str(npi)+'npi_'+str(trials)+'trials.pkl') for f in factors]
    frame = pd.concat(frames)

    max_RT = np.amax(frame.values)
    min_RT = np.amin(frame.values)

    frame['factor'] = np.repeat(factors, trials)

    bins = 155

    plt.figure()
    sns.histplot(frame[['conflict', 'agreement']], alpha=0.5, bins=bins, binrange=[min_RT,max_RT],edgecolor='black')#, common_bins=False)#
    # for f in frames:
    #     sns.histplot(f[['conflict', 'agreement']], alpha=0.5, bins=100, binrange=[min_RT,max_RT],edgecolor='black', common_bins=True)#
    plt.savefig('RT_tests_histogram_conflict_agreement_all_'+str(npi)+'npi_'+str(trials)+'trials.svg',dpi=600)
    plt.xlim(0,1600)
    plt.ylim([0,trials])
    plt.xlabel('RT (#samples)')
    plt.show()

    plt.figure()
    sns.histplot(frame[["goal", "habit"]], alpha=0.5, bins=bins, binrange=[min_RT,max_RT],edgecolor='black')#, common_bins=False)#
    plt.savefig('RT_tests_histogram_goal_habit_all_'+str(npi)+'npi_'+str(trials)+'trials.svg',dpi=600)
    plt.xlim(0,1600)
    plt.ylim([0,trials])
    plt.xlabel('RT (#samples)')
    plt.show()

    plt.figure()
    sns.histplot(frame, alpha=0.6, bins=bins, binrange=[min_RT,max_RT],edgecolor='black')#, common_bins=False)#
    plt.savefig('RT_tests_histogram_all_all_'+str(npi)+'npi_'+str(trials)+'trials.svg',dpi=600)
    plt.xlim(0,1600)
    plt.ylim([0,trials])
    plt.xlabel('RT (#samples)')
    plt.show()

def evaluate_DKL(num_tests, trials, conflict):
    factors = np.arange(0.05,0.65,0.05)
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
    plt.savefig('dkl_threshold_factor_'+str(npi)+'npi_'+str(trials)+'trials.svg', dpi=600)
    plt.show()


#evaluate_DKL(num_tests, trials, test_vals[0])
#plot_RT_distributions(num_tests, trials, test_vals, 0.1)

factors = [0.1,0.3,0.5]
plot_common_histogram(factors, trials)




