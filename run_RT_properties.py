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

# test function
def run_action_selection(post, prior, like, trials = 100):

    ac_sel = asl.DirichletSelector(trials, 2, npi)
    for t in range(trials):
        ac_sel.select_desired_action(t, 0, post, list(range(npi)), like, prior)

    return ac_sel.RT.squeeze()

# set up number of trials
trials = 1000

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

RT = np.zeros((num_tests, trials))
for i, test in enumerate(tests):
    post, prior, like = test_vals[i]
    RT[i] = run_action_selection(post, prior, like, trials)
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
sns.histplot(RT[['conflict', 'agreement']], alpha=0.5, bins=50, binrange=[min_RT,max_RT])#, common_bins=False)#
plt.savefig('RT_tests_histogram_conflict_agreement_'+str(npi)+'npi_'+str(trials)+'trials.svg',dpi=600)
plt.xlim(min_RT,max_RT)
plt.ylim([0,trials*0.9])
plt.show()

plt.figure()
sns.histplot(RT[["goal", "habit"]], alpha=0.5, bins=50, binrange=[min_RT,max_RT])#, common_bins=False)#
plt.savefig('RT_tests_histogram_goal_habit_'+str(npi)+'npi_'+str(trials)+'trials.svg',dpi=600)
plt.xlim(min_RT,max_RT)
plt.ylim([0,trials*0.9])
plt.show()

plt.figure()
sns.histplot(RT, alpha=0.6, bins=50, binrange=[min_RT,max_RT])#, common_bins=False)#
plt.savefig('RT_tests_histogram_all_'+str(npi)+'npi_'+str(trials)+'trials.svg',dpi=600)
plt.xlim(min_RT,max_RT)
plt.ylim([0,trials*0.9])
plt.show()