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
tests = ["conflict", "agreement", "goal", "habit", "uncertainty"]
num_tests = len(tests)
test_vals = []

# test values
vals = [0.8, 0.2]
conflict = vals.copy()
conflict.reverse()
flat = [0.5, 0.5]

# test function
def run_action_selection(post, prior, like, trials = 100):

    ac_sel = asl.DirichletSelector(trials, 2, len(vals))
    for t in range(trials):
        ac_sel.select_desired_action(t, 0, post, [0,1], like, prior)

    return ac_sel.RT.squeeze()

# set up number of trials
trials = 1000

# conflict
prior = np.array(vals)
like = np.array(conflict)
post = prior*like
post /= post.sum()
test_vals.append([post,prior,like])

# agreement
prior = np.array(vals)
like = np.array(vals)
post = prior*like
post /= post.sum()
test_vals.append([post,prior,like])

# goal
prior = np.array(flat)
like = np.array(vals)
post = prior*like
post /= post.sum()
test_vals.append([post,prior,like])

# habit
prior = np.array(vals)
like = np.array(flat)
post = prior*like
post /= post.sum()
test_vals.append([post,prior,like])

# uncertainty
prior = np.array(flat)
like = np.array(flat)
post = prior*like
post /= post.sum()
test_vals.append([post,prior,like])

RT = np.zeros((num_tests, trials))
for i, test in enumerate(tests):
    post, prior, like = test_vals[i]
    RT[i] = run_action_selection(post, prior, like, trials)
RT = pd.DataFrame(data=RT.T, columns=tests)

# plt.figure()
# for i, test in enumerate(tests):
#     plt.plot(RT[i],label=test)
# plt.legend()
# plt.show()

plt.figure()
sns.histplot(RT, bins=60, alpha=0.5)
plt.savefig('RT_tests_histogram_'+str(trials)+'trials.svg',dpi=600)
plt.show()