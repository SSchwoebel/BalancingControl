#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 17:56:24 2017

@author: sarah
"""

import numpy as np
from misc import *
import world
import environment as env
import agent as agt
import perception as prc
import action_selection as asl
import itertools
import matplotlib.pylab as plt
from multiprocessing import Pool
from matplotlib.colors import LinearSegmentedColormap
import jsonpickle as pickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import json
import seaborn as sns
import os
import pandas as pd
import gc
np.set_printoptions(threshold = 100000, precision = 5)
plt.style.use('seaborn-whitegrid')


"""
set parameters
"""

agent = 'bethe'
#agent = 'meanfield'

save_data = False
data_folder = os.path.join('/home/yourname/yourproject','data_folder')

trials = 200 #number of trials
T = 5 #number of time steps in each trial
L = 4 #grid length
no = L**2 #number of observations
ns = L**2 #number of states
na = 4 #number of actions
npi = na**(T-1)
nr = 2
nc = 1
actions = np.array([[-1,0],[0,-1], [1,0], [0,1]])

"""
run function
"""
def run_agent(par_list, trials=trials, T=T, L = L, ns=ns, na=na):

    #set parameters:
    #obs_unc: observation uncertainty condition
    #state_unc: state transition uncertainty condition
    #goal_pol: evaluate only policies that lead to the goal
    #utility: goal prior, preference p(o)
    obs_unc, state_unc, goal_pol, avg, utility, h = par_list


    """
    create matrices
    """

    vals = np.array([1., 2/3., 1/2., 1./2.])

    #generating probability of observations in each state
    A = np.eye(ns)

    #generate horizontal gradient for observation uncertainty condition
    if obs_unc:

        condition = 'obs'

        for s in range(ns):
            x = s//L
            y = s%L

            c = vals[L - y - 1]

            # look for neighbors
            neighbors = []
            if (s-4)>=0 and (s-4)!=11:
                neighbors.append(s-4)

            if (s%4)!=0 and (s-1)!=11:
                neighbors.append(s-1)

            if (s+4)<=(ns-1) and (s+4)!=11:
                neighbors.append(s+4)

            if ((s+1)%4)!=0 and (s+1)!=11:
                neighbors.append(s+1)

            A[s,s] = c
            for n in neighbors:
                A[n,s] = (1-c)/len(neighbors)


    #state transition generative probability (matrix)
    B = np.zeros((ns, ns, na))

    cert_arr = np.zeros(ns)
    for s in range(ns):
        x = s//L
        y = s%L

        #state uncertainty condition
        if state_unc:
            if (x==0) or (y==3):
                c = vals[0]
            elif (x==1) or (y==2):
                c = vals[1]
            elif (x==2) or (y==1):
                c = vals[2]
            else:
                c = vals[3]

            condition = 'state'

        else:
            c = 1.

        cert_arr[s] = c
        for u in range(na):
            x = s//L+actions[u][0]
            y = s%L+actions[u][1]

            #check if state goes over boundary
            if x < 0:
                x = 0
            elif x == L:
                x = L-1

            if y < 0:
                y = 0
            elif y == L:
                y = L-1

            s_new = L*x + y
            if s_new == s:
                B[s, s, u] = 1
            else:
                B[s, s, u] = 1-c
                B[s_new, s, u] = c


    """
    create environment (grid world)
    """
    Rho = np.zeros((nr,ns))
    Rho[0,:] = 1
    Rho[:,np.argmax(utility)] = [0, 1]
    print(Rho)
    util = np.array([1-np.amax(utility), np.amax(utility)])

    environment = env.GridWorld(A, B, Rho, trials = trials, T = T)

    Rho_agent = np.ones((nr,ns,nc))/ nr

    dirichlet_rew_params = np.ones_like(Rho_agent)

    """
    create policies
    """

    if goal_pol:
        pol = []
        su = 3
        for p in itertools.product([0,1], repeat=T-1):
            if (np.array(p)[0:6].sum() == su) and (np.array(p)[-1]!=1):
                pol.append(list(p))

        pol = np.array(pol) + 2
    else:
        pol = np.array(list(itertools.product(list(range(na)), repeat=T-1)))

    pol = pol[np.where(pol[:,0]>1)]

    npi = pol.shape[0]

    prior_policies = np.ones((npi,nc)) / npi
    dirichlet_pol_param = np.zeros_like(prior_policies) + h

    """
    set state prior (where agent thinks it starts)
    """

    state_prior = np.zeros((ns))

    state_prior[0] = 1./4.
    state_prior[1] = 1./4.
    state_prior[4] = 1./4.
    state_prior[5] = 1./4.

    """
    set action selection method
    """

    if avg:

        sel = 'avg'

        ac_sel = asl.DirichletSelector(trials = trials, T = T,
                                      number_of_actions = na)
    else:

        sel = 'max'

        ac_sel = asl.MaxSelector(trials = trials, T = T,
                                      number_of_actions = na)

#    ac_sel = asl.AveragedPolicySelector(trials = trials, T = T,
#                                        number_of_policies = npi,
#                                        number_of_actions = na)


    """
    set up agent
    """
    #bethe agent
    if agent == 'bethe':

        agnt = 'bethe'

        # perception and planning

        bayes_prc = prc.HierarchicalPerception(A, B, Rho_agent, [1], state_prior,
                                               util, prior_policies,
                                               dirichlet_pol_params = dirichlet_pol_param,
                                               dirichlet_rew_params = dirichlet_rew_params)

        bayes_pln = agt.BayesianPlanner(bayes_prc, ac_sel, pol,
                      trials = trials, T = T,
                      prior_states = state_prior,
                      prior_policies = prior_policies,
                      number_of_states = ns,
                      learn_habit = True,
                      learn_rew = True,
                      #save_everything = True,
                      number_of_policies = npi,
                      number_of_rewards = nr)
    #MF agent
    else:

        agnt = 'mf'

        # perception and planning

        bayes_prc = prc.MFPerception(A, B, state_prior, utility, T = T)

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

    w.simulate_experiment()


    """
    plot and evaluate results
    """
    #find successful and unsuccessful runs
    goal = np.argmax(utility)
    successfull = np.where(environment.hidden_states[:,-1]==goal)[0]
    unsuccessfull = np.where(environment.hidden_states[:,-1]!=goal)[0]
    total  = len(successfull)

    #set up figure params
    factor = 3
    grid_plot_kwargs = {'vmin': -2, 'vmax': 2, 'center': 0, 'linecolor': '#D3D3D3',
                        'linewidths': 7, 'alpha': 1, 'xticklabels': False,
                        'yticklabels': False, 'cbar': False,
                        'cmap': sns.diverging_palette(120, 45, as_cmap=True)} #"RdBu_r",

    #plot start and goal state
    start_goal = np.zeros((L,L))

    start_goal[0,1] = 1.
    start_goal[-2,-1] = -1.

    # plot grid
    fig = plt.figure(figsize=[factor*5,factor*5])

    ax = fig.gca()

    annot = np.zeros((L,L))
    for i in range(L):
        for j in range(L):
            annot[i,j] = i*L+j

    u = sns.heatmap(start_goal, ax = ax, **grid_plot_kwargs, annot=annot, annot_kws={"fontsize": 40})
    ax.invert_yaxis()
    plt.savefig('grid.svg', dpi=600)
    plt.show()

    # set up paths figure
    fig = plt.figure(figsize=[factor*5,factor*5])

    ax = fig.gca()

    u = sns.heatmap(start_goal, zorder=2, ax = ax, **grid_plot_kwargs)
    ax.invert_yaxis()

    #find paths and count them
    n = np.zeros((ns, na))

    for i in successfull:

        for j in range(T-1):
            d = environment.hidden_states[i, j+1] - environment.hidden_states[i, j]
            if d not in [1,-1,4,-4,0]:
                print("ERROR: beaming")
            if d == 1:
                n[environment.hidden_states[i, j],0] +=1
            if d == -1:
                n[environment.hidden_states[i, j]-1,0] +=1
            if d == 4:
                n[environment.hidden_states[i, j],1] +=1
            if d == -4:
                n[environment.hidden_states[i, j]-4,1] +=1

    un = np.zeros((ns, na))

    for i in unsuccessfull:

        for j in range(T-1):
            d = environment.hidden_states[i, j+1] - environment.hidden_states[i, j]
            if d not in [1,-1,L,-L,0]:
                print("ERROR: beaming")
            if d == 1:
                un[environment.hidden_states[i, j],0] +=1
            if d == -1:
                un[environment.hidden_states[i, j]-1,0] +=1
            if d == 4:
                un[environment.hidden_states[i, j],1] +=1
            if d == -4:
                un[environment.hidden_states[i, j]-4,1] +=1

    total_num = n.sum() + un.sum()

    if np.any(n > 0):
        n /= total_num

    if np.any(un > 0):
        un /= total_num

    #plotting
    for i in range(ns):

        x = [i%L + .5]
        y = [i//L + .5]

        #plot uncertainties
        if obs_unc:
            plt.plot(x,y, 'o', color=(219/256,122/256,147/256), markersize=factor*12/(A[i,i])**2, alpha=1.)
        if state_unc:
            plt.plot(x,y, 'o', color=(100/256,149/256,237/256), markersize=factor*12/(cert_arr[i])**2, alpha=1.)

        #plot unsuccessful paths
        for j in range(2):

            if un[i,j]>0.0:
                if j == 0:
                    xp = x + [x[0] + 1]
                    yp = y + [y[0] + 0]
                if j == 1:
                    xp = x + [x[0] + 0]
                    yp = y + [y[0] + 1]

                plt.plot(xp,yp, '-', color='#D5647C', linewidth=factor*75*un[i,j],
                         zorder = 9, alpha=1)

    #set plot title
    #plt.title("Planning: successful "+str(round(100*total/trials))+"%", fontsize=factor*9)

    #plot successful paths on top
    for i in range(ns):

        x = [i%L + .5]
        y = [i//L + .5]

        for j in range(2):

            if n[i,j]>0.0:
                if j == 0:
                    xp = x + [x[0] + 1]
                    yp = y + [y[0]]
                if j == 1:
                    xp = x + [x[0] + 0]
                    yp = y + [y[0] + 1]
                plt.plot(xp,yp, '-', color='#4682B4', linewidth=factor*75*n[i,j],
                         zorder = 10, alpha=1)


    print("percent won", total/trials, "state prior", np.amax(utility))


    plt.savefig('chosen_paths_h'+str(h)+'.svg')
    plt.show()

    # max_RT = np.amax(w.agent.action_selection.RT[:,0])
    # plt.figure()
    # plt.plot(w.agent.action_selection.RT[:,0], '.')
    # plt.ylim([0,1.05*max_RT])
    # plt.xlim([0,trials])
    # plt.savefig("Gridworld_Dir_h"+str(h)+".svg")
    # plt.show()

    """
    save data
    """

    if save_data:
        jsonpickle_numpy.register_handlers()

        ut = np.amax(utility)
        p_o = '{:02d}'.format(round(ut*10).astype(int))
        fname = agnt+'_'+condition+'_'+sel+'_initUnc_'+p_o+'.json'
        fname = os.path.join(data_folder, fname)
        pickled = pickle.encode(w)
        with open(fname, 'w') as outfile:
            json.dump(pickled, outfile)

    return w

"""
set condition dependent up parameters
"""
repetitions = 1
# prior over outcomes: encodes utility
utility = []

#ut = [0.5, 0.6, 0.7, 0.8, 0.9, 1-1e-3]
u = 0.999
utility = np.zeros(ns)
utility[-5] = u
utility[:-5] = (1-u)/(ns-1)
utility[-4:] = (1-u)/(ns-1)

# action selection: avergaed or max selection
avg = True
tendencies = [1,10000]
# parameter list
l = []

# either observation uncertainty
#l.append([True, False, False, avg, utility])

# or state uncertainty
#l.append([False, True, False, avg, utility])

# or no uncertainty
l.append([False, False, False, avg, utility])

par_list = []

for p in itertools.product(l, tendencies):
    par_list.append(p[0]+[p[1]])

# num_threads = 11
# pool = Pool(num_threads)

RTs = np.zeros((repetitions*trials*len(tendencies)))
for n,pars in enumerate(par_list):
    h = pars[-1]
    for i in range(repetitions):
        w = run_agent(pars)
        RTs[i*trials+n*(repetitions*trials):(i+1)*trials+n*(repetitions*trials)] = w.agent.action_selection.RT[:,0].copy()
        w = 0
        gc.collect()


runs = np.tile(np.tile(np.arange(repetitions), (trials, 1)).reshape(-1, order='f'),len(tendencies))
times = np.tile(np.arange(trials), repetitions*len(tendencies))
tend_idx = np.array(['h = '+str(1./tendencies[i//(repetitions*trials)]) for i in range(repetitions*trials*len(tendencies))])
DataFrame = pd.DataFrame({'trial': times, 'run': runs,'tendency h':tend_idx, 'RT': RTs})

plt.figure()
sns.lineplot(data=DataFrame, x='trial', y='RT', style='tendency h', estimator=np.nanmedian, linewidth=2, err_kws={'alpha':0.4}, ci=99)
#plt.legend()
plt.xlim([0,trials])
plt.ylim([0,0.75*np.amax(RTs)])
plt.ylabel('RT (#sampples)')
plt.savefig('Dir_gridworld_RT_stats_'+str(repetitions)+'repetitions.svg', dpi=600)
plt.show()


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
