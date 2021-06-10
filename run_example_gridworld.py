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

const = 0#1e-10

trials = 200 #number of trials
T = 5 #number of time steps in each trial
Lx = 4 #grid length
Ly = 5
no = Lx*Ly#Lx*Ly #number of observations
ns = Lx*Ly#Lx*Ly #number of states
na = 3 #number of actions
npi = na**(T-1)
nr = 2
nc = ns
actions = np.array([[0,-1], [1,0], [0,1]])#[-1,0],
g1 = 14
g2 = 10
start = 2

"""
run function
"""
def run_agent(par_list, trials=trials, T=T, Lx = Lx, Ly = Ly, ns=ns, na=na):

    #set parameters:
    #obs_unc: observation uncertainty condition
    #state_unc: state transition uncertainty condition
    #goal_pol: evaluate only policies that lead to the goal
    #utility: goal prior, preference p(o)
    obs_unc, state_unc, goal_pol, avg, context, utility, h, q = par_list


    """
    create matrices
    """

    vals = np.array([1., 2/3., 1/2., 1./2.])

    #generating probability of observations in each state
    A = np.eye(ns) + const
    np.fill_diagonal(A, 1-(ns-1)*const)

    #generate horizontal gradient for observation uncertainty condition
    # if obs_unc:

    #     condition = 'obs'

    #     for s in range(ns):
    #         x = s//Ly
    #         y = s%Ly

    #         c = 1#vals[L - y - 1]

    #         # look for neighbors
    #         neighbors = []
    #         if (s-4)>=0 and (s-4)!=g1:
    #             neighbors.append(s-4)

    #         if (s%4)!=0 and (s-1)!=g1:
    #             neighbors.append(s-1)

    #         if (s+4)<=(ns-1) and (s+4)!=g1:
    #             neighbors.append(s+4)

    #         if ((s+1)%4)!=0 and (s+1)!=g1:
    #             neighbors.append(s+1)

    #         A[s,s] = c
    #         for n in neighbors:
    #             A[n,s] = (1-c)/len(neighbors)


    #state transition generative probability (matrix)
    B = np.zeros((ns, ns, na)) + const

    cert_arr = np.zeros(ns)
    for s in range(ns):
        x = s//Ly
        y = s%Ly

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
            x = s//Ly+actions[u][0]
            y = s%Ly+actions[u][1]

            #check if state goes over boundary
            if x < 0:
                x = 0
            elif x == Lx:
                x = Lx-1

            if y < 0:
                y = 0
            elif y == Ly:
                y = Ly-1

            s_new = Ly*x + y
            if s_new == s:
                B[s, s, u] = 1 - (ns-1)*const
            else:
                B[s, s, u] = 1-c + const
                B[s_new, s, u] = c - (ns-1)*const


    """
    create environment (grid world)
    """
    Rho = np.zeros((nr,ns)) + const
    Rho[0,:] = 1 - (nr-1)*const
    Rho[:,np.argmax(utility)] = [0+const, 1-(nr-1)*const]
    print(Rho)
    util = np.array([1-np.amax(utility), np.amax(utility)])

    environment = env.GridWorld(A, B, Rho, trials = trials, T = T, initial_state=start)

    Rho_agent = np.ones((nr,ns,nc))/ nr


    if True:
        templates = np.ones_like(Rho_agent)
        templates[0] *= 100
        assert ns == nc
        for s in range(ns):
            templates[0,s,s] = 1
            templates[1,s,s] = 100
        dirichlet_rew_params = templates
    else:
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

    #pol = pol[np.where(pol[:,0]>1)]

    npi = pol.shape[0]

    prior_policies = np.ones((npi,nc)) / npi
    dirichlet_pol_param = np.zeros_like(prior_policies) + h

    """
    set state prior (where agent thinks it starts)
    """

    state_prior = np.zeros((ns))

    # state_prior[0] = 1./4.
    # state_prior[1] = 1./4.
    # state_prior[4] = 1./4.
    # state_prior[5] = 1./4.
    state_prior[start] = 1

    """
    set context prior and matrix
    """

    context_prior = np.ones(nc)
    trans_matrix_context = np.ones((nc,nc))
    if nc > 1:
        # context_prior[0] = 0.9
        # context_prior[1:] = 0.1 / (nc-1)
        context_prior /= nc
        trans_matrix_context[:] = (1-q) / (nc-1)
        np.fill_diagonal(trans_matrix_context, q)

    """
    set action selection method
    """

    if avg:

        sel = 'avg'

        ac_sel = asl.DirichletSelector(trials = trials, T = T, factor=0.5, draw_true_post=True,
                                      number_of_actions = na, calc_entropy=True, calc_dkl=True)
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

        bayes_prc = prc.HierarchicalPerception(A, B, Rho_agent, trans_matrix_context, state_prior,
                                               util, prior_policies,
                                               dirichlet_pol_params = dirichlet_pol_param,
                                               dirichlet_rew_params = dirichlet_rew_params)

        bayes_pln = agt.BayesianPlanner(bayes_prc, ac_sel, pol,
                      trials = trials, T = T,
                      prior_states = state_prior,
                      prior_policies = prior_policies,
                      prior_context = context_prior,
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

    if not context:
        w.simulate_experiment()
    else:
        w.simulate_experiment(curr_trials=range(0, trials//2))
        Rho_new = np.zeros((nr,ns)) + const
        Rho_new[0,:] = 1 - (nr-1)*const
        Rho_new[:,g2] = [0+const, 1-(nr-1)*const]
        print(Rho_new)
        w.environment.Rho[:] = Rho_new
        #w.agent.perception.generative_model_rewards = Rho_new
        w.simulate_experiment(curr_trials=range(trials//2, trials))

    """
    plot and evaluate results
    """
    #find successful and unsuccessful runs
    #goal = np.argmax(utility)
    successfull_g1 = np.where(environment.hidden_states[:,-1]==g1)[0]
    if context:
        successfull_g2 = np.where(environment.hidden_states[:,-1]==g2)[0]
        unsuccessfull1 = np.where(environment.hidden_states[:,-1]!=g1)[0]
        unsuccessfull2 = np.where(environment.hidden_states[:,-1]!=g2)[0]
        unsuccessfull = np.intersect1d(unsuccessfull1, unsuccessfull2)
    else:
        unsuccessfull = np.where(environment.hidden_states[:,-1]!=g1)[0]

    #total  = len(successfull)

    #plot start and goal state
    start_goal = np.zeros((Lx,Ly))

    x_y_start = (start//Ly, start%Ly)
    start_goal[x_y_start] = 1.
    x_y_g1 = (g1//Ly, g1%Ly)
    start_goal[x_y_g1] = -1.
    x_y_g2 = (g2//Ly, g2%Ly)
    start_goal[x_y_g2] = -2.

    palette = [(159/255, 188/255, 147/255),
 (135/255, 170/255, 222/255),
 (242/255, 241/255, 241/255),
 (242/255, 241/255, 241/255),
 (199/255, 174/255, 147/255),
 (199/255, 174/255, 147/255)]

    #set up figure params
    factor = 3
    grid_plot_kwargs = {'vmin': -2, 'vmax': 2, 'center': 0, 'linecolor': '#D3D3D3',
                        'linewidths': 7, 'alpha': 1, 'xticklabels': False,
                        'yticklabels': False, 'cbar': False,
                        'cmap': palette}#sns.diverging_palette(120, 45, as_cmap=True)} #"RdBu_r",

    # plot grid
    fig = plt.figure(figsize=[factor*5,factor*4])

    ax = fig.gca()

    annot = np.zeros((Lx,Ly))
    for i in range(Lx):
        for j in range(Ly):
            annot[i,j] = i*Ly+j +1

    u = sns.heatmap(start_goal, ax = ax, **grid_plot_kwargs, annot=annot, annot_kws={"fontsize": 40})
    ax.invert_yaxis()
    plt.savefig('grid.svg', dpi=600)
    plt.show()

    # set up paths figure
    fig = plt.figure(figsize=[factor*5,factor*4])

    ax = fig.gca()

    u = sns.heatmap(start_goal, zorder=2, ax = ax, **grid_plot_kwargs)
    ax.invert_yaxis()

    #find paths and count them
    n1 = np.zeros((ns, na))

    for i in successfull_g1:

        for j in range(T-1):
            d = environment.hidden_states[i, j+1] - environment.hidden_states[i, j]
            if d not in [1,-1,Ly,-Ly,0]:
                print("ERROR: beaming")
            if d == 1:
                n1[environment.hidden_states[i, j],0] +=1
            if d == -1:
                n1[environment.hidden_states[i, j]-1,0] +=1
            if d == Ly:
                n1[environment.hidden_states[i, j],1] +=1
            if d == -Ly:
                n1[environment.hidden_states[i, j]-Ly,1] +=1

    n2 = np.zeros((ns, na))

    if context:
        for i in successfull_g2:

            for j in range(T-1):
                d = environment.hidden_states[i, j+1] - environment.hidden_states[i, j]
                if d not in [1,-1,Ly,-Ly,0]:
                    print("ERROR: beaming")
                if d == 1:
                    n2[environment.hidden_states[i, j],0] +=1
                if d == -1:
                    n2[environment.hidden_states[i, j]-1,0] +=1
                if d == Ly:
                    n2[environment.hidden_states[i, j],1] +=1
                if d == -Ly:
                    n2[environment.hidden_states[i, j]-Ly,1] +=1

    un = np.zeros((ns, na))

    for i in unsuccessfull:

        for j in range(T-1):
            d = environment.hidden_states[i, j+1] - environment.hidden_states[i, j]
            if d not in [1,-1,Ly,-Ly,0]:
                print("ERROR: beaming")
            if d == 1:
                un[environment.hidden_states[i, j],0] +=1
            if d == -1:
                un[environment.hidden_states[i, j]-1,0] +=1
            if d == Ly:
                un[environment.hidden_states[i, j],1] +=1
            if d == -Ly:
                un[environment.hidden_states[i, j]-4,1] +=1

    total_num = n1.sum() + n2.sum() + un.sum()

    if np.any(n1 > 0):
        n1 /= total_num

    if np.any(n2 > 0):
        n2 /= total_num

    if np.any(un > 0):
        un /= total_num

    #plotting
    for i in range(ns):

        x = [i%Ly + .5]
        y = [i//Ly + .5]

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

        x = [i%Ly + .5]
        y = [i//Ly + .5]

        for j in range(2):

            if n1[i,j]>0.0:
                if j == 0:
                    xp = x + [x[0] + 1]
                    yp = y + [y[0]]
                if j == 1:
                    xp = x + [x[0] + 0]
                    yp = y + [y[0] + 1]
                plt.plot(xp,yp, '-', color='#4682B4', linewidth=factor*75*n1[i,j],
                         zorder = 10, alpha=1)

    #plot successful paths on top
    if context:
        for i in range(ns):

            x = [i%Ly + .5]
            y = [i//Ly + .5]

            for j in range(2):

                if n2[i,j]>0.0:
                    if j == 0:
                        xp = x + [x[0] + 1]
                        yp = y + [y[0]]
                    if j == 1:
                        xp = x + [x[0] + 0]
                        yp = y + [y[0] + 1]
                    plt.plot(xp,yp, '-', color='#55ab75', linewidth=factor*75*n2[i,j],
                             zorder = 10, alpha=1)


    #print("percent won", total/trials, "state prior", np.amax(utility))


    plt.savefig('chosen_paths_'+name_str+'h'+str(h)+'.svg')
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
utility[g1] = u
utility[:g1] = (1-u)/(ns-1)
utility[g1+1:] = (1-u)/(ns-1)

# action selection: avergaed or max selection
avg = True
tendencies = [1,1000]
context = True
if context:
    name_str = "context_"
else:
    name_str = ""
# parameter list
l = []

# either observation uncertainty
#l.append([True, False, False, avg, utility])

# or state uncertainty
#l.append([False, True, False, avg, utility])

# or no uncertainty
l.append([False, False, False, avg, context, utility])

par_list = []

for p in itertools.product(l, tendencies):
    par_list.append(p[0]+[p[1]])

qs = [0.97, 0.97]
# num_threads = 11
# pool = Pool(num_threads)
# worlds = []
RTs = np.zeros((repetitions*trials*len(tendencies)))
for n,pars in enumerate(par_list):
    h = pars[-1]
    q = qs[n]
    for i in range(repetitions):
        # worlds.append(run_agent(pars+[q]))
        # w = worlds[-1]
        w = run_agent(pars+[q])
        RTs[i*trials+n*(repetitions*trials):(i+1)*trials+n*(repetitions*trials)] = w.agent.action_selection.RT[:,0].copy()
        if i == repetitions-1:
            if context:
                plt.figure()
                plt.plot(w.agent.posterior_context[:,0,:])
                #plt.plot(w.agent.posterior_context[:,0,g2])
                plt.show()
            plt.figure()
            rew_prob = np.einsum('tsc,tc->ts', w.agent.posterior_dirichlet_rew[:,0,1,:,:],w.agent.posterior_context[:,0])
            rew_prob /= rew_prob.sum(axis=1)[:,None]
            plt.plot(rew_prob)
            plt.ylim([0, .75])
            plt.show()
            plt.figure()
            plt.plot(np.einsum('tsc,tc->ts', w.agent.posterior_dirichlet_rew[:,0,1,:,:],w.agent.posterior_context[:,0]))
            plt.ylim([0, 40])
            plt.show()
            plt.figure()
            plt.plot(w.agent.action_selection.entropy_post[:,0])
            plt.ylim([2.5,5])
            plt.show()
            plt.figure()
            plt.plot(w.agent.action_selection.entropy_like[:,0])
            plt.ylim([2.5,5])
            plt.show()
            plt.figure()
            plt.plot(w.agent.action_selection.entropy_prior[:,0])
            plt.ylim([2.5,5])
            plt.show()
            plt.figure()
            plt.figure()
            plt.plot(w.agent.action_selection.DKL_post[:,0])
            plt.show()
            plt.figure()
            plt.plot(w.agent.action_selection.DKL_prior[:,0])
            plt.show()
            posterior_policies = np.einsum('tpc,tc->tp', w.agent.posterior_policies[:,0], w.agent.posterior_context[:,0])
            k=6
            ind = np.argpartition(posterior_policies, -k, axis=1)[:,-k:]
            max_pol = np.array([posterior_policies[i,ind[i]] for i in range(trials)])
            plt.figure()
            plt.plot(max_pol)
            plt.figure()
            plt.figure()
            plt.plot(posterior_policies.argmax(axis=1))
            plt.show()
            plt.figure()
            plt.plot(w.agent.action_selection.RT[:,0], 'x')
            plt.show()
            like = np.einsum('tpc,tc->tp', w.agent.likelihood[:,0], w.agent.posterior_context[:,0])
            prior = np.einsum('tpc,tc->tp', w.agent.prior_policies[:], w.agent.posterior_context[:,0])
            for i in [20,40,100,trials-1]:
                plt.figure()
                plt.plot(np.sort(like[i]), linewidth=3, label='likelihood')
                plt.plot(np.sort(prior[i]), linewidth=3, label='prior')
                plt.title("trial "+str(i))
                plt.ylim([0,0.25])
                plt.xlim([0,len(prior[i])])
                plt.xlabel('policy', fontsize=16)
                plt.ylabel('probability', fontsize=16)
                plt.legend()
                plt.savefig('underlying_prior_like_trial_'+str(i)+'_h_'+str(h)+'.svg')
                plt.show()
            # for i in [trials-1]:
            #     plt.figure()
            #     plt.plot(np.sort(prior[i]))
            #     plt.title("trial "+str(i))
            #     plt.ylim([0,1])
            #     plt.show()
        w = 0
        gc.collect()


runs = np.tile(np.tile(np.arange(repetitions), (trials, 1)).reshape(-1, order='f'),len(tendencies))
times = np.tile(np.arange(trials), repetitions*len(tendencies))
tend_idx = np.array(['h = '+str(1./tendencies[i//(repetitions*trials)]) for i in range(repetitions*trials*len(tendencies))])
DataFrame = pd.DataFrame({'trial': times, 'run': runs,'tendency h':tend_idx, 'RT': RTs})

# plt.figure()
# sns.lineplot(data=DataFrame, x='trial', y='RT', style='tendency h', estimator=np.nanmean, linewidth=2, err_kws={'alpha':0.4}, ci=99)
# #plt.legend()
# plt.xlim([0,trials])
# plt.ylim([0,0.5*np.amax(RTs)])#0.75*
# plt.ylabel('RT (#sampples)')
# plt.savefig('Dir_gridworld_RT_stats_context_'+str(repetitions)+'repetitions.svg', dpi=600)
# plt.show()

plt.figure()
palette = [(38/255,99/255,141/255),(54/255, 142/255, 201/255)]#[(30/255,82/255,225/255),(30/255, 164/255, 255/255)]#[(31/255,119/255,180/255),(79/255, 128/255, 23/255)]
sns.lineplot(data=DataFrame, x='trial', y='RT', hue='tendency h', style='tendency h', estimator=np.nanmedian, linewidth=2, err_kws={'alpha':0.4}, ci=99, palette=palette)
#plt.legend()
plt.ylim([0,np.amax(RTs)])
plt.xlim([0,trials])
plt.ylabel('RT (#sampples)')
plt.savefig('Dir_gridworld_RT_stats_'+name_str+str(repetitions)+'repetitions_median.svg', dpi=600)
plt.show()

plt.figure()
palette = [(38/255,99/255,141/255),(54/255, 142/255, 201/255)]#[(31/255,119/255,180/255),(79/255, 128/255, 23/255)]
sns.lineplot(data=DataFrame, x='trial', y='RT', hue='tendency h', style='tendency h', estimator=np.nanmean, linewidth=2, err_kws={'alpha':0.4}, ci=99, palette=palette)
#plt.legend()
plt.ylim([0,np.amax(RTs)])
plt.xlim([0,trials])
plt.ylabel('RT (#sampples)')
plt.savefig('Dir_gridworld_RT_stats_'+name_str+str(repetitions)+'repetitions_mean.svg', dpi=600)
plt.show()


def run_action_selection(post, prior, like, trials = trials, crit_factor = 0.4, calc_dkl = False):

    ac_sel = asl.DirichletSelector(trials, 2, npi, factor=crit_factor, calc_dkl=calc_dkl)
    for t in range(trials):
        ac_sel.select_desired_action(t, 0, post[t], list(range(npi)), like[t], prior[t])

    if calc_dkl:
        return ac_sel.RT.squeeze(), ac_sel.DKL_post.squeeze(), ac_sel.DKL_prior.squeeze()
    else:
        return ac_sel.RT.squeeze()
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
