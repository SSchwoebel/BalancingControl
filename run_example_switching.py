#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 15:33:22 2021

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
np.set_printoptions(threshold = 100000, precision = 5)

"""
run function
"""

def run_agent(par_list, trials, T, ns, na, nr, nc, f, contexts, states, \
              state_trans=None, correct_choice=None, congruent=None,\
              num_in_run=None, random_draw=False, pol_lambda=0, r_lambda=0,
              one_context=False):
    #set parameters:
    #learn_pol: initial concentration paramter for policy prior
    #trans_prob: reward probability
    #avg: True for average action selection, False for maximum selection
    #Rho: Environment's reward generation probabilities as a function of time
    #utility: goal prior, preference p(o)
    learn_pol, trans_prob, Rho, utility, unc = par_list


    """
    create matrices
    """


    #generating probability of observations in each state
    A = np.eye(ns)


    #state transition generative probability (matrix)
    if state_trans is None:
        B = np.zeros((ns, ns, na))

        for i in range(0,na):
            B[i+1,:,i] += 1
    else:
        B = state_trans.copy()

    # agent's beliefs about reward generation

    # concentration parameters
    C_alphas = np.ones((nr, ns, nc))
    # initialize state in front of levers so that agent knows it yields no reward
    C_alphas[:,:4,:] = np.array([100,1])[:,None,None]
    # C_alphas[:,4:,0] = np.array([[1, 2],print(self.Rho.shape)
    #                               [2, 1]])
    # C_alphas[:,4:,1] = np.array([[2, 1],
    #                               [1, 2]])

    # agent's initial estimate of reward generation probability
    C_agent = np.zeros((nr, ns, nc))
    for c in range(nc):
        C_agent[:,:,c] = np.array([(C_alphas[:,i,c])/(C_alphas[:,i,c]).sum() for i in range(ns)]).T


    # context transition matrix

    if nc>1:
        p = trans_prob
        q = 1.-p
        transition_matrix_context = np.zeros((nc, nc))
        transition_matrix_context += q/(nc-1)
        for i in range(nc):
            transition_matrix_context[i,i] = p
    else:
        transition_matrix_context = np.array([[1]])

    # context observation matrix

    if nc > 1:
        D = np.zeros((nc,nc)) + unc
        for c in range(nc):
            D[c,c] = 1-(unc*(nc-1))
    else:
        D = np.array([[1]])

    """
    create environment (grid world)
    """
    if not one_context:
        
        environment = env.TaskSwitching(A, B, Rho, D, states, contexts, \
                                    trials = trials, T = T,\
                                    correct_choice=correct_choice, \
                                    congruent=congruent, \
                                    num_in_run=num_in_run)
            
    else:
        
        environment = env.TaskSwitchingOneConext(A, B, Rho, D, states, contexts, \
                                    trials = trials, T = T,\
                                    correct_choice=correct_choice, \
                                    congruent=congruent, \
                                    num_in_run=num_in_run)
        


    """
    create policies
    """

    pol = np.array(list(itertools.product(list(range(na)), repeat=T-1)))

    npi = pol.shape[0]

    # concentration parameters
    alphas = np.zeros((npi, nc)) + learn_pol

    prior_pi = alphas / alphas.sum(axis=0)


    """
    set state prior (where agent thinks it starts)
    """

    state_prior = np.zeros((ns))

    state_prior[:4] = 1./4

    """
    set action selection method
    """

    ac_sel = asl.DirichletSelector(trials=trials, T=T, number_of_actions=na, factor=f, calc_dkl=False, calc_entropy=False, draw_true_post=random_draw)

    """
    set context prior
    """

    if nc > 1:
        prior_context = np.zeros((nc)) + 0.1/(nc-1)
        prior_context[0] = 0.9
    else:
        prior_context = np.array([1])

    """
    set up agent
    """

    # perception
    bayes_prc = prc.HierarchicalPerception(A, B, C_agent, transition_matrix_context, 
                                        state_prior, utility, prior_pi, alphas, 
                                        C_alphas, T=T, generative_model_context=D, 
                                        pol_lambda=pol_lambda, r_lambda=r_lambda,
                                        non_decaying=4)

    # agent
    bayes_pln = agt.BayesianPlanner(bayes_prc, ac_sel, pol,
                      trials = trials, T = T,
                      prior_states = state_prior,
                      prior_policies = prior_pi,
                      number_of_states = ns,
                      prior_context = prior_context,
                      learn_habit = True,
                      learn_rew = True,
                      #save_everything = True,
                      number_of_policies = npi,
                      number_of_rewards = nr)


    """
    create world
    """

    w = world.World(environment, bayes_pln, trials = trials, T = T)

    """
    simulate experiment
    """
    w.simulate_experiment(range(trials))

    return w



"""
set condition dependent up parameters
"""

def run_switching_simulations(repetitions, folder):

    trials = 100
    T = 2
    ns = 6
    na = 2
    nr = 2
    nc = 2
    u = 0.99
    utility = np.array([1-u,u])
    f = 3.5
    random_draw = False
    pol_lambda = 0
    r_lambda = 0

    Rho = np.zeros((trials, nr, ns))

    for tendency in [1000]:#[1,1000]:#,3,5,10,30,50,100]: #1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]:
        for trans in [94,95,96]:#[80,85,90,91,92,93,94,95,96,97,98,99]:
            for unc in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,8,10]:
                print(tendency, trans, unc)

                # Rho[:], contexts, states, state_trans, correct_choice, congruent, num_in_run = \
                #     switching_timeseries(trials, nr=nr, ns=ns, na=na, nc=nc, stable_length=5)

                # plt.figure()
                # plt.plot(Rho[:,2,2])
                # plt.plot(Rho[:,1,1])
                # plt.show()

                if random_draw==False:
                    prefix="acsel_"
                else:
                    prefix=""
                if pol_lambda > 0:
                    s = "alpha_"
                else:
                    s = ""
                if r_lambda > 0:
                    s += "beta_"
                else:
                    s += ""
                run_name = prefix+"switching_"+s+"h"+str(int(tendency))+"_t"+str(trans)+"_u"+str(unc)+"_f"+str(f)+"_ut"+str(u)+"_test.json"
                print(run_name)
                fname = os.path.join(folder, run_name)

                jsonpickle_numpy.register_handlers()

                if False:#run_name in os.listdir(folder):
                    with open(fname, 'r') as infile:
                        data = json.load(infile)

                    worlds = pickle.decode(data)
                    print(len(worlds))
                    num_w_old = len(worlds)
                else:
                    worlds = []
                    num_w_old = 0

                learn_pol = tendency
                parameters = [learn_pol, trans/100., Rho, utility, unc/100.]

                for i in range(num_w_old, repetitions):
                    Rho[:], contexts, states, state_trans, correct_choice, congruent, num_in_run = \
                    switching_timeseries(trials, nr=nr, ns=ns, na=na, nc=nc, stable_length=5)
                    worlds.append(run_agent(parameters, trials, T, ns, na, nr, nc,\
                                            f, contexts, states, \
                                            state_trans=state_trans, \
                                            correct_choice=correct_choice, \
                                            congruent=congruent, \
                                            num_in_run=num_in_run, \
                                            random_draw=random_draw, \
                                            pol_lambda=pol_lambda, \
                                            r_lambda=r_lambda))
                    # w = worlds[-1]
                    # choices = w.actions[:,0]
                    # correct = (choices == w.environment.correct_choice).sum()
                    # print("percent correct:", correct/trials)
                    # correct_cong = (choices[w.environment.congruent==1] == w.environment.correct_choice[w.environment.congruent==1]).sum()
                    # print("percent correct congruent:", correct_cong/(w.environment.congruent==1).sum())
                    # correct_incong = (choices[w.environment.congruent==0] == w.environment.correct_choice[w.environment.congruent==0]).sum()
                    # print("percent correct incongruent:", correct_incong/(w.environment.congruent==0).sum())
                    # RTs = w.agent.action_selection.RT[:,0]
                    # RT_cong = np.median(RTs[w.environment.congruent==1])
                    # RT_incong = np.median(RTs[w.environment.congruent==0])
                    # print("congruent RT:", RT_cong)
                    # print("incongruent RT:", RT_incong)
                    # length = int(np.amax(w.environment.num_in_run)) + 1
                    # numbers = w.environment.num_in_run
                    # numbers_cong = numbers[w.environment.congruent==1]
                    # numbers_incong = numbers[w.environment.congruent==0]
                    # RT_medians_cong = [np.median(RTs[w.environment.congruent==1][numbers_cong==i]) for i in range(length)]
                    # RT_medians_incong = [np.median(RTs[w.environment.congruent==0][numbers_incong==i]) for i in range(length)]
                    # plt.figure()
                    # plt.plot(RT_medians_cong, 'x')
                    # plt.plot(RT_medians_incong, 'x')
                    # plt.show()
                    # plt.figure()
                    # post_pol = np.einsum('tpc,tc->tp', w.agent.posterior_policies[:,0,:,:], w.agent.posterior_context[:,0,:])
                    # like = np.einsum('tpc,tc->tp', w.agent.likelihood[:,0,:,:], w.agent.posterior_context[:,0,:])
                    # plt.plot(post_pol[:,1], '.')
                    # plt.plot(like[:,1], 'x')
                    # plt.ylim([0,1])
                    # plt.show()
                    # plt.figure()
                    # plt.plot(w.agent.action_selection.RT[:,0], '.')
                    # #plt.plot(Rho[:,2,2])
                    # #plt.plot(Rho[:,1,1])
                    # #plt.ylim([ESS*10,2000])
                    # plt.ylim([0,2000])
                    # plt.savefig("Dir_h"+str(int(learn_pol))+"_RT_timecourse"+str(i)+".svg")#"ESS"+str(ESS)+"_h"+str(int(learn_pol))+"_RT_timecourse"+str(i)+".svg")#
                    # plt.show()
                    # plt.figure()
                    # plt.hist(w.agent.action_selection.RT[:,0])
                    # plt.savefig("uncertain_Dir_h"+str(int(learn_pol))+"_RT_hist"+str(i)+"_1000trials.svg")#"ESS"+str(ESS)+"_h"+str(int(learn_pol))+"_RT_hist"+str(i)+".svg")#
                    # plt.show()
                    # plt.figure()
                    # plt.plot(w.agent.posterior_context[:,0,:], 'x')
                    # plt.show()


                if random_draw==False:
                    prefix="acsel_"
                else:
                    prefix=""
                #run_name = prefix+"switching_h"+str(int(learn_pol))+"_t"+str(trans)+"_u"+str(unc)+".json"
                fname = os.path.join(folder, run_name)

                jsonpickle_numpy.register_handlers()
                pickled = pickle.encode(worlds)
                with open(fname, 'w') as outfile:
                    json.dump(pickled, outfile)

                pickled = 0
                worlds = 0

                gc.collect()


def run_single_task_simulations(repetitions, folder):

    trials = 100
    T = 2
    ns = 6
    na = 2
    nr = 2
    nc = 1
    u = 0.99
    utility = np.array([1-u,u])
    f = 3.5
    random_draw = False
    pol_lambda = 0
    r_lambda = 0

    Rho = np.zeros((trials, nr, ns))

    for tendency in [1000]:#,3,5,10,30,50,100]: #1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]:
        for trans in [97,98,99]:#[80,85,90,91,92,93,94,95,96,97,98,99]:
            for unc in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,8,10]:
                print(tendency, trans, unc)

                # Rho[:], contexts, states, state_trans, correct_choice, congruent, num_in_run = \
                #     switching_timeseries(trials, nr=nr, ns=ns, na=na, nc=nc, stable_length=5)

                # plt.figure()
                # plt.plot(Rho[:,2,2])
                # plt.plot(Rho[:,1,1])
                # plt.show()
                if random_draw==False:
                    prefix="acsel_"
                else:
                    prefix=""
                if pol_lambda > 0:
                    s = "alpha_"
                else:
                    s = ""
                if r_lambda > 0:
                    s += "beta_"
                else:
                    s += ""
                run_name = prefix+"single_prior"+s+"h"+str(int(tendency))+"_t"+str(trans)+"_u"+str(unc)+"_f"+str(f)+"_ut"+str(u)+"_test.json"
                fname = os.path.join(folder, run_name)

                jsonpickle_numpy.register_handlers()

                if run_name in os.listdir(folder):
                    with open(fname, 'r') as infile:
                        data = json.load(infile)

                    worlds = pickle.decode(data)
                    print(len(worlds))
                    num_w_old = len(worlds)
                else:
                    worlds = []
                    num_w_old = 0

                learn_pol = tendency
                parameters = [learn_pol, trans/100., Rho, utility, unc/100.]

                for i in range(num_w_old, repetitions):
                    Rho[:], contexts, states, state_trans, correct_choice, congruent, num_in_run = \
                    single_task_timeseries(trials, nr=nr, ns=ns, na=na, nc=nc)
                    worlds.append(run_agent(parameters, trials, T, ns, na, nr, nc,\
                                            f, contexts, states, \
                                            state_trans=state_trans, \
                                            correct_choice=correct_choice, \
                                            congruent=congruent, \
                                            num_in_run=num_in_run, \
                                            random_draw=random_draw, \
                                            pol_lambda=pol_lambda))
                    # w = worlds[-1]
                    # choices = w.actions[:,0]
                    # correct = (choices == w.environment.correct_choice).sum()
                    # print("percent correct:", correct/trials)
                    # correct_cong = (choices[w.environment.congruent==1] == w.environment.correct_choice[w.environment.congruent==1]).sum()
                    # print("percent correct congruent:", correct_cong/(w.environment.congruent==1).sum())
                    # correct_incong = (choices[w.environment.congruent==0] == w.environment.correct_choice[w.environment.congruent==0]).sum()
                    # print("percent correct incongruent:", correct_incong/(w.environment.congruent==0).sum())
                    # RTs = w.agent.action_selection.RT[:,0]
                    # RT_cong = np.median(RTs[w.environment.congruent==1])
                    # RT_incong = np.median(RTs[w.environment.congruent==0])
                    # print("congruent RT:", RT_cong)
                    # print("incongruent RT:", RT_incong)
                    # length = int(np.amax(w.environment.num_in_run)) + 1
                    # numbers = w.environment.num_in_run
                    # numbers_cong = numbers[w.environment.congruent==1]
                    # numbers_incong = numbers[w.environment.congruent==0]
                    # RT_medians_cong = [np.median(RTs[w.environment.congruent==1][numbers_cong==i]) for i in range(length)]
                    # RT_medians_incong = [np.median(RTs[w.environment.congruent==0][numbers_incong==i]) for i in range(length)]
                    # plt.figure()
                    # plt.plot(RT_medians_cong, 'x')
                    # plt.plot(RT_medians_incong, 'x')
                    # plt.show()
                    # plt.figure()
                    # post_pol = np.einsum('tpc,tc->tp', w.agent.posterior_policies[:,0,:,:], w.agent.posterior_context[:,0,:])
                    # like = np.einsum('tpc,tc->tp', w.agent.likelihood[:,0,:,:], w.agent.posterior_context[:,0,:])
                    # plt.plot(post_pol[:,1], '.')
                    # plt.plot(like[:,1], 'x')
                    # plt.ylim([0,1])
                    # plt.show()
                    # plt.figure()
                    # plt.plot(w.agent.action_selection.RT[:,0], '.')
                    # #plt.plot(Rho[:,2,2])
                    # #plt.plot(Rho[:,1,1])
                    # #plt.ylim([ESS*10,2000])
                    # plt.ylim([0,2000])
                    # plt.savefig("Dir_h"+str(int(learn_pol))+"_RT_timecourse"+str(i)+".svg")#"ESS"+str(ESS)+"_h"+str(int(learn_pol))+"_RT_timecourse"+str(i)+".svg")#
                    # plt.show()
                    # plt.figure()
                    # plt.hist(w.agent.action_selection.RT[:,0])
                    # plt.savefig("uncertain_Dir_h"+str(int(learn_pol))+"_RT_hist"+str(i)+"_1000trials.svg")#"ESS"+str(ESS)+"_h"+str(int(learn_pol))+"_RT_hist"+str(i)+".svg")#
                    # plt.show()
                    # plt.figure()
                    # plt.plot(w.agent.posterior_context[:,0,:], 'x')
                    # plt.show()

                jsonpickle_numpy.register_handlers()
                pickled = pickle.encode(worlds)
                with open(fname, 'w') as outfile:
                    json.dump(pickled, outfile)

                pickled = 0
                worlds = 0

                gc.collect()
                
                
def run_switching_simulations_one_context(repetitions, folder):

    trials = 100
    T = 2
    ns = 8
    na = 2
    nr = 2
    nc = 1
    u = 0.99
    utility = np.array([1-u,u])
    f = 3.5
    random_draw = False
    pol_lambda = 0
    r_lambda = 0

    Rho = np.zeros((nr, ns, 2))

    for tendency in [1000]:#[1,1000]:#,3,5,10,30,50,100]: #1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]:
        # these two params dont exist anyways when there is only one context. 
        # just setting them for naming consistency with the other simulations
        for trans in [95]:#[80,85,90,91,92,93,94,95,96,97,98,99]:
            for unc in [1]:#,2,3,4,5,6,8,10]:
                print(tendency, trans, unc)

                if random_draw==False:
                    prefix="acsel_"
                else:
                    prefix=""
                if pol_lambda > 0:
                    s = "alpha_"
                else:
                    s = ""
                if r_lambda > 0:
                    s += "beta_"
                else:
                    s += ""
                run_name = prefix+"switching_"+s+"h"+str(int(tendency))+"_t"+str(trans)+"_u"+str(unc)+"_f"+str(f)+"_ut"+str(u)+"_onecontext.json"
                print(run_name)
                fname = os.path.join(folder, run_name)

                jsonpickle_numpy.register_handlers()

                if False:#run_name in os.listdir(folder):
                    with open(fname, 'r') as infile:
                        data = json.load(infile)

                    worlds = pickle.decode(data)
                    print(len(worlds))
                    num_w_old = len(worlds)
                else:
                    worlds = []
                    num_w_old = 0

                learn_pol = tendency
                parameters = [learn_pol, trans/100., Rho, utility, unc/100.]

                for i in range(num_w_old, repetitions):
                    Rho[:], contexts, states, state_trans, correct_choice, congruent, num_in_run = \
                    switching_timeseries_onecontext(trials, nr=nr, ns=ns, na=na, stable_length=5)
                    worlds.append(run_agent(parameters, trials, T, ns, na, nr, nc,\
                                            f, contexts, states, \
                                            state_trans=state_trans, \
                                            correct_choice=correct_choice, \
                                            congruent=congruent, \
                                            num_in_run=num_in_run, \
                                            random_draw=random_draw, \
                                            pol_lambda=pol_lambda, \
                                            r_lambda=r_lambda,
                                            one_context=True))


                if random_draw==False:
                    prefix="acsel_"
                else:
                    prefix=""
                #run_name = prefix+"switching_h"+str(int(learn_pol))+"_t"+str(trans)+"_u"+str(unc)+".json"
                fname = os.path.join(folder, run_name)

                jsonpickle_numpy.register_handlers()
                pickled = pickle.encode(worlds)
                with open(fname, 'w') as outfile:
                    json.dump(pickled, outfile)

                pickled = 0
                worlds = 0

                gc.collect()




def analyze_switching_simulations(folder):

    tendencies = [1000]
    probs = [80,85,90,91,92,93,94,95,96,97,98,99]#
    uncertainties = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,8,10]
    run_name = "acsel_switching_h"+str(int(tendencies[0]))+"_t"+str(probs[0])+"_u"+str(uncertainties[0])+"_f3.5_ut0.99.json"
    print(run_name)
    fname = os.path.join(folder, run_name)

    jsonpickle_numpy.register_handlers()

    with open(fname, 'r') as infile:
        data = json.load(infile)

    worlds_old = pickle.decode(data)

    repetitions = len(worlds_old)
    trials = worlds_old[0].trials
    num_types = len(tendencies)*len(probs)*len(uncertainties)
    correct = np.zeros(repetitions*trials*num_types)
    RT = np.zeros(repetitions*trials*num_types)
    agent = np.zeros(repetitions*trials*num_types)
    num_in_run = np.zeros(repetitions*trials*num_types)
    congruent = np.zeros(repetitions*trials*num_types)
    trial_num = np.zeros(repetitions*trials*num_types)
    epoch = np.zeros(repetitions*trials*num_types)
    tend_arr = np.zeros(repetitions*trials*num_types)
    prob_arr = np.zeros(repetitions*trials*num_types)
    unc_arr  = np.zeros(repetitions*trials*num_types)
    non_dec_time = 100
    t_s = 0.2

    sim_type = 0
    for tendency in tendencies:#,3,5,10,30,50,100]: #1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]:
        for trans in probs:#[100,99,98,97,96,95,94]:
            for unc in uncertainties:

                run_name = "acsel_switching_h"+str(int(tendency))+"_t"+str(trans)+"_u"+str(unc)+"_f3.5_ut0.99.json"
                fname = os.path.join(folder, run_name)

                jsonpickle_numpy.register_handlers()

                with open(fname, 'r') as infile:
                    data = json.load(infile)

                worlds_old = pickle.decode(data)

                repetitions = len(worlds_old)
                trials = worlds_old[0].trials
                
                print("switching", len(worlds_old), tendency, trans, unc)

                offset = sim_type*repetitions*trials

                for i in range(repetitions):
                    w = worlds_old[i]
                    correct[offset+i*trials:offset+(i+1)*trials] = (w.actions[:,0] == w.environment.correct_choice).astype(int)
                    RT[offset+i*trials:offset+(i+1)*trials] = t_s*w.agent.action_selection.RT[:,0] + non_dec_time
                    agent[offset+i*trials:offset+(i+1)*trials] = i
                    num_in_run[offset+i*trials:offset+(i+1)*trials] = w.environment.num_in_run
                    congruent[offset+i*trials:offset+(i+1)*trials] = np.logical_not(w.environment.congruent)
                    trial_num[offset+i*trials:offset+(i+1)*trials] = np.arange(0,trials)
                    epoch[offset+i*trials:offset+(i+1)*trials] = [-1]*10 + [0]*10 + [1]*20 + [2]*30 + [3]*(trials-70)#
                    tend_arr[offset+i*trials:offset+(i+1)*trials] = tendency
                    prob_arr[offset+i*trials:offset+(i+1)*trials] = 100-trans
                    unc_arr[offset+i*trials:offset+(i+1)*trials] = unc

                sim_type+=1

    data_dict = {"correct": correct, "RT": RT, "agent": agent,
                 "num_in_run": num_in_run, "congruent": congruent,
                 "trial_num": trial_num, "epoch": epoch,
                 "uncertainty": unc_arr, "tendencies": tend_arr,
                 "trans_probs": prob_arr}
    data = pd.DataFrame(data_dict)

    data_single = analyze_single_simulations(folder, plot=False, non_dec_time=non_dec_time, t_s=t_s)

    # plt.figure()
    # for i in range(0,3):
    #     sns.lineplot(x='num_in_run', y='RT', data=data.query('epoch == @i'), style='congruent', label=str(i), ci = 95, estimator=np.nanmean, linewidth=3)
    # plt.show()
    tendency=1000
    trans=95
    unc=0.5
    trans = 100-trans

    # RT & accuracy as a function of num in run for congruent and incongruent trials and for different training durations (Fig 3 in Steyvers 2019)
    #sns.set_style("white")switching
    current_palette = sns.color_palette("colorblind")
    plot_palette = [(236/255,98/255,103/255), current_palette[2], current_palette[0]]
    sns.set_style("ticks")
    plt.figure(figsize=(3.5,5))
    #plt.title("tendency "+str(tendency)+", trans "+str(trans)+", unc "+str(unc))
    sns.lineplot(x='num_in_run', y='RT', data=data.query('epoch >= 0 and epoch < 3 and tendencies==@tendency and uncertainty==@unc and trans_probs==@trans'), style='congruent', hue='epoch', markers=True, ci = 95, estimator=np.nanmean, linewidth=3, palette=plot_palette)
    #plt.ylim([600,2000])
    plt.xticks([1,2,3,4,5], fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Trial after switch", fontsize=16)
    plt.ylabel("RT", fontsize=16)
    plt.savefig("numinrun_congruent_RT.svg")
    plt.show()
    plt.figure(figsize=(3.5,5))
    #plt.title("tendency "+str(tendency)+", trans "+str(trans)+", unc "+str(unc))
    sns.lineplot(x='num_in_run', y='correct', data=data.query('epoch >= 0 and epoch < 3 and tendencies==@tendency and uncertainty==@unc and trans_probs==@trans'), style='congruent', hue='epoch', markers=True, ci = 95, estimator=np.nanmean, linewidth=3, palette=plot_palette)
    plt.ylim([0.65,1.])
    plt.xticks([1,2,3,4,5], fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Trial after switch", fontsize=16)
    plt.ylabel("Prop correct", fontsize=16)
    plt.savefig("numinrun_congruent_correct.svg")
    plt.show()

    # CTI
    plt.figure()
    CTI_palette = [plot_palette[2], plot_palette[0]]
    #plt.title("tendency "+str(tendency)+", trans "+str(trans))
    sns.lineplot(x='uncertainty', y='RT', data=data.query('tendencies==@tendency and trans_probs==@trans and num_in_run<3 and uncertainty<1.1 and uncertainty>0'), style='num_in_run', hue='num_in_run', markers=True, ci = 95, estimator=np.nanmean, linewidth=3, palette=CTI_palette[:2])
    sns.lineplot(x='uncertainty', y='RT', data=data_single.query('tendencies==@tendency and trans_probs==@trans and uncertainty<1.1 and uncertainty>0'), style='num_in_run', hue='num_in_run', ci = 95, markers=True, estimator=np.nanmean, linewidth=3, palette=[plot_palette[1]])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("uncertainty (in %)", fontsize=16)
    plt.ylabel("RT", fontsize=16)
    plt.gca().invert_xaxis()
    #plt.ylim([200,1400])
    plt.savefig("CTI.svg")
    plt.show()
    
    # RTI
    palette = [(220/255, 149/255, 166/255), (104/255,98/255,166/255)]
    plt.figure()
    #plt.title("tendency "+str(tendency)+", unc "+str(unc))
    sns.lineplot(x='trans_probs', y='RT', data=data.query('tendencies==@tendency and num_in_run<3 and uncertainty==@unc and trans_probs>1'), hue='num_in_run', style='tendencies', markers=True, ci = 95, estimator=np.nanmean, linewidth=3, palette=palette)
    plt.xlabel("change probability", fontsize=16)
    plt.ylabel("RT", fontsize=16)
    plt.xticks([5,10,15,20], fontsize=16)
    plt.yticks(fontsize=16)
    #plt.ylim([600,900])
    plt.savefig("RTI.svg")
    plt.show()


    plt.figure()
    plt.title("tendency "+str(tendency)+", trans "+str(trans))
    sns.lineplot(x='num_in_run', y='RT', data=data.query('tendencies==@tendency and trans_probs==@trans and uncertainty<11 and epoch>2'), hue='uncertainty', style='congruent', markers=True, ci = 95, estimator=np.nanmean, linewidth=3, palette="crest")
    #plt.ylim([400,1800])
    plt.show()
    plt.figure()
    plt.title("tendency "+str(tendency)+", trans "+str(trans))
    sns.lineplot(x='num_in_run', y='correct', data=data.query('tendencies==@tendency and trans_probs==@trans and uncertainty<11 and epoch>2'), hue='uncertainty', style='congruent', markers=True, ci = 95, estimator=np.nanmean, linewidth=3, palette="crest")
    plt.ylim([0,1])
    plt.show()

    plt.figure(figsize=(3.5,5))
    #plt.title("tendency "+str(tendency)+", trans "+str(trans))
    sns.lineplot(x='num_in_run', y='RT', data=data.query('tendencies==@tendency and uncertainty==@unc and trans_probs<15 and epoch>2 and correct==1'), hue='trans_probs', style='trans_probs', dashes=False, markers=True, ci = 95, estimator=np.nanmean, linewidth=3, palette="crest")
    #plt.ylim([600,1500])
    plt.savefig("numinrun_RTI_RT.svg")
    plt.xticks([1,2,3,4,5])
    plt.show()
    plt.figure(figsize=(3.5,2.5))
    #plt.title("tendency "+str(tendency)+", trans "+str(trans))
    sns.lineplot(x='num_in_run', y='correct', data=data.query('tendencies==@tendency and uncertainty==@unc and trans_probs<15 and epoch>2'), hue='trans_probs', style='trans_probs', dashes=False, markers=True, ci = 95, estimator=np.nanmean, linewidth=3, palette="crest")
    plt.ylim([0.5,1.0])
    plt.gca().invert_yaxis()
    plt.savefig("numinrun_RTI_correct.svg")
    plt.xticks([1,2,3,4,5])
    plt.show()
    # plt.figure()
    # sns.lineplot(x='num_in_run', y='RT', data=data.query('congruent == 1 and trial_num > 50'), ci = 95, estimator=np.nanmedian, linewidth=3)
    # sns.lineplot(x='num_in_run', y='RT', data=data.query('congruent == 0 and trial_num > 50'), ci = 95, estimator=np.nanmedian, linewidth=3)
    # plt.show()

    return data, data_single#

def analyze_single_simulations(folder, plot=True, non_dec_time=0, t_s=1):

    tendencies = [1000]
    probs = [90,95,99]#[80,85,90,91,92,93,94,95,96,97,98,99]
    uncertainties = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,8,10]
    run_name = "acsel_single_h"+str(int(tendencies[0]))+"_t"+str(probs[0])+"_u"+str(uncertainties[0])+"_f3.5_ut0.99.json"
    fname = os.path.join(folder, run_name)

    jsonpickle_numpy.register_handlers()

    with open(fname, 'r') as infile:
        data = json.load(infile)

    worlds_old = pickle.decode(data)

    repetitions = len(worlds_old)
    trials = worlds_old[0].trials
    num_types = len(tendencies)*len(probs)*len(uncertainties)
    correct = np.zeros(repetitions*trials*num_types)
    RT = np.zeros(repetitions*trials*num_types)
    agent = np.zeros(repetitions*trials*num_types)
    num_in_run = np.zeros(repetitions*trials*num_types)
    congruent = np.zeros(repetitions*trials*num_types)
    trial_num = np.zeros(repetitions*trials*num_types)
    epoch = np.zeros(repetitions*trials*num_types)
    tend_arr = np.zeros(repetitions*trials*num_types)
    prob_arr = np.zeros(repetitions*trials*num_types)
    unc_arr  = np.zeros(repetitions*trials*num_types)

    sim_type = 0
    for tendency in tendencies:#,3,5,10,30,50,100]: #1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]:
        for trans in probs:#[100,99,98,97,96,95,94]:
            for unc in uncertainties:

                run_name = "acsel_single_h"+str(int(tendency))+"_t"+str(trans)+"_u"+str(unc)+"_f3.5_ut0.99.json"
                fname = os.path.join(folder, run_name)

                jsonpickle_numpy.register_handlers()

                with open(fname, 'r') as infile:
                    data = json.load(infile)

                worlds_old = pickle.decode(data)

                repetitions = len(worlds_old)
                trials = worlds_old[0].trials
                
                print("single", len(worlds_old), tendency, trans, unc)

                offset = sim_type*repetitions*trials

                for i in range(repetitions):
                    w = worlds_old[i]
                    correct[offset+i*trials:offset+(i+1)*trials] = (w.actions[:,0] == w.environment.correct_choice).astype(int)
                    RT[offset+i*trials:offset+(i+1)*trials] = t_s*w.agent.action_selection.RT[:,0] + non_dec_time
                    agent[offset+i*trials:offset+(i+1)*trials] = i
                    num_in_run[offset+i*trials:offset+(i+1)*trials] = w.environment.num_in_run
                    congruent[offset+i*trials:offset+(i+1)*trials] = np.logical_not(w.environment.congruent)
                    trial_num[offset+i*trials:offset+(i+1)*trials] = np.arange(0,trials)
                    epoch[offset+i*trials:offset+(i+1)*trials] = [-1]*10 + [0]*10 + [1]*20 + [2]*30 + [3]*(trials-70)
                    tend_arr[offset+i*trials:offset+(i+1)*trials] = tendency
                    prob_arr[offset+i*trials:offset+(i+1)*trials] = 100-trans
                    unc_arr[offset+i*trials:offset+(i+1)*trials] = unc

                sim_type+=1

    data_dict = {"correct": correct, "RT": RT, "agent": agent,
                 "num_in_run": num_in_run, "congruent": congruent,
                 "trial_num": trial_num, "epoch": epoch,
                 "uncertainty": unc_arr, "tendencies": tend_arr,
                 "trans_probs": prob_arr}
    data = pd.DataFrame(data_dict)

    if plot:
        # plt.figure()
        # for i in range(0,3):
        #     sns.lineplot(x='num_in_run', y='RT', data=data.query('epoch == @i'), style='congruent', label=str(i), ci = 95, estimator=np.nanmean, linewidth=3)
        # plt.show()
        tendency=1000
        trans=95
        unc=1
        trans = 100-trans
        
        plt.figure()
        plt.title("tendency "+str(tendency)+", trans "+str(trans)+", unc "+str(unc))
        sns.lineplot(x='num_in_run', y='RT', data=data.query('epoch >= 0 and epoch < 3 and tendencies==@tendency and uncertainty==@unc and trans_probs==@trans'), style='congruent', hue='epoch', ci = 95, estimator=np.nanmean, linewidth=3)
        plt.ylim([0,1800])
        plt.show()
        plt.figure()
        plt.title("tendency "+str(tendency)+", trans "+str(trans)+", unc "+str(unc))
        sns.lineplot(x='num_in_run', y='correct', data=data.query('epoch >= 0 and epoch < 3 and tendencies==@tendency and uncertainty==@unc and trans_probs==@trans'), style='congruent', hue='epoch', ci = 95, estimator=np.nanmean, linewidth=3)
        plt.ylim([0.,1.])
        plt.show()
        plt.figure()
        #plt.title("tendency "+str(tendency)+", trans "+str(trans))
        sns.lineplot(x='uncertainty', y='RT', data=data.query('tendencies==@tendency and trans_probs==@trans and uncertainty<1 and uncertainty>0'), style='num_in_run', ci = 95, estimator=np.nanmean, linewidth=3)
        plt.ylim([600,1500])
        plt.gca().invert_xaxis()
        plt.savefig("CTI_single.svg")
        plt.show()
        plt.figure()
        plt.title("tendency "+str(tendency)+", trans "+str(trans))
        sns.lineplot(x='num_in_run', y='RT', data=data.query('tendencies==@tendency and trans_probs==@trans and uncertainty<11 and epoch>2'), hue='uncertainty', style='congruent', ci = 95, estimator=np.nanmean, linewidth=3)
        plt.ylim([400,1800])
        plt.show()
        plt.figure()
        plt.title("tendency "+str(tendency)+", trans "+str(trans))
        sns.lineplot(x='num_in_run', y='correct', data=data.query('tendencies==@tendency and trans_probs==@trans and uncertainty<11 and epoch>2'), hue='uncertainty', style='congruent', ci = 95, estimator=np.nanmean, linewidth=3)
        plt.ylim([0,1])
        plt.show()
        # plt.figure()
        # sns.lineplot(x='num_in_run', y='RT', data=data.query('congruent == 1 and trial_num > 50'), ci = 95, estimator=np.nanmedian, linewidth=3)
        # sns.lineplot(x='num_in_run', y='RT', data=data.query('congruent == 0 and trial_num > 50'), ci = 95, estimator=np.nanmedian, linewidth=3)
        # plt.show()

    return data


def analyze_onecontext_simulations(folder):

    tendencies = [1000]
    probs = [95]#
    uncertainties = [1]
    run_name = "acsel_switching_h"+str(int(tendencies[0]))+"_t"+str(probs[0])+"_u"+str(uncertainties[0])+"_f3.5_ut0.99_onecontext.json"
    print(run_name)
    fname = os.path.join(folder, run_name)

    jsonpickle_numpy.register_handlers()

    with open(fname, 'r') as infile:
        data = json.load(infile)

    worlds_old = pickle.decode(data)

    repetitions = len(worlds_old)
    trials = worlds_old[0].trials
    num_types = len(tendencies)*len(probs)*len(uncertainties)
    correct = np.zeros(repetitions*trials*num_types)
    RT = np.zeros(repetitions*trials*num_types)
    agent = np.zeros(repetitions*trials*num_types)
    num_in_run = np.zeros(repetitions*trials*num_types)
    congruent = np.zeros(repetitions*trials*num_types)
    trial_num = np.zeros(repetitions*trials*num_types)
    epoch = np.zeros(repetitions*trials*num_types)
    tend_arr = np.zeros(repetitions*trials*num_types)
    prob_arr = np.zeros(repetitions*trials*num_types)
    unc_arr  = np.zeros(repetitions*trials*num_types)
    non_dec_time = 100
    t_s = 0.2

    sim_type = 0
    for tendency in tendencies:#,3,5,10,30,50,100]: #1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]:
        for trans in probs:#[100,99,98,97,96,95,94]:
            for unc in uncertainties:

                run_name = "acsel_switching_h"+str(int(tendency))+"_t"+str(trans)+"_u"+str(unc)+"_f3.5_ut0.99_onecontext.json"
                fname = os.path.join(folder, run_name)

                jsonpickle_numpy.register_handlers()

                with open(fname, 'r') as infile:
                    data = json.load(infile)

                worlds_old = pickle.decode(data)

                repetitions = len(worlds_old)
                trials = worlds_old[0].trials
                
                print("switching", len(worlds_old), tendency, trans, unc)

                offset = sim_type*repetitions*trials

                for i in range(repetitions):
                    w = worlds_old[i]
                    correct[offset+i*trials:offset+(i+1)*trials] = (w.actions[:,0] == w.environment.correct_choice).astype(int)
                    RT[offset+i*trials:offset+(i+1)*trials] = t_s*w.agent.action_selection.RT[:,0] + non_dec_time
                    agent[offset+i*trials:offset+(i+1)*trials] = i
                    num_in_run[offset+i*trials:offset+(i+1)*trials] = w.environment.num_in_run
                    congruent[offset+i*trials:offset+(i+1)*trials] = np.logical_not(w.environment.congruent)
                    trial_num[offset+i*trials:offset+(i+1)*trials] = np.arange(0,trials)
                    epoch[offset+i*trials:offset+(i+1)*trials] = [-1]*10 + [0]*10 + [1]*20 + [2]*30 + [3]*(trials-70)#
                    tend_arr[offset+i*trials:offset+(i+1)*trials] = tendency
                    prob_arr[offset+i*trials:offset+(i+1)*trials] = 100-trans
                    unc_arr[offset+i*trials:offset+(i+1)*trials] = unc

                sim_type+=1

    data_dict = {"correct": correct, "RT": RT, "agent": agent,
                 "num_in_run": num_in_run, "congruent": congruent,
                 "trial_num": trial_num, "epoch": epoch,
                 "uncertainty": unc_arr, "tendencies": tend_arr,
                 "trans_probs": prob_arr}
    data = pd.DataFrame(data_dict)

    # plt.figure()
    # for i in range(0,3):
    #     sns.lineplot(x='num_in_run', y='RT', data=data.query('epoch == @i'), style='congruent', label=str(i), ci = 95, estimator=np.nanmean, linewidth=3)
    # plt.show()
    tendency=1000
    trans=95
    unc=1
    trans = 100-trans

    # RT & accuracy as a function of num in run for congruent and incongruent trials and for different training durations (Fig 3 in Steyvers 2019)
    #sns.set_style("white")switching
    current_palette = sns.color_palette("colorblind")
    plot_palette = [(236/255,98/255,103/255), current_palette[2], current_palette[0]]
    sns.set_style("ticks")
    plt.figure(figsize=(3.5,5))
    #plt.title("tendency "+str(tendency)+", trans "+str(trans)+", unc "+str(unc))
    sns.lineplot(x='num_in_run', y='RT', data=data.query('epoch >= 0 and epoch < 3 and tendencies==@tendency and uncertainty==@unc and trans_probs==@trans'), style='congruent', hue='epoch', markers=True, ci = 95, estimator=np.nanmean, linewidth=3, palette=plot_palette)
    #plt.ylim([600,2000])
    plt.xticks([1,2,3,4,5], fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Trial after switch", fontsize=16)
    plt.ylabel("RT", fontsize=16)
    plt.savefig("numinrun_congruent_RT.svg")
    plt.show()
    plt.figure(figsize=(3.5,5))
    #plt.title("tendency "+str(tendency)+", trans "+str(trans)+", unc "+str(unc))
    sns.lineplot(x='num_in_run', y='correct', data=data.query('epoch >= 0 and epoch < 3 and tendencies==@tendency and uncertainty==@unc and trans_probs==@trans'), style='congruent', hue='epoch', markers=True, ci = 95, estimator=np.nanmean, linewidth=3, palette=plot_palette)
    #plt.ylim([0.65,1.])
    plt.xticks([1,2,3,4,5], fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Trial after switch", fontsize=16)
    plt.ylabel("Prop correct", fontsize=16)
    plt.savefig("numinrun_congruent_correct.svg")
    plt.show()

    return data, data_single


def main():

    """
    set parameters
    """

    folder = "data"
    if not os.path.isdir(folder):
        os.mkdir(folder)

    repetitions = 50

    """
    run simulations
    """
    # runs simulations with varying habitual tendency and reward probability
    # results are stored in data folder
    #run_switching_simulations(repetitions, folder)
    #run_single_task_simulations(repetitions, folder)
    #run_switching_simulations_one_context(repetitions, folder)

    #data, data_single = analyze_switching_simulations(folder) #
    #data = analyze_single_simulations(folder)
    data = analyze_onecontext_simulations(folder)
    return data, data_single


if __name__ == "__main__":
    data, data_single = main() #