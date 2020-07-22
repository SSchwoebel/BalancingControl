# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 14:26:41 2020

@author: Maarten
"""

import numpy as np

#from analysis import plot_analyses, plot_analyses_training, plot_analyses_deval
#from misc import *

import world 
import environment as env
import agent as agt
import perception as prc
import action_selection as asl
import itertools
import matplotlib.pylab as plt
#from matplotlib.animation import FuncAnimation
#from multiprocessing import Pool
#from matplotlib.colors import LinearSegmentedColormap
import jsonpickle as pickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import json
import seaborn as sns
#import pandas as pd
import os
#import scipy as sc
#import scipy.signal as ss
#import bottleneck as bn
import gc
np.set_printoptions(threshold = 100000, precision = 5)



"""
run function
"""
def run_agent(par_list, trials, T, ns, na, nr, nc, deval=False):
    
    #set parameters:
    #learn_pol: initial concentration paramter for policy prior
    #avg: True for average action selection, False for maximum selection
    #Rho: Environment's reward generation probabilities as a function of time
    #utility: goal prior, preference p(o)
    learn_pol, avg, Rho, utility = par_list
    
    
    """
    create matrices
    """
    
    
    #generating probability of observations in each state
    A = np.eye(ns) # observation uncertainty = 0
        
    
    #state transition generative probability (matrix)
    B = np.zeros((ns, ns, na))
    
    L = 4 # grid length
    c = 1.0 # transition uncertainty = 0
    actions = np.array([[-1,0],[0,-1], [1,0], [0,1]])
    
    cert_arr = np.zeros(ns)
    for s in range(ns):
        x = s//L
        y = s%L
            
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
                
            
    # agent's initial estimate of reward generation probability
    C_agent = np.zeros((nr, ns, nc))
    for c in range(nc):
        C_agent[:,:,c] = np.array([[0.0, 1.0] if state == 11 else [1.0, 0.0] for state in range(ns)]).T
    
    # context transition matrix
    transition_matrix_context = np.zeros((nc, nc))
    transition_matrix_context[:,:] = 1.0                            
    """
    create environment (grid world)
    """
    
    environment = env.GridWorld(A, B, Rho, trials = trials, T = T)
    
    
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
    
    state_prior[1] = 1.0

    """
    set action selection method
    """

    if avg:
    
        ac_sel = asl.AveragedSelector(trials = trials, T = T, 
                                      number_of_actions = na)
    else:
        
        ac_sel = asl.MaxSelector(trials = trials, T = T, 
                                      number_of_actions = na)
    
  
    """
    set up agent
    """

    # perception
    bayes_prc = prc.HierarchicalPerception(A, B, C_agent, 
                                           transition_matrix_context, 
                                           state_prior, 
                                           utility, 
                                           prior_pi, 
                                           alphas, 
                                           dirichlet_rew_params=None, 
                                           T=T)
    
    # agent
    bayes_pln = agt.BayesianPlanner(bayes_prc, ac_sel, pol,
                      trials = trials, T = T,
                      prior_states = state_prior,
                      prior_policies = prior_pi,
                      number_of_states = ns, 
                      prior_context = None,
                      learn_habit = True,
                      number_of_policies = npi,
                      number_of_rewards = nr)
    

    """
    create world
    """
    
    w = world.World(environment, bayes_pln, trials = trials, T = T)
    
    """
    simulate experiment
    """
    if not deval:       
        w.simulate_experiment(range(trials))
        
    else:
        w.simulate_experiment(range(trials//2))
        # reset utility to implement devaluation
        ut = utility[1:].sum()
        bayes_prc.prior_rewards[2:] = ut / (nr-2)
        bayes_prc.prior_rewards[:2] = (1-ut) / 2
        
        w.simulate_experiment(range(trials//2, trials))
    

############################################################################## 
    """
    plot and evaluate results
    """
    #find successful and unsuccessful runs
    goal = 11
    successfull = np.where(environment.hidden_states[:,-1]==goal)[0]
    unsuccessfull = np.where(environment.hidden_states[:,-1]!=goal)[0]
    total  = len(successfull)
    
    #set up figure
    factor = 2
    fig = plt.figure(figsize=[factor*5,factor*5])
    
    ax = fig.gca()
        
    #plot start and goal state
    start_goal = np.zeros((L,L))
    
    start_goal[0,1] = 1.
    start_goal[-2,-1] = -1.
    
    u = sns.heatmap(start_goal, vmin=-1, vmax=1, zorder=2,
                    ax = ax, linewidths = 2, alpha=0.7, cmap="RdBu_r",
                    xticklabels = False,
                    yticklabels = False,
                    cbar=False)
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
        
        # #plot uncertainties
        # if obs_unc:
        #     plt.plot(x,y, 'o', color=(219/256,122/256,147/256), markersize=factor*12/(A[i,i])**2, alpha=1.)
        # if state_unc:
        #     plt.plot(x,y, 'o', color=(100/256,149/256,237/256), markersize=factor*12/(cert_arr[i])**2, alpha=1.)
                
        #plot unsuccessful paths    
        for j in range(2):
            
            if un[i,j]>0.0:
                if j == 0:
                    xp = x + [x[0] + 1]
                    yp = y + [y[0] + 0]
                if j == 1:
                    xp = x + [x[0] + 0]
                    yp = y + [y[0] + 1]
                    
                plt.plot(xp,yp, '-', color='r', linewidth=factor*30*un[i,j],
                         zorder = 9, alpha=0.6)
    
    #set plot title
    plt.title("Planning: successful "+str(round(100*total/trials))+"%", fontsize=factor*9)           
    
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
                plt.plot(xp,yp, '-', color='c', linewidth=factor*30*n[i,j],
                         zorder = 10, alpha=0.6)
                
                
    print("percent won", total/trials, "state prior", np.amax(utility))
    
    
    plt.show()
##############################################################################    
    
    return w

def main():

    """
    set parameters
    """
    na = 4 # number of actions
    T = na + 1 # number of time steps in each miniblock
    ns = 16 # number of states    
    nr = 2 # number of rewards (0 and 1)
    nc = 1 # number of contexts
    # conditional distribution over rewards given states
    Rho = np.array([[0.0, 1.0] if state == 11 else [1.0, 0.0] for state in range(ns)]).T 
    
    avg = True # type of action selection (average vs. mode)
    u = 0.99 
    utility = np.array([1-u, u]) # prior over rewards given observations
    # modifies concentration parameters of dirichlet distribution
    # smaller values -> stronger habits
    learn_pol = 1000.0
    
    mb = 50 # number of miniblocks
    rep = 1 # number of repetitions
    
   
    """
    run simulations
    """
    
    folder = "data"
    if not os.path.isdir(folder):
        os.mkdir(folder)
            
    worlds = [None]*rep    
    for i in range(rep):
        worlds[i] = run_agent((learn_pol, avg, Rho, utility), mb, T, ns, na, nr, nc)
        
    run_name = "h_" + str(round(learn_pol, 1)) + "_mb_" + str(mb) + "_rep_" + str(rep) + ".json"
    fname = os.path.join(folder, run_name)
    
    jsonpickle_numpy.register_handlers()
    pickled = pickle.encode(worlds)
    with open(fname, 'w') as outfile:
        json.dump(pickled, outfile)    
 
    gc.collect()   
    
    return worlds 
 

if __name__ == "__main__":
    results = main()

# """
# reload data
# """
# with open(fname, 'r') as infile:
#     data = json.load(infile)
    
# w_new = pickle.decode(data)
