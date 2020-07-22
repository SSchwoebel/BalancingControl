#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 17:44:57 2020

@author: sarah
"""

import numpy as np

import world 
import agent as agt
import perception as prc
import inference as infer
import matplotlib.pylab as plt
import jsonpickle as pickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import json
#import seaborn as sns
import pandas as pd
import os
np.set_printoptions(threshold = 100000, precision = 5)

import pymc3 as pm
import theano
import theano.tensor as tt
import gc
    


def run_fitting(folder):
    
    for tendency in [1]:
        for trans in [99]:
            print(tendency, trans)
            traces = []
                
            run_name ="h"+str(tendency)+"_t"+str(trans)+"_p90_train100.json"
            fname = os.path.join(folder, run_name)
            
            jsonpickle_numpy.register_handlers()
            
            with open(fname, 'r') as infile:
                data = json.load(infile)
                
            worlds_old = pickle.decode(data)
            
            inferrer = infer.Inferrer(worlds_old)
            
            traces = inferrer.run_single_inference(5, ndraws=3000, nburn=1000, cores=4)
            
            #traces = inferrer.run_group_inference(ndraws=3000, nburn=1000, cores=4)
                
            fname = os.path.join(folder, run_name[:-5]+"_traces.json")
                    
            jsonpickle_numpy.register_handlers()
            pickled = pickle.encode(traces)
            with open(fname, 'w') as outfile:
                json.dump(pickled, outfile)
            
            pickled = 0
            traces = 0
            
            gc.collect()
            
            
def load_fitting(folder):
    
    for tendency in [1]:
        for trans in [99]:
            print(tendency, trans)
            traces = []
                
            run_name ="h"+str(tendency)+"_t"+str(trans)+"_p90_train100.json"
            fname = os.path.join(folder, run_name)
            
            jsonpickle_numpy.register_handlers()
            
            with open(fname, 'r') as infile:
                data = json.load(infile)
                
            worlds_old = pickle.decode(data)
            
            repetitions = len(worlds_old)
            
            inferrer = infer.Inferrer(worlds_old)
                
            run_name ="h"+str(tendency)+"_t"+str(trans)+"_p90_train100.json"
            
            fname = os.path.join(folder, run_name[:-5]+"_traces.json")
            
            with open(fname, 'r') as infile:
                data = json.load(infile)
                
            trace_name = pickle.decode(data)
            
            inferrer.plot_inference(trace_name, model='single', idx=5)
            
            pickled = 0
            traces = 0
            
            gc.collect()
                        


def main():

    """
    set parameters
    """
    
    T = 2 #number of time steps in each trial
    nb = 2
    no = nb+1 #number of observations
    ns = nb+1 #number of states
    na = nb #number of actions
    npi = na**(T-1)
    nr = nb+1
    nc = nb #1
    n_parallel = 1
    
    folder = "data"
    if not os.path.isdir(folder):
        raise Exception("run_rew_prob_simulations() needs to be run first")
        
    run_args = [T, ns, na, nr, nc]
    
    u = 0.99
    utility = np.zeros(nr)
    for i in range(1,nr):
        utility[i] = u/(nr-1)
    utility[0] = (1.-u)
    
    repetitions = 20
    
    avg = True
    
    #run_fitting(folder)
    
    load_fitting(folder)
    
    
if __name__ == "__main__":
    

    """
    set parameters
    """
    
    T = 2 #number of time steps in each trial
    nb = 2
    no = nb+1 #number of observations
    ns = nb+1 #number of states
    na = nb #number of actions
    npi = na**(T-1)
    nr = nb+1
    nc = nb #1
    n_parallel = 1
    
    folder = "data"
    if not os.path.isdir(folder):
        raise Exception("run_rew_prob_simulations() needs to be run first")
        
    run_args = [T, ns, na, nr, nc]
    
    u = 0.99
    utility = np.zeros(nr)
    for i in range(1,nr):
        utility[i] = u/(nr-1)
    utility[0] = (1.-u)
    
    repetitions = 20
    
    avg = True
    
    run_fitting(folder)
    
    load_fitting(folder)
    
