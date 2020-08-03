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
import seaborn as sns
import pandas as pd
import os
np.set_printoptions(threshold = 100000, precision = 5)

import pymc3 as pm
import theano
import theano.tensor as tt
import gc



def run_fitting(folder):

    samples = []
    tendencies = [1, 3, 5, 10, 30, 50, 100]
    for tendency in tendencies:
        for trans in [99]:
            for prob in [90]:
                for train in [562]:
                    print(tendency, trans)

                    run_name ="h"+str(tendency)+"_t"+str(trans)+"_p"+str(prob)+"_train"+str(train)+".json"
                    fname = os.path.join(folder, run_name)

                    jsonpickle_numpy.register_handlers()

                    with open(fname, 'r') as infile:
                        data = json.load(infile)

                    worlds_old = pickle.decode(data)

                    test_trials = list(range(0,50)) + list(range(train,150))

                    inferrer = infer.Inferrer(worlds_old[:20], 0.01, 1., test_trials=test_trials)
                    # print(1./inferrer.sample_space)
                    # print(inferrer.likelihood.mean(axis=0))
                    # plt.figure()
                    # plt.plot(inferrer.likelihood.mean(axis=0), '.')
                    # plt.show()

                    inferrer.run_single_inference(ndraws=15000, nburn=5000, cores=4)
                    samples.append(inferrer.samples)

                    fname = os.path.join(folder, run_name[:-5]+"_samples.json")

                    jsonpickle_numpy.register_handlers()
                    pickled = pickle.encode([samples[-1], inferrer.sample_space])
                    with open(fname, 'w') as outfile:
                        json.dump(pickled, outfile)

                    pickled = 0

                    gc.collect()

                    # plt.figure()
                    # plt.hist(samples[-1])
                    # plt.show()


    # labels = np.tile(1./np.array(tendencies), (samples[-1].shape[0], 1)).reshape(-1, order='f')
    # data = -np.array(samples).flatten()
    # pd_h_samples = pd.DataFrame(data={'inferred tendencies': data, 'true tendencies': labels})

    # plt.figure()
    # ax = plt.gca()
    # ax.set_ylim([-12,0])
    # yticklabels = [""]*len(inferrer.sample_space)
    # yticklabels[0] = 0.01
    # yticklabels[-1] = 1.
    # yticklabels[len(inferrer.sample_space)//2] = 0.1
    # ax.set_yticklabels(yticklabels)
    # sns.boxenplot(data=pd_h_samples, x='true tendencies', y='inferred tendencies', ax=ax)
    # #sns.stripplot(data=pd_h_samples, x='tendencies', y='samples', size=4, color='grey')
    # #plt.ylim([0,1])
    # plt.show()

    #return samples, inferrer


def load_fitting(folder):

    samples = []
    tendencies = [1, 3, 5, 10, 30, 50, 100]
    for tendency in tendencies:
        for trans in [99]:
            for prob in [90]:
                for train in [562]:
                    print(tendency, trans)
                    traces = []

                    run_name ="h"+str(tendency)+"_t"+str(trans)+"_p"+str(prob)+"_train"+str(train)+".json"
                    fname = os.path.join(folder, run_name)

                    fname = os.path.join(folder, run_name[:-5]+"_samples.json")

                    jsonpickle_numpy.register_handlers()
                    with open(fname, 'r') as infile:
                        data = json.load(infile)

                    curr_samples, sample_space = pickle.decode(data)

                    samples.append(curr_samples)

    labels = np.tile(1./np.array(tendencies), (samples[-1].shape[0], 1)).reshape(-1, order='f')
    data = -np.array(samples).flatten()
    pd_h_samples = pd.DataFrame(data={'inferred tendencies': data, 'true tendencies': labels})

    plt.figure()
    ax = plt.gca()
    # ax.set_ylim([-13,1])
    ax.set_yticks(np.arange(-2.,0.5,0.25))
    yticklabels = [""]*len(sample_space)
    yticklabels[0] = 0.01
    yticklabels[-1] = 1.
    yticklabels[len(sample_space)//2] = 0.1
    ax.set_yticklabels(yticklabels)
    sns.boxenplot(data=pd_h_samples, x='true tendencies', y='inferred tendencies', ax=ax)
    #sns.stripplot(data=pd_h_samples, x='tendencies', y='samples', size=4, color='grey')
    #plt.ylim([0,1])
    plt.savefig('train_inference.svg')
    plt.show()

    plt.figure()
    ax = plt.gca()
    # ax.set_ylim([-len(sample_space)-1,1])
    # ax.set_yticks(range(-len(sample_space),0+1,1))
    # yticklabels = [""]*len(sample_space)
    # yticklabels[0] = 0.01
    # yticklabels[-1] = 1.
    # yticklabels[len(sample_space)//2] = 0.1
    # ax.set_yticklabels(yticklabels)
    sns.violinplot(data=pd_h_samples, x='true tendencies', y='inferred tendencies', ax=ax)
    #sns.stripplot(data=pd_h_samples, x='tendencies', y='samples', size=4, color='grey')
    #plt.ylim([0,1])
    plt.show()



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

