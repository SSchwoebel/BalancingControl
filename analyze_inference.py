#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 11:46:14 2022

@author: sarah
"""

import torch as ar
array = ar.tensor

import pyro
import pyro.distributions as dist
import agent as agt
import perception as prc
import action_selection as asl
import inference as inf

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

#device = ar.device("cuda") if ar.cuda.is_available() else ar.device("cpu")
#device = ar.device("cuda")
#device = ar.device("cpu")

from inference import device
import distributions as analytical_dists
import numpy as np

folder = "data"

tendencies = []
pol_lambdas = []
rew_lambdas = []
dec_temps = []

alpha_lamb_pis = []
beta_lamb_pis = []
alpha_lamb_rs = []
beta_lamb_rs = []
alpha_hs = []
beta_hs = []
concentration_dec_temps = []
rate_dec_temps = []

for i in range(2):
    for tend in [1, 2, 3, 4, 5, 10, 100]:
        for pl in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for rl in [0.1, 0.3, 0.5, 0.7, 0.9]:
                for dt in [1.0, 3.0, 5.0, 7.0]:
                    
                    inf_name = "twostage_inference"+str(i)+"_pl"+str(pl)+"_rl"+str(rl)+"_dt"+str(dt)+"_tend"+str(tend)+".json"
                    fname = os.path.join(folder, inf_name)
    
                    jsonpickle_numpy.register_handlers()
                    
                    if os.path.isfile(fname):
                        
                        with open(fname, 'r') as infile:
                            data = json.load(infile)
        
                        param_dict = pickle.decode(data)
                        
                        tendencies.append(1./tend)
                        pol_lambdas.append(pl)
                        rew_lambdas.append(rl)
                        dec_temps.append(dt)
                        
                        alpha_lamb_pis.append(param_dict["alpha_lamb_pi"][0])
                        beta_lamb_pis.append(param_dict["beta_lamb_pi"][0])
                        alpha_lamb_rs.append(param_dict["alpha_lamb_r"][0])
                        beta_lamb_rs.append(param_dict["beta_lamb_r"][0])
                        alpha_hs.append(param_dict["alpha_h"][0])
                        beta_hs.append(param_dict["beta_h"][0])
                        concentration_dec_temps.append(param_dict["concentration_dec_temp"][0])
                        rate_dec_temps.append(param_dict["rate_dec_temp"][0])
                    

mean_lamb_pis = analytical_dists.BetaMean(np.array(alpha_lamb_pis), np.array(beta_lamb_pis))
mode_lamb_pis = analytical_dists.BetaMode(np.array(alpha_lamb_pis), np.array(beta_lamb_pis))

mean_lamb_rs = analytical_dists.BetaMean(np.array(alpha_lamb_rs), np.array(beta_lamb_rs))
mode_lamb_rs = analytical_dists.BetaMode(np.array(alpha_lamb_rs), np.array(beta_lamb_rs))

mean_hs = analytical_dists.BetaMean(np.array(alpha_hs), np.array(beta_hs))
mode_hs = analytical_dists.BetaMode(np.array(alpha_hs), np.array(beta_hs))

mean_dec_temps = analytical_dists.GammaMean(np.array(concentration_dec_temps), np.array(rate_dec_temps))
mode_dec_temps = analytical_dists.GammaMode(np.array(concentration_dec_temps), np.array(rate_dec_temps))

DataFrame = pd.DataFrame({'pol_lambdas': pol_lambdas, 'rew_lambdas': rew_lambdas, 
                          'dec_temps': dec_temps, 'tendencies': tendencies,
                          'alpha_lamb_pis': alpha_lamb_pis, 'beta_lamb_pis': beta_lamb_pis, 
                          'mean_lamb_pis': mean_lamb_pis, 'mode_lamb_pis': mode_lamb_pis,
                          'alpha_lamb_rs': alpha_lamb_rs, 'beta_lamb_rs': beta_lamb_rs, 
                          'mean_lamb_rs': mean_lamb_rs, 'mode_lamb_rs': mode_lamb_rs,
                          'alpha_hs': alpha_hs, 'beta_hs': beta_hs, 
                          'mean_hs': mean_hs, 'mode_hs': mode_hs,
                          'concentration_dec_temps': concentration_dec_temps, 'rate_dec_temps': rate_dec_temps,
                          'mean_dec_temps': mean_dec_temps, 'mode_dec_temps': mode_dec_temps})

plt.figure()
sns.scatterplot(data=DataFrame, x='pol_lambdas', y='mean_lamb_pis')
plt.show()

plt.figure()
sns.scatterplot(data=DataFrame, x='rew_lambdas', y='mean_lamb_rs')
plt.show()

plt.figure()
sns.scatterplot(data=DataFrame, x='tendencies', y='mean_hs')
plt.show()