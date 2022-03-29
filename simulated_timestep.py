#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 16:10:45 2022

@author: sarah
"""

import jax.numpy as jnp
import jax.scipy.special as scs


class Agent():
    def __init__(self, trans_matrix, obs_matrix, opt_params):
        
        self.trans_matrix = trans_matrix
        self.obs_matrix = obs_matrix
        
        for 