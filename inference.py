#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 18:31:45 2020

@author: sarah
"""

import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-darkgrid')
print('Running on PyMC3 v{}'.format(pm.__version__))

# for reproducibility here's some version info for modules used in this notebook
import theano
import theano.tensor as tt



class LogLike(tt.Op):

    """
    Specify what type of object will be passed and returned to the Op when it is
    called. In our case we will be passing it a vector of values (the parameters
    that define our model) and returning a single "scalar" value (the
    log-likelihood)
    """
    itypes = [tt.dvector] # expects a vector of parameter values when called
    otypes = [tt.dmatrix, tt.dmatrix] # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike, fixed):
        """
        Initialise the Op with various things that our log-likelihood function
        requires. Below are the things that are needed in this particular
        example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that our function requires.
        """

        # add inputs as class attributes
        self.likelihood = loglike
        self.fixed = fixed

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        parameters, = inputs  # this will contain my variables

        # call the log-likelihood function
        probs_a, probs_r = self.likelihood(parameters,self.fixed)

        outputs[0][0] = np.array(probs_a) # output the log-likelihood
        outputs[1][0] = np.array(probs_r) # output the log-likelihood