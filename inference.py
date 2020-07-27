#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 18:31:45 2020

@author: sarah
"""

import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt

import world
import agent as agt
import perception as prc

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


class Inferrer:
    def __init__(self, worlds, min_h, max_h, nvals=13, test_trials=None):

        self.nruns = len(worlds)
        self.nvals = nvals

        w = worlds[0]

        self.create_sample_space(min_h, max_h)

        self.setup_agent(w, test_trials=test_trials)

        self.actions = np.array([w.actions[:,0] for w in worlds])

        self.rewards = np.array([w.rewards[:,1] for w in worlds])

        self.inferrer = LogLike(self.agent.fit_model, self.fixed)

    def setup_agent(self, w, first_trial=0, test_trials=None):

        ns = w.environment.Theta.shape[0]
        nr = w.environment.Rho.shape[1]
        na = w.environment.Theta.shape[2]
        nc = w.agent.perception.generative_model_rewards.shape[2]
        T = w.T
        trials = w.trials
        observations = w.observations.copy()
        rewards = w.rewards.copy()
        actions = w.actions.copy()
        utility = w.agent.perception.prior_rewards.copy()
        A = w.agent.perception.generative_model_observations.copy()
        B = w.agent.perception.generative_model_states.copy()

        if test_trials is None:
            test_trials = np.arange(0, trials, 1, dtype=int)

        transition_matrix_context = w.agent.perception.transition_matrix_context.copy()

        # concentration parameters
        C_alphas = np.ones((nr, ns, nc))
        # initialize state in front of levers so that agent knows it yields no reward
        C_alphas[0,0,:] = 100
        for i in range(1,nr):
            C_alphas[i,0,:] = 1

        # agent's initial estimate of reward generation probability
        C_agent = np.zeros((nr, ns, nc))
        for c in range(nc):
            C_agent[:,:,c] = np.array([(C_alphas[:,i,c])/(C_alphas[:,i,c]).sum() for i in range(ns)]).T

        pol = w.agent.policies.copy()

        #pol = pol[-2:]
        npi = pol.shape[0]

        # prior over policies

        alpha = 1
        alphas = np.zeros_like(w.agent.perception.dirichlet_pol_params.copy()) + alpha

        prior_pi = alphas.copy()
        prior_pi /= prior_pi.sum(axis=0)

        state_prior = np.zeros((ns))

        state_prior[0] = 1.

        prior_context = np.zeros((nc)) + 1./(nc)#np.dot(transition_matrix_context, w.agent.posterior_context[-1,-1])

    #    prior_context[0] = 1.

        pol_par = alphas

        # perception
        bayes_prc = prc.HierarchicalPerception(A, B, C_agent, transition_matrix_context, state_prior, utility, prior_pi, pol_par, C_alphas, T=T)

        bayes_pln = agt.BayesianPlanner(bayes_prc, None, pol,
                          trials = trials, T = T,
                          prior_states = state_prior,
                          prior_policies = prior_pi,
                          number_of_states = ns,
                          prior_context = prior_context,
                          learn_habit = True,
                          #save_everything = True,
                          number_of_policies = npi,
                          number_of_rewards = nr)

        self.agent = world.FakeWorld(bayes_pln, observations, rewards, actions, trials = trials, T = T)

        self.fixed = {'rew_mod': C_agent, 'beta_rew': C_alphas}

        self.likelihood = np.zeros((self.nruns, len(self.sample_space)), dtype=np.float64)

        for i in range(self.nruns):
            print("precalculating likelihood run ", i)
            for j,h in enumerate(self.sample_space):
                alpha = 1./h
                self.likelihood[i,j] \
                    = self.agent.fit_model(alpha, self.fixed, test_trials)
            self.likelihood[i] /= self.likelihood[i].sum()
            #print(self.likelihood[i])

    def group_likelihood(self, p):

        def func(likelihood):
            #p_D_p = tt.log(tt.dot(likelihood, p).prod())
            logps = tt.log(p) + likelihood
            ps = tt.exp(logps).sum(axis=1)
            p_D_p = tt.log(ps).sum()
            return p_D_p

        return func


    def run_single_inference(self, idx=None, ndraws=300, nburn=100, cores=4):

        minimum = 0.
        maximum = len(self.sample_space) - 1

        # if idx is not None:
        #     runs = [idx]
        # else:
        #     runs = range(self.nruns)

        # with pm.Model() as smodel:

        #     a = pm.Gamma('a', alpha=1., beta=1., shape=self.nvals)
        #     for i in runs:
        #         p_i = pm.Dirichlet('p_{}'.format(i), a=a, shape=self.nvals, observed=self.likelihood[i])

        #     p = pm.Dirichlet('p', a=a, shape=self.nvals)
        #     h = pm.Categorical('h', p)

        if idx is not None:
            runs = [idx]
        else:
            runs = range(self.nruns)

        with pm.Model() as smodel:

            a = [1]*self.nvals #pm.Gamma('a', alpha=1., beta=1., shape=self.nvals)

            p = pm.Dirichlet('p', a=a, shape=self.nvals)

            group_p = pm.DensityDist('gp', self.group_likelihood(p), observed=tt.log(self.likelihood))

            h = pm.Categorical('h', p)

            # uniform priors on h
            #hab_ten = pm.Categorical('h')

            # # convert to a tensor
            # alpha = tt.as_tensor_variable([10**(hab_ten/4.)])
            # probs_a, probs_r = self.inferrer(alpha)

            # # use a DensityDist
            # pm.Categorical('actions', probs_a, observed=self.actions[idx])
            # pm.Categorical('rewards', probs_r, observed=self.rewards[idx])

            # step = pm.Metropolis()#S=np.ones(1)*0.01)

            trace = pm.sample(ndraws, tune=nburn, discard_tuned_samples=True, cores=cores)#, step=step

            # plot the traces
            plt.figure()
            _ = pm.traceplot(trace)#, lines=('h', 1./alpha_true))
            plt.show()
            # plt.figure()
            # _ = pm.plot_posterior(trace, var_names=['h'], ref_val=(1./alpha_true))
            # plt.show()

            # save the traces
            self.samples = trace['h', nburn:]

    def analyze_samples(self, samples):

        dist = np.array([len(np.where(samples==i)[0]) for i in range(len(self.sample_space))], dtype=np.float64)

        dist /= dist.sum()

        return dist

    def analyze_dist(self, dist):

        mean = (dist * self.sample_space).sum()

        variance = (dist * np.abs((self.sample_space - mean)**2)).sum()

        mode = self.sample_space[np.argmax(dist)]

        return mode, mean, variance

    def create_alpha_val(self, sample):

        return 10**(sample / self.factor)

    def create_sample_space(self, min_h, max_h):

        min_exponent = -np.log10(max_h)
        max_exponent = -np.log10(min_h)

        delta = max_exponent - min_exponent

        step = delta / (self.nvals - 1)

        self.sample_space = 10**(-np.arange(min_exponent, max_exponent+step, step))

        self.factor = (self.nvals - 1.) / delta

    def run_group_inference(self, ndraws=300, nburn=100, cores=5):

        curr_model = self.group_model()

        with curr_model:

            step = pm.Metropolis()#S=np.ones(1)*0.01)

            trace = pm.sample(ndraws, tune=nburn, discard_tuned_samples=True, step=step, cores=cores)

            # plot the traces
#            plt.figure()
#            _ = pm.traceplot(trace)#, lines=('h', 1./alpha_true))
#            plt.show()
#            plt.figure()
#            _ = pm.plot_posterior(trace, var_names=['h'], ref_val=(1./alpha_true))
#            plt.show()

            # save the traces
            fname = pm.save_trace(trace)

        return fname

    def single_model(self, idx):

        minimum = 0.
        maximum = 8.
        sample_space = np.arange(minimum, maximum+1, 1)
        sample_space = 1./10**(sample_space/4.)

        with pm.Model() as smodel:
            # uniform priors on h
            hab_ten = pm.DiscreteUniform('h', 0., 8.)

            # convert to a tensor
            alpha = tt.as_tensor_variable([10**(hab_ten/4.)])
            probs_a, probs_r = self.inferrer(alpha)

            # use a DensityDist
            pm.Categorical('actions', probs_a, observed=self.actions[idx])
            pm.Categorical('rewards', probs_r, observed=self.rewards[idx])

        return smodel, sample_space

    def group_model(self):

        with pm.Model() as gmodel:
            # uniform priors on h
            m = pm.DiscreteUniform('h', 0., 20.)
            std = pm.InverseGamma('s', 3., 0.5)
            mean = 2*m+1
            alphas = np.arange(1., 101., 5.)
            p = self.discreteNormal(alphas, mean, std)

            for i in range(self.nruns):
                hab_ten = pm.Categorical('h_{}'.format(i), p)

                alpha = tt.as_tensor_variable([hab_ten])
                probs_a, probs_r = self.inferrer(alpha)

                # use a DensityDist
                pm.Categorical('actions_{}'.format(i), probs_a, observed=self.actions[i])
                pm.Categorical('rewards_{}'.format(i), probs_r, observed=self.rewards[i])

        return gmodel

    def discreteNormal(self, x, mean, std):

        p = np.exp(-(x - mean)**2/(2*std**2))
        p /= p.sum()
        return p

    def plot_inference(self, samples, model='single', idx=None):

        pass
        # if model=='single':
        #     curr_model = self.single_model(idx)
        # elif model=='group':
        #     curr_model = self.group_model()

        # with curr_model:

        #     # save the traces
        #     trace = pm.load_trace(trace_name)

        #     # plot the traces
        #     plt.figure()
        #     _ = pm.traceplot(trace)#, lines=('h', 1./alpha_true))
        #     plt.show()
    #        plt.figure()
    #        _ = pm.plot_posterior(trace, var_names=['h'], ref_val=(1./alpha_true))
    #        plt.show()

