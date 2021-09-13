"""This module contains the class that defines the interaction between
different modules that govern agent's behavior.
"""
arr_type = "torch"
if arr_type == "numpy":
    import numpy as ar
    array = ar.array
else:
    import torch as ar
    array = ar.tensor
from perception import HierarchicalPerception
from misc import ln, softmax, own_logical_and
import scipy.special as scs


class BayesianPlanner(object):

    def __init__(self, perception, action_selection, policies,
                 prior_states = None, prior_policies = None,
                 prior_context = None,
                 learn_habit = False,
                 learn_rew = False,
                 trials = 1, T = 10, number_of_states = 6,
                 number_of_rewards = 2,
                 number_of_policies = 10):

        #set the modules of the agent
        self.perception = perception
        self.action_selection = action_selection

        #set parameters of the agent
        self.nh = number_of_states #number of states
        self.npi = number_of_policies #number of policies
        self.nr = number_of_rewards

        self.T = T

        if policies is not None:
            self.policies = policies
        else:
            #make action sequences for each policy
            self.policies = ar.eye(self.npi, dtype = int)

        self.possible_polcies = self.policies.clone().detach()

        self.actions = ar.unique(self.policies)
        self.na = len(self.actions)

        if prior_states is not None:
            self.prior_states = prior_states
        else:
            self.prior_states = ar.ones(self.nh)
            self.prior_states /= self.prior_states.sum()

        if prior_context is not None:
            self.prior_context = prior_context
            self.nc = prior_context.shape[0]
        else:
            self.prior_context = ar.ones(1)
            self.nc = 1

        if prior_policies is not None:
            self.prior_policies = prior_policies[:,None]#ar.tile(prior_policies, (1,self.nc)).T
        else:
            self.prior_policies = ar.ones((self.npi,self.nc))/self.npi

        self.learn_habit = learn_habit
        self.learn_rew = learn_rew

        #set various data structures
        self.actions = ar.zeros((trials, T), dtype = int)
        self.posterior_states = ar.zeros((trials, T, self.nh, T, self.npi, self.nc))
        self.posterior_policies = ar.zeros((trials, T, self.npi, self.nc))
        self.posterior_dirichlet_pol = ar.zeros((trials, self.npi, self.nc))
        self.posterior_dirichlet_rew = ar.zeros((trials, T, self.nr, self.nh, self.nc))
        self.observations = ar.zeros((trials, T), dtype = int)
        self.rewards = ar.zeros((trials, T), dtype = int)
        self.posterior_context = ar.ones((trials, T, self.nc))
        self.posterior_context[:,:,:] = self.prior_context[None,None,:]
        self.likelihood = ar.zeros((trials, T, self.npi, self.nc))
        self.prior_policies = ar.zeros((trials, self.npi, self.nc))
        self.prior_policies[:] = prior_policies[None,:,:]
        self.posterior_actions = ar.zeros((trials, T-1, self.na))
        self.posterior_rewards = ar.zeros((trials, T, self.nr))
        self.log_probability = 0
        if hasattr(self.perception, 'generative_model_context'):
            self.context_obs = ar.zeros(trials, dtype=int)


    def reset(self, params, fixed):

        self.actions[:] = 0
        self.posterior_states[:] = 0
        self.posterior_policies[:] = 0
        self.posterior_dirichlet_pol[:] = 0
        self.posterior_dirichlet_rew[:] =0
        self.observations[:] = 0
        self.rewards[:] = 0
        self.posterior_context[:,:,:] = self.prior_context[None,None,:]
        self.likelihood[:] = 0
        self.posterior_actions[:] = 0
        self.posterior_rewards[:] = 0
        self.log_probability = 0

        self.perception.reset(params, fixed)


    def update_beliefs(self, tau, t, observation, reward, response, context=None):
        self.observations[tau,t] = observation
        self.rewards[tau,t] = reward
        if context is not None:
            self.context_obs[tau] = context

        if t == 0:
            self.possible_polcies = ar.arange(0,self.npi,1, dtype=ar.long)
        else:
            #TODO!
            # wow so inefficient. probably rather remove for fitting...
            possible_policies = self.policies[:,t-1]==response
            prev_pols = ar.zeros(self.npi, dtype=bool)
            prev_pols[:] = False
            prev_pols[self.possible_polcies] = True
            new_pols = own_logical_and(possible_policies, prev_pols)
            self.possible_polcies = ar.where(new_pols==True)[0]
            
            # TODO once 1D intersect exists
            #self.possible_polcies = ar.intersect1d(self.possible_polcies, possible_policies)
            self.log_probability += ln(self.posterior_actions[tau,t-1,response])

        self.posterior_states[tau, t] = self.perception.update_beliefs_states(
                                         tau, t,
                                         observation,
                                         reward,
                                         self.policies,
                                         self.possible_polcies)

        #update beliefs about policies
        self.posterior_policies[tau, t], self.likelihood[tau,t] = self.perception.update_beliefs_policies(tau, t)

        if tau == 0:
            prior_context = self.prior_context
        else: #elif t == 0:
            prior_context = ar.dot(self.perception.transition_matrix_context, self.posterior_context[tau-1, -1]).reshape((self.nc))
#            else:
#                prior_context = ar.dot(self.perception.transition_matrix_context, self.posterior_context[tau, t-1])

        # check here what to do with the greater and equal sign
        if self.nc>1 and t>=0:
            
            if hasattr(self, 'context_obs'):
                c_obs = self.context_obs[tau]
            else:
                c_obs = None
            self.posterior_context[tau, t] = \
            self.perception.update_beliefs_context(tau, t, \
                                                   reward, \
                                                   self.posterior_states[tau, t], \
                                                   self.posterior_policies[tau, t], \
                                                   prior_context, \
                                                   self.policies,\
                                                   context=c_obs)
        elif self.nc>1 and t==0:
            self.posterior_context[tau, t] = prior_context
        else:
            self.posterior_context[tau,t] = 1

        # print(tau,t)
        # print("prior", prior_context)
        # print("post", self.posterior_context[tau, t])

        if t < self.T-1:
            #post_pol = ar.matmul(self.posterior_policies[tau, t], self.posterior_context[tau, t])
            self.posterior_actions[tau, t] = self.estimate_action_probability(tau, t)

        if t == self.T-1 and self.learn_habit:
            self.posterior_dirichlet_pol[tau], self.prior_policies[tau] = self.perception.update_beliefs_dirichlet_pol_params(tau, t, \
                                                            self.posterior_policies[tau,t], \
                                                            self.posterior_context[tau,t])

        if False:
            self.posterior_rewards[tau, t-1] = ar.einsum('rsc,spc,pc,c->r',
                                                  self.perception.generative_model_rewards,
                                                  self.posterior_states[tau,t,:,t],
                                                  self.posterior_policies[tau,t],
                                                  self.posterior_context[tau,t])
        #if reward > 0:
        # check later if stuff still works!
        if self.learn_rew:# and t>0:#==self.T-1:
            self.posterior_dirichlet_rew[tau,t] = self.perception.update_beliefs_dirichlet_rew_params(tau, t, \
                                                            reward, \
                                                   self.posterior_states[tau, t], \
                                                   self.posterior_policies[tau, t], \
                                                   self.posterior_context[tau,t])

    def generate_response(self, tau, t):

        #get response probability
        posterior_states = self.posterior_states[tau, t]
        posterior_policies = ar.einsum('pc,c->p', self.posterior_policies[tau, t], self.posterior_context[tau, 0])
        posterior_policies /= posterior_policies.sum()
        avg_likelihood = ar.einsum('pc,c->p', self.likelihood[tau,t], self.posterior_context[tau, 0])
        avg_likelihood /= avg_likelihood.sum()
        prior = ar.einsum('pc,c->p', self.prior_policies[tau-1], self.posterior_context[tau, 0])
        prior /= prior.sum()
        #print(self.posterior_context[tau, t])
        non_zero = posterior_policies > 0
        controls = self.policies[:, t]#[non_zero]
        actions = ar.unique(controls)
        # posterior_policies = posterior_policies[non_zero]
        # avg_likelihood = avg_likelihood[non_zero]
        # prior = prior[non_zero]

        self.actions[tau, t] = self.action_selection.select_desired_action(tau,
                                        t, posterior_policies, controls, avg_likelihood, prior)


        return self.actions[tau, t]


    def estimate_action_probability(self, tau, t):

        # TODO: should this be t=0 or t=t?
        posterior_policies = ar.einsum('pc,c->p', self.posterior_policies[tau, t], self.posterior_context[tau, t])
        posterior_policies /= posterior_policies.sum()
        
        #estimate action probability
        control_prob = ar.zeros(self.na)
        for a in range(self.na):
            control_prob[a] = posterior_policies[self.policies[:,t] == a].sum()


        return control_prob
    
    def set_parameters(self, **kwargs):
        
        if 'pol_lambda' in kwargs.keys():
            self.perception.pol_lambda = kwargs['pol_lambda']
        if 'r_lambda' in kwargs.keys():
            self.perception.r_lambda = kwargs['r_lambda']
        if 'dec_temp' in kwargs.keys():
            self.perception.dec_temp = kwargs['dec_temp']


class BayesianPlanner_old(object):

    def __init__(self, perception, action_selection, policies,
                 prior_states = None, prior_policies = None,
                 prior_context = None,
                 learn_habit = False,
                 learn_rew = False,
                 trials = 1, T = 10, number_of_states = 6,
                 number_of_rewards = 2,
                 number_of_policies = 10):

        #set the modules of the agent
        self.perception = perception
        self.action_selection = action_selection

        #set parameters of the agent
        self.nh = number_of_states #number of states
        self.npi = number_of_policies #number of policies
        self.nr = number_of_rewards

        self.T = T

        if policies is not None:
            self.policies = policies
        else:
            #make action sequences for each policy
            self.policies = ar.eye(self.npi, dtype = int)

        self.possible_polcies = self.policies.copy()

        self.actions = ar.unique(self.policies)
        self.na = len(self.actions)

        if prior_states is not None:
            self.prior_states = prior_states
        else:
            self.prior_states = ar.ones(self.nh)
            self.prior_states /= self.prior_states.sum()

        if prior_context is not None:
            self.prior_context = prior_context
            self.nc = prior_context.shape[0]
        else:
            self.prior_context = ar.ones(1)
            self.nc = 1

        if prior_policies is not None:
            self.prior_policies = ar.tile(prior_policies, (1,self.nc)).T
        else:
            self.prior_policies = ar.ones((self.npi,self.nc))/self.npi

        self.learn_habit = learn_habit
        self.learn_rew = learn_rew

        #set various data structures
        self.actions = ar.zeros((trials, T), dtype = int)
        self.posterior_states = ar.zeros((trials, T, self.nh, T, self.npi, self.nc))
        self.posterior_policies = ar.zeros((trials, T, self.npi, self.nc))
        self.posterior_dirichlet_pol = ar.zeros((trials, self.npi, self.nc))
        self.posterior_dirichlet_rew = ar.zeros((trials, T, self.nr, self.nh, self.nc))
        self.observations = ar.zeros((trials, T), dtype = int)
        self.rewards = ar.zeros((trials, T), dtype = int)
        self.posterior_context = ar.ones((trials, T, self.nc))
        self.posterior_context[:,:,:] = self.prior_context[None,None,:]
        self.likelihood = ar.zeros((trials, T, self.npi, self.nc))
        self.prior_policies = ar.zeros((trials, self.npi, self.nc))
        self.prior_policies[:] = prior_policies[None,:,:]
        self.posterior_actions = ar.zeros((trials, T-1, self.na))
        self.posterior_rewards = ar.zeros((trials, T, self.nr))
        self.log_probability = 0


    def reset(self, params, fixed):

        self.actions[:] = 0
        self.posterior_states[:] = 0
        self.posterior_policies[:] = 0
        self.posterior_dirichlet_pol[:] = 0
        self.posterior_dirichlet_rew[:] =0
        self.observations[:] = 0
        self.rewards[:] = 0
        self.posterior_context[:,:,:] = self.prior_context[None,None,:]
        self.likelihood[:] = 0
        self.posterior_actions[:] = 0
        self.posterior_rewards[:] = 0
        self.log_probability = 0

        self.perception.reset(params, fixed)


    def update_beliefs(self, tau, t, observation, reward, response):
        self.observations[tau,t] = observation
        self.rewards[tau,t] = reward

        if t == 0:
            self.possible_polcies = ar.arange(0,self.npi,1).astype(ar.int32)
        else:
            possible_policies = ar.where(self.policies[:,t-1]==response)[0]
            self.possible_polcies = ar.intersect1d(self.possible_polcies, possible_policies)
            self.log_probability += ln(self.posterior_actions[tau,t-1,response])

        self.posterior_states[tau, t] = self.perception.update_beliefs_states(
                                         tau, t,
                                         observation,
                                         reward,
                                         self.policies,
                                         self.possible_polcies)

        #update beliefs about policies
        self.posterior_policies[tau, t], self.likelihood[tau,t] = self.perception.update_beliefs_policies(tau, t)

        if tau == 0:
            prior_context = self.prior_context
        else: #elif t == 0:
            prior_context = ar.dot(self.perception.transition_matrix_context, self.posterior_context[tau-1, -1]).reshape((self.nc))
#            else:
#                prior_context = ar.dot(self.perception.transition_matrix_context, self.posterior_context[tau, t-1])

        if self.nc>1 and t>0:
            self.posterior_context[tau, t] = \
            self.perception.update_beliefs_context(tau, t, \
                                                   reward, \
                                                   self.posterior_states[tau, t], \
                                                   self.posterior_policies[tau, t], \
                                                   prior_context, \
                                                   self.policies)
        elif self.nc>1 and t==0:
            self.posterior_context[tau, t] = prior_context
        else:
            self.posterior_context[tau,t] = 1

        # print(tau,t)
        # print("prior", prior_context)
        # print("post", self.posterior_context[tau, t])

        if t < self.T-1:
            post_pol = ar.dot(self.posterior_policies[tau, t], self.posterior_context[tau, t])
            self.posterior_actions[tau, t] = self.estimate_action_probability(tau, t, post_pol)

        if t == self.T-1 and self.learn_habit:
            self.posterior_dirichlet_pol[tau], self.prior_policies[tau] = self.perception.update_beliefs_dirichlet_pol_params(tau, t, \
                                                            self.posterior_policies[tau,t], \
                                                            self.posterior_context[tau,t])

        if False:
            self.posterior_rewards[tau, t-1] = ar.einsum('rsc,spc,pc,c->r',
                                                  self.perception.generative_model_rewards,
                                                  self.posterior_states[tau,t,:,t],
                                                  self.posterior_policies[tau,t],
                                                  self.posterior_context[tau,t])
        #if reward > 0:
        if self.learn_rew:
            self.posterior_dirichlet_rew[tau,t] = self.perception.update_beliefs_dirichlet_rew_params(tau, t, \
                                                            reward, \
                                                   self.posterior_states[tau, t], \
                                                   self.posterior_policies[tau, t], \
                                                   self.posterior_context[tau,t])

    def generate_response(self, tau, t):

        #get response probability
        posterior_states = self.posterior_states[tau, t]
        posterior_policies = ar.einsum('pc,c->p', self.posterior_policies[tau, t], self.posterior_context[tau, 0])
        posterior_policies /= posterior_policies.sum()
        avg_likelihood = ar.einsum('pc,c->p', self.likelihood[tau,t], self.posterior_context[tau, 0])
        avg_likelihood /= avg_likelihood.sum()
        prior = ar.einsum('pc,c->p', self.prior_policies[tau-1], self.posterior_context[tau, 0])
        prior /= prior.sum()
        #print(self.posterior_context[tau, t])
        non_zero = posterior_policies > 0
        controls = self.policies[:, t]#[non_zero]
        actions = ar.unique(controls)
        # posterior_policies = posterior_policies[non_zero]
        # avg_likelihood = avg_likelihood[non_zero]
        # prior = prior[non_zero]

        self.actions[tau, t] = self.action_selection.select_desired_action(tau,
                                        t, posterior_policies, controls, avg_likelihood, prior)


        return self.actions[tau, t]


    def estimate_action_probability(self, tau, t, posterior_policies):

        #estimate action probability
        control_prob = ar.zeros(self.na)
        for a in range(self.na):
            control_prob[a] = posterior_policies[self.policies[:,t] == a].sum()


        return control_prob