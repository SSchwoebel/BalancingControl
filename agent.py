"""This module contains the class that defines the interaction between
different modules that govern agent's behavior.
"""
#import numpy as np
import torch
from perception import HierarchicalPerception
from misc import ln, softmax, intersect1d
import scipy.special as scs
import gc

        
class BayesianPlanner(object):
    
    def __init__(self, perception, action_selection, policies,
                 prior_states = None, prior_policies = None, 
                 prior_context = None,
                 learn_habit = False,
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
        self.trials = trials
        
        if policies is not None:
            self.policies = policies
        else:
            #make action sequences for each policy
            self.policies = torch.eye(self.npi, dtype = int)
        
        self.na = len(torch.unique(policies))
            
        self.possible_polcies = self.policies.clone().detach()
        
        self.actions = torch.unique(self.policies)
        self.na = len(self.actions)
        
        if prior_states is not None:
            self.prior_states = prior_states
        else:
            self.prior_states = torch.ones(self.nh)
            self.prior_states /= self.prior_states.sum()
            
        if prior_context is not None:
            self.prior_context = prior_context
            self.nc = prior_context.shape[0]
        else:
            self.prior_context = torch.ones(1)
            
        if prior_policies is not None:
            self.prior_policies = prior_policies#.repeat(self.nc, 1)
        else:
            self.prior_policies = torch.ones((self.npi,self.nc))/self.npi
            
        self.learn_habit = learn_habit
            
        #set various data structures
        self.actions = torch.zeros((trials, T), dtype = int)
        self.posterior_states = torch.zeros((trials, T, self.nh, T, self.npi, self.nc))
        self.posterior_policies = torch.zeros((trials, T, self.npi, self.nc))
        self.posterior_dirichlet_pol = torch.zeros((trials, self.npi, self.nc))
        self.posterior_dirichlet_rew = torch.zeros((trials, T, self.nr, self.nh, self.nc))
        self.observations = torch.zeros((trials, T), dtype = int)
        self.rewards = torch.zeros((trials, T), dtype = int)
        self.posterior_context = torch.ones((trials, T, self.nc))
        self.posterior_context[:,:,:] = self.prior_context[None,None,:]
        self.likelihood = torch.zeros((trials, T, self.npi, self.nc))
        #self.posterior_rewards = torch.zeros((trials, T, self.nr))
        self.posterior_actions = torch.zeros((trials, T-1, self.na))
        
    def reset(self, parameters):
        
        self.reset_beliefs()
        self.reset_parameters(parameters)

    def reset_beliefs(self):
        
        self.actions[:] = torch.zeros((self.trials, self.T), dtype = int)
        self.posterior_states[:] = torch.zeros((self.trials, self.T, self.nh, self.T, self.npi, self.nc))
        self.posterior_policies[:] = torch.zeros((self.trials, self.T, self.npi, self.nc))
        self.posterior_dirichlet_pol[:] = torch.zeros((self.trials, self.npi, self.nc))
        self.posterior_dirichlet_rew[:] = torch.zeros((self.trials, self.T, self.nr, self.nh, self.nc))
        self.observations[:] = torch.zeros((self.trials, self.T), dtype = int)
        self.rewards[:] = torch.zeros((self.trials, self.T), dtype = int)
        self.posterior_context[:] = torch.ones((self.trials, self.T, self.nc))
        self.posterior_context[:,:,:] = self.prior_context[None,None,:]
        self.likelihood[:] = torch.zeros((self.trials, self.T, self.npi, self.nc))
        #self.posterior_rewards = torch.zeros((self.trials, self.T, self.nr))
        self.posterior_actions[:] = torch.zeros((self.trials, self.T-1, self.na))
        
    def reset_parameters(self, parameters):
        self.perception.reset_parameters(parameters)
        
        
    def update_beliefs(self, tau, t, observation, reward, response):
        self.observations[tau,t] = observation
        self.rewards[tau,t] = reward
        
        if t == 0:
            self.possible_polcies = torch.arange(0,self.npi,1, dtype=torch.int32)
        else:
            possible_policies = torch.where(self.policies[:,t-1]==response)[0].type(torch.int32)
            self.possible_polcies = intersect1d(self.possible_polcies, possible_policies)
            
        self.posterior_states[tau, t] = self.perception.update_beliefs_states(
                                         tau, t,
                                         observation,
                                         reward,
                                         self.policies,
                                         self.possible_polcies)
        
        #update beliefs about policies
        self.posterior_policies[tau, t], self.likelihood[tau,t] = self.perception.update_beliefs_policies(tau, t)
        
        if t == 0 and tau == 0:
            prior_context = self.prior_context
        else: #elif t == 0:
            prior_context = torch.einsum('ck,k->c', self.perception.transition_matrix_context, self.posterior_context[tau-1, -1])#.reshape((self.nc))
#            else:
#                prior_context = torch.dot(self.perception.transition_matrix_context, self.posterior_context[tau, t-1])
        
        if t>=0:
            self.posterior_context[tau, t] = \
            self.perception.update_beliefs_context(tau, t, \
                                                   reward, \
                                                   self.posterior_states[tau, t], \
                                                   self.posterior_policies[tau, t], \
                                                   prior_context, \
                                                   self.policies)
            
        
        if t < self.T-1:
            post_pol = torch.einsum('pc,c->p', self.posterior_policies[tau, t], self.posterior_context[tau, t])
            self.posterior_actions[tau, t] = torch.tensor([post_pol[self.policies[:,t]==a] for a in range(self.na)])
            
        if t == self.T-1 and self.learn_habit:
            self.posterior_dirichlet_pol[tau] = self.perception.update_beliefs_dirichlet_pol_params(tau, t, \
                                                            possible_policies, \
                                                            self.posterior_context[tau,t])
        #if reward > 0:    
        self.posterior_dirichlet_rew[tau,t] = self.perception.update_beliefs_dirichlet_rew_params(tau, t, \
                                                            reward, \
                                                   self.posterior_states[tau, t], \
                                                   self.posterior_policies[tau, t], \
                                                   self.posterior_context[tau,t])
        
        #gc.collect()
        
#        self.posterior_rewards =  self.perception.update_beliefs_rewards(tau, t, \
#                                                                         self.posterior_states[tau, t], \
#                                                                         self.posterior_policies[tau, t], \
#                                                                         self.posterior_context[tau, t])
        
    
    def generate_response(self, tau, t):
        
        #get response probability
        posterior_states = self.posterior_states[tau, t]
        posterior_policies = torch.einsum('pc,c->p', self.posterior_policies[tau, t], self.posterior_context[tau, t])
        #print(self.posterior_context[tau, t])
        posterior_policies /= posterior_policies.sum()
        non_zero = posterior_policies > 0
        controls = self.policies[:, t][non_zero]
        posterior_policies = posterior_policies[non_zero]
        actions = torch.unique(controls)

        self.actions[tau, t] = self.action_selection.select_desired_action(tau, 
                                        t, posterior_policies, controls)
            
        
        return self.actions[tau, t]
    

