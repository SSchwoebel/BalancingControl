"""This module contains the class that defines the interaction between
different modules that govern agent's behavior.
"""
import numpy as np
from perception import HierarchicalPerception
from misc import ln, softmax
import scipy.special as scs

        
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
        
        if policies is not None:
            self.policies = policies
        else:
            #make action sequences for each policy
            self.policies = np.eye(self.npi, dtype = int)
            
        self.possible_polcies = self.policies.copy()
        
        self.actions = np.unique(self.policies)
        self.na = len(self.actions)
        
        if prior_states is not None:
            self.prior_states = prior_states
        else:
            self.prior_states = np.ones(self.nh)
            self.prior_states /= self.prior_states.sum()
            
        if prior_context is not None:
            self.prior_context = prior_context
            self.nc = prior_context.shape[0]
        else:
            self.prior_context = np.ones(1)
            
        if prior_policies is not None:
            self.prior_policies = np.tile(prior_policies, (1,self.nc)).T
        else:
            self.prior_policies = np.ones((self.npi,self.nc))/self.npi
            
        self.learn_habit = learn_habit
            
        #set various data structures
        self.actions = np.zeros((trials, T), dtype = int)
        self.posterior_states = np.zeros((trials, T, self.nh, T, self.npi, self.nc))
        self.posterior_policies = np.zeros((trials, T, self.npi, self.nc))
        self.posterior_dirichlet_pol = np.zeros((trials, self.npi, self.nc))
        self.posterior_dirichlet_rew = np.zeros((trials, T, self.nr, self.nh, self.nc))
        self.observations = np.zeros((trials, T), dtype = int)
        self.rewards = np.zeros((trials, T), dtype = int)
        self.posterior_context = np.ones((trials, T, self.nc))
        self.posterior_context[:,:,:] = self.prior_context[np.newaxis,np.newaxis,:]
        self.likelihood = np.zeros((trials, T, self.npi, self.nc))
        self.posterior_actions = np.zeros((trials, T-1, self.na))
        self.log_probability = 0
        

    def reset(self, params, fixed):
        
        self.actions[:] = 0
        self.posterior_states[:] = 0
        self.posterior_policies[:] = 0
        self.posterior_dirichlet_pol[:] = 0
        self.posterior_dirichlet_rew[:] =0
        self.observations[:] = 0
        self.rewards[:] = 0
        self.posterior_context[:,:,:] = self.prior_context[np.newaxis,np.newaxis,:]
        self.likelihood[:] = 0
        self.log_probability = 0
        
        self.perception.reset(params, fixed)
        
        
    def update_beliefs(self, tau, t, observation, reward, response):
        self.observations[tau,t] = observation
        self.rewards[tau,t] = reward
            
        if t == 0:
            self.possible_polcies = np.arange(0,self.npi,1).astype(np.int32)
        else:
            possible_policies = np.where(self.policies[:,t-1]==response)[0]
            self.possible_polcies = np.intersect1d(self.possible_polcies, possible_policies)
            self.log_probability += ln(self.posterior_actions[tau,t-1,response])
            
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
            prior_context = np.dot(self.perception.transition_matrix_context, self.posterior_context[tau-1, -1]).reshape((self.nc))
#            else:
#                prior_context = np.dot(self.perception.transition_matrix_context, self.posterior_context[tau, t-1])
        
        if t>=0:
            self.posterior_context[tau, t] = \
            self.perception.update_beliefs_context(tau, t, \
                                                   reward, \
                                                   self.posterior_states[tau, t], \
                                                   self.posterior_policies[tau, t], \
                                                   prior_context, \
                                                   self.policies)
            
        if t < self.T-1:
            post_pol = np.dot(self.posterior_policies[tau, t], self.posterior_context[tau, t])
            self.posterior_actions[tau, t] = self.estimate_action_probability(tau, t, post_pol)
            
        if t == self.T-1 and self.learn_habit:
            self.posterior_dirichlet_pol[tau] = self.perception.update_beliefs_dirichlet_pol_params(tau, t, \
                                                            self.posterior_policies[tau,t], \
                                                            self.posterior_context[tau,t])
        #if reward > 0:    
        self.posterior_dirichlet_rew[tau,t] = self.perception.update_beliefs_dirichlet_rew_params(tau, t, \
                                                            reward, \
                                                   self.posterior_states[tau, t], \
                                                   self.posterior_policies[tau, t], \
                                                   self.posterior_context[tau,t])
    
    def generate_response(self, tau, t):
        
        #get response probability
        posterior_states = self.posterior_states[tau, t]
        posterior_policies = np.dot(self.posterior_policies[tau, t], self.posterior_context[tau, t])
        #print(self.posterior_context[tau, t])
        posterior_policies /= posterior_policies.sum()
        non_zero = posterior_policies > 0
        controls = self.policies[:, t][non_zero]
        posterior_policies = posterior_policies[non_zero]
        actions = np.unique(controls)

        self.actions[tau, t] = self.action_selection.select_desired_action(tau, 
                                        t, posterior_policies, controls)
            
        
        return self.actions[tau, t]
    
    
    def estimate_action_probability(self, tau, t, posterior_policies):
        
        #estimate action probability
        control_prob = np.zeros(self.na)
        for a in range(self.na):
            control_prob[a] = posterior_policies[self.policies[:,t] == a].sum()

            
        return control_prob
    

