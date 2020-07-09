"""This module contains the World class that defines interactions between 
the environment and the agent. It also keeps track of all observations and 
actions generated during a single experiment. To initiate it one needs to 
provide the environment class and the agent class that will be used for the 
experiment.
"""
import numpy as np
from misc import ln

class World(object):
    
    def __init__(self, environment, agent, trials = 1, T = 10):
        #set inital elements of the world to None        
        self.environment = environment
        self.agent = agent

        self.trials = trials # number of trials in the experiment
        self.T = T # number of time steps in each trial
        
        self.free_parameters = {}
        
        #container for observations
        self.observations = np.zeros((self.trials, self.T), dtype = int)
                
        #container for agents actions
        self.actions = np.zeros((self.trials, self.T), dtype = int)
        
        #container for rewards
        self.rewards = np.zeros((self.trials, self.T), dtype = int)
        
    def simulate_experiment(self, curr_trials=None):
        """This methods evolves all the states of the world by iterating 
        through all the trials and time steps of each trial.
        """
        if curr_trials is not None:
            trials = curr_trials
        else:
            trials = range(self.trials)
        for tau in trials:
            for t in range(self.T):
                self.__update_world(tau, t)
 
    
    def estimate_par_evidence(self, params, method='MLE'):

        
        val = np.zeros(params.shape[0])
        for i, par in enumerate(params):
            if method == 'MLE':
                val[i] = self.__get_log_likelihood(par)
            else:
                val[i] = self.__get_log_jointprobability(par)
        
        return val
    
    def fit_model(self, bounds, n_pars, method='MLE'):
        """This method uses the existing observation and response data to 
        determine the set of parameter values that are most likely to cause 
        the meassured behavior. 
        """
        
        inference = Inference(ftol = 1e-4, xtol = 1e-8, bounds = bounds, 
                           opts = {'np': n_pars})
        
        if method == 'MLE':
            return inference.infer_posterior(self.__get_log_likelihood)
        else:
            return inference.infer_posterior(self.__get_log_jointprobability)
        
        
    #this is a private method do not call it outside of the class
    def __get_log_likelihood(self, params):
        self.agent.set_free_parameters(params)
        self.agent.reset_beliefs(self.actions)
        self.__update_model()
        
        p1 = np.tile(np.arange(self.trials), (self.T, 1)).T
        p2 = np.tile(np.arange(self.T), (self.trials, 1))
        p3 = self.actions.astype(int)
        
        return ln(self.agent.asl.control_probability[p1, p2, p3]).sum()
    
    def __get_log_jointprobability(self, params):
        self.agent.set_free_parameters(params)
        self.agent.reset_beliefs(self.actions)
        self.__update_model()
        
        p1 = np.tile(np.arange(self.trials), (self.T, 1)).T
        p2 = np.tile(np.arange(self.T), (self.trials, 1))
        p3 = self.actions.astype(int)
        
        ll = ln(self.agent.asl.control_probability[p1, p2, p3]).sum()
        
        return  ll + self.agent.log_prior()
    
    #this is a private method do not call it outside of the class    
    def __update_model(self):
        """This private method updates the internal states of the behavioral 
        model given the avalible set of observations and actions.
        """

        for tau in range(self.trials):
            for t in range(self.T):
                if t == 0:
                    response = None
                else:
                    response = self.actions[tau, t-1]
                
                observation = self.observations[tau,t]
                
                self.agent.update_beliefs(tau, t, observation, response)
                self.agent.plan_behavior(tau, t)
                self.agent.estimate_response_probability(tau, t)
    
    #this is a private method do not call it outside of the class    
    def __update_world(self, tau, t):
        """This private method performs a signel time step update of the 
        whole world. Here we update the hidden state(s) of the environment, 
        the perceptual and planning states of the agent, and in parallel we 
        generate observations and actions.
        """
        
        if t==0:
            self.environment.set_initial_states(tau)
            response = None
        else:
            response = self.actions[tau, t-1]
            self.environment.update_hidden_states(tau, t, response)
                                                      
        self.observations[tau, t] = \
            self.environment.generate_observations(tau, t)
        
        if t>0:
            self.rewards[tau, t] = self.environment.generate_rewards(tau, t)
            
        observation = self.observations[tau, t]
        
        reward = self.rewards[tau, t]
    
        self.agent.update_beliefs(tau, t, observation, reward, response)
        
        
        if t < self.T-1:
            self.actions[tau, t] = self.agent.generate_response(tau, t)
        else:
            self.actions[tau, t] = -1

        

class FakeWorld(object):
    
    def __init__(self, agent, observations, rewards, actions, trials = 1, T = 10, log_prior=0):
        #set inital elements of the world to None        
        self.agent = agent

        self.trials = trials # number of trials in the experiment
        self.T = T # number of time steps in each trial
        
        self.free_parameters = {}
        
        #container for observations
        self.observations = observations.copy()
                
        #container for agents actions
        self.actions = actions.copy()
        
        #container for rewards
        self.rewards = rewards.copy()
        
        self.log_prior = log_prior
        
    def __simulate_agent(self):
        """This methods evolves all the states of the world by iterating 
        through all the trials and time steps of each trial.
        """
        
        for tau in range(self.trials):
            for t in range(self.T):
                self.__update_model(tau, t)
 
    
    def estimate_par_evidence(self, params, fixed):

        
        val = self.__get_log_jointprobability(params, fixed)
        
        return val
    
    def fit_model(self, bounds, n_pars, method='MLE'):
        """This method uses the existing observation and response data to 
        determine the set of parameter values that are most likely to cause 
        the meassured behavior. 
        """
        
        raise NotImplementedError
    
    def __get_log_jointprobability(self, params, fixed):
        
        self.agent.reset(params, fixed)
        
        self.__simulate_agent()
        
        p1 = np.tile(np.arange(self.trials), (self.T-1, 1)).T
        p2 = np.tile(np.arange(self.T-1), (self.trials, 1))
        p3 = self.actions.astype(int)
        #self.agent.log_probability
        ll = self.agent.log_probability#ln(self.agent.posterior_actions[p1, p2, p3].prod())
        
        return  ll + self.log_prior
    
    #this is a private method do not call it outside of the class    
    def __update_model(self, tau, t):
        """This private method updates the internal states of the behavioral 
        model given the avalible set of observations and actions.
        """
        if t==0:
            response = None
        else:
            response = self.actions[tau, t-1]
                                                      
            
        observation = self.observations[tau, t]
        
        reward = self.rewards[tau, t]
        
    
        self.agent.update_beliefs(tau, t, observation, reward, response)
        