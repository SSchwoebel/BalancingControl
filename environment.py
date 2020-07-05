"""This module contains various experimental environments used for testing 
human behavior."""
#import numpy as np
import torch
      
"""
test: please ignore
"""
class FakeGridWorld(object):
    
    def __init__(self, Omega, Theta,
                 hidden_states, trials = 1, T = 10):
        
        #set probability distribution used for generating observations
        self.Omega = Omega.clone()
        
        #set probability distribution used for generating state transitions
        self.Theta = Theta.clone()
    
        #set container that keeps track the evolution of the hidden states
        self.hidden_states = torch.zeros((trials, T), dtype = int)
        self.hidden_states[:] = torch.array([hidden_states for i in range(trials)])
    
    def set_initial_states(self, tau):
        #start in lower corner
        self.hidden_states[tau, 0] = 1
        
        #print("trial:", tau)
        
    
    def generate_observations(self, tau, t):
        #generate one sample from multinomial distribution
        o = torch.multinomial(self.Omega[:, self.hidden_states[tau, t]], 1)
        return o

    
    def update_hidden_states(self, tau, t, response):
        
        current_state = self.hidden_states[tau, t-1]        
        
        self.hidden_states[tau, t] = torch.multinomial(self.Theta[:, current_state, int(response)], 1)
        
        
class MultiArmedBandid(object):
    
    def __init__(self, Omega, Theta, Rho,
                 trials = 1, T = 10):
        
        #set probability distribution used for generating observations
        self.Omega = Omega.clone()
        
        #set probability distribution used for generating rewards
#        self.Rho = np.zeros((trials, Rho.shape[0], Rho.shape[1]))
#        self.Rho[0] = Rho.clone()
        self.Rho = Rho.clone()
        
        #set probability distribution used for generating state transitions
        self.Theta = Theta.clone()
        
        self.nh = Theta.shape[0]
        
#        self.changes = np.array([0.01, -0.01])
    
        #set container that keeps track the evolution of the hidden states
        self.hidden_states = torch.zeros((trials, T), dtype = int)
        
        self.trials = trials
    
    def set_initial_states(self, tau):
        #start in lower corner
        self.hidden_states[tau, 0] = 0
        
#        if tau%100==0:
#            print("trial:", tau)
        
    
    def generate_observations(self, tau, t):
        #generate one sample from multinomial distribution
        o = torch.multinomial(self.Omega[:, self.hidden_states[tau, t]], 1)
        return o

    
    def update_hidden_states(self, tau, t, response):
        
        current_state = self.hidden_states[tau, t-1]        
        
        self.hidden_states[tau, t] = torch.multinomial(self.Theta[:, current_state, int(response)], 1)
        
    def generate_rewards(self, tau, t):
        #generate one sample from multinomial distribution
        r = torch.multinomial(self.Rho[tau, :, self.hidden_states[tau, t]], 1)
        
        return r