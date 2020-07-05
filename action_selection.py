from misc import ln
import numpy as np

class AveragedSelector(object):
    
    def __init__(self, trials = 1, T = 10, number_of_actions = 2):
        self.n_pars = 0
        
        self.na = number_of_actions
        self.control_probability = np.zeros((trials, T, self.na))
    
    def reset_beliefs(self):
        self.control_probability[:,:,:] = 0
        
    def set_pars(self, pars):
        pass
        
    def log_prior(self):
        return 0

    def select_desired_action(self, tau, t, posterior_policies, actions, *args):
        
        #estimate action probability
        self.estimate_action_probability(tau, t, posterior_policies, actions)
        
        #generate the desired response from action probability
        u = np.random.choice(self.na, p = self.control_probability[tau, t])
        
        return u
    
    def estimate_action_probability(self, tau, t, posterior_policies, actions, *args):
        
        #estimate action probability
        control_prob = np.zeros(self.na)
        for a in range(self.na):
            control_prob[a] = posterior_policies[actions == a].sum()

            
        self.control_probability[tau, t] = control_prob
        

class MaxSelector(object):
    
    def __init__(self, trials = 1, T = 10, number_of_actions = 2):
        self.n_pars = 0
        
        self.na = number_of_actions
        self.control_probability = np.zeros((trials, T, self.na))
    
    def reset_beliefs(self):
        self.control_probability[:,:,:] = 0
        
    def set_pars(self, pars):
        pass
        
    def log_prior(self):
        return 0

    def select_desired_action(self, tau, t, posterior_policies, actions, *args):
        
        #estimate action probability
        self.estimate_action_probability(tau, t, posterior_policies, actions)
        
        #generate the desired response from maximum policy probability
        indices = np.where(posterior_policies == np.amax(posterior_policies))
        u = np.random.choice(actions[indices])
        
        return u
    
    def estimate_action_probability(self, tau, t, posterior_policies, actions, *args):
        
        #estimate action probability
        control_prob = np.zeros(self.na)
        for a in range(self.na):
            control_prob[a] = posterior_policies[actions == a].sum()
            
        self.control_probability[tau, t] = control_prob
        
        
class AveragedPolicySelector(object):
    
    def __init__(self, trials = 1, T = 10, number_of_policies = 10, number_of_actions = 2):
        self.n_pars = 0
        
        self.na = number_of_actions
        
        self.npi = number_of_policies
    
    def reset_beliefs(self):
        self.control_probability[:,:,:] = 0
        
    def set_pars(self, pars):
        pass
        
    def log_prior(self):
        return 0

    def select_desired_action(self, tau, t, posterior_policies, actions, *args):
        
        #generate the desired response from policy probability
        npi = posterior_policies.shape[0]
        pi = np.random.choice(npi, p = posterior_policies)
        
        u = actions[pi]
        
        return u
    