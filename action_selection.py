from misc import ln
import numpy as np
from statsmodels.tsa.stattools import acovf as acov

class MCMCSelector(object):

    def __init__(self, trials = 1, T = 10, number_of_actions = 2, ESS = 50):
        self.n_pars = 0

        self.na = number_of_actions
        self.control_probability = np.zeros((trials, T, self.na))
        self.ess = ESS
        self.RT = np.zeros((trials, T-1))

    def reset_beliefs(self):
        self.control_probability[:,:,:] = 0

    def set_pars(self, pars):
        pass

    def log_prior(self):
        return 0

    def select_desired_action(self, tau, t, posterior_policies, actions, *args):

        npi = posterior_policies.shape[0]
        likelihood = args[0]
        prior = args[1]
        accepted_pis = np.zeros(50000, dtype=np.int32) - 1

        curr_ess = 0
        i = 0

        pi = np.random.choice(npi, p=prior)
        accepted_pis[i] = pi
        i += 1
        while (curr_ess < self.ess) or (i<10*self.ess):

            pi = np.random.choice(npi, p=prior)
            r = np.random.rand()
            #print(i, curr_ess)

            if likelihood[pi]/likelihood[accepted_pis[i-1]] > r:#posterior_policies[pi]/posterior_policies[accepted_pis[i-1]] > r:
                accepted_pis[i] = pi
            else:
                accepted_pis[i] = accepted_pis[i-1]

            autocorr = acov(accepted_pis[:i+1])
            #print(autocorr)

            if autocorr[0] > 0:
                ACT = 1 + 2*np.abs(autocorr[1:]).sum()/autocorr[0]
                curr_ess = i/ACT
            else:
                ACT = 0
                curr_ess = 1

            i += 1

        self.RT[tau,t] = i-1
        print(tau, t, i-1)

        u = actions[accepted_pis[i-1]]

        #estimate action probability
        self.estimate_action_probability(tau, t, posterior_policies, actions)

        return u

    def estimate_action_probability(self, tau, t, posterior_policies, actions, *args):

        #estimate action probability
        control_prob = np.zeros(self.na)
        for a in range(self.na):
            control_prob[a] = posterior_policies[actions == a].sum()


        self.control_probability[tau, t] = control_prob


class DKLSelector(object):

    def __init__(self, trials = 1, T = 10, number_of_actions = 2, ESS = 50):
        self.n_pars = 0

        self.na = number_of_actions
        self.control_probability = np.zeros((trials, T, self.na))
        self.ess = ESS
        self.RT = np.zeros((trials, T-1))

    def reset_beliefs(self):
        self.control_probability[:,:,:] = 0

    def set_pars(self, pars):
        pass

    def log_prior(self):
        return 0

    def select_desired_action(self, tau, t, posterior_policies, actions, *args):

        npi = posterior_policies.shape[0]
        likelihood = args[0]
        prior = args[1]

        DKL = (likelihood * ln(likelihood/prior)).sum()
        H = - (posterior_policies * ln(posterior_policies)).sum()
        H_p = - (prior * ln(prior)).sum()

        self.RT[tau,t] = np.exp(H_p + np.random.normal(H, DKL))

        #estimate action probability
        self.estimate_action_probability(tau, t, posterior_policies, actions)
        u = np.random.choice(self.na, p = self.control_probability[tau, t])

        return u

    def estimate_action_probability(self, tau, t, posterior_policies, actions, *args):

        #estimate action probability
        control_prob = np.zeros(self.na)
        for a in range(self.na):
            control_prob[a] = posterior_policies[actions == a].sum()


        self.control_probability[tau, t] = control_prob


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
