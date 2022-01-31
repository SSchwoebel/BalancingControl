from misc import ln, logBeta, Beta_function
arr_type = "jnp"
if arr_type == "numpy":
    import numpy as ar
    array = ar.array
    import scipy.special as scs
elif arr_type == "torch":
    import torch as ar
    array = ar.tensor
    import scipy.special as scs
elif arr_type == "jnp":
    import jax.numpy as ar
    array = ar.array
    import jax.scipy.special as scs

from misc import entropy
from statsmodels.tsa.stattools import acovf as acov
import matplotlib.pylab as plt

class MCMCSelector(object):

    def __init__(self, trials = 1, T = 10, number_of_actions = 2, ESS = 50):
        self.n_pars = 0

        self.na = number_of_actions
        self.control_probability = ar.zeros((trials, T, self.na))
        self.ess = ESS
        self.RT = ar.zeros((trials, T-1))

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
        accepted_pis = ar.zeros(50000, dtype=ar.int32) - 1

        curr_ess = 0
        i = 0

        pi = ar.random.choice(npi, p=prior)
        accepted_pis[i] = pi
        i += 1
        while (curr_ess < self.ess) or (i<10*self.ess):

            pi = ar.random.choice(npi, p=prior)
            r = ar.random.rand()
            #print(i, curr_ess)

            if likelihood[pi]/likelihood[accepted_pis[i-1]] > r:#posterior_policies[pi]/posterior_policies[accepted_pis[i-1]] > r:
                accepted_pis[i] = pi
            else:
                accepted_pis[i] = accepted_pis[i-1]

            autocorr = acov(accepted_pis[:i+1])
            #print(autocorr)

            if autocorr[0] > 0:
                ACT = 1 + 2*ar.abs(autocorr[1:]).sum()/autocorr[0]
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
        control_prob = ar.zeros(self.na)
        for a in range(self.na):
            control_prob[a] = posterior_policies[actions == a].sum()


        self.control_probability[tau, t] = control_prob


class DirichletSelector(object):

    def __init__(self, trials = 1, T = 10, number_of_actions = 2, factor=0.4, calc_dkl=False, calc_entropy=False, draw_true_post=False):
        self.n_pars = 0

        self.na = number_of_actions
        self.control_probability = ar.zeros((trials, T, self.na))
        self.RT = ar.zeros((trials, T-1))
        self.factor = factor
        self.draw_true_post = draw_true_post

        self.calc_dkl = calc_dkl
        if calc_dkl:
            self.DKL_post = ar.zeros((trials, T-1))
            self.DKL_prior = ar.zeros((trials, T-1))
        self.calc_entropy = calc_entropy
        if calc_entropy:
            self.entropy_post = ar.zeros((trials, T-1))
            self.entropy_prior = ar.zeros((trials, T-1))
            self.entropy_like = ar.zeros((trials, T-1))

    def reset_beliefs(self):
        self.control_probability[:,:,:] = 0

    def set_pars(self, pars):
        pass

    def log_prior(self):
        return 0

    def select_desired_action(self, tau, t, posterior_policies, actions, *args):

        npi = posterior_policies.shape[0]
        likelihood = args[0]
        prior = args[1] #ar.ones_like(likelihood)/npi #
        # likelihood = ar.array([0.5,0.5])
        # prior = ar.array([0.5,0.5])
        # posterior_policies = prior * likelihood
        # posterior_policies /= posterior_policies.sum()
        #print(posterior_policies, prior, likelihood)
        self.accepted_pis = ar.zeros(100000, dtype=ar.int32) - 1
        dir_counts = ar.ones(npi, ar.double)

        curr_ess = 0
        i = 0

        H_0 =         + (dir_counts.sum()-npi)*scs.digamma(dir_counts.sum()) \
                        - ((dir_counts - 1)*scs.digamma(dir_counts)).sum() \
                        + logBeta(dir_counts)
        #print("H", H_0)

        pi = ar.random.choice(npi, p=prior)
        self.accepted_pis[i] = pi
        dir_counts[pi] += 1
        H_dir =         + (dir_counts.sum()-npi)*scs.digamma(dir_counts.sum()) \
                        - ((dir_counts - 1)*scs.digamma(dir_counts)).sum() \
                        + logBeta(dir_counts)
        #print("H", H_dir)

        if t == 0:
            i += 1
            while H_dir>H_0 - self.factor + self.factor*H_0:

                pi = ar.random.choice(npi, p=prior)
                r = ar.random.rand()
                #print(i, curr_ess)

                #acc_prob = min(1, posterior_policies[pi]/posterior_policies[self.accepted_pis[i-1]])
                if likelihood[self.accepted_pis[i-1]]>0:
                    acc_prob = min(1, likelihood[pi]/likelihood[self.accepted_pis[i-1]])
                else:
                    acc_prob = 1
                if acc_prob >= r:#posterior_policies[pi]/posterior_policies[self.accepted_pis[i-1]] > r:
                    self.accepted_pis[i] = pi
                    dir_counts[pi] += 1#acc_prob
                else:
                    self.accepted_pis[i] = self.accepted_pis[i-1]
                    dir_counts[self.accepted_pis[i-1]] += 1#1-acc_prob

                H_dir =     + (dir_counts.sum()-npi)*scs.digamma(dir_counts.sum()) \
                            - ((dir_counts - 1)*scs.digamma(dir_counts)).sum() \
                            + logBeta(dir_counts)
                #print("H", H_dir)

                i += 1

            self.RT[tau,t] = i-1
            #print(tau, t, i-1)
        else:
            self.RT[tau,t] = 0

        if self.draw_true_post:
            chosen_pol = ar.random.choice(npi, p=posterior_policies)
        else:
            chosen_pol = self.accepted_pis[i-1]

        u = actions[chosen_pol]
        #print(tau,t,iself.accepted_pis[i-1],u,H_rel)
        # if tau in range(100,110) and t==0:
        #     plt.figure()
        #     plt.plot(posterior_policies)
        #     plt.show()

        if self.calc_dkl:
            # autocorr = acov(self.accepted_pis[:i+1])

            # if autocorr[0] > 0:
            #     ACT = 1 + 2*ar.abs(autocorr[1:]).sum()/autocorr[0]
            #     ess = i/ACT
            #     ess = round(ess)
            # else:
            #     ess = 1

            dist = dir_counts / dir_counts.sum()
            D_KL = entropy(posterior_policies, dist)
            self.DKL_post[tau,t] = D_KL
            D_KL = entropy(prior, dist)
            self.DKL_prior[tau,t] = D_KL

        if self.calc_entropy:
            self.entropy_post[tau,t] = entropy(posterior_policies)
            self.entropy_prior[tau,t] = entropy(prior)
            self.entropy_like[tau,t] = entropy(likelihood)
            # if t==0:
            #     print(tau)
            #     n = 12
            #     ind = ar.argpartition(posterior_policies, -n)[-n:]
            #     print(ar.sort(ind))
            #     print(ar.sort(posterior_policies[ind]))

        #estimate action probability
        self.estimate_action_probability(tau, t, posterior_policies, actions)

        return u

    def estimate_action_probability(self, tau, t, posterior_policies, actions, *args):

        #estimate action probability
        control_prob = ar.zeros(self.na)
        for a in range(self.na):
            control_prob[a] = posterior_policies[actions == a].sum()


        self.control_probability[tau, t] = control_prob


class DKLSelector(object):

    def __init__(self, trials = 1, T = 10, number_of_actions = 2, ESS = 50):
        self.n_pars = 0

        self.na = number_of_actions
        self.control_probability = ar.zeros((trials, T, self.na))
        self.ess = ESS
        self.RT = ar.zeros((trials, T-1))

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

        self.RT[tau,t] = ar.exp(H_p + ar.random.normal(H, DKL))

        #estimate action probability
        self.estimate_action_probability(tau, t, posterior_policies, actions)
        u = ar.random.choice(self.na, p = self.control_probability[tau, t])

        return u

    def estimate_action_probability(self, tau, t, posterior_policies, actions, *args):

        #estimate action probability
        control_prob = ar.zeros(self.na)
        for a in range(self.na):
            control_prob[a] = posterior_policies[actions == a].sum()


        self.control_probability[tau, t] = control_prob


class AveragedSelector(object):

    def __init__(self, trials = 1, T = 10, number_of_actions = 2):
        self.n_pars = 0

        self.na = number_of_actions
        self.control_probability = ar.zeros((trials, T, self.na))

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
        u = ar.distributions.Categorical(self.control_probability[tau, t]).sample()

        return u

    def estimate_action_probability(self, tau, t, posterior_policies, actions, *args):
        #estimate action probability
        control_prob = ar.zeros(self.na)
        for a in range(self.na):
            control_prob[a] = posterior_policies[actions == a].sum()


        self.control_probability[tau, t] = control_prob


class MaxSelector(object):

    def __init__(self, trials = 1, T = 10, number_of_actions = 2):
        self.n_pars = 0

        self.na = number_of_actions
        self.control_probability = ar.zeros((trials, T, self.na))

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
        indices = ar.where(posterior_policies == ar.amax(posterior_policies))
        u = ar.random.choice(actions[indices])

        return u

    def estimate_action_probability(self, tau, t, posterior_policies, actions, *args):

        #estimate action probability
        control_prob = ar.zeros(self.na)
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
        pi = ar.random.choice(npi, p = posterior_policies)

        u = actions[pi]

        return u
