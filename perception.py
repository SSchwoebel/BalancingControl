from misc import ln, softmax
#import numpy as np
import torch
import scipy.special as scs
from misc import D_KL_nd_dirichlet, D_KL_dirichlet_categorical
import gc
    
    
class HierarchicalPerception(object):
    def __init__(self,
                 generative_model_observations, 
                 generative_model_states,
                 generative_model_rewards,
                 transition_matrix_context,
                 prior_states,
                 prior_rewards,
                 prior_policies,
                 dirichlet_pol_params = None,
                 dirichlet_rew_params = None,
                 T=5):

        self.generative_model_observations = generative_model_observations.clone().detach()
        self.generative_model_states = generative_model_states.clone().detach()
        self.generative_model_rewards = generative_model_rewards.clone().detach()
        self.transition_matrix_context = transition_matrix_context.clone().detach()
        self.prior_rewards = prior_rewards.clone().detach()
        self.prior_states = prior_states.clone().detach()
        self.prior_policies = prior_policies.clone().requires_grad_(True)
        self.T = T
        self.nh = prior_states.shape[0]
        self.npi = prior_policies.shape[0]
        if len(generative_model_rewards.shape) > 2:
            self.infer_context = True
            self.nc = generative_model_rewards.shape[2]
        else:
            self.nc = 1
            self.generative_model_rewards = self.generative_model_rewards[:,:,None]
        if dirichlet_pol_params is not None:
            self.dirichlet_pol_params = dirichlet_pol_params.clone().requires_grad_(True)
        if dirichlet_rew_params is not None:
            self.dirichlet_rew_params = dirichlet_rew_params.clone().detach()
            
        self.generative_model_rewards_init = generative_model_rewards.clone().detach()
        self.dirichlet_rew_params_init = dirichlet_rew_params.clone().detach()
        
        for c in range(self.nc):
            for state in range(self.nh):
                self.generative_model_rewards[:,state,c] =\
                torch.exp(torch.digamma(self.dirichlet_rew_params[:,state,c])\
                       -torch.digamma(self.dirichlet_rew_params[:,state,c].sum()))
                self.generative_model_rewards[:,state,c] /= self.generative_model_rewards[:,state,c].sum()
                
        self.bwd_messages = torch.zeros((self.nh, self.T, self.npi, self.nc))
        self.fwd_messages = torch.zeros((self.nh, self.T, self.npi, self.nc))
        self.fwd_norms = torch.zeros((self.T+1, self.npi, self.nc))
        self.obs_messages = torch.zeros((self.nh, self.T, self.nc)) + 1/self.nh
        self.rew_messages = torch.zeros((self.nh, self.T, self.nc))
        
                
    def reset_parameters(self, parameters):
        
        self.dirichlet_rew_params[:] = parameters['dir_rew'].clone().detach()
        self.generative_model_rewards[:] = parameters['rew_gen_mod'].clone().detach()
        self.prior_policies = parameters['prior_pol'].clone().requires_grad_(True)
        self.dirichlet_pol_params = parameters['alpha'].clone().requires_grad_(True)
            
        
    def instantiate_messages(self, policies):
        self.npi = policies.shape[0]
        
        self.bwd_messages[:] = torch.zeros((self.nh, self.T, self.npi, self.nc))
        self.bwd_messages[:,-1,:, :] = 1./self.nh
        self.fwd_messages[:] = torch.zeros((self.nh, self.T, self.npi, self.nc))
        self.fwd_messages[:, 0, :, :] = self.prior_states[:, None, None]
        
        self.fwd_norms[:] = torch.zeros((self.T+1, self.npi, self.nc))
        self.fwd_norms[0,:,:] = 1.
        
        self.obs_messages[:] = torch.zeros((self.nh, self.T, self.nc)) + 1/self.nh#self.prior_observations.dot(self.generative_model_observations)
        #self.obs_messages = torch.tile(self.obs_messages,(self.T,1)).T
        
        self.rew_messages[:] = torch.zeros((self.nh, self.T, self.nc))
        #self.rew_messages[:] = torch.tile(self.prior_rewards.dot(self.generative_model_rewards),(self.T,1)).T
        
        for c in range(self.nc):
            self.rew_messages[:,:,c] = torch.matmul(self.prior_rewards, self.generative_model_rewards[:,:,c])[:,None]
            for pi, cstates in enumerate(policies):
                for t, u in enumerate(torch.flip(cstates, dims = [0])):
                    tp = self.T - 2 - t
                    self.bwd_messages[:,tp,pi,c] = self.bwd_messages[:,tp+1,pi,c]*\
                                                self.obs_messages[:, tp+1,c]*\
                                                self.rew_messages[:, tp+1,c]
                    self.bwd_messages[:,tp,pi,c] = torch.matmul(self.bwd_messages[:,tp,pi,c],\
                                     self.generative_model_states[:,:,u])
                    self.bwd_messages[:,tp, pi,c] /= self.bwd_messages[:,tp,pi,c].sum()
                
    def update_messages(self, t, pi, cs, c=0):
        if t > 0:
            for i, u in enumerate(torch.flip(cs[:t], dims = [0])):
                self.bwd_messages[:,t-1-i,pi,c] = self.bwd_messages[:,t-i,pi,c]*\
                                                self.obs_messages[:,t-i,c]*\
                                                self.rew_messages[:, t-i,c]
                self.bwd_messages[:,t-1-i,pi,c] = torch.matmul(self.bwd_messages[:,t-1-i,pi,c],\
                    self.generative_model_states[:,:,u])
                
                norm = self.bwd_messages[:,t-1-i,pi,c].sum()
                if norm > 0:
                    self.bwd_messages[:,t-1-i, pi,c] /= norm
        
        if len(cs[t:]) > 0:
           for i, u in enumerate(cs[t:]):
               self.fwd_messages[:, t+1+i, pi,c] = self.fwd_messages[:,t+i, pi,c]*\
                                                self.obs_messages[:, t+i,c]*\
                                                self.rew_messages[:, t+i,c]
               self.fwd_messages[:, t+1+i, pi,c] = torch.matmul( \
                                                self.generative_model_states[:,:,u], \
                                                self.fwd_messages[:, t+1+i, pi,c])
               self.fwd_norms[t+1+i,pi,c] = self.fwd_messages[:,t+1+i,pi,c].sum()
               if self.fwd_norms[t+1+i, pi,c] > 0: #???? Shouldn't this not happen?
                   self.fwd_messages[:,t+1+i, pi,c] /= self.fwd_norms[t+1+i,pi,c]
                   
        #gc.collect()
                   
    def reset_preferences(self, t, new_preference, policies):
        
        self.prior_rewards = new_preference.clone().detach()
        
        for c in range(self.nc):
            self.rew_messages[:,:,c] = self.prior_rewards.dot(self.generative_model_rewards[:,:,c])[:,None]
            for pi, cstates in enumerate(policies[t:]):
                for i, u in enumerate(torch.flip(cstates, dims = [0])):
                    tp = self.T - 2 - i
                    self.bwd_messages[:,tp,pi,c] = self.bwd_messages[:,tp+1,pi,c]*\
                                                self.obs_messages[:, tp+1,c]*\
                                                self.rew_messages[:, tp+1,c]
                    self.bwd_messages[:,tp,pi,c] = self.bwd_messages[:,tp,pi,c]\
                        .dot(self.generative_model_states[:,:,u])
                    self.bwd_messages[:,tp, pi,c] /= self.bwd_messages[:,tp,pi,c].sum()

    def update_beliefs_states(self, tau, t, observation, reward, policies, possible_policies):
        #estimate expected state distribution
        if t == 0:
            self.instantiate_messages(policies)
        
        self.obs_messages[:,t,:] = self.generative_model_observations[observation][:,None]
        
        self.rew_messages[:,t,:] = self.generative_model_rewards[reward]
        
        for c in range(self.nc):
            for pi, cs in enumerate(policies):
                if self.prior_policies[pi,c] > 1e-15 and pi in possible_policies:
                    self.update_messages(t, pi, cs, c)
                else:
                    self.fwd_messages[:,:,pi,c] = 1./self.nh
        
        #estimate posterior state distribution
        posterior = self.fwd_messages*self.bwd_messages*self.obs_messages[:,:,None,:]*self.rew_messages[:,:,None,:]
        norm = posterior.sum(axis = 0)
        self.fwd_norms[-1] = norm[-1]
        posterior /= norm
        #gc.collect()
        return posterior
        
    def update_beliefs_policies(self, tau, t):
        
        #print((prior_policies>1e-4).sum())
        likelihood = self.fwd_norms.prod(axis=0)
        posterior = likelihood * self.prior_policies
        posterior/= posterior.sum(axis=0)[None,:]
        #posterior = softmax(ln(self.fwd_norms).sum(axis = 0)+ln(self.prior_policies))
        
        #np.testing.assert_allclose(post, posterior)
        
        return posterior, likelihood
    
    
    def update_beliefs_context(self, tau, t, reward, posterior_states, posterior_policies, prior_context, policies):
        
        post_policies = (prior_context[None,:] * posterior_policies).sum(axis=1)
        beta = self.dirichlet_rew_params.clone().detach()
        states = (posterior_states[:,t,:] * post_policies[None,:,None]).sum(axis=1)
        #beta_prime = self.dirichlet_rew_params.clone()
        #beta_prime[reward] = beta[reward] + states
        
#        for c in range(self.nc):
#            for state in range(self.nh):
#                self.generative_model_rewards[:,state,c] =\
#                np.exp(scs.digamma(beta_prime[:,state,c])\
#                       -scs.digamma(beta_prime[:,state,c].sum()))
#                self.generative_model_rewards[:,state,c] /= self.generative_model_rewards[:,state,c].sum()
#                
#            self.rew_messages[:,t+1:,c] = self.prior_rewards.dot(self.generative_model_rewards[:,:,c])[:,None]
#        
#        for c in range(self.nc):
#            for pi, cs in enumerate(policies):
#                if self.prior_policies[pi,c] > 1e-15:
#                    self.update_messages(t, pi, cs, c)
#                else:
#                    self.fwd_messages[:,:,pi,c] = 1./self.nh #0
        
        alpha = self.dirichlet_pol_params.clone()
        if t == self.T-1:
            chosen_pol = torch.argmax(post_policies)
            inf_context = torch.argmax(prior_context)
            alpha_prime = self.dirichlet_pol_params + torch.eye(self.npi)[chosen_pol]*prior_context
            #alpha_prime[chosen_pol,inf_context] = self.dirichlet_pol_params[chosen_pol,inf_context] + 1
        else:
            alpha_prime = alpha
        

        if self.nc == 1:
            posterior = torch.ones(1)
        else:
            # todo: recalc
            #outcome_surprise = ((states * prior_context[None,:]).sum(axis=1)[:,None] * (scs.digamma(beta_prime[reward]) - scs.digamma(beta_prime.sum(axis=0)))).sum(axis=0)
            outcome_surprise = (posterior_policies * ln(self.fwd_norms.prod(axis=0))).sum(axis=0)
            entropy = - (posterior_policies * ln(posterior_policies)).sum(axis=0)
            #policy_surprise = (post_policies[:,None] * scs.digamma(alpha_prime)).sum(axis=0) - scs.digamma(alpha_prime.sum(axis=0))
            policy_surprise = (posterior_policies * torch.digamma(alpha_prime)).sum(axis=0) - torch.digamma(alpha_prime.sum(axis=0))
            posterior = outcome_surprise + policy_surprise + entropy

                        #+ torch.nan_to_num((posterior_policies * ln(self.fwd_norms).sum(axis = 0))).sum(axis=0)#\
                        
#            if tau in range(90,120) and t == 1:
#                #print(tau, torch.exp(outcome_surprise), torch.exp(policy_surprise))
#                print(tau, torch.exp(outcome_surprise[1])/torch.exp(outcome_surprise[0]), torch.exp(policy_surprise[1])/torch.exp(policy_surprise[0]))

                                                
            posterior = softmax(posterior+ln(prior_context))
            
        return posterior
    
    
    def update_beliefs_dirichlet_pol_params(self, tau, t, possible_policies, posterior_context = [1]):
        assert(t == self.T-1)
        #chosen_pol = torch.argmax(posterior_policies, axis=0)
        #print("chosen", chosen_pol)
#        self.dirichlet_pol_params[chosen_pol,:] += posterior_context.sum(axis=0)/posterior_context.sum()
        self.dirichlet_pol_params= self.dirichlet_pol_params + torch.eye(self.npi)[possible_policies[0]] * posterior_context
        prior_policies = torch.exp(torch.digamma(self.dirichlet_pol_params) - torch.digamma(self.dirichlet_pol_params.sum(axis=0))[None,:])
        self.prior_policies = prior_policies/prior_policies.sum(axis=0)
        
        return self.dirichlet_pol_params
    
    def update_beliefs_dirichlet_rew_params(self, tau, t, reward, posterior_states, posterior_policies, posterior_context = [1]):
        states = (posterior_states[:,t,:,:] * posterior_policies[None,:,:]).sum(axis=1)
        old = self.dirichlet_rew_params.clone().detach()
        self.dirichlet_rew_params[reward,:,:] += states * posterior_context[None,:]
        for c in range(self.nc):
            for state in range(self.nh):
                self.generative_model_rewards[:,state,c] =\
                torch.exp(torch.digamma(self.dirichlet_rew_params[:,state,c])\
                       -torch.digamma(self.dirichlet_rew_params[:,state,c].sum()))
                self.generative_model_rewards[:,state,c] /= self.generative_model_rewards[:,state,c].sum()
                
            self.rew_messages[:,t+1:,c] = torch.matmul(self.prior_rewards, self.generative_model_rewards[:,:,c])[:,None]
            
#        for c in range(self.nc):
#            for pi, cs in enumerate(policies):
#                if self.prior_policies[pi,c] > 1e-15:
#                    self.update_messages(t, pi, cs, c)
#                else:
#                    self.fwd_messages[:,:,pi,c] = 1./self.nh #0
            
        return self.dirichlet_rew_params
    
#    def update_beliefs_rewards(self, tau, t, posterior_states, posterior_policies, posterior_contexts):
#        
#        posterior = torch.einsum('spc,pc,c->sc', posterior_states[:,t], posterior_policies, posterior_contexts)
#        posterior = torch.einsum('sc,rsc->r', posterior, self.generative_model_rewards)
#        
#        return posterior
    
class TwoStepPerception(object):
    def __init__(self,
                 generative_model_observations, 
                 generative_model_states,
                 generative_model_rewards,
                 transition_matrix_context,
                 prior_states,
                 prior_rewards,
                 prior_policies,
                 dirichlet_pol_params = None,
                 dirichlet_rew_params = None,
                 T=5):

        self.generative_model_observations = generative_model_observations.clone().detach()
        self.generative_model_states = generative_model_states.clone().detach()
        self.generative_model_rewards = generative_model_rewards.clone().detach()
        self.transition_matrix_context = transition_matrix_context.clone().detach()
        self.prior_rewards = prior_rewards.clone().detach()
        self.prior_states = prior_states.clone().detach()
        self.prior_policies = prior_policies.clone().detach()
        self.T = T
        self.nh = prior_states.shape[0]
        if len(generative_model_rewards.shape) > 2:
            self.infer_context = True
            self.nc = generative_model_rewards.shape[2]
        else:
            self.nc = 1
            self.generative_model_rewards = self.generative_model_rewards[:,:,None]
        if dirichlet_pol_params is not None:
            self.dirichlet_pol_params = dirichlet_pol_params.clone().detach()
        if dirichlet_rew_params is not None:
            self.dirichlet_rew_params = dirichlet_rew_params.clone().detach()
            
        
        for c in range(self.nc):
            for state in range(self.nh):
                self.generative_model_rewards[:,state,c] =\
                torch.exp(torch.digamma(self.dirichlet_rew_params[:,state,c])\
                       -torch.digamma(self.dirichlet_rew_params[:,state,c].sum()))
                self.generative_model_rewards[:,state,c] /= self.generative_model_rewards[:,state,c].sum()
            
        
    def instantiate_messages(self, policies):
        npi = policies.shape[0]
        
        self.bwd_messages = torch.zeros((self.nh, self.T, npi, self.nc))
        self.bwd_messages[:,-1,:, :] = 1./self.nh
        self.fwd_messages = torch.zeros((self.nh, self.T, npi, self.nc))
        self.fwd_messages[:, 0, :, :] = self.prior_states[:, None, None]
        
        self.fwd_norms = torch.zeros((self.T+1, npi, self.nc))
        self.fwd_norms[0,:,:] = 1.
        
        self.obs_messages = torch.zeros((self.nh, self.T, self.nc)) + 1/self.nh#self.prior_observations.dot(self.generative_model_observations)
        #self.obs_messages = torch.tile(self.obs_messages,(self.T,1)).T
        
        self.rew_messages = torch.zeros((self.nh, self.T, self.nc))
        #self.rew_messages[:] = torch.tile(self.prior_rewards.dot(self.generative_model_rewards),(self.T,1)).T
        
        for c in range(self.nc):
            self.rew_messages[:,:,c] = self.prior_rewards.dot(self.generative_model_rewards[:,:,c])[:,None]
            for pi, cstates in enumerate(policies):
                for t, u in enumerate(torch.flip(cstates, dims = 0)):
                    tp = self.T - 2 - t
                    self.bwd_messages[:,tp,pi,c] = self.bwd_messages[:,tp+1,pi,c]*\
                                                self.obs_messages[:, tp+1,c]*\
                                                self.rew_messages[:, tp+1,c]
                    self.bwd_messages[:,tp,pi,c] = self.bwd_messages[:,tp,pi,c]\
                        .dot(self.generative_model_states[:,:,u])
                    self.bwd_messages[:,tp, pi,c] /= self.bwd_messages[:,tp,pi,c].sum()
                
    def update_messages(self, t, pi, cs, c=0):
        if t > 0:
            for i, u in enumerate(torch.flip(cs[:t], dims = 0)):
                self.bwd_messages[:,t-1-i,pi,c] = self.bwd_messages[:,t-i,pi,c]*\
                                                self.obs_messages[:,t-i,c]*\
                                                self.rew_messages[:, t-i,c]
                self.bwd_messages[:,t-1-i,pi,c] = self.bwd_messages[:,t-1-i,pi,c]\
                    .dot(self.generative_model_states[:,:,u])
                
                norm = self.bwd_messages[:,t-1-i,pi,c].sum()
                if norm > 0:
                    self.bwd_messages[:,t-1-i, pi,c] /= norm
        
        if len(cs[t:]) > 0:
           for i, u in enumerate(cs[t:]):
               self.fwd_messages[:, t+1+i, pi,c] = self.fwd_messages[:,t+i, pi,c]*\
                                                self.obs_messages[:, t+i,c]*\
                                                self.rew_messages[:, t+i,c]
               self.fwd_messages[:, t+1+i, pi,c] = \
                                                self.generative_model_states[:,:,u].\
                                                dot(self.fwd_messages[:, t+1+i, pi,c])
               self.fwd_norms[t+1+i,pi,c] = self.fwd_messages[:,t+1+i,pi,c].sum()
               if self.fwd_norms[t+1+i, pi,c] > 0: #???? Shouldn't this not happen?
                   self.fwd_messages[:,t+1+i, pi,c] /= self.fwd_norms[t+1+i,pi,c]
                   
    def reset_preferences(self, t, new_preference, policies):
        
        self.prior_rewards = new_preference.clone().detach()
        
        for c in range(self.nc):
            self.rew_messages[:,:,c] = self.prior_rewards.dot(self.generative_model_rewards[:,:,c])[:,None]
            for pi, cstates in enumerate(policies[t:]):
                for i, u in enumerate(torch.flip(cstates, dims = 0)):
                    tp = self.T - 2 - i
                    self.bwd_messages[:,tp,pi,c] = self.bwd_messages[:,tp+1,pi,c]*\
                                                self.obs_messages[:, tp+1,c]*\
                                                self.rew_messages[:, tp+1,c]
                    self.bwd_messages[:,tp,pi,c] = self.bwd_messages[:,tp,pi,c]\
                        .dot(self.generative_model_states[:,:,u])
                    self.bwd_messages[:,tp, pi,c] /= self.bwd_messages[:,tp,pi,c].sum()

    def update_beliefs_states(self, tau, t, observation, reward, policies, possible_policies):
        #estimate expected state distribution
        if t == 0:
            self.instantiate_messages(policies)
        
        self.obs_messages[:,t,:] = self.generative_model_observations[observation][:,None]
        
        self.rew_messages[:,t,:] = self.generative_model_rewards[reward]
        
        for c in range(self.nc):
            for pi, cs in enumerate(policies):
                if self.prior_policies[pi,c] > 1e-15 and pi in possible_policies:
                    self.update_messages(t, pi, cs, c)
                else:
                    self.fwd_messages[:,:,pi,c] = 0#1./self.nh #0
        
        #estimate posterior state distribution
        posterior = self.fwd_messages*self.bwd_messages*self.obs_messages[:,:,None,:]*self.rew_messages[:,:,None,:]
        norm = posterior.sum(axis = 0)
        self.fwd_norms[-1] = norm[-1]
        posterior /= norm
        return posterior
        
    def update_beliefs_policies(self, tau, t, gamma=4):
        
        #print((prior_policies>1e-4).sum())
        likelihood = self.fwd_norms.prod(axis=0)
        posterior = torch.power(likelihood,gamma) * self.prior_policies
        posterior/= posterior.sum(axis=0)[None,:]
        #posterior = softmax(ln(self.fwd_norms).sum(axis = 0)+ln(self.prior_policies))
        
        #np.testing.assert_allclose(post, posterior)
        
        return posterior, likelihood
    
    
    def update_beliefs_context(self, tau, t, reward, posterior_states, posterior_policies, prior_context, policies):
        
        post_policies = (prior_context[None,:] * posterior_policies).sum(axis=1)
        beta = self.dirichlet_rew_params.clone().detach()
        states = (posterior_states[:,t,:] * post_policies[None,:,None]).sum(axis=1)
        beta_prime = self.dirichlet_rew_params.clone().detach()
        beta_prime[reward] = beta[reward] + states
        
#        for c in range(self.nc):
#            for state in range(self.nh):
#                self.generative_model_rewards[:,state,c] =\
#                torch.exp(scs.digamma(beta_prime[:,state,c])\
#                       -scs.digamma(beta_prime[:,state,c].sum()))
#                self.generative_model_rewards[:,state,c] /= self.generative_model_rewards[:,state,c].sum()
#                
#            self.rew_messages[:,t+1:,c] = self.prior_rewards.dot(self.generative_model_rewards[:,:,c])[:,None]
#        
#        for c in range(self.nc):
#            for pi, cs in enumerate(policies):
#                if self.prior_policies[pi,c] > 1e-15:
#                    self.update_messages(t, pi, cs, c)
#                else:
#                    self.fwd_messages[:,:,pi,c] = 1./self.nh #0
        
        alpha = self.dirichlet_pol_params.clone().detach()
        if t == self.T-1:
            chosen_pol = torch.argmax(post_policies)
            inf_context = torch.argmax(prior_context)
            alpha_prime = self.dirichlet_pol_params.clone().detach()
            alpha_prime[chosen_pol,:] += prior_context
            #alpha_prime[chosen_pol,inf_context] = self.dirichlet_pol_params[chosen_pol,inf_context] + 1
        else:
            alpha_prime = alpha
        

        if self.nc == 1:
            posterior = torch.ones(1)
        else:
            # todo: recalc
            #outcome_surprise = ((states * prior_context[None,:]).sum(axis=1)[:,None] * (scs.digamma(beta_prime[reward]) - scs.digamma(beta_prime.sum(axis=0)))).sum(axis=0)
            outcome_surprise = (posterior_policies * ln(self.fwd_norms.prod(axis=0))).sum(axis=0)
            entropy = - (posterior_policies * ln(posterior_policies)).sum(axis=0)
            #policy_surprise = (post_policies[:,None] * scs.digamma(alpha_prime)).sum(axis=0) - scs.digamma(alpha_prime.sum(axis=0))
            policy_surprise = (posterior_policies * scs.digamma(alpha_prime)).sum(axis=0) - scs.digamma(alpha_prime.sum(axis=0))
            posterior = outcome_surprise + policy_surprise + entropy

                        #+ torch.nan_to_num((posterior_policies * ln(self.fwd_norms).sum(axis = 0))).sum(axis=0)#\
                        
#            if tau in range(90,120) and t == 1:
#                #print(tau, torch.exp(outcome_surprise), torch.exp(policy_surprise))
#                print(tau, torch.exp(outcome_surprise[1])/torch.exp(outcome_surprise[0]), torch.exp(policy_surprise[1])/torch.exp(policy_surprise[0]))

                                                
            posterior = softmax(posterior+ln(prior_context))
            
        return posterior
    
    
    def update_beliefs_dirichlet_pol_params(self, tau, t, posterior_policies, posterior_context = [1]):
        assert(t == self.T-1)
        chosen_pol = torch.argmax(posterior_policies, axis=0)
#        self.dirichlet_pol_params[chosen_pol,:] += posterior_context.sum(axis=0)/posterior_context.sum()
        alpha = 0.3#0.3
        self.dirichlet_pol_params = (1-alpha) * self.dirichlet_pol_params + 1 - (1-alpha)
        self.dirichlet_pol_params[chosen_pol,:] += posterior_context
        self.prior_policies[:] = torch.exp(torch.digamma(self.dirichlet_pol_params) - torch.digamma(self.dirichlet_pol_params.sum(axis=0))[None,:])
        self.prior_policies /= self.prior_policies.sum(axis=0)
        
        return self.dirichlet_pol_params
    
    def update_beliefs_dirichlet_rew_params(self, tau, t, reward, posterior_states, posterior_policies, posterior_context = [1]):
        states = (posterior_states[:,t,:,:] * posterior_policies[None,:,:]).sum(axis=1)
        state = torch.argmax(states)
        old = self.dirichlet_rew_params.clone().detach()
#        self.dirichlet_rew_params[:,state,:] = (1-0.4) * self.dirichlet_rew_params[:,state,:] #+1 - (1-0.4)
#        self.dirichlet_rew_params[reward,state,:] += 1#states * posterior_context[None,:]  
        alpha = 0.6#0.3#1#0.3#0.05
        self.dirichlet_rew_params[:,3:,:] = (1-alpha) * self.dirichlet_rew_params[:,3:,:] +1 - (1-alpha)
        self.dirichlet_rew_params[reward,:,:] += states * posterior_context[None,:]  
        for c in range(self.nc):
            for state in range(self.nh):
                self.generative_model_rewards[:,state,c] =\
                torch.exp(torch.digamma(self.dirichlet_rew_params[:,state,c])\
                       -torch.digamma(self.dirichlet_rew_params[:,state,c].sum()))
                self.generative_model_rewards[:,state,c] /= self.generative_model_rewards[:,state,c].sum()
                
            self.rew_messages[:,t+1:,c] = self.prior_rewards.dot(self.generative_model_rewards[:,:,c])[:,None]
            
#        for c in range(self.nc):
#            for pi, cs in enumerate(policies):
#                if self.prior_policies[pi,c] > 1e-15:
#                    self.update_messages(t, pi, cs, c)
#                else:
#                    self.fwd_messages[:,:,pi,c] = 1./self.nh #0
            
        return self.dirichlet_rew_params
    
