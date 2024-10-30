from misc import ln, softmax
import numpy as np
import scipy.special as scs
from misc import normalize

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
                 generative_model_context = None,
                 hidden_state_mapping = False,       
                 learn_rew = True,
                 learn_pol = True,
                 learn_context_obs = False,
                 infer_context = False,
                 prior_context = np.array([1]),
                 T=5,
                 trials=10, 
                 all_rewards = None,
                 all_policies = None,
                 pol_lambda=0, 
                 r_lambda=0, 
                 non_decaying=0, 
                 dec_temp=1.,
                 dirichlet_context_obs_params = None,
                 **pars
                ):

        self.generative_model_observations = generative_model_observations
        self.generative_model_states = generative_model_states # ghost context dimension should be added in the run_agent function, same for generative_model_rewards 
        self.generative_model_rewards = generative_model_rewards
        self.transition_matrix_context = transition_matrix_context
       
        self.prior_rewards = prior_rewards
        self.prior_states = prior_states
        
        self.npi = prior_policies.shape[0]
        self.nh = prior_states.shape[0]
        self.nr = generative_model_rewards.shape[0]
        self.na = np.size(np.unique(all_policies))
        self.nco, self.nc = generative_model_context.shape
        self.T = T
        self.trials = trials
        self.all_policies = all_policies
        self.all_rewards = all_rewards
        
        self.hidden_state_mapping = hidden_state_mapping
        self.infer_context = infer_context
        self.learn_rew = learn_rew
        self.learn_habit = learn_pol
        self.learn_context_obs = learn_context_obs
        
        
        self.pol_lambda = pol_lambda
        self.r_lambda = r_lambda
        self.dec_temp = dec_temp
        self.non_decaying = non_decaying

        self.generative_model_context = generative_model_context
        self.generative_model_rewards = generative_model_rewards


        self.dirichlet_pol_params = dirichlet_pol_params
        self.dirichlet_rew_params = dirichlet_rew_params
        self.dirichlet_context_obs_params = dirichlet_context_obs_params
        

        if hidden_state_mapping:
            self.nh = generative_model_rewards.shape[1]
            self.nm = prior_states.size
        else:
            self.nh, self.nm = [prior_states.size]*2

        ### posteriors log
        self.posterior_states = np.zeros((trials, T, self.nm, T, self.npi, self.nc))
        self.posterior_policies = np.zeros((trials, T, self.npi, self.nc))
        self.posterior_dirichlet_pol = np.zeros((trials, self.npi, self.nc))
        self.posterior_dirichlet_context_obs = np.zeros((trials, self.nco, self.nc))
        self.posterior_context = np.ones((trials, T, self.nc)) 
        self.posterior_context[0,:,:] = prior_context[None,:]
        self.outcome_surprise_log = np.ones((trials, T, self.nc)) 
        self.policy_entropy_log = np.ones((trials, T, self.nc)) 
        self.policy_surprise_log = np.ones((trials, T, self.nc)) 
        self.context_obs_surprise_log = np.ones((trials, T, self.nc))
        # self.posterior_actions = np.zeros((trials, T, self.na))
        # self.posterior_actions[:,0,:] = np.nan
        self.posterior_rewards = np.zeros((trials, T, self.nr))
        self.posterior_dirichlet_rew = np.zeros((trials, T, self.nr, self.nh, self.nc))
        self.likelihood_policies = np.zeros((trials, T, self.npi,self.nc))
        ### observations, actions and rewards log
        self.observations = np.zeros((trials, self.T), dtype = int)
        self.actions = np.zeros((trials, self.T), dtype = int)
        self.rewards = np.zeros((trials, self.T), dtype = int)
        self.context_obs = np.zeros((trials, self.T), dtype=int)

        self.prior_policies = np.zeros([trials, self.npi, self.nc]) 
        self.prior_policies[:] = prior_policies[None,:,:]
        self.prior_context = np.zeros((trials,T, self.nc))
        self.prior_context[0,:,:] = prior_context[None,:]
        
        self.pars = pars
   
   
    def reset(self, params, fixed):

        alphas = np.zeros((self.npi, self.nc)) + params
        self.generative_model_rewards[:] = fixed['rew_mod'].copy()
        self.dirichlet_rew_params[:] = fixed['beta_rew'].copy()
        self.prior_policies[:] = alphas / alphas.sum(axis=0)[None,:]
        self.dirichlet_pol_params = alphas
        

    def instantiate_messages(self):

        self.bwd_messages = np.zeros((self.nm, self.T, self.npi, self.nc))
        self.bwd_messages[:,-1,:, :] = 1./self.nm
        self.fwd_messages = np.zeros((self.nm, self.T, self.npi, self.nc))
        self.fwd_messages[:, 0, :, :] = self.prior_states[:, None, None]
        self.fwd_norms = np.zeros((self.T+1, self.npi, self.nc))
        self.fwd_norms[0,:,:] = 1.

        self.obs_messages = np.zeros((self.nm, self.T, self.nc)) + 1/self.nm
        self.rew_messages = np.zeros((self.nm, self.T, self.nc))               #set t=0 to uniform?

        
        for c in range(self.nc):

            rew_message = self.prior_rewards.dot(self.generative_model_rewards[:,:,c])[:,np.newaxis]
            self.rew_messages[:,1:,c] = rew_message[[self.curr_states]][:,None]

            for pi, cstates in enumerate(self.all_policies):
                for t, u in enumerate(np.flip(cstates, axis = 0)):
                    tp = self.T - 2 - t
                    self.bwd_messages[:,tp,pi,c] = self.bwd_messages[:,tp+1,pi,c]*\
                                                self.obs_messages[:, tp+1,c]*\
                                                self.rew_messages[:, tp+1,c]
                    self.bwd_messages[:,tp,pi,c] = self.bwd_messages[:,tp,pi,c]\
                        .dot(self.generative_model_states[:,:,u,c])
                    self.bwd_messages[:,tp, pi,c] /= self.bwd_messages[:,tp,pi,c].sum()

        # tgt specific!
        self.rew_messages[:,0,:] = 1./self.nm  

        
    def update_messages(self, t, pi, cs, c=0):
        if t > 0:
            for i, u in enumerate(np.flip(cs[:t], axis = 0)):
                self.bwd_messages[:,t-1-i,pi,c] = self.bwd_messages[:,t-i,pi,c]*\
                                                self.obs_messages[:,t-i,c]*\
                                                self.rew_messages[:, t-i,c]
                self.bwd_messages[:,t-1-i,pi,c] = self.bwd_messages[:,t-1-i,pi,c]\
                    .dot(self.generative_model_states[:,:,u,c])

                norm = self.bwd_messages[:,t-1-i,pi,c].sum()
                if norm > 0:
                    self.bwd_messages[:,t-1-i, pi,c] /= norm

        if len(cs[t:]) > 0:
           for i, u in enumerate(cs[t:]):
               self.fwd_messages[:, t+1+i, pi,c] = self.fwd_messages[:,t+i, pi,c]*\
                                                self.obs_messages[:, t+i,c]*\
                                                self.rew_messages[:, t+i,c]

               self.fwd_messages[:, t+1+i, pi,c] = \
                                                self.generative_model_states[:,:,u,c].\
                                                dot(self.fwd_messages[:, t+1+i, pi,c])
               self.fwd_norms[t+1+i,pi,c] = self.fwd_messages[:,t+1+i,pi,c].sum()
               if self.fwd_norms[t+1+i, pi,c] > 0: #???? Shouldn't this not happen?
                   self.fwd_messages[:,t+1+i, pi,c] /= self.fwd_norms[t+1+i,pi,c]


    def reset_preferences(self, t, new_preference, policies):

        self.prior_rewards = new_preference.copy()

        for c in range(self.nc):
            self.rew_messages[:,:,c] = self.prior_rewards.dot(self.generative_model_rewards[:,:,c])[:,np.newaxis]
            for pi, cstates in enumerate(policies[t:]):
                for i, u in enumerate(np.flip(cstates, axis = 0)):
                    tp = self.T - 2 - i
                    self.bwd_messages[:,tp,pi,c] = self.bwd_messages[:,tp+1,pi,c]*\
                                                self.obs_messages[:, tp+1,c]*\
                                                self.rew_messages[:, tp+1,c]
                    self.bwd_messages[:,tp,pi,c] = self.bwd_messages[:,tp,pi,c]\
                        .dot(self.generative_model_states[:,:,u,c])
                    self.bwd_messages[:,tp, pi,c] /= self.bwd_messages[:,tp,pi,c].sum()


    def update_beliefs_states(self, tau, t, observation, reward):
        #estimate expected state distribution
        if t == 0:
            self.instantiate_messages()
        else:
            reward_ind = np.argmax(reward == self.all_rewards)
            self.rew_messages[:,t,:] = self.generative_model_rewards[:,self.curr_states,:][reward_ind]
            self.rew_messages[:,t+1:,:] = np.einsum("r,rsc->sc",self.prior_rewards,self.generative_model_rewards)[self.curr_states,None,:]

        self.obs_messages[:,t,:] = self.generative_model_observations[observation][:,None]

        for c in range(self.nc):
            for pi, cs in enumerate(self.all_policies):
                if self.prior_policies[tau,pi,c] > 1e-15 and self.possible_policies[pi]:
                    self.update_messages(t, pi, cs, c)
                else:
                    self.fwd_messages[:,:,pi,c] = 0#1./self.nh
        #estimate posterior state distribution
        posterior = self.fwd_messages*self.bwd_messages*self.obs_messages[:,:,np.newaxis,:]*self.rew_messages[:,:,np.newaxis,:]
        norm = posterior.sum(axis = 0)
        self.fwd_norms[-1] = norm[-1]
        non_zero = norm > 0
        posterior[:,non_zero] /= norm[non_zero]
        return np.nan_to_num(posterior)


    def update_beliefs_policies(self, tau, t):

        #print((prior_policies>1e-4).sum())
        likelihood = self.fwd_norms.prod(axis=0)
        posterior = np.power(likelihood, self.dec_temp) * self.prior_policies[tau]
        likelihood /= likelihood.sum(axis=0)[None,:]
        posterior/= posterior.sum(axis=0)[None,:]
        posterior = np.nan_to_num(posterior)
        #posterior = softmax(ln(self.fwd_norms).sum(axis = 0)+ln(self.prior_policies))
        #np.testing.assert_allclose(post, posterior)

        return posterior, likelihood


    def update_beliefs_context(self, tau, t, reward, posterior_states, posterior_policies, prior_context, policies, context=None):

        post_policies = (prior_context[None,:] * posterior_policies).sum(axis=1)
        # beta = self.dirichlet_rew_params.copy()
        # states = (posterior_states[:,t,:] * post_policies[None,:,None]).sum(axis=1)
        # beta_prime = self.dirichlet_rew_params.copy()
        # beta_prime[reward] = beta[reward] + states

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

        alpha = self.dirichlet_pol_params.copy()
        if t == self.T-1:
            chosen_pol = np.argmax(post_policies)
            inf_context = np.argmax(prior_context)
            alpha_prime = self.dirichlet_pol_params.copy()
            alpha_prime[chosen_pol,:] += prior_context
            #alpha_prime[chosen_pol,inf_context] = self.dirichlet_pol_params[chosen_pol,inf_context] + 1
        else:
            alpha_prime = alpha


        if self.nc == 1:
            posterior = np.ones(1)
        else:
            if t>0:
                outcome_surprise = (posterior_policies * ln(self.fwd_norms.prod(axis=0))).sum(axis=0)
                entropy = - (posterior_policies * ln(posterior_policies)).sum(axis=0)
                policy_surprise = (posterior_policies * scs.digamma(alpha_prime)).sum(axis=0) - scs.digamma(alpha_prime.sum(axis=0))
            else:
                outcome_surprise = np.zeros(self.nc)
                entropy = np.zeros(self.nc)
                policy_surprise = np.zeros(self.nc)
                
            if context is not None:
                context_obs_suprise = ln(self.generative_model_context[context]+1e-10)
            else:
                context_obs_suprise = np.zeros(self.nc)
                
            posterior = outcome_surprise + policy_surprise + entropy + context_obs_suprise +ln(prior_context)
        
            # if self.pars["context"][tau] % 2 == 0:
            #     i1 = 2
            #     i0 = 0
            # else:
            #     i1 = 3
            #     i0 = 1
            # if t==0:
            #     print(f"-----------------------------------------------------------")
                
            # print(f"\n{self.pars['trial_type'][tau]}, {tau},{t}")
            # print(f"inferred: {np.argmax(np.nan_to_num(softmax(posterior))) == self.pars['context'][tau]}")
            # print(f"context : {context}")
            # print(f"rewards : {self.rewards[tau,:t+1]}")
            # print(f"planets : {self.curr_states[self.observations[tau,t]]}")
            # print(f"actions : {self.actions[tau,:t+1]}")
            # print(f"inferred: {np.argmax(posterior)}")
            # print(f"true    : {self.pars['context'][tau]}\n")
            

            # print(f"      F(pi,c): {outcome_surprise.round(3)}")    
            # print(f"q(p) ln p'(p): {policy_surprise.round(3)}")
            # print(f"q(p) ln q (p): {(entropy.round(3))}")
            # print(f"    ln p(d|c): {context_obs_suprise.round(3)}")
            # print(f"    ln  p'(c): {ln(prior_context).round(3)}")      
            # print(f"   summed log: {posterior.round(3)}")
            # print(f"       summed: {np.nan_to_num(softmax(posterior)).round(3)}\n")

            # # print(f"      F(pi,c): {(outcome_surprise[i0] - outcome_surprise[i1]).round(3)}")    
            # # print(f"q(p) ln p'(p): {(policy_surprise[i0] -policy_surprise[i1]).round(3)}")
            # # print(f"q(p) ln q (p): {((entropy[i0] -entropy[i1]).round(3))}")
            # # print(f"    ln p(d|c): {(context_obs_suprise[i0] -context_obs_suprise[i1]).round(3)}")
            # # print(f"    ln  p'(c): {(ln(prior_context)[i0] -ln(prior_context)[i1]).round(3)}")      
            # # print(f"   summed log: {posterior.round(3)}")
            # # print(f"       summed: {np.nan_to_num(softmax(posterior)).round(3)}\n")

            # # print(posterior_policies.round(3))
            # # print(ln(self.fwd_norms.prod(axis=0)).round(3))
            # # print(self.generative_model_rewards[:,:,0].round(5))
            # # print(self.generative_model_rewards[:,:,1].round(5))
            # # print(self.generative_model_rewards[:,:,2].round(5))
            # # print(self.generative_model_rewards[:,:,3].round(5))
            # print(self.generative_model_context.round(3))
            # try:
            #     print(self.posterior_dirichlet_context_obs[tau-1].round(3))
            # except:
            #     pass
            self.prior_context[tau,t] = prior_context
            self.outcome_surprise_log[tau,t] = outcome_surprise
            self.policy_entropy_log[tau,t] = entropy
            self.policy_surprise_log[tau,t] = policy_surprise 
            self.context_obs_surprise_log[tau,t] = context_obs_suprise
            
            posterior = np.nan_to_num(softmax(posterior))
            
   

        return posterior


    def update_beliefs_dirichlet_pol_params(self, tau, t, posterior_policies, posterior_context = [1]):
        assert(t == self.T-1)
        
        chosen_pol = np.argmax(posterior_policies, axis=0)
#        self.dirichlet_pol_params[chosen_pol,:] += posterior_context.sum(axis=0)/posterior_context.sum()
        self.dirichlet_pol_params = (1-self.pol_lambda) * self.dirichlet_pol_params + 1 - (1-self.pol_lambda)
        self.dirichlet_pol_params[chosen_pol,:] += posterior_context
        prior_policies = self.dirichlet_pol_params.copy() #np.exp(scs.digamma(self.dirichlet_pol_params) - scs.digamma(self.dirichlet_pol_params.sum(axis=0))[None,:])
        prior_policies /= prior_policies.sum(axis=0)[None,:]

        return self.dirichlet_pol_params, prior_policies


    def update_beliefs_dirichlet_rew_params(self, tau, t, reward, posterior_states, posterior_policies, posterior_context = [1]):

        reward_ind = np.argmax(self.all_rewards == reward)
        states = np.einsum('spc,pc -> sc', posterior_states[:,t,:,:], posterior_policies)      # sum_{pi}q(s|pi,c)q(pi|c) = q(s|c)

        posterior_states_given_context = np.zeros([self.nh, self.nc])
        for h in range(self.nh):
            posterior_states_given_context[h,:] = states[self.curr_states == h].sum(axis=0)

        self.dirichlet_rew_params[:,self.non_decaying:,:] = (1-self.r_lambda) * self.dirichlet_rew_params[:,self.non_decaying:,:] +1 - (1-self.r_lambda)
        self.dirichlet_rew_params[reward_ind,:,:] += posterior_states_given_context * posterior_context[None,:]      #  phi_ijk' = phi_ijk + delta_{i,r} q(s=j)q(c=k)

        for c in range(self.nc):
            for state in range(self.nh):
                #self.generative_model_rewards[:,state,c] = self.dirichlet_rew_params[:,state,c] / self.dirichlet_rew_params[:,state,c].sum()
                self.generative_model_rewards[:,state,c] =\
                np.exp(scs.digamma(self.dirichlet_rew_params[:,state,c])\
                        -scs.digamma(self.dirichlet_rew_params[:,state,c].sum()))
                self.generative_model_rewards[:,state,c] /= self.generative_model_rewards[:,state,c].sum()
                
            # self.rew_messages[:,t+1:,c] = self.prior_rewards.dot(self.generative_model_rewards[:,:,c])[:,None]
#        for c in range(self.nc):
#            for pi, cs in enumerate(policies):
#                if self.prior_policies[pi,c] > 1e-15:
#                    self.update_messages(t, pi, cs, c)
#                else:
#                    self.fwd_messages[:,:,pi,c] = 1./self.nh #0

        return self.dirichlet_rew_params    
    
    
    def update_beliefs_dirichlet_context_obs_params(self, tau, t, posterior_context=[1], context_obs=None):
        
        inferred_context = np.argmax(posterior_context)
        # print(self.dirichlet_context_obs_params.round(4))
        self.dirichlet_context_obs_params[context_obs,:] += posterior_context      #  phi_ijk' = phi_ijk + delta_{i,r} q(s=j)q(c=k)
        # self.dirichlet_context_obs_params[context_obs,inferred_context] += 1
        
        # print(tau,t)
        # print(self.dirichlet_context_obs_params.round(3))
        for c in range(self.nc):
            self.generative_model_context[:,c] = np.exp(scs.digamma(self.dirichlet_context_obs_params[:,c])\
                                                -scs.digamma(self.dirichlet_context_obs_params[:,c].sum()))
            self.generative_model_context[:,c] /= self.generative_model_context[:,c].sum()
                
        return self.dirichlet_context_obs_params 
    
    
    def update_beliefs(self, tau, t, observation, reward, prev_response, context_observation=None):
        
        self.observations[tau,t] = observation
        self.rewards[tau,t] = reward
        self.actions[tau,t] = prev_response
        self.context_obs[tau] = context_observation

        if tau == 0:
            prior_context = self.prior_context[tau,-1]
        else: #elif t == 0:
            prior_context = np.dot(self.transition_matrix_context, self.posterior_context[tau-1, -1])#.reshape((self.nc))

            # if self.pars["trial_type"][tau] == 0 and tau % self.pars["miniblock_size"] == 0 and not tau == 0:
            #     print("\n")
            #     print(context_observation,tau,t)
            #     print(prior_context)
            #     prior_context = (np.eye(4)[context_observation] + 0.02) / (np.eye(4)[context_observation] + 0.02).sum()
            #     print(prior_context)

                
        if t==0:
            self.possible_policies = np.ones(self.npi, dtype=bool)
        else:
            curr_policies = (self.all_policies[:,t-1] == prev_response)
            self.possible_policies = np.logical_and(self.possible_policies, curr_policies)


        self.posterior_states[tau, t] = self.update_beliefs_states(
                                         tau, t,
                                         observation,
                                         reward)

        #update beliefs about policies
        self.posterior_policies[tau, t],\
        self.likelihood_policies[tau,t] = self.update_beliefs_policies(tau, t)

        if self.nc>1:
            self.posterior_context[tau, t] = \
            self.update_beliefs_context(tau, t, \
                                                   reward, \
                                                   self.posterior_states[tau, t], \
                                                   self.posterior_policies[tau, t], \
                                                   prior_context, \
                                                   self.all_policies,\
                                                   context=context_observation)

        if t == self.T-1 and tau < self.trials-1:
            
            if self.learn_context_obs:
                self.posterior_dirichlet_context_obs[tau] =\
                    self.update_beliefs_dirichlet_context_obs_params(tau,t,self.posterior_context[tau,t],context_observation)
            
            if self.learn_habit:
                self.posterior_dirichlet_pol[tau], self.prior_policies[tau+1] = self.update_beliefs_dirichlet_pol_params(tau, t, \
                                                                self.posterior_policies[tau,t], \
                                                                self.posterior_context[tau,t])

        if False:
            self.posterior_rewards[tau, t-1] = np.einsum('rsc,spc,pc,c->r',
                                                  self.generative_model_rewards,
                                                  self.posterior_states[tau,t,:,t],
                                                  self.posterior_policies[tau,t],
                                                  self.posterior_context[tau,t])
        #if reward > 0:
        # check later if stuff still works!
        if self.learn_rew and t>0:#==self.T-1:
            self.posterior_dirichlet_rew[tau,t] = self.update_beliefs_dirichlet_rew_params(tau, t, \
                                                            reward, \
                                                   self.posterior_states[tau, t], \
                                                   self.posterior_policies[tau, t], \
                                                   self.posterior_context[tau,t])


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

        self.generative_model_observations = generative_model_observations.copy()
        self.generative_model_states = generative_model_states.copy()
        self.generative_model_rewards = generative_model_rewards.copy()
        self.transition_matrix_context = transition_matrix_context.copy()
        self.prior_rewards = prior_rewards.copy()
        self.prior_states = prior_states.copy()
        self.prior_policies = prior_policies.copy()
        self.T = T
        self.nh = prior_states.shape[0]
        if len(generative_model_rewards.shape) > 2:
            self.infer_context = True
            self.nc = generative_model_rewards.shape[2]
        else:
            self.nc = 1
            self.generative_model_rewards = self.generative_model_rewards[:,:,np.newaxis]
        if dirichlet_pol_params is not None:
            self.dirichlet_pol_params = dirichlet_pol_params.copy()
        if dirichlet_rew_params is not None:
            self.dirichlet_rew_params = dirichlet_rew_params.copy()


        for c in range(self.nc):
            for state in range(self.nh):
                self.generative_model_rewards[:,state,c] =\
                np.exp(scs.digamma(self.dirichlet_rew_params[:,state,c])\
                       -scs.digamma(self.dirichlet_rew_params[:,state,c].sum()))
                self.generative_model_rewards[:,state,c] /= self.generative_model_rewards[:,state,c].sum()


    def instantiate_messages(self, policies):
        npi = policies.shape[0]

        self.bwd_messages = np.zeros((self.nh, self.T, npi, self.nc))
        self.bwd_messages[:,-1,:, :] = 1./self.nh
        self.fwd_messages = np.zeros((self.nh, self.T, npi, self.nc))
        self.fwd_messages[:, 0, :, :] = self.prior_states[:, np.newaxis, np.newaxis]

        self.fwd_norms = np.zeros((self.T+1, npi, self.nc))
        self.fwd_norms[0,:,:] = 1.

        self.obs_messages = np.zeros((self.nh, self.T, self.nc)) + 1/self.nh#self.prior_observations.dot(self.generative_model_observations)
        #self.obs_messages = np.tile(self.obs_messages,(self.T,1)).T

        self.rew_messages = np.zeros((self.nh, self.T, self.nc))
        #self.rew_messages[:] = np.tile(self.prior_rewards.dot(self.generative_model_rewards),(self.T,1)).T

        for c in range(self.nc):
            self.rew_messages[:,:,c] = self.prior_rewards.dot(self.generative_model_rewards[:,:,c])[:,np.newaxis]
            for pi, cstates in enumerate(policies):
                for t, u in enumerate(np.flip(cstates, axis = 0)):
                    tp = self.T - 2 - t
                    self.bwd_messages[:,tp,pi,c] = self.bwd_messages[:,tp+1,pi,c]*\
                                                self.obs_messages[:, tp+1,c]*\
                                                self.rew_messages[:, tp+1,c]
                    self.bwd_messages[:,tp,pi,c] = self.bwd_messages[:,tp,pi,c]\
                        .dot(self.generative_model_states[:,:,u])
                    self.bwd_messages[:,tp, pi,c] /= self.bwd_messages[:,tp,pi,c].sum()

    def update_messages(self, t, pi, cs, c=0):
        if t > 0:
            for i, u in enumerate(np.flip(cs[:t], axis = 0)):
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

        self.prior_rewards = new_preference.copy()

        for c in range(self.nc):
            self.rew_messages[:,:,c] = self.prior_rewards.dot(self.generative_model_rewards[:,:,c])[:,np.newaxis]
            for pi, cstates in enumerate(policies[t:]):
                for i, u in enumerate(np.flip(cstates, axis = 0)):
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

        self.obs_messages[:,t,:] = self.generative_model_observations[observation][:,np.newaxis]

        self.rew_messages[:,t,:] = self.generative_model_rewards[reward]

        for c in range(self.nc):
            for pi, cs in enumerate(policies):
                if self.prior_policies[pi,c] > 1e-15 and pi in possible_policies:
                    self.update_messages(t, pi, cs, c)
                else:
                    self.fwd_messages[:,:,pi,c] = 0#1./self.nh #0

        #estimate posterior state distribution
        posterior = self.fwd_messages*self.bwd_messages*self.obs_messages[:,:,np.newaxis,:]*self.rew_messages[:,:,np.newaxis,:]
        norm = posterior.sum(axis = 0)
        self.fwd_norms[-1] = norm[-1]
        posterior /= norm
        return np.nan_to_num(posterior)

    def update_beliefs_policies(self, tau, t, gamma=4):

        #print((prior_policies>1e-4).sum())
        likelihood = self.fwd_norms.prod(axis=0)
        posterior = np.power(likelihood,gamma) * self.prior_policies
        posterior/= posterior.sum(axis=0)[np.newaxis,:]
        #posterior = softmax(ln(self.fwd_norms).sum(axis = 0)+ln(self.prior_policies))

        #np.testing.assert_allclose(post, posterior)

        return posterior, likelihood


    def update_beliefs_context(self, tau, t, reward, posterior_states, posterior_policies, prior_context, policies):

        post_policies = (prior_context[np.newaxis,:] * posterior_policies).sum(axis=1)
        beta = self.dirichlet_rew_params.copy()
        states = (posterior_states[:,t,:] * post_policies[np.newaxis,:,np.newaxis]).sum(axis=1)
        beta_prime = self.dirichlet_rew_params.copy()
        beta_prime[reward] = beta[reward] + states

#        for c in range(self.nc):
#            for state in range(self.nh):
#                self.generative_model_rewards[:,state,c] =\
#                np.exp(scs.digamma(beta_prime[:,state,c])\
#                       -scs.digamma(beta_prime[:,state,c].sum()))
#                self.generative_model_rewards[:,state,c] /= self.generative_model_rewards[:,state,c].sum()
#
#            self.rew_messages[:,t+1:,c] = self.prior_rewards.dot(self.generative_model_rewards[:,:,c])[:,np.newaxis]
#
#        for c in range(self.nc):
#            for pi, cs in enumerate(policies):
#                if self.prior_policies[pi,c] > 1e-15:
#                    self.update_messages(t, pi, cs, c)
#                else:
#                    self.fwd_messages[:,:,pi,c] = 1./self.nh #0

        alpha = self.dirichlet_pol_params.copy()
        if t == self.T-1:
            chosen_pol = np.argmax(post_policies)
            inf_context = np.argmax(prior_context)
            alpha_prime = self.dirichlet_pol_params.copy()
            alpha_prime[chosen_pol,:] += prior_context
            #alpha_prime[chosen_pol,inf_context] = self.dirichlet_pol_params[chosen_pol,inf_context] + 1
        else:
            alpha_prime = alpha


        if self.nc == 1:
            posterior = np.ones(1)
        else:
            # todo: recalc
            #outcome_surprise = ((states * prior_context[np.newaxis,:]).sum(axis=1)[:,np.newaxis] * (scs.digamma(beta_prime[reward]) - scs.digamma(beta_prime.sum(axis=0)))).sum(axis=0)
            outcome_surprise = (posterior_policies * ln(self.fwd_norms.prod(axis=0))).sum(axis=0)
            entropy = - (posterior_policies * ln(posterior_policies)).sum(axis=0)
            #policy_surprise = (post_policies[:,np.newaxis] * scs.digamma(alpha_prime)).sum(axis=0) - scs.digamma(alpha_prime.sum(axis=0))
            policy_surprise = (posterior_policies * scs.digamma(alpha_prime)).sum(axis=0) - scs.digamma(alpha_prime.sum(axis=0))
            posterior = outcome_surprise + policy_surprise + entropy

                        #+ np.nan_to_num((posterior_policies * ln(self.fwd_norms).sum(axis = 0))).sum(axis=0)#\

#            if tau in range(90,120) and t == 1:
#                #print(tau, np.exp(outcome_surprise), np.exp(policy_surprise))
#                print(tau, np.exp(outcome_surprise[1])/np.exp(outcome_surprise[0]), np.exp(policy_surprise[1])/np.exp(policy_surprise[0]))


            posterior = np.nan_to_num(softmax(posterior+ln(prior_context)))

        return posterior


    def update_beliefs_dirichlet_pol_params(self, tau, t, posterior_policies, posterior_context = [1]):
        assert(t == self.T-1)
        chosen_pol = np.argmax(posterior_policies, axis=0)
#        self.dirichlet_pol_params[chosen_pol,:] += posterior_context.sum(axis=0)/posterior_context.sum()
        alpha = 0.3#0.3
        self.dirichlet_pol_params = (1-alpha) * self.dirichlet_pol_params + 1 - (1-alpha)
        self.dirichlet_pol_params[chosen_pol,:] += posterior_context
        self.prior_policies[:] = np.exp(scs.digamma(self.dirichlet_pol_params) - scs.digamma(self.dirichlet_pol_params.sum(axis=0))[np.newaxis,:])
        self.prior_policies /= self.prior_policies.sum(axis=0)

        return self.dirichlet_pol_params

    def update_beliefs_dirichlet_rew_params(self, tau, t, reward, posterior_states, posterior_policies, posterior_context = [1]):
        states = (posterior_states[:,t,:,:] * posterior_policies[np.newaxis,:,:]).sum(axis=1)
        state = np.argmax(states)
        old = self.dirichlet_rew_params.copy()
#        self.dirichlet_rew_params[:,state,:] = (1-0.4) * self.dirichlet_rew_params[:,state,:] #+1 - (1-0.4)
#        self.dirichlet_rew_params[reward,state,:] += 1#states * posterior_context[np.newaxis,:]
        alpha = 0.6#0.3#1#0.3#0.05
        self.dirichlet_rew_params[:,3:,:] = (1-alpha) * self.dirichlet_rew_params[:,3:,:] +1 - (1-alpha)
        self.dirichlet_rew_params[reward,:,:] += states * posterior_context[np.newaxis,:]
        for c in range(self.nc):
            for state in range(self.nh):
                self.generative_model_rewards[:,state,c] =\
                np.exp(scs.digamma(self.dirichlet_rew_params[:,state,c])\
                       -scs.digamma(self.dirichlet_rew_params[:,state,c].sum()))
                self.generative_model_rewards[:,state,c] /= self.generative_model_rewards[:,state,c].sum()

            self.rew_messages[:,t+1:,c] = self.prior_rewards.dot(self.generative_model_rewards[:,:,c])[:,np.newaxis]

#        for c in range(self.nc):
#            for pi, cs in enumerate(policies):
#                if self.prior_policies[pi,c] > 1e-15:
#                    self.update_messages(t, pi, cs, c)
#                else:
#                    self.fwd_messages[:,:,pi,c] = 1./self.nh #0

        return self.dirichlet_rew_params
