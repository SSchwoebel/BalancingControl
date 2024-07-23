from misc import ln, softmax
arr_type = "torch"
if arr_type == "numpy":
    import numpy as ar
    array = ar.array
else:
    import torch as ar
    array = ar.tensor

import numpy as np
import scipy.special as scs
from misc import D_KL_nd_dirichlet, D_KL_dirichlet_categorical
from opt_einsum import contract

#device = ar.device("cuda") if ar.cuda.is_available() else ar.device("cpu")
#device = ar.device("cuda")
#device = ar.device("cpu")

try:
    from inference import device
except:
    device = ar.device("cpu")


class Group2Perception(object):
    def __init__(self,
                 generative_model_observations,
                 generative_model_states,
                 generative_model_rewards,
                 transition_matrix_context,
                 prior_states,
                 prior_rewards,
                 prior_policies,
                 policies,
                 alpha_0 = ar.tensor([1]),
                 dirichlet_rew_params = None,
                 generative_model_context = None,
                 learn_habit = False,
                 learn_rew = False,
                 mask=None,
                 T=5, trials=10, pol_lambda=0, r_lambda=0, non_decaying=0,
                 dec_temp=1., npart=1, nsubs=1, infer_alpha_0=False, use_h=True):

        self.generative_model_observations = generative_model_observations
        self.generative_model_states = generative_model_states
        self.prior_rewards = prior_rewards
        self.nr = prior_rewards.shape[0]
        self.prior_states = prior_states
        self.T = T
        self.trials = trials
        self.nh = prior_states.shape[0]
        self.learn_habit = learn_habit
        self.learn_rew = learn_rew
        self.pol_lambda = pol_lambda
        self.r_lambda = r_lambda
        self.non_decaying = non_decaying
        self.dec_temp = dec_temp
        self.policies = policies
        self.npi = policies.shape[0]
        self.actions = ar.unique(policies)
        self.na = len(self.actions)
        self.npart = npart
        self.nsubs = nsubs
        # infer_alpha_0 says whether to infer alpha_0 at all
        self.infer_alpha_0 = infer_alpha_0
        # use_h says whether to use h or alpha_0 for inference
        self.use_h = use_h
        self.alpha_0 = alpha_0
        
        if mask is None:
            self.mask = ar.ones(trials, nsubs).bool()
        else:
            self.mask = mask.long()[:,None,:]

        if self.infer_alpha_0:
            self.npars = 4
        else:
            self.npars = 3
        self.param_names = list(self.locs_to_pars(ar.zeros(self.npars)).keys())

        self.dirichlet_rew_params_init = dirichlet_rew_params#ar.stack([dirichlet_rew_params]*self.npart, dim=-1)
        self.dirichlet_pol_params_init = ar.zeros((self.npi,self.npart, self.nsubs)).to(device) + self.alpha_0[None,...]#ar.stack([dirichlet_pol_params]*self.npart, dim=-1)

        self.dirichlet_rew_params = [ar.stack([ar.stack([self.dirichlet_rew_params_init for k in range(self.npart)], dim=-1) for j in range(self.nsubs)], dim=-1)]
        self.dirichlet_pol_params = [self.dirichlet_pol_params_init]

        #self.prior_policies_init = self.dirichlet_pol_params[0] / self.dirichlet_pol_params[0].sum(axis=0)[None,...]
        self.prior_policies = [self.dirichlet_pol_params[0] / self.dirichlet_pol_params[0].sum(axis=0)[None,...]]

        #self.generative_model_rewards_init = self.dirichlet_rew_params[0] / self.dirichlet_rew_params[0].sum(axis=0)[None,...]
        self.generative_model_rewards = [self.dirichlet_rew_params[0] / self.dirichlet_rew_params[0].sum(axis=0)[None,...]]

        self.observations = []
        self.rewards = []

        #self.instantiate_messages()
        self.bwd_messages = []
        self.fwd_messages = []
        self.obs_messages = []
        self.rew_messages = []
        self.fwd_norms = []

        self.posterior_states = []
        self.posterior_policies = []
        self.posterior_actions = []

        self.big_trans_matrix = ar.stack([ar.stack([generative_model_states[:,:,policies[pi,t]] for pi in range(self.npi)]) for t in range(self.T-1)]).T.to(device)
        #print(self.big_trans_matrix.shape)

        # self.reset()

    def locs_to_pars(self, locs):

        if self.infer_alpha_0:
            if self.use_h:
                par_dict = {"policy rate": ar.sigmoid(locs[...,0]),
                            "reward rate": ar.sigmoid(locs[...,1]),
                            "dec temp": 10*ar.sigmoid(locs[...,2]),
                            "habitual tendency": ar.sigmoid(locs[...,3])}
            else:
                par_dict = {"policy rate": ar.sigmoid(locs[...,0]),
                            "reward rate": ar.sigmoid(locs[...,1]),
                            "dec temp": 10*ar.sigmoid(locs[...,2]),
                            "habitual tendency": ar.exp(locs[...,3])}
        else:
            par_dict = {"policy rate": ar.sigmoid(locs[...,0]),
                        "reward rate": ar.sigmoid(locs[...,1]),
                        "dec temp": 10*ar.sigmoid(locs[...,2])}

        return par_dict

    def set_parameters(self, locs=None, par_dict=None):

        if locs is not None:
            par_dict = self.locs_to_pars(locs)

        if 'policy rate' in par_dict.keys():
            self.pol_lambda = par_dict['policy rate']
        if 'reward rate' in par_dict.keys():
            self.r_lambda = par_dict['reward rate']
        if 'dec temp' in par_dict.keys():
            self.dec_temp = par_dict['dec temp']
        if 'habitual tendency' in par_dict.keys():
            if self.use_h:
                self.alpha_0 = 1./par_dict['habitual tendency']
            else:
                self.alpha_0 = par_dict['habitual tendency']

    def reset(self):
        if len(self.dec_temp.shape) > 1:
            self.npart = self.dec_temp.shape[0]
            self.nsubs = self.dec_temp.shape[1]
        else:
            self.nsubs = self.dec_temp.shape[0]
            self.npart = 1
            #self.alpha_0 = self.alpha_0[None,:]
            self.pol_lambda = self.pol_lambda[None,:]
            self.r_lambda = self.r_lambda[None,:]
            self.dec_temp = self.dec_temp[None,:]

        # print(self.alpha_0.shape)
        # print(self.npart, self.nsubs)
        self.dirichlet_pol_params_init = ar.zeros((self.npi,self.npart, self.nsubs)).to(device) + self.alpha_0[None,...].to(device)

        self.dirichlet_rew_params = [ar.stack([ar.stack([self.dirichlet_rew_params_init for k in range(self.npart)], dim=-1) for j in range(self.nsubs)], dim=-1)]
        self.dirichlet_pol_params = [self.dirichlet_pol_params_init]

        self.prior_policies = [self.dirichlet_pol_params[0] / self.dirichlet_pol_params[0].sum(axis=0)[None,...]]

        self.generative_model_rewards = [self.dirichlet_rew_params[0] / self.dirichlet_rew_params[0].sum(axis=0)[None,...]]

        self.observations = []
        self.rewards = []

        #self.instantiate_messages()
        self.bwd_messages = []
        self.fwd_messages = []
        self.obs_messages = []
        self.rew_messages = []
        self.fwd_norms = []

        self.posterior_states = []
        self.posterior_policies = []
        self.posterior_actions = []


    def make_current_messages(self, tau, t):

        generative_model_rewards = self.generative_model_rewards[-1].to(device)

        #obs_messages = ar.zeros((self.nh, self.T)) + 1/self.nh

        # rew_messages = ar.zeros((self.nh, self.T))
        # rew_messages[:] = self.prior_rewards.matmul(generative_model_rewards)[:,None]
        observations = ar.stack(self.observations[-t-1:])*self.mask[tau]
        # obs_messages = []
        # for n in range(self.nsubs):
        #     prev_obs = [self.generative_model_observations[o] for o in observations[-t-1:,n]]
        #     obs = prev_obs + [ar.zeros((self.nh)).to(device)+1./self.nh]*(self.T-t-1)
        #     obs = [ar.stack(obs).T.to(device)]*self.npart
        #     obs_messages.append(ar.stack(obs, dim=-1))
        # old_obs_messages = ar.stack(obs_messages, dim=-1).to(device)
        
        prev_obs = self.generative_model_observations[observations].permute((2,0,1))[:,:,None,:]
        exp_obs = ar.zeros(self.nh, self.T-t-1, 1, self.nsubs).to(device)+1./self.nh
        new_obs_messages = ar.cat((prev_obs, exp_obs), dim=1)
        obs_messages = ar.cat([new_obs_messages]*self.npart, dim=-2)
        # print("obs correct", ar.allclose(obs_messages, old_obs_messages))

        # prev_obs = [self.generative_model_observations[o] for o in self.observations[-t-1:]]
        # obs = prev_obs + [ar.zeros((self.nh)).to(device)+1./self.nh]*(self.T-t-1)
        # obs = [ar.stack(obs).T.to(device)]*self.npart
        # obs = [ar.stack(obs,dim=-1).to(device)]*self.nsubs
        # obs_messages = ar.stack(obs,dim=-1).to(device)

        # prev_obs = [[self.generative_model_observations[o] for o in obs_vec] for obs_vec in self.observations[-t-1:]]
        # obs = prev_obs + [[ar.zeros((self.nh))+1./self.nh]*(self.T-t-1)]*n
        # obs_messages = ar.stack(obs).T

        # prev_rew = [generative_model_rewards[r] for r in self.rewards[-t-1:]]
        # rew = prev_rew + [self.prior_rewards.matmul(generative_model_rewards)]*(self.T-t-1)
        # rew_messages = ar.stack(rew).T

        # prev_rew = [generative_model_rewards[r] for r in self.rewards[-t-1:]]
        # rew = prev_rew + [self.prior_rewards.matmul(generative_model_rewards)]*(self.T-t-1)
        # rew_messages = ar.stack(rew).T
        rewards = ar.stack(self.rewards[-t-1:])*self.mask[tau]

        # rew_messages = []
        # for n in range(self.nsubs):
        #     rew_messages.append(ar.stack([ar.stack([generative_model_rewards[r,:,i,n].to(device) for r in rewards[-t-1:,n]]  \
        #                                            + [self.prior_rewards.matmul(generative_model_rewards[:,:,i,n].to(device)).to(device)]*(self.T-t-1)).T.to(device) for i in range(self.npart)], dim=-1).to(device))
        # old_rew_messages = ar.stack(rew_messages, dim=-1).to(device)
        

        one_hot_rews = ar.nn.functional.one_hot(rewards, num_classes=self.nr).float()
        prev_rew = ar.einsum('tnr,rspn->tspn', one_hot_rews, generative_model_rewards)
        exp_rew = ar.einsum('r,rspn->spn', self.prior_rewards, generative_model_rewards)
        exp_rews = ar.cat([exp_rew[None,...]]*(self.T), dim=0)
        rew_messages = ar.cat((prev_rew, exp_rews[:self.T-t-1]), dim=0).permute((1,0,2,3))
        # print("rew correct", ar.allclose(rew_messages, old_rew_messages))
        #print(rew.shape)

        # for i in range(t):
        #     tp = -t-1+i
            # observation = self.observations[tp]
            # obs_messages[:,i] = self.generative_model_observations[observation]

            # reward = self.rewards[tp]
            # rew_messages[:,i] = generative_model_rewards[reward]

        self.obs_messages.append(obs_messages)
        self.rew_messages.append(rew_messages)

    def update_messages(self, tau, t, possible_policies):

        # bwd_messages = ar.zeros((self.nh, self.T,self.npi)) #+ 1./self.nh
        # bwd_messages[:,-1,:] = 1./self.nh
        bwd = [ar.zeros((self.nh, self.npi, self.npart, self.nsubs)).to(device)+1./self.nh]
        # fwd_messages = ar.zeros((self.nh, self.T, self.npi))
        # fwd_messages[:,0,:] = self.prior_states[:,None]
        fwd = [ar.zeros((self.nh, self.npi, self.npart, self.nsubs)).to(device)+self.prior_states[:,None,None,None]]
        # fwd_norms = ar.zeros((self.T+1, self.npi))
        # fwd_norms[0,:] = 1.
        fwd_norm = [ar.ones(self.npi, self.npart, self.nsubs).to(device)]

        self.make_current_messages(tau,t)

        obs_messages = self.obs_messages[-1]
        rew_messages = self.rew_messages[-1]

        for i in range(self.T-2,-1,-1):
            tmp = ar.einsum('hpnk,shp,hnk,hnk->spnk',bwd[-1],self.big_trans_matrix[...,i],obs_messages[:,i+1],rew_messages[:,i+1]).to(device)
            bwd.append(tmp)
            #bwd_messages[:,i,:] = ar.einsum('hp,shp,h,h->sp',bwd_messages[:,i+1,:],self.big_trans_matrix[...,i],obs_messages[:,i+1],rew_messages[:,i+1])
            # bwd_messages[:,-2-i,pi] = bwd_messages[:,-1-i,pi]*\
            #                             obs_messages[:,t-i]*\
            #                             rew_messages[:, t-i]
            # bwd_messages[:,-2-i,pi] = bwd_messages[:,-2-i,pi]\
            #      .matmul(self.generative_model_states[:,:,u])
            #bwd_messages[:,i,:] = test[-1]
            norm = bwd[-1].sum(axis=0)
            mask = norm > 0
            bwd[-1][:,mask] /= norm[None,mask]
            # norm = bwd_messages[:,i,:].sum(axis=0)
            # mask = norm > 0
            # bwd_messages[:,i,:][:,mask] /= norm[None,mask]

        bwd.reverse()
        bwd_messages = ar.stack(bwd).permute(1,0,2,3,4).to(device)

        #     norm = bwd_messages[-1].sum(axis=0)
        #     mask = norm > 0
        #     bwd_messages[-1][:,mask] /= norm[None,mask]

        # bwd_messages = ar.stack(bwd_messages).permute((1,0,2))

        for i in range(self.T-1):
            tmp = ar.einsum('spnk,shp,snk,snk->hpnk',fwd[-1],self.big_trans_matrix[...,i],obs_messages[:,i],rew_messages[:,i]).to(device)
            fwd.append(tmp)
            # fwd_messages[:, 1+i, pi] = fwd_messages[:,i, pi]*\
            #                              obs_messages[:, i]*\
            #                              rew_messages[:, i]
            # fwd_messages[:, 1+i, pi] = self.generative_model_states[:,:,u].\
            #                              matmul(fwd_messages[:, 1+i, pi])
            norm = fwd[-1].sum(axis=0)
            mask = norm > 0
            fwd[-1][:,mask] /= norm[None,mask]
            zeros = ar.zeros((self.npi, self.npart, self.nsubs))
            fwd_norm.append(ar.where(possible_policies[:,None,:], norm, zeros))
            # fwd_norm.append(ar.zeros((self.npi,self.npart)).to(device))
            # fwd_norm[-1][possible_policies] = norm[possible_policies]
            # if fwd_norms[1+i, pi] > 0: #???? Shouldn't this not happen?
            #     fwd_messages[:,1+i, pi] /= fwd_messages[:,1+i,pi].sum()

            # else:
            #     fwd_messages[:,:,pi] = 0#1./self.nh

        fwd_messages = ar.stack(fwd).permute(1,0,2,3,4).to(device)

        # for pi, cs in enumerate(self.policies):
        #     if self.prior_policies[-1][pi] > 1e-15 and pi in possible_policies:

        #         for i, u in enumerate(ar.flip(cs[:], [0])):
        #             bwd_messages[:,-2-i,pi] = bwd_messages[:,-1-i,pi]*\
        #                                         obs_messages[:,t-i]*\
        #                                         rew_messages[:, t-i]
        #             bwd_messages[:,-2-i,pi] = bwd_messages[:,-2-i,pi]\
        #                  .matmul(self.generative_model_states[:,:,u])

        #             norm = bwd_messages[:,-2-i,pi].sum()
        #             if norm > 0:
        #                 bwd_messages[:,-2-i, pi] /= norm

        #         for i, u in enumerate(cs[:]):
        #             fwd_messages[:, 1+i, pi] = fwd_messages[:,i, pi]*\
        #                                          obs_messages[:, i]*\
        #                                          rew_messages[:, i]
        #             fwd_messages[:, 1+i, pi] = self.generative_model_states[:,:,u].\
        #                                          matmul(fwd_messages[:, 1+i, pi])
        #             fwd_norms[1+i,pi] = fwd_messages[:,1+i,pi].sum()
        #             if fwd_norms[1+i, pi] > 0: #???? Shouldn't this not happen?
        #                 fwd_messages[:,1+i, pi] /= fwd_messages[:,1+i,pi].sum()

        #     else:
        #         fwd_messages[:,:,pi] = 0#1./self.nh

        posterior = fwd_messages*bwd_messages*obs_messages[:,:,None,:]*rew_messages[:,:,None,:]
        norm = posterior.sum(axis = 0)
        #fwd_norms[-1] = norm[-1]
        fwd_norm.append(norm[-1])
        fwd_norms = ar.stack(fwd_norm).to(device)
        # print(tau,t,self.fwd_norms[tau,t])
        non_zero = norm > 0
        posterior[:,non_zero] /= norm[non_zero]

        self.bwd_messages.append(bwd_messages)
        self.fwd_messages.append(fwd_messages)
        self.fwd_norms.append(fwd_norms)
        self.posterior_states.append(posterior)

        return posterior

    def update_beliefs(self, tau, t, observation, reward, prev_response, possible_policies):

        self.update_beliefs_states(tau, t, observation, reward, possible_policies)

        #update beliefs about policies
        self.update_beliefs_policies(tau, t) #self.posterior_policies[tau, t], self.likelihood[tau,t]
        # if tau == 0:
        #     prior_context = self.prior_context
        # else: #elif t == 0:
        #     prior_context = ar.dot(self.perception.transition_matrix_context, self.posterior_context[tau-1, -1]).reshape((self.nc))
#            else:
#                prior_context = ar.dot(self.perception.transition_matrix_context, self.posterior_context[tau, t-1])

        # print(tau,t)
        # print("prior", prior_context)
        # print("post", self.posterior_context[tau, t])

        # if t < self.T-1:
        #     #post_pol = ar.matmul(self.posterior_policies[tau, t], self.posterior_context[tau, t])
        #     self.posterior_actions[tau, t] = self.estimate_action_probability(tau, t)

        if t == self.T-1 and self.learn_habit:
            self.update_beliefs_dirichlet_pol_params(tau, t)

        if False:
            self.posterior_rewards[tau, t-1] = ar.einsum('rsc,spc,pc,c->r',
                                                  self.perception.generative_model_rewards,
                                                  self.posterior_states[tau,t,:,t],
                                                  self.posterior_policies[tau,t])
        #if reward > 0:
        # check later if stuff still works!
        if self.learn_rew:# and t==self.T-1:
            self.update_beliefs_dirichlet_rew_params(tau, t, reward)

    def update_beliefs_states(self, tau, t, observation, reward, possible_policies):
        #estimate expected state distribution
        # if t == 0:
        #     self.instantiate_messages(policies)
        self.observations.append(observation)
        self.rewards.append(reward)

        self.update_messages(tau, t, possible_policies)

        #return posterior#ar.nan_to_num(posterior)

    def update_beliefs_policies(self, tau, t):

        #print((prior_policies>1e-4).sum())

        likelihood = (self.fwd_norms[-1]+1e-10).prod(axis=0).to(device)
        norm = likelihood.sum(axis=0).to(device)
        log_like = ar.log(likelihood/norm[None,...]+1e-10).to(device)
        likelihood = ar.exp(self.dec_temp[None,...]*self.mask[tau]*log_like).to(device)
        # print("like", likelihood)
        # Fe = ar.log((self.fwd_norms[-1]+1e-10).prod(axis=0))
        # softplus = ar.nn.Softplus(beta=self.dec_temp)
        # likelihood = softplus(Fe)
        # posterior_policies = likelihood * self.prior_policies[-1] / (likelihood * self.prior_policies[-1]).sum(axis=0)

        # likelihood = ar.pow(likelihood/norm[None,...],self.dec_temp[None,...]).to(device) #* ar.pow(norm,self.dec_temp)

        # print("softmax", likelihood)
        # print("norm1", ar.pow(norm,self.dec_temp))
        posterior_policies = likelihood * self.prior_policies[-1]*self.mask[tau] / (likelihood * self.prior_policies[-1]).sum(axis=0)
        # print("unnorm", likelihood * self.prior_policies[-1])
        # print("norm", (likelihood * self.prior_policies[-1]).sum(axis=0))
        # print("post", posterior_policies)
        #likelihood /= likelihood.sum(axis=0)[None,:]
        #posterior/= posterior.sum(axis=0)[None,:]
        #posterior = ar.nan_to_num(posterior)
        #posterior = softmax(ln(self.fwd_norms).sum(axis = 0)+ln(self.prior_policies))

        #ar.testing.assert_allclose(post, posterior)

        self.posterior_policies.append(posterior_policies)

        if t<self.T-1:
            posterior_actions = ar.zeros((self.na,self.npart, self.nsubs)).to(device)
            for a in range(self.na):
                posterior_actions[a] = posterior_policies[self.policies[:,t] == a,...].sum(axis=0)*self.mask[tau]

            posterior_actions = ar.where(self.mask[tau]>0, posterior_actions, 1./self.na)

            self.posterior_actions.append(posterior_actions)

        #return posterior, likelihood


    def update_beliefs_dirichlet_pol_params(self, tau, t):
        assert(t == self.T-1)
        chosen = ar.eye(self.npi)[ar.argmax(self.posterior_policies[-1], axis=0)].to(device)
        chosen_pol = chosen.permute((2,0,1))
        # print(chosen_pol.shape)
        #print(chosen_pol)
#        self.dirichlet_pol_params[chosen_pol,:] += posterior_context.sum(axis=0)/posterior_context.sum()
        dirichlet_pol_params = (1-self.pol_lambda*self.mask[tau])[None,:,:] * self.dirichlet_pol_params[-1] + (1 - (1-self.pol_lambda*self.mask[tau]))[None,:,:]*self.dirichlet_pol_params_init + chosen_pol#*self.dirichlet_pol_params_init
        #dirichlet_pol_params[(chosen_pol[0],list(range(self.npart)))] += 1#posterior_context

        prior_policies = dirichlet_pol_params / dirichlet_pol_params.sum(axis=0)[None,...]#ar.exp(scs.digamma(self.dirichlet_pol_params) - scs.digamma(self.dirichlet_pol_params.sum(axis=0))[None,:])
        #prior_policies /= prior_policies.sum(axis=0)[None,:]

        self.dirichlet_pol_params.append(dirichlet_pol_params.to(device))
        self.prior_policies.append(prior_policies.to(device))

        #return dirichlet_pol_params, prior_policies

    def update_beliefs_dirichlet_rew_params(self, tau, t, reward):
        posterior_states = self.posterior_states[-1]
        posterior_policies = self.posterior_policies[-1]
        states = (posterior_states[:,t,:,:,:] * posterior_policies[None,:,:,:]).sum(axis=1)
        # c = ar.argmax(posterior_context)
        # self.dirichlet_rew_params[reward,:,c] += states[:,c]

#         self.dirichlet_rew_params[tau,t,:,self.non_decaying:,:] = (1-self.r_lambda) * self.dirichlet_rew_params[tau,t,:,self.non_decaying:,:] +1 - (1-self.r_lambda)
#         self.dirichlet_rew_params[tau,t,reward,:,:] += states * posterior_context[None,:]
#         for c in range(self.nc):
#             for state in range(self.nh):
#                 #self.generative_model_rewards[:,state,c] = self.dirichlet_rew_params[:,state,c] / self.dirichlet_rew_params[:,state,c].sum()
#                 self.generative_model_rewards[tau,t,:,state,c] = self.dirichlet_rew_params[tau,t,:,state,c]#\
#                 # ar.exp(scs.digamma(self.dirichlet_rew_params[:,state,c])\
#                 #         -scs.digamma(self.dirichlet_rew_params[:,state,c].sum()))
#                 self.generative_model_rewards[tau,t,:,state,c] /= self.generative_model_rewards[tau,t,:,state,c].sum()
#             self.rew_messages[tau,t+1:,:,t+1:,c] = self.prior_rewards.matmul(self.generative_model_rewards[tau,t,:,:,c])[None,:,None]

        dirichlet_rew_params = self.dirichlet_rew_params[0].clone().to(device)#.detach()
        # dirichlet_rew_params = ar.ones_like(self.dirichlet_rew_params_init)#self.dirichlet_rew_params_init.clone()
        # dirichlet_rew_params[:,:self.non_decaying] = self.dirichlet_rew_params[-1][:,:self.non_decaying]
        dirichlet_rew_params[:,self.non_decaying:,:,:] = (1-self.r_lambda*self.mask[tau])[None,None,:,:] * self.dirichlet_rew_params[-1][:,self.non_decaying:,:,:] +1 - (1-self.r_lambda*self.mask[tau])[None,None,:,:]
        #dirichlet_rew_params[reward[0],:,:,:] += states #* posterior_context[None,:]

        # TODO: EVIL LOOP!
        for s in range(self.nsubs):
            dirichlet_rew_params[reward[s],:,:,s] += states[...,s]

        #one_hot_rews = ar.nn.functional.one_hot(reward, num_classes=self.nr).float()
        #new_rew_params = ar.einsum('rsmn,nr->rsmn', dirichlet_rew_params, one_hot_rews)
        #new_rew_params = dirichlet_rew_params + ar.einsum('rsmn,smn->rsmn', new_rew_params, states)
        #dirichlet_rew_params = dirichlet_rew_params[one_hot_rews,:,:,:] + states
        #ar.testing.assert_allclose(dirichlet_rew_params, new_rew_params)

        generative_model_rewards = dirichlet_rew_params / dirichlet_rew_params.sum(axis=0)[None,...]
        self.dirichlet_rew_params.append(dirichlet_rew_params.to(device))
        self.generative_model_rewards.append(generative_model_rewards.to(device))

        #return dirichlet_rew_params


class GroupPerception(object):
    def __init__(self,
                 generative_model_observations,
                 generative_model_states,
                 generative_model_rewards,
                 transition_matrix_context,
                 prior_states,
                 prior_rewards,
                 prior_policies,
                 policies,
                 observations,
                 rewards,
                 responses,
                 alpha_0 = None,
                 dirichlet_rew_params = None,
                 generative_model_context = None,
                 T=5, trials=10, pol_lambda=0, r_lambda=0, non_decaying=0,
                 dec_temp=1., npart=1, nsubs=1):

        self.generative_model_observations = generative_model_observations
        self.generative_model_states = generative_model_states
        self.prior_rewards = prior_rewards
        self.prior_states = prior_states
        self.T = T
        self.trials = trials
        self.nh = prior_states.shape[0]
        self.pol_lambda = pol_lambda
        self.r_lambda = r_lambda
        self.non_decaying = non_decaying
        self.dec_temp = dec_temp
        self.policies = policies
        self.npi = policies.shape[0]
        self.actions = ar.unique(policies)
        self.na = len(self.actions)

        self.npart = npart
        self.nsubs = nsubs
        self.alpha_0 = alpha_0
        self.dirichlet_rew_params_init = dirichlet_rew_params#ar.stack([dirichlet_rew_params]*self.npart, dim=-1)
        self.dirichlet_pol_params_init = ar.zeros((self.nsubs, self.npi,self.npart)).to(device) + self.alpha_0#ar.stack([dirichlet_pol_params]*self.npart, dim=-1)

        self.dirichlet_rew_params = [ar.stack([ar.stack([self.dirichlet_rew_params_init]*self.npart, dim=-1)]*self.nsubs).to(device)]
        self.dirichlet_pol_params = [self.dirichlet_pol_params_init]

        #self.prior_policies_init = self.dirichlet_pol_params[0] / self.dirichlet_pol_params[0].sum(axis=0)[None,...]
        self.prior_policies = [self.dirichlet_pol_params[0] / self.dirichlet_pol_params[0].sum(axis=1)[:,None,...]]

        #self.generative_model_rewards_init = self.dirichlet_rew_params[0] / self.dirichlet_rew_params[0].sum(axis=0)[None,...]
        self.generative_model_rewards = [self.dirichlet_rew_params[0] / self.dirichlet_rew_params[0].sum(axis=1)[:,None,...]]

        # obs_messages = []
        # for n in range(nsubs):
        #     sub_messages = []
        #     for tau in range(trials):
        #         trial_messages = []
        #         for t in range(T):
        #             prev_obs = [self.generative_model_observations[o] for o in observations[n,tau, -t-1:]]
        #             obs = prev_obs + [ar.zeros((self.nh)).to(device)+1./self.nh]*(self.T-t-1)
        #             obs = [ar.stack(obs).T.to(device)]*npart
        #             trial_messages.append(ar.stack(obs,dim=-1).to(device))

        #         sub_messages.append(ar.stack(trial_messages))
        #     obs_messages.append(ar.stack(sub_messages))

        # self.obs_messages = ar.stack(obs_messages)
        # print(self.obs_messages.shape)

        self.obs_messages = []

        self.rew_messages = []

        # possible_policies = []
        # for n in range(nsubs):
        #     sub_policies = []
        #     for tau in range(trials):
        #         trial_policies = []
        #         for t in range(T):
        #             if t == 0:
        #                 trial_policies.append(ar.ones(self.npi, dtype=bool))
        #             else:
        #                 response = actions[n,tau,t-1]
        #                 possible_pols = policies[:,t-1]==response
        #                 prev_pols = trial_policies[-1]
        #                 new_pols = ar.logical_and(possible_pols, prev_pols).to(device)
        #                 trial_policies.append(new_pols)
        #         sub_policies.append(ar.stack(trial_policies))
        #     possible_policies.append(ar.stack(sub_policies))

        # self.possible_policies = ar.stack(possible_policies)
        self.possible_policies = []

        self.observations = observations
        self.rewards = rewards
        self.responses = responses

        #self.instantiate_messages()
        self.bwd_messages = []
        self.fwd_messages = []
        self.fwd_norms = []

        self.posterior_states = []
        self.posterior_policies = []
        self.posterior_actions = []

        self.big_trans_matrix = ar.stack([ar.stack([generative_model_states[:,:,policies[pi,t]] for pi in range(self.npi)]) for t in range(self.T-1)]).T.to(device)
        #print(self.big_trans_matrix.shape)

    def reset(self):

        if len(self.dec_temp.shape) > 1:
            self.nsubs = self.dec_temp.shape[1]
            self.npart = self.dec_temp.shape[0]
            self.alpha_0 = self.alpha_0.permute(1,0)
            self.pol_lambda = self.pol_lambda.permute(1,0)
            self.r_lambda = self.r_lambda.permute(1,0)
            self.dec_temp = self.dec_temp.permute(1,0)
        else:
            self.nsubs = self.dec_temp.shape[0]
            self.npart = 1
            self.alpha_0 = self.alpha_0[:,None]
            self.pol_lambda = self.pol_lambda[:,None]
            self.r_lambda = self.r_lambda[:,None]
            self.dec_temp = self.dec_temp[:,None]

        self.dirichlet_pol_params_init = ar.zeros((self.nsubs, self.npi,self.npart)).to(device) + self.alpha_0[:,None,:].to(device)

        self.dirichlet_rew_params = [ar.stack([ar.stack([self.dirichlet_rew_params_init]*self.npart, dim=-1)]*self.nsubs).to(device)]
        self.dirichlet_pol_params = [self.dirichlet_pol_params_init]

        self.prior_policies = [self.dirichlet_pol_params[0] / self.dirichlet_pol_params[0].sum(axis=1)[:,None,...]]

        self.generative_model_rewards = [self.dirichlet_rew_params[0] / self.dirichlet_rew_params[0].sum(axis=1)[:,None,...]]

        #self.instantiate_messages()
        self.bwd_messages = []
        self.fwd_messages = []
        self.fwd_norms = []
        self.rew_messages = []
        self.obs_messages = []
        self.possible_policies = []

        self.posterior_states = []
        self.posterior_policies = []
        self.posterior_actions = []

    def make_current_messages(self, tau, t):

        rew_messages = []
        for n in range(self.nsubs):

            rew_messages.append(ar.stack([ar.stack([self.generative_model_rewards[-1][n,r,:,i].to(device) for r in self.rewards[n,tau, -t-1:]] \
                                                          + [self.prior_rewards.matmul(self.generative_model_rewards[-1][n,:,:,i].to(device)).to(device)]*(self.T-t-1)).T.to(device) for i in range(self.npart)], dim=-1).to(device))

        self.rew_messages.append(ar.stack(rew_messages))

        obs_messages = []
        for n in range(self.nsubs):
            prev_obs = [self.generative_model_observations[o] for o in self.observations[n,tau, -t-1:]]
            obs = prev_obs + [ar.zeros((self.nh)).to(device)+1./self.nh]*(self.T-t-1)
            obs = [ar.stack(obs).T.to(device)]*self.npart
            obs_messages.append(ar.stack(obs,dim=-1).to(device))

        self.obs_messages.append(ar.stack(obs_messages))

        possible_policies = []
        for n in range(self.nsubs):
            if t == 0:
                possible_policies.append(ar.ones(self.npi, dtype=bool))
            else:
                response = self.responses[n,tau,t-1]
                possible_pols = self.policies[:,t-1]==response
                prev_pols = self.possible_policies[-1][n]
                new_pols = ar.logical_and(possible_pols, prev_pols).to(device)
                possible_policies.append(new_pols)

        self.possible_policies.append(ar.stack(possible_policies))

    def update_messages(self, tau, t):

        # bwd_messages = ar.zeros((self.nh, self.T,self.npi)) #+ 1./self.nh
        # bwd_messages[:,-1,:] = 1./self.nh
        bwd = [ar.zeros((self.nsubs, self.nh, self.npi, self.npart)).to(device)+1./self.nh]
        # fwd_messages = ar.zeros((self.nh, self.T, self.npi))
        # fwd_messages[:,0,:] = self.prior_states[:,None]
        fwd = [ar.zeros((self.nsubs, self.nh, self.npi, self.npart)).to(device)+self.prior_states[None,:,None,None]]
        # fwd_norms = ar.zeros((self.T+1, self.npi))
        # fwd_norms[0,:] = 1.
        fwd_norm = [ar.ones(self.nsubs, self.npi, self.npart).to(device)]

        self.make_current_messages(tau,t)

        obs_messages = self.obs_messages[-1]
        rew_messages = self.rew_messages[-1]

        for i in range(self.T-2,-1,-1):
            tmp = ar.einsum('Nhpn,shp,Nhn,Nhn->Nspn',bwd[-1],self.big_trans_matrix[...,i],obs_messages[:,:,i+1],rew_messages[:,:,i+1]).to(device)
            #bwd_messages[:,i,:] = ar.einsum('hp,shp,h,h->sp',bwd_messages[:,i+1,:],self.big_trans_matrix[...,i],obs_messages[:,i+1],rew_messages[:,i+1])
            # bwd_messages[:,-2-i,pi] = bwd_messages[:,-1-i,pi]*\
            #                             obs_messages[:,t-i]*\
            #                             rew_messages[:, t-i]
            # bwd_messages[:,-2-i,pi] = bwd_messages[:,-2-i,pi]\
            #      .matmul(self.generative_model_states[:,:,u])
            #bwd_messages[:,i,:] = test[-1]
            norm = tmp.sum(axis=1)
            bwd.append(ar.where(norm[:,None,...]>0, tmp/norm[:,None,...], tmp))
            # norm = bwd_messages[:,i,:].sum(axis=0)
            # mask = norm > 0
            # bwd_messages[:,i,:][:,mask] /= norm[None,mask]

        bwd.reverse()
        bwd_messages = ar.stack(bwd).permute(1, 2, 0, 3, 4).to(device)

        #     norm = bwd_messages[-1].sum(axis=0)
        #     mask = norm > 0
        #     bwd_messages[-1][:,mask] /= norm[None,mask]

        # bwd_messages = ar.stack(bwd_messages).permute((1,0,2))

        for i in range(self.T-1):
            tmp = ar.einsum('Nspn,shp,Nsn,Nsn->Nhpn',fwd[-1],self.big_trans_matrix[...,i],obs_messages[:,:,i],rew_messages[:,:,i]).to(device)
            # fwd_messages[:, 1+i, pi] = fwd_messages[:,i, pi]*\
            #                              obs_messages[:, i]*\
            #                              rew_messages[:, i]
            # fwd_messages[:, 1+i, pi] = self.generative_model_states[:,:,u].\
            #                              matmul(fwd_messages[:, 1+i, pi])
            norm = fwd[-1].sum(axis=1)
            fwd.append(ar.where(norm[:,None,...]>0, tmp/norm[:,None,...], tmp))
            zeros = ar.zeros((self.nsubs, self.npi, self.npart))
            fwd_norm.append(ar.where(self.possible_policies[-1][:,:,None], norm, zeros))
            # fwd_norm.append(ar.zeros((self.npi,self.npart)).to(device))
            # fwd_norm[-1][possible_policies] = norm[possible_policies]
            # if fwd_norms[1+i, pi] > 0: #???? Shouldn't this not happen?
            #     fwd_messages[:,1+i, pi] /= fwd_messages[:,1+i,pi].sum()

            # else:
            #     fwd_messages[:,:,pi] = 0#1./self.nh

        fwd_messages = ar.stack(fwd).permute(1, 2, 0, 3, 4).to(device)

        # for pi, cs in enumerate(self.policies):
        #     if self.prior_policies[-1][pi] > 1e-15 and pi in possible_policies:

        #         for i, u in enumerate(ar.flip(cs[:], [0])):
        #             bwd_messages[:,-2-i,pi] = bwd_messages[:,-1-i,pi]*\
        #                                         obs_messages[:,t-i]*\
        #                                         rew_messages[:, t-i]
        #             bwd_messages[:,-2-i,pi] = bwd_messages[:,-2-i,pi]\
        #                  .matmul(self.generative_model_states[:,:,u])

        #             norm = bwd_messages[:,-2-i,pi].sum()
        #             if norm > 0:
        #                 bwd_messages[:,-2-i, pi] /= norm

        #         for i, u in enumerate(cs[:]):
        #             fwd_messages[:, 1+i, pi] = fwd_messages[:,i, pi]*\
        #                                          obs_messages[:, i]*\
        #                                          rew_messages[:, i]
        #             fwd_messages[:, 1+i, pi] = self.generative_model_states[:,:,u].\
        #                                          matmul(fwd_messages[:, 1+i, pi])
        #             fwd_norms[1+i,pi] = fwd_messages[:,1+i,pi].sum()
        #             if fwd_norms[1+i, pi] > 0: #???? Shouldn't this not happen?
        #                 fwd_messages[:,1+i, pi] /= fwd_messages[:,1+i,pi].sum()

        #     else:
        #         fwd_messages[:,:,pi] = 0#1./self.nh

        posterior = fwd_messages*bwd_messages*obs_messages[:,:,:,None,:]*rew_messages[:,:,:,None,:]
        norm = posterior.sum(axis = 1)
        #fwd_norms[-1] = norm[-1]
        fwd_norm.append(norm[:,-1])
        fwd_norms = ar.stack(fwd_norm, dim=1).to(device)
        # print(tau,t,self.fwd_norms[tau,t])
        non_zero = norm > 0
        posterior = ar.where(non_zero[:,None,...], posterior/norm[:,None,...], posterior)
        # posterior[:,non_zero] /= norm[non_zero]

        self.bwd_messages.append(bwd_messages)
        self.fwd_messages.append(fwd_messages)
        self.fwd_norms.append(fwd_norms)
        self.posterior_states.append(posterior)

        return posterior

    def update_beliefs_states(self, tau, t):
        #estimate expected state distribution
        # if t == 0:
        #     self.instantiate_messages(policies)

        self.update_messages(tau, t)

        #return posterior#ar.nan_to_num(posterior)

    def update_beliefs_policies(self, tau, t):

        #print((prior_policies>1e-4).sum())

        likelihood = (self.fwd_norms[-1]+1e-10).prod(axis=1).to(device)
        norm = likelihood.sum(axis=1)
        # print("like", likelihood)
        # Fe = ar.log((self.fwd_norms[-1]+1e-10).prod(axis=0))
        # softplus = ar.nn.Softplus(beta=self.dec_temp)
        # likelihood = softplus(Fe)
        # posterior_policies = likelihood * self.prior_policies[-1] / (likelihood * self.prior_policies[-1]).sum(axis=0)
        likelihood = ar.pow(likelihood/norm[:,None,...],self.dec_temp[:,None,...]).to(device) #* ar.pow(norm,self.dec_temp)
        # print("softmax", likelihood)
        # print("norm1", ar.pow(norm,self.dec_temp))
        posterior_policies = likelihood * self.prior_policies[-1] / (likelihood * self.prior_policies[-1]).sum(axis=1)[:,None,...]
        # print("unnorm", likelihood * self.prior_policies[-1])
        # print("norm", (likelihood * self.prior_policies[-1]).sum(axis=0))
        # print("post", posterior_policies)
        #likelihood /= likelihood.sum(axis=0)[None,:]
        #posterior/= posterior.sum(axis=0)[None,:]
        #posterior = ar.nan_to_num(posterior)
        #posterior = softmax(ln(self.fwd_norms).sum(axis = 0)+ln(self.prior_policies))

        #ar.testing.assert_allclose(post, posterior)

        self.posterior_policies.append(posterior_policies)

        if t<self.T-1:
            posterior_actions = ar.zeros((self.nsubs, self.na,self.npart)).to(device)
            for a in range(self.na):
                posterior_actions[:, a] = posterior_policies[:, self.policies[:,t] == a, :].sum(axis=1)

            self.posterior_actions.append(posterior_actions)

        #return posterior, likelihood


    def update_beliefs_dirichlet_pol_params(self, tau, t):
        assert(t == self.T-1)
        chosen_pol = ar.argmax(self.posterior_policies[-1], axis=1).to(device)
        # print(self.posterior_policies)
        #print(chosen_pol)
#        self.dirichlet_pol_params[chosen_pol,:] += posterior_context.sum(axis=0)/posterior_context.sum()
        dirichlet_pol_params = (1-self.pol_lambda[:,None,:]) * self.dirichlet_pol_params[-1] + (1 - (1-self.pol_lambda[:,None,:]))*self.dirichlet_pol_params_init + self.posterior_policies[-1]
        # dirichlet_pol_params[(chosen_pol,list(range(self.npart)))] += 1#posterior_context

        prior_policies = dirichlet_pol_params / dirichlet_pol_params.sum(axis=1)[:,None,...]#ar.exp(scs.digamma(self.dirichlet_pol_params) - scs.digamma(self.dirichlet_pol_params.sum(axis=0))[None,:])
        #prior_policies /= prior_policies.sum(axis=0)[None,:]

        self.dirichlet_pol_params.append(dirichlet_pol_params.to(device))
        self.prior_policies.append(prior_policies.to(device))

        #return dirichlet_pol_params, prior_policies

    def update_beliefs_dirichlet_rew_params(self, tau, t):
        posterior_states = self.posterior_states[-1]
        posterior_policies = self.posterior_policies[-1]
        states = (posterior_states[:,:,t,:,:] * posterior_policies[:,None,:,:]).sum(axis=2)
        reward = self.rewards[:,tau,t]
        rew_indicies = (list(range(self.nsubs)), reward)
        # c = ar.argmax(posterior_context)
        # self.dirichlet_rew_params[reward,:,c] += states[:,c]

#         self.dirichlet_rew_params[tau,t,:,self.non_decaying:,:] = (1-self.r_lambda) * self.dirichlet_rew_params[tau,t,:,self.non_decaying:,:] +1 - (1-self.r_lambda)
#         self.dirichlet_rew_params[tau,t,reward,:,:] += states * posterior_context[None,:]
#         for c in range(self.nc):
#             for state in range(self.nh):
#                 #self.generative_model_rewards[:,state,c] = self.dirichlet_rew_params[:,state,c] / self.dirichlet_rew_params[:,state,c].sum()
#                 self.generative_model_rewards[tau,t,:,state,c] = self.dirichlet_rew_params[tau,t,:,state,c]#\
#                 # ar.exp(scs.digamma(self.dirichlet_rew_params[:,state,c])\
#                 #         -scs.digamma(self.dirichlet_rew_params[:,state,c].sum()))
#                 self.generative_model_rewards[tau,t,:,state,c] /= self.generative_model_rewards[tau,t,:,state,c].sum()
#             self.rew_messages[tau,t+1:,:,t+1:,c] = self.prior_rewards.matmul(self.generative_model_rewards[tau,t,:,:,c])[None,:,None]

        dirichlet_rew_params = self.dirichlet_rew_params[0].clone().to(device)
        dirichlet_rew_params[:,:,self.non_decaying:,:] = (1-self.r_lambda)[:,None,None,:] * self.dirichlet_rew_params[-1][:,:,self.non_decaying:,:] +1 - (1-self.r_lambda)[:,None,None,:]

        for s in range(self.nsubs):
            dirichlet_rew_params[s,reward[s],:,:] += states[s] #* posterior_context[None,:]

        generative_model_rewards = dirichlet_rew_params / dirichlet_rew_params.sum(axis=1)[:,None,...]
        self.dirichlet_rew_params.append(dirichlet_rew_params.to(device))
        self.generative_model_rewards.append(generative_model_rewards.to(device))

        #return dirichlet_rew_params


class FittingPerception(object):
    def __init__(self,
                 generative_model_observations,
                 generative_model_states,
                 generative_model_rewards,
                 transition_matrix_context,
                 prior_states,
                 prior_rewards,
                 prior_policies,
                 policies,
                 alpha_0 = None,
                 dirichlet_rew_params = None,
                 generative_model_context = None,
                 T=5, trials=10, pol_lambda=0, r_lambda=0, non_decaying=0,
                 dec_temp=1., npart=1):

        self.generative_model_observations = generative_model_observations
        self.generative_model_states = generative_model_states
        self.prior_rewards = prior_rewards
        self.prior_states = prior_states
        self.T = T
        self.trials = trials
        self.nh = prior_states.shape[0]
        self.pol_lambda = pol_lambda
        self.r_lambda = r_lambda
        self.non_decaying = non_decaying
        self.dec_temp = dec_temp
        self.policies = policies
        self.npi = policies.shape[0]
        self.actions = ar.unique(policies)
        self.na = len(self.actions)
        self.npart = npart
        self.alpha_0 = alpha_0
        self.dirichlet_rew_params_init = dirichlet_rew_params#ar.stack([dirichlet_rew_params]*self.npart, dim=-1)
        self.dirichlet_pol_params_init = ar.zeros((self.npi,self.npart)).to(device) + self.alpha_0[None,:]#ar.stack([dirichlet_pol_params]*self.npart, dim=-1)

        self.dirichlet_rew_params = [ar.stack([self.dirichlet_rew_params_init]*self.npart, dim=-1).to(device)]
        self.dirichlet_pol_params = [self.dirichlet_pol_params_init]

        #self.prior_policies_init = self.dirichlet_pol_params[0] / self.dirichlet_pol_params[0].sum(axis=0)[None,...]
        self.prior_policies = [self.dirichlet_pol_params[0] / self.dirichlet_pol_params[0].sum(axis=0)[None,...]]

        #self.generative_model_rewards_init = self.dirichlet_rew_params[0] / self.dirichlet_rew_params[0].sum(axis=0)[None,...]
        self.generative_model_rewards = [self.dirichlet_rew_params[0] / self.dirichlet_rew_params[0].sum(axis=0)[None,...]]

        self.observations = []
        self.rewards = []

        #self.instantiate_messages()
        self.bwd_messages = []
        self.fwd_messages = []
        self.obs_messages = []
        self.rew_messages = []
        self.fwd_norms = []

        self.posterior_states = []
        self.posterior_policies = []
        self.posterior_actions = []

        self.big_trans_matrix = ar.stack([ar.stack([generative_model_states[:,:,policies[pi,t]] for pi in range(self.npi)]) for t in range(self.T-1)]).T.to(device)
        #print(self.big_trans_matrix.shape)

    def reset(self):

        self.npart = self.dec_temp.shape[0]

        self.dirichlet_pol_params_init = ar.zeros((self.npi,self.npart)).to(device) + self.alpha_0[None,:].to(device)

        self.dirichlet_rew_params = [ar.stack([self.dirichlet_rew_params_init]*self.npart, dim=-1).to(device)]
        self.dirichlet_pol_params = [self.dirichlet_pol_params_init]

        self.prior_policies = [self.dirichlet_pol_params[0] / self.dirichlet_pol_params[0].sum(axis=0)[None,...]]

        self.generative_model_rewards = [self.dirichlet_rew_params[0] / self.dirichlet_rew_params[0].sum(axis=0)[None,...]]

        self.observations = []
        self.rewards = []

        #self.instantiate_messages()
        self.bwd_messages = []
        self.fwd_messages = []
        self.obs_messages = []
        self.rew_messages = []
        self.fwd_norms = []

        self.posterior_states = []
        self.posterior_policies = []
        self.posterior_actions = []


    # def instantiate_messages(self):

    #     self.bwd_messages_init = ar.zeros((self.nh, self.T, self.npi))
    #     self.bwd_messages_init[:,-1,:] = 1./self.nh
    #     self.fwd_messages_init = ar.zeros((self.nh, self.T, self.npi))
    #     self.fwd_messages_init[:,0,:] = self.prior_states[:,None]
    #     self.fwd_norms_init = ar.zeros((self.T+1, self.npi))
    #     self.fwd_norms_init[0,:] = 1.

    #     self.obs_messages_init = ar.zeros((self.nh, self.T)) + 1/self.nh

    #     self.rew_messages_init = ar.zeros((self.nh, self.T))
    #     self.rew_messages_init[:] = self.prior_rewards.matmul(self.generative_model_rewards_init)[:,None]

    #     for pi, cstates in enumerate(self.policies):
    #         for t, u in enumerate(ar.flip(cstates, [0])):
    #             tp = self.T - 2 - t
    #             self.bwd_messages_init[:,tp,pi] = (self.bwd_messages_init[:,tp+1,pi]*\
    #                                         self.obs_messages_init[:,tp+1]*\
    #                                         self.rew_messages_init[:,tp+1])
    #             bwd_message = (self.bwd_messages_init[:,tp,pi]\
    #                 .matmul(self.generative_model_states[:,:,u]))
    #             bwd_message /= bwd_message.sum()
    #             self.bwd_messages_init[:,tp,pi] = bwd_message

    #     self.bwd_messages = [self.bwd_messages_init]
    #     self.fwd_messages = [self.fwd_messages_init]
    #     self.fwd_norms = [self.fwd_norms_init]
    #     self.obs_messages = [self.obs_messages_init]
    #     self.rew_messages = [self.rew_messages_init]


    def make_current_messages(self, tau, t):

        generative_model_rewards = self.generative_model_rewards[-1].to(device)

        #obs_messages = ar.zeros((self.nh, self.T)) + 1/self.nh

        # rew_messages = ar.zeros((self.nh, self.T))
        # rew_messages[:] = self.prior_rewards.matmul(generative_model_rewards)[:,None]

        prev_obs = [self.generative_model_observations[o] for o in self.observations[-t-1:]]
        obs = prev_obs + [ar.zeros((self.nh)).to(device)+1./self.nh]*(self.T-t-1)
        obs = [ar.stack(obs).T.to(device)]*self.npart
        obs_messages = ar.stack(obs,dim=-1).to(device)

        # prev_obs = [[self.generative_model_observations[o] for o in obs_vec] for obs_vec in self.observations[-t-1:]]
        # obs = prev_obs + [[ar.zeros((self.nh))+1./self.nh]*(self.T-t-1)]*n
        # obs_messages = ar.stack(obs).T

        # prev_rew = [generative_model_rewards[r] for r in self.rewards[-t-1:]]
        # rew = prev_rew + [self.prior_rewards.matmul(generative_model_rewards)]*(self.T-t-1)
        # rew_messages = ar.stack(rew).T

        # prev_rew = [generative_model_rewards[r] for r in self.rewards[-t-1:]]
        # rew = prev_rew + [self.prior_rewards.matmul(generative_model_rewards)]*(self.T-t-1)
        # rew_messages = ar.stack(rew).T

        rew_messages = ar.stack([ar.stack([generative_model_rewards[r,:,i].to(device) for r in self.rewards[-t-1:]] + [self.prior_rewards.matmul(generative_model_rewards[:,:,i].to(device)).to(device)]*(self.T-t-1)).T.to(device) for i in range(self.npart)], dim=-1).to(device)
        #print(rew.shape)

        # for i in range(t):
        #     tp = -t-1+i
            # observation = self.observations[tp]
            # obs_messages[:,i] = self.generative_model_observations[observation]

            # reward = self.rewards[tp]
            # rew_messages[:,i] = generative_model_rewards[reward]

        self.obs_messages.append(obs_messages)
        self.rew_messages.append(rew_messages)

    def update_messages(self, tau, t, possible_policies):

        # bwd_messages = ar.zeros((self.nh, self.T,self.npi)) #+ 1./self.nh
        # bwd_messages[:,-1,:] = 1./self.nh
        bwd = [ar.zeros((self.nh, self.npi, self.npart)).to(device)+1./self.nh]
        # fwd_messages = ar.zeros((self.nh, self.T, self.npi))
        # fwd_messages[:,0,:] = self.prior_states[:,None]
        fwd = [ar.zeros((self.nh, self.npi, self.npart)).to(device)+self.prior_states[:,None,None]]
        # fwd_norms = ar.zeros((self.T+1, self.npi))
        # fwd_norms[0,:] = 1.
        fwd_norm = [ar.ones(self.npi, self.npart).to(device)]

        self.make_current_messages(tau,t)

        obs_messages = self.obs_messages[-1]
        rew_messages = self.rew_messages[-1]

        for i in range(self.T-2,-1,-1):
            tmp = ar.einsum('hpn,shp,hn,hn->spn',bwd[-1],self.big_trans_matrix[...,i],obs_messages[:,i+1],rew_messages[:,i+1]).to(device)
            bwd.append(tmp)
            #bwd_messages[:,i,:] = ar.einsum('hp,shp,h,h->sp',bwd_messages[:,i+1,:],self.big_trans_matrix[...,i],obs_messages[:,i+1],rew_messages[:,i+1])
            # bwd_messages[:,-2-i,pi] = bwd_messages[:,-1-i,pi]*\
            #                             obs_messages[:,t-i]*\
            #                             rew_messages[:, t-i]
            # bwd_messages[:,-2-i,pi] = bwd_messages[:,-2-i,pi]\
            #      .matmul(self.generative_model_states[:,:,u])
            #bwd_messages[:,i,:] = test[-1]
            norm = bwd[-1].sum(axis=0)
            mask = norm > 0
            bwd[-1][:,mask] /= norm[None,mask]
            # norm = bwd_messages[:,i,:].sum(axis=0)
            # mask = norm > 0
            # bwd_messages[:,i,:][:,mask] /= norm[None,mask]

        bwd.reverse()
        bwd_messages = ar.stack(bwd).permute(1,0,2,3).to(device)

        #     norm = bwd_messages[-1].sum(axis=0)
        #     mask = norm > 0
        #     bwd_messages[-1][:,mask] /= norm[None,mask]

        # bwd_messages = ar.stack(bwd_messages).permute((1,0,2))

        for i in range(self.T-1):
            tmp = ar.einsum('spn,shp,sn,sn->hpn',fwd[-1],self.big_trans_matrix[...,i],obs_messages[:,i],rew_messages[:,i]).to(device)
            fwd.append(tmp)
            # fwd_messages[:, 1+i, pi] = fwd_messages[:,i, pi]*\
            #                              obs_messages[:, i]*\
            #                              rew_messages[:, i]
            # fwd_messages[:, 1+i, pi] = self.generative_model_states[:,:,u].\
            #                              matmul(fwd_messages[:, 1+i, pi])
            norm = fwd[-1].sum(axis=0)
            mask = norm > 0
            fwd[-1][:,mask] /= norm[None,mask]
            zeros = ar.zeros((self.npi, self.npart))
            fwd_norm.append(ar.where(possible_policies, norm, zeros))#[:,None]
            # fwd_norm.append(ar.zeros((self.npi,self.npart)).to(device))
            # fwd_norm[-1][possible_policies] = norm[possible_policies]
            # if fwd_norms[1+i, pi] > 0: #???? Shouldn't this not happen?
            #     fwd_messages[:,1+i, pi] /= fwd_messages[:,1+i,pi].sum()

            # else:
            #     fwd_messages[:,:,pi] = 0#1./self.nh

        fwd_messages = ar.stack(fwd).permute(1,0,2,3).to(device)

        # for pi, cs in enumerate(self.policies):
        #     if self.prior_policies[-1][pi] > 1e-15 and pi in possible_policies:

        #         for i, u in enumerate(ar.flip(cs[:], [0])):
        #             bwd_messages[:,-2-i,pi] = bwd_messages[:,-1-i,pi]*\
        #                                         obs_messages[:,t-i]*\
        #                                         rew_messages[:, t-i]
        #             bwd_messages[:,-2-i,pi] = bwd_messages[:,-2-i,pi]\
        #                  .matmul(self.generative_model_states[:,:,u])

        #             norm = bwd_messages[:,-2-i,pi].sum()
        #             if norm > 0:
        #                 bwd_messages[:,-2-i, pi] /= norm

        #         for i, u in enumerate(cs[:]):
        #             fwd_messages[:, 1+i, pi] = fwd_messages[:,i, pi]*\
        #                                          obs_messages[:, i]*\
        #                                          rew_messages[:, i]
        #             fwd_messages[:, 1+i, pi] = self.generative_model_states[:,:,u].\
        #                                          matmul(fwd_messages[:, 1+i, pi])
        #             fwd_norms[1+i,pi] = fwd_messages[:,1+i,pi].sum()
        #             if fwd_norms[1+i, pi] > 0: #???? Shouldn't this not happen?
        #                 fwd_messages[:,1+i, pi] /= fwd_messages[:,1+i,pi].sum()

        #     else:
        #         fwd_messages[:,:,pi] = 0#1./self.nh

        posterior = fwd_messages*bwd_messages*obs_messages[:,:,None,:]*rew_messages[:,:,None,:]
        norm = posterior.sum(axis = 0)
        #fwd_norms[-1] = norm[-1]
        fwd_norm.append(norm[-1])
        fwd_norms = ar.stack(fwd_norm).to(device)
        # print(tau,t,self.fwd_norms[tau,t])
        non_zero = norm > 0
        posterior[:,non_zero] /= norm[non_zero]

        self.bwd_messages.append(bwd_messages)
        self.fwd_messages.append(fwd_messages)
        self.fwd_norms.append(fwd_norms)
        self.posterior_states.append(posterior)

        return posterior

    def update_beliefs_states(self, tau, t, observation, reward, possible_policies):
        #estimate expected state distribution
        # if t == 0:
        #     self.instantiate_messages(policies)
        self.observations.append(observation.to(device))
        self.rewards.append(reward.to(device))

        self.update_messages(tau, t, possible_policies)

        #return posterior#ar.nan_to_num(posterior)

    def update_beliefs_policies(self, tau, t):

        #print((prior_policies>1e-4).sum())

        likelihood = (self.fwd_norms[-1]+1e-10).prod(axis=0).to(device)
        norm = likelihood.sum(axis=0)
        # print("like", likelihood)
        # Fe = ar.log((self.fwd_norms[-1]+1e-10).prod(axis=0))
        # softplus = ar.nn.Softplus(beta=self.dec_temp)
        # likelihood = softplus(Fe)
        # posterior_policies = likelihood * self.prior_policies[-1] / (likelihood * self.prior_policies[-1]).sum(axis=0)
        likelihood = ar.pow(likelihood/norm,self.dec_temp[None,:]).to(device) #* ar.pow(norm,self.dec_temp)
        # print("softmax", likelihood)
        # print("norm1", ar.pow(norm,self.dec_temp))
        posterior_policies = likelihood * self.prior_policies[-1] / (likelihood * self.prior_policies[-1]).sum(axis=0)
        # print("unnorm", likelihood * self.prior_policies[-1])
        # print("norm", (likelihood * self.prior_policies[-1]).sum(axis=0))
        # print("post", posterior_policies)
        #likelihood /= likelihood.sum(axis=0)[None,:]
        #posterior/= posterior.sum(axis=0)[None,:]
        #posterior = ar.nan_to_num(posterior)
        #posterior = softmax(ln(self.fwd_norms).sum(axis = 0)+ln(self.prior_policies))

        #ar.testing.assert_allclose(post, posterior)

        self.posterior_policies.append(posterior_policies)

        if t<self.T-1:
            posterior_actions = ar.zeros((self.na,self.npart)).to(device)
            for a in range(self.na):
                posterior_actions[a] = posterior_policies[self.policies[:,t] == a].sum(axis=0)

            self.posterior_actions.append(posterior_actions)

        #return posterior, likelihood


    def update_beliefs_dirichlet_pol_params(self, tau, t):
        assert(t == self.T-1)
        chosen_pol = ar.argmax(self.posterior_policies[-1], axis=0).to(device)
        #print(chosen_pol)
#        self.dirichlet_pol_params[chosen_pol,:] += posterior_context.sum(axis=0)/posterior_context.sum()
        dirichlet_pol_params = (1-self.pol_lambda) * self.dirichlet_pol_params[-1] + (1 - (1-self.pol_lambda))*self.dirichlet_pol_params_init
        dirichlet_pol_params[(chosen_pol,list(range(self.npart)))] += 1#posterior_context

        prior_policies = dirichlet_pol_params / dirichlet_pol_params.sum(axis=0)[None,...]#ar.exp(scs.digamma(self.dirichlet_pol_params) - scs.digamma(self.dirichlet_pol_params.sum(axis=0))[None,:])
        #prior_policies /= prior_policies.sum(axis=0)[None,:]

        self.dirichlet_pol_params.append(dirichlet_pol_params.to(device))
        self.prior_policies.append(prior_policies.to(device))

        #return dirichlet_pol_params, prior_policies

    def update_beliefs_dirichlet_rew_params(self, tau, t, reward):
        posterior_states = self.posterior_states[-1]
        posterior_policies = self.posterior_policies[-1]
        states = (posterior_states[:,t,:,:] * posterior_policies[None,:,:]).sum(axis=1)
        # c = ar.argmax(posterior_context)
        # self.dirichlet_rew_params[reward,:,c] += states[:,c]

#         self.dirichlet_rew_params[tau,t,:,self.non_decaying:,:] = (1-self.r_lambda) * self.dirichlet_rew_params[tau,t,:,self.non_decaying:,:] +1 - (1-self.r_lambda)
#         self.dirichlet_rew_params[tau,t,reward,:,:] += states * posterior_context[None,:]
#         for c in range(self.nc):
#             for state in range(self.nh):
#                 #self.generative_model_rewards[:,state,c] = self.dirichlet_rew_params[:,state,c] / self.dirichlet_rew_params[:,state,c].sum()
#                 self.generative_model_rewards[tau,t,:,state,c] = self.dirichlet_rew_params[tau,t,:,state,c]#\
#                 # ar.exp(scs.digamma(self.dirichlet_rew_params[:,state,c])\
#                 #         -scs.digamma(self.dirichlet_rew_params[:,state,c].sum()))
#                 self.generative_model_rewards[tau,t,:,state,c] /= self.generative_model_rewards[tau,t,:,state,c].sum()
#             self.rew_messages[tau,t+1:,:,t+1:,c] = self.prior_rewards.matmul(self.generative_model_rewards[tau,t,:,:,c])[None,:,None]

        dirichlet_rew_params = self.dirichlet_rew_params[0].clone().to(device)#.detach()
        # dirichlet_rew_params = ar.ones_like(self.dirichlet_rew_params_init)#self.dirichlet_rew_params_init.clone()
        # dirichlet_rew_params[:,:self.non_decaying] = self.dirichlet_rew_params[-1][:,:self.non_decaying]
        dirichlet_rew_params[:,self.non_decaying:,:] = (1-self.r_lambda)[None,None,:] * self.dirichlet_rew_params[-1][:,self.non_decaying:,:] +1 - (1-self.r_lambda)[None,None,:]
        dirichlet_rew_params[reward,:,:] += states #* posterior_context[None,:]

        generative_model_rewards = dirichlet_rew_params / dirichlet_rew_params.sum(axis=0)[None,...]
        self.dirichlet_rew_params.append(dirichlet_rew_params.to(device))
        self.generative_model_rewards.append(generative_model_rewards.to(device))

        #return dirichlet_rew_params


class HierarchicalPerception(object):
    def __init__(self,
                 generative_model_observations,
                 generative_model_states,
                 generative_model_rewards,
                 transition_matrix_context,
                 prior_states,
                 prior_rewards,
                 prior_policies,
                 policies,
                 dirichlet_pol_params = None,
                 dirichlet_rew_params = None,
                 generative_model_context = None,
                 T=5, trials=10, pol_lambda=0, r_lambda=0, non_decaying=0,
                 dec_temp=1.):

        self.generative_model_observations = generative_model_observations
        if len(generative_model_states.shape)<4:
            self.generative_model_states = generative_model_states[:,:,:,None]
        else:
            self.generative_model_states = generative_model_states
        # self.generative_model_rewards = generative_model_rewards
        self.transition_matrix_context = transition_matrix_context
        self.prior_rewards = prior_rewards
        self.prior_states = prior_states
        dims = list(prior_policies.shape)
        dims = [trials] + dims
        self.prior_policies = ar.zeros(dims)
        self.prior_policies[:] = prior_policies[None,...]
        self.npi = prior_policies.shape[0]
        self.T = T
        self.trials = trials
        self.nh = prior_states.shape[0]
        self.pol_lambda = pol_lambda
        self.r_lambda = r_lambda
        self.non_decaying = non_decaying
        self.dec_temp = dec_temp
        self.policies = policies

        if len(generative_model_rewards.shape) > 2:
            dims = generative_model_rewards.shape
            self.generative_model_rewards =  ar.zeros((self.trials, self.T, dims[0], dims[1], dims[2]))
            self.infer_context = True
            self.nc = generative_model_rewards.shape[2]
            self.generative_model_rewards[:] = generative_model_rewards[None,None,:,:,:]
        else:
            self.nc = 1
            dims = generative_model_rewards.shape
            self.generative_model_rewards =  ar.zeros((self.trials, self.T, dims[0], dims[1], 1))
            self.generative_model_rewards[:] = self.generative_model_rewards[None,None,:,:,None]
        if dirichlet_pol_params is not None:
            dims = list(dirichlet_pol_params.shape)
            dims = [self.trials] + dims
            self.dirichlet_pol_params = ar.zeros(dims)
            self.dirichlet_pol_params[:] = dirichlet_pol_params[None,None,...]
        if dirichlet_rew_params is not None:
            dims = list(dirichlet_rew_params.shape)
            dims = [self.trials, self.T] + dims
            self.dirichlet_rew_params = ar.zeros(dims)
            self.dirichlet_rew_params[:] = dirichlet_rew_params[None,None,...]
        if generative_model_context is not None:
            self.generative_model_context = generative_model_context


            for c in range(self.nc):
                for state in range(self.nh):
                    self.generative_model_rewards[:,:,:,state,c] = (self.dirichlet_rew_params[:,state,c] / self.dirichlet_rew_params[:,state,c].sum())[None,None,...]
                    # self.generative_model_rewards[:,state,c] =\
                    # ar.exp(scs.digamma(self.dirichlet_rew_params[:,state,c])\
                    #        -scs.digamma(self.dirichlet_rew_params[:,state,c].sum()))
                    # self.generative_model_rewards[:,state,c] /= self.generative_model_rewards[:,state,c].sum()
        self.instantiate_messages()


    def reset(self):

        self.generative_model_rewards[:] = self.generative_model_rewards[0,0].clone().detach()[None,None,...]
        self.dirichlet_rew_params[:] = self.dirichlet_rew_params[0,0].clone().detach()[None,None,...]
        self.dirichlet_pol_params = ar.ones_like(self.dirichlet_pol_params)
        self.prior_policies = ar.ones_like(self.prior_policies) / self.npi

        for c in range(self.nc):
            for state in range(self.nh):
                self.generative_model_rewards[:,:,:,state,c] = (self.dirichlet_rew_params[0,0,:,state,c] / self.dirichlet_rew_params[0,0,:,state,c].sum())[None,None,...]
                # self.generative_model_rewards[:,state,c] =\
                # ar.exp(scs.digamma(self.dirichlet_rew_params[:,state,c])\
                #        -scs.digamma(self.dirichlet_rew_params[:,state,c].sum()))
                # self.generative_model_rewards[:,state,c] /= self.generative_model_rewards[:,state,c].sum()

        self.instantiate_messages()


    def instantiate_messages(self):

        self.bwd_messages = ar.zeros((self.trials, self.T, self.nh, self.T, self.npi, self.nc))
        self.bwd_messages[:,:,:,-1,:, :] = 1./self.nh
        self.fwd_messages = ar.zeros((self.trials, self.T, self.nh, self.T, self.npi, self.nc))
        self.fwd_messages[:,:,:, 0, :, :] = self.prior_states[None,None,:, None, None]
        self.fwd_norms = ar.zeros((self.trials, self.T, self.T+1, self.npi, self.nc))
        self.fwd_norms[:,:,0,:,:] = 1.

        self.obs_messages = ar.zeros((self.trials, self.T, self.nh, self.T, self.nc)) + 1/self.nh#self.prior_observations.dot(self.generative_model_observations)
        #self.obs_messages = ar.tile(self.obs_messages,(self.T,1)).T

        self.rew_messages = ar.zeros((self.trials, self.T, self.nh, self.T, self.nc))
        #self.rew_messages[:] = ar.tile(self.prior_rewards.dot(self.generative_model_rewards),(self.T,1)).T

        for c in range(self.nc):
            self.rew_messages[:,:,:,:,c] = self.prior_rewards.matmul(self.generative_model_rewards[0,0,:,:,c])[None,None,:,None]
            for pi, cstates in enumerate(self.policies):
                for t, u in enumerate(ar.flip(cstates, [0])):
                    tp = self.T - 2 - t
                    self.bwd_messages[0,0,:,tp,pi,c] = (self.bwd_messages[0,0,:,tp+1,pi,c]*\
                                                self.obs_messages[0,0,:, tp+1,c]*\
                                                self.rew_messages[0,0,:, tp+1,c])[None,None,...]
                    bwd_message = (self.bwd_messages[0,0,:,tp,pi,c]\
                        .matmul(self.generative_model_states[:,:,u,c]))
                    bwd_message /= bwd_message.sum()
                    self.bwd_messages[:,:,:,tp, pi,c] = bwd_message[None,None,...]

    def update_messages(self, tau, t, pi, cs, c=0):
        #print(tau,t)
        if t > 0:
            for i, u in enumerate(ar.flip(cs[:], [0])):
                self.bwd_messages[tau,t,:,self.T-2-i,pi,c] = self.bwd_messages[tau,t,:,self.T-1-i,pi,c]*\
                                                self.obs_messages[tau,t,:,self.T-1-i,c]*\
                                                self.rew_messages[tau,t,:, self.T-1-i,c]
                self.bwd_messages[tau,t,:,self.T-2-i,pi,c] = self.bwd_messages[tau,t,:,self.T-2-i,pi,c]\
                    .matmul(self.generative_model_states[:,:,u,c])

                norm = self.bwd_messages[tau,t,:,self.T-2-i,pi,c].sum()
                if norm > 0:
                    self.bwd_messages[tau,t,:,self.T-2-i, pi,c] /= norm

        if len(cs[:]) > 0:
           for i, u in enumerate(cs[:]):
               self.fwd_messages[tau,t,:, 1+i, pi,c] = (self.fwd_messages[tau,t,:,i, pi,c]*\
                                                self.obs_messages[tau,t,:, i,c]*\
                                                self.rew_messages[tau,t,:, i,c])

               self.fwd_messages[tau,t,:, 1+i, pi,c] = \
                                                (self.generative_model_states[:,:,u,c].\
                                                matmul(self.fwd_messages[tau,t,:, 1+i, pi,c]))
               self.fwd_norms[tau,t,1+i,pi,c] = self.fwd_messages[tau,t,:,1+i,pi,c].sum()
               if self.fwd_norms[tau,t,1+i, pi,c] > 0: #???? Shouldn't this not happen?
                   self.fwd_messages[tau,t,:,1+i, pi,c] /= self.fwd_norms[tau,t,1+i,pi,c]

    def reset_preferences(self, t, new_preference, policies):

        self.prior_rewards = new_preference

        for c in range(self.nc):
            self.rew_messages[:,:,c] = self.prior_rewards.matmul(self.generative_model_rewards[:,:,c])[:,None]
            for pi, cstates in enumerate(policies[t:]):
                for i, u in enumerate(ar.flip(cstates, axis = 0)):
                    tp = self.T - 2 - i
                    self.bwd_messages[:,tp,pi,c] = self.bwd_messages[:,tp+1,pi,c]*\
                                                self.obs_messages[:, tp+1,c]*\
                                                self.rew_messages[:, tp+1,c]
                    self.bwd_messages[:,tp,pi,c] = self.bwd_messages[:,tp,pi,c]\
                        .matmul(self.generative_model_states[:,:,u,c])
                    self.bwd_messages[:,tp, pi,c] /= self.bwd_messages[:,tp,pi,c].sum()

    def update_beliefs_states(self, tau, t, observation, reward, possible_policies):
        #estimate expected state distribution
        # if t == 0:
        #     self.instantiate_messages(policies)

        if t>0:
            self.obs_messages[tau,t,:,:t,:] = self.obs_messages[tau,t-1,:,:t,:].clone()
            self.rew_messages[tau,t,:,:t,:] = self.rew_messages[tau,t-1,:,:t,:].clone()

        self.obs_messages[tau,t,:,t,:] = self.generative_model_observations[observation][:,None]
        self.rew_messages[tau,t,:,t,:] = self.generative_model_rewards[tau,t,reward]

        for c in range(self.nc):
            for pi, cs in enumerate(self.policies):
                if self.prior_policies[tau,pi,c] > 1e-15 and pi in possible_policies:
                    self.update_messages(tau, t, pi, cs, c)
                else:
                    self.fwd_messages[tau,t,:,:,pi,c] = 0#1./self.nh

        # print(tau,t,self.fwd_messages[tau,t])
        # print(tau,t,self.bwd_messages[tau,t])
        # print(tau,t,self.obs_messages[tau,t])
        # print(tau,t,self.rew_messages[tau,t])
        #estimate posterior state distribution
        posterior = self.fwd_messages[tau,t]*self.bwd_messages[tau,t]*self.obs_messages[tau,t,:,:,None,:]*self.rew_messages[tau,t,:,:,None,:]
        norm = posterior.sum(axis = 0)
        self.fwd_norms[tau,t,-1] = norm[-1]
        # print(tau,t,self.fwd_norms[tau,t])
        non_zero = norm > 0
        posterior[:,non_zero] /= norm[non_zero]
        return posterior#ar.nan_to_num(posterior)

    def update_beliefs_policies(self, tau, t):

        #print((prior_policies>1e-4).sum())
        likelihood = self.fwd_norms[tau,t].prod(axis=0)
        posterior = ar.pow(likelihood,self.dec_temp) * self.prior_policies[tau] / (ar.pow(likelihood,self.dec_temp) * self.prior_policies[tau]).sum(axis=0)[None,:]
        #likelihood /= likelihood.sum(axis=0)[None,:]
        #posterior/= posterior.sum(axis=0)[None,:]
        #posterior = ar.nan_to_num(posterior)

        #posterior = softmax(ln(self.fwd_norms).sum(axis = 0)+ln(self.prior_policies))

        #ar.testing.assert_allclose(post, posterior)

        return posterior, likelihood


    def update_beliefs_context(self, tau, t, reward, posterior_states, posterior_policies, prior_context, policies, context=None):

        post_policies = (prior_context[None,:] * posterior_policies).sum(axis=1)
        beta = self.dirichlet_rew_params
        states = (posterior_states[:,t,:] * post_policies[None,:,None]).sum(axis=1)
        beta_prime = self.dirichlet_rew_params.deepcopy()
        beta_prime[reward] = beta[reward] + states

#        for c in range(self.nc):
#            for state in range(self.nh):
#                self.generative_model_rewards[:,state,c] =\
#                ar.exp(scs.digamma(beta_prime[:,state,c])\
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
            chosen_pol = ar.argmax(post_policies)
            inf_context = ar.argmax(prior_context)
            alpha_prime = self.dirichlet_pol_params.deepcopy()
            alpha_prime[chosen_pol,:] += prior_context
            #alpha_prime[chosen_pol,inf_context] = self.dirichlet_pol_params[chosen_pol,inf_context] + 1
        else:
            alpha_prime = alpha


        if self.nc == 1:
            posterior = ar.ones(1)
        else:
            # todo: recalc
            #outcome_surprise = ((states * prior_context[None,:]).sum(axis=1)[:,None] * (scs.digamma(beta_prime[reward]) - scs.digamma(beta_prime.sum(axis=0)))).sum(axis=0)
            if t>0:
                outcome_surprise = (posterior_policies * ln(self.fwd_norms.prod(axis=0))).sum(axis=0)
                entropy = - (posterior_policies * ln(posterior_policies)).sum(axis=0)
                #policy_surprise = (post_policies[:,None] * scs.digamma(alpha_prime)).sum(axis=0) - scs.digamma(alpha_prime.sum(axis=0))
                policy_surprise = (posterior_policies * scs.digamma(alpha_prime)).sum(axis=0) - scs.digamma(alpha_prime.sum(axis=0))
            else:
                outcome_surprise = 0
                entropy = 0
                policy_surprise = 0

            if context is not None:
                context_obs_suprise = ln(self.generative_model_context[context]+1e-10)
            else:
                context_obs_suprise = 0
            posterior = outcome_surprise + policy_surprise + entropy + context_obs_suprise

                        #+ ar.nan_to_num((posterior_policies * ln(self.fwd_norms).sum(axis = 0))).sum(axis=0)#\

#            if tau in range(90,120) and t == 1:
#                #print(tau, ar.exp(outcome_surprise), ar.exp(policy_surprise))
#                print(tau, ar.exp(outcome_surprise[1])/ar.exp(outcome_surprise[0]), ar.exp(policy_surprise[1])/ar.exp(policy_surprise[0]))

            posterior = ar.nan_to_num(softmax(posterior+ln(prior_context)))

        return posterior


    def update_beliefs_dirichlet_pol_params(self, tau, t, posterior_policies, posterior_context = [1]):
        assert(t == self.T-1)
        chosen_pol = ar.argmax(posterior_policies, axis=0)
#        self.dirichlet_pol_params[chosen_pol,:] += posterior_context.sum(axis=0)/posterior_context.sum()
        if tau < self.trials-1:
            self.dirichlet_pol_params[tau+1] = (1-self.pol_lambda) * self.dirichlet_pol_params[tau] + 1 - (1-self.pol_lambda)
            self.dirichlet_pol_params[tau+1,chosen_pol,:] += posterior_context
            self.prior_policies[tau+1] = self.dirichlet_pol_params[tau+1].clone()#ar.exp(scs.digamma(self.dirichlet_pol_params) - scs.digamma(self.dirichlet_pol_params.sum(axis=0))[None,:])
            self.prior_policies[tau+1] /= self.prior_policies[tau+1].sum(axis=0)[None,:]
            # if tau < self.trials-1:
            #     self.dirichlet_pol_params[tau+1] = self.dirichlet_pol_params[tau]

            return self.dirichlet_pol_params[tau+1], self.prior_policies[tau+1]

        else:
            return self.dirichlet_pol_params[tau], self.prior_policies[tau]

    def update_beliefs_dirichlet_rew_params(self, tau, t, reward, posterior_states, posterior_policies, posterior_context = [1]):
        states = (posterior_states[:,t,:,:] * posterior_policies[None,:,:]).sum(axis=1)
        # c = ar.argmax(posterior_context)
        # self.dirichlet_rew_params[reward,:,c] += states[:,c]

#         self.dirichlet_rew_params[tau,t,:,self.non_decaying:,:] = (1-self.r_lambda) * self.dirichlet_rew_params[tau,t,:,self.non_decaying:,:] +1 - (1-self.r_lambda)
#         self.dirichlet_rew_params[tau,t,reward,:,:] += states * posterior_context[None,:]
#         for c in range(self.nc):
#             for state in range(self.nh):
#                 #self.generative_model_rewards[:,state,c] = self.dirichlet_rew_params[:,state,c] / self.dirichlet_rew_params[:,state,c].sum()
#                 self.generative_model_rewards[tau,t,:,state,c] = self.dirichlet_rew_params[tau,t,:,state,c]#\
#                 # ar.exp(scs.digamma(self.dirichlet_rew_params[:,state,c])\
#                 #         -scs.digamma(self.dirichlet_rew_params[:,state,c].sum()))
#                 self.generative_model_rewards[tau,t,:,state,c] /= self.generative_model_rewards[tau,t,:,state,c].sum()
#             self.rew_messages[tau,t+1:,:,t+1:,c] = self.prior_rewards.matmul(self.generative_model_rewards[tau,t,:,:,c])[None,:,None]

        dirichlet_rew_params = self.dirichlet_rew_params[tau,t,:,:,:].clone()
        dirichlet_rew_params[:,self.non_decaying:,:] = (1-self.r_lambda) * dirichlet_rew_params[:,self.non_decaying:,:] +1 - (1-self.r_lambda)
        dirichlet_rew_params[reward,:,:] += states * posterior_context[None,:]
        generative_model_rewards = dirichlet_rew_params.clone()
        rew_messages = ar.zeros_like(self.rew_messages[tau,t])
        for c in range(self.nc):
            for state in range(self.nh):
                #self.generative_model_rewards[:,state,c] = self.dirichlet_rew_params[:,state,c] / self.dirichlet_rew_params[:,state,c].sum()
                #generative_model_rewards[tau,t,:,state,c] = dirichlet_rew_params[:,state,c]#\
                # ar.exp(scs.digamma(self.dirichlet_rew_params[:,state,c])\
                #         -scs.digamma(self.dirichlet_rew_params[:,state,c].sum()))
                generative_model_rewards[:,state,c] /= generative_model_rewards[:,state,c].sum()
            rew_messages[:,:,c] = self.prior_rewards.matmul(generative_model_rewards[:,:,c])[:,None]
        #rew_messages = ar.einsum('r,rsc->sc',self.prior_rewards,generative_model_rewards)
            # rew_messages = self.prior_rewards.matmul(generative_model_rewards[:,:,c])
            # self.rew_messages[tau,t+1:,:,t+1:,c] = self.prior_rewards.matmul(self.generative_model_rewards[tau,t,:,:,c])[None,:,None]

        # print(dirichlet_rew_params)
        # print(generative_model_rewards)
        # print(rew_messages)

        #print(self.T)
        if t<self.T-1:
            self.dirichlet_rew_params[tau,t+1] = dirichlet_rew_params
            self.generative_model_rewards[tau,t+1] = generative_model_rewards
            self.rew_messages[tau,t+1:,:,t+1:,:] = rew_messages[None,:,t+1:,:]
        elif tau<self.trials-1:
            self.dirichlet_rew_params[tau+1,0] = dirichlet_rew_params
            self.generative_model_rewards[tau+1,0] = generative_model_rewards
            self.rew_messages[tau+1,0:,:,0:,:] = rew_messages[None,:,0:,:]

        return dirichlet_rew_params





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
            self.generative_model_rewards = self.generative_model_rewards[:,:,None]
        if dirichlet_pol_params is not None:
            self.dirichlet_pol_params = dirichlet_pol_params.copy()
        if dirichlet_rew_params is not None:
            self.dirichlet_rew_params = dirichlet_rew_params.copy()


        for c in range(self.nc):
            for state in range(self.nh):
                self.generative_model_rewards[:,state,c] =\
                ar.exp(scs.digamma(self.dirichlet_rew_params[:,state,c])\
                       -scs.digamma(self.dirichlet_rew_params[:,state,c].sum()))
                self.generative_model_rewards[:,state,c] /= self.generative_model_rewards[:,state,c].sum()


    def instantiate_messages(self, policies):
        npi = policies.shape[0]

        self.bwd_messages = ar.zeros((self.nh, self.T, npi, self.nc))
        self.bwd_messages[:,-1,:, :] = 1./self.nh
        self.fwd_messages = ar.zeros((self.nh, self.T, npi, self.nc))
        self.fwd_messages[:, 0, :, :] = self.prior_states[:, None, None]

        self.fwd_norms = ar.zeros((self.T+1, npi, self.nc))
        self.fwd_norms[0,:,:] = 1.

        self.obs_messages = ar.zeros((self.nh, self.T, self.nc)) + 1/self.nh#self.prior_observations.dot(self.generative_model_observations)
        #self.obs_messages = ar.tile(self.obs_messages,(self.T,1)).T

        self.rew_messages = ar.zeros((self.nh, self.T, self.nc))
        #self.rew_messages[:] = ar.tile(self.prior_rewards.dot(self.generative_model_rewards),(self.T,1)).T

        for c in range(self.nc):
            self.rew_messages[:,:,c] = self.prior_rewards.dot(self.generative_model_rewards[:,:,c])[:,None]
            for pi, cstates in enumerate(policies):
                for t, u in enumerate(ar.flip(cstates, axis = 0)):
                    tp = self.T - 2 - t
                    self.bwd_messages[:,tp,pi,c] = self.bwd_messages[:,tp+1,pi,c]*\
                                                self.obs_messages[:, tp+1,c]*\
                                                self.rew_messages[:, tp+1,c]
                    self.bwd_messages[:,tp,pi,c] = self.bwd_messages[:,tp,pi,c]\
                        .dot(self.generative_model_states[:,:,u])
                    self.bwd_messages[:,tp, pi,c] /= self.bwd_messages[:,tp,pi,c].sum()

    def update_messages(self, t, pi, cs, c=0):
        if t > 0:
            for i, u in enumerate(ar.flip(cs[:t], axis = 0)):
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
            self.rew_messages[:,:,c] = self.prior_rewards.dot(self.generative_model_rewards[:,:,c])[:,None]
            for pi, cstates in enumerate(policies[t:]):
                for i, u in enumerate(ar.flip(cstates, axis = 0)):
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
        return ar.nan_to_num(posterior)

    def update_beliefs_policies(self, tau, t, gamma=4):

        #print((prior_policies>1e-4).sum())
        likelihood = self.fwd_norms.prod(axis=0)
        posterior = ar.power(likelihood,gamma) * self.prior_policies
        posterior/= posterior.sum(axis=0)[None,:]
        #posterior = softmax(ln(self.fwd_norms).sum(axis = 0)+ln(self.prior_policies))

        #ar.testing.assert_allclose(post, posterior)

        return posterior, likelihood


    def update_beliefs_context(self, tau, t, reward, posterior_states, posterior_policies, prior_context, policies):

        post_policies = (prior_context[None,:] * posterior_policies).sum(axis=1)
        beta = self.dirichlet_rew_params.copy()
        states = (posterior_states[:,t,:] * post_policies[None,:,None]).sum(axis=1)
        beta_prime = self.dirichlet_rew_params.copy()
        beta_prime[reward] = beta[reward] + states

#        for c in range(self.nc):
#            for state in range(self.nh):
#                self.generative_model_rewards[:,state,c] =\
#                ar.exp(scs.digamma(beta_prime[:,state,c])\
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
            chosen_pol = ar.argmax(post_policies)
            inf_context = ar.argmax(prior_context)
            alpha_prime = self.dirichlet_pol_params.copy()
            alpha_prime[chosen_pol,:] += prior_context
            #alpha_prime[chosen_pol,inf_context] = self.dirichlet_pol_params[chosen_pol,inf_context] + 1
        else:
            alpha_prime = alpha


        if self.nc == 1:
            posterior = ar.ones(1)
        else:
            # todo: recalc
            #outcome_surprise = ((states * prior_context[None,:]).sum(axis=1)[:,None] * (scs.digamma(beta_prime[reward]) - scs.digamma(beta_prime.sum(axis=0)))).sum(axis=0)
            outcome_surprise = (posterior_policies * ln(self.fwd_norms.prod(axis=0))).sum(axis=0)
            entropy = - (posterior_policies * ln(posterior_policies)).sum(axis=0)
            #policy_surprise = (post_policies[:,None] * scs.digamma(alpha_prime)).sum(axis=0) - scs.digamma(alpha_prime.sum(axis=0))
            policy_surprise = (posterior_policies * scs.digamma(alpha_prime)).sum(axis=0) - scs.digamma(alpha_prime.sum(axis=0))
            posterior = outcome_surprise + policy_surprise + entropy

                        #+ ar.nan_to_num((posterior_policies * ln(self.fwd_norms).sum(axis = 0))).sum(axis=0)#\

#            if tau in range(90,120) and t == 1:
#                #print(tau, ar.exp(outcome_surprise), ar.exp(policy_surprise))
#                print(tau, ar.exp(outcome_surprise[1])/ar.exp(outcome_surprise[0]), ar.exp(policy_surprise[1])/ar.exp(policy_surprise[0]))


            posterior = ar.nan_to_num(softmax(posterior+ln(prior_context)))

        return posterior


    def update_beliefs_dirichlet_pol_params(self, tau, t, posterior_policies, posterior_context = [1]):
        assert(t == self.T-1)
        chosen_pol = ar.argmax(posterior_policies, axis=0)
#        self.dirichlet_pol_params[chosen_pol,:] += posterior_context.sum(axis=0)/posterior_context.sum()
        alpha = 0.3#0.3
        self.dirichlet_pol_params = (1-alpha) * self.dirichlet_pol_params + 1 - (1-alpha)
        self.dirichlet_pol_params[chosen_pol,:] += posterior_context
        self.prior_policies[:] = ar.exp(scs.digamma(self.dirichlet_pol_params) - scs.digamma(self.dirichlet_pol_params.sum(axis=0))[None,:])
        self.prior_policies /= self.prior_policies.sum(axis=0)

        return self.dirichlet_pol_params

    def update_beliefs_dirichlet_rew_params(self, tau, t, reward, posterior_states, posterior_policies, posterior_context = [1]):
        states = (posterior_states[:,t,:,:] * posterior_policies[None,:,:]).sum(axis=1)
        state = ar.argmax(states)
        old = self.dirichlet_rew_params.copy()
#        self.dirichlet_rew_params[:,state,:] = (1-0.4) * self.dirichlet_rew_params[:,state,:] #+1 - (1-0.4)
#        self.dirichlet_rew_params[reward,state,:] += 1#states * posterior_context[None,:]
        alpha = 0.6#0.3#1#0.3#0.05
        self.dirichlet_rew_params[:,3:,:] = (1-alpha) * self.dirichlet_rew_params[:,3:,:] +1 - (1-alpha)
        self.dirichlet_rew_params[reward,:,:] += states * posterior_context[None,:]
        for c in range(self.nc):
            for state in range(self.nh):
                self.generative_model_rewards[:,state,c] =\
                ar.exp(scs.digamma(self.dirichlet_rew_params[:,state,c])\
                       -scs.digamma(self.dirichlet_rew_params[:,state,c].sum()))
                self.generative_model_rewards[:,state,c] /= self.generative_model_rewards[:,state,c].sum()

            self.rew_messages[:,t+1:,c] = self.prior_rewards.dot(self.generative_model_rewards[:,:,c])[:,None]

#        for c in range(self.nc):
#            for pi, cs in enumerate(policies):
#                if self.prior_policies[pi,c] > 1e-15:
#                    self.update_messages(t, pi, cs, c)
#                else:
#                    self.fwd_messages[:,:,pi,c] = 1./self.nh #0

        return self.dirichlet_rew_params


class mfmbPerception(object):
    def __init__(self,
                 generative_model_states,
                 policies,
                 Q_mf_init,
                 Q_mb_init,
                 utility,
                 lamb = 0.9,
                 alpha = 0.1,
                 beta_mf = 2.,
                 beta_mb = 2.,
                 p = 0.1,
                 T=3,
                 npart=1, nsubs=1):

        self.generative_model_states = generative_model_states
        self.alpha = alpha
        self.beta_mb = beta_mb
        self.beta_mf = beta_mf
        self.lamb = lamb
        self.p = p
        self.ns = generative_model_states.shape[0]
        self.utility = utility
        self.nr = utility.shape[0]
        self.na = len(ar.unique(policies))
        self.T = T
        self.prev_first_action = []
        self.action_probs = []

        self.Q_mf_init = Q_mf_init
        self.Q_mb_init = Q_mb_init
        self.Q_mf = [Q_mf_init] #sxa
        self.Q_mb = [Q_mb_init] #sxa

        self.observations = []
        self.rewards = []
        self.actions = []

        self.posterior_actions = [ar.zeros((self.na))+1./self.na]

    def set_parameters(self, **kwargs):

        if 'lamb' in kwargs.keys():
            self.lamb = kwargs['lamb']
        if 'alpha' in kwargs.keys():
            self.alpha = kwargs['alpha']
        if 'beta_mf' in kwargs.keys():
            self.beta_mf = kwargs['beta_mf']
        if 'beta_mb' in kwargs.keys():
            self.beta_mb = 1./kwargs['beta_mb']
        if 'p' in kwargs.keys():
            self.p = kwargs['p']

    def reset(self):

        if len(self.alpha.shape) > 1:
            self.npart = self.alpha.shape[0]
            self.nsubs = self.dec_temp.shape[1]
        else:
            self.nsubs = self.alpha.shape[0]
            self.npart = 1
            #self.alpha_0 = self.alpha_0[None,:]
            self.lamb = self.lamb[None,:]
            self.alpha = self.alpha[None,:]
            self.beta_mf = self.beta_mf[None,:]
            self.beta_mb = self.beta_mb[None,:]
            self.p = self.p[None,:]

        self.prev_first_action = []
        self.action_probs = []

        self.Q_mf = [[ar.stack([ar.stack([self.Q_mf_init[k]]*self.npart)]*self.nsubs).permute(2,3,1,0) for k in range(3)]] #sxa
        self.Q_mb = [[ar.stack([ar.stack([self.Q_mb_init[k]]*self.npart)]*self.nsubs).permute(2,3,1,0) for k in range(3)]] #sxa

        self.posterior_actions = [ar.zeros(self.na,self.npart,self.nsubs)+1./self.na]

        self.observations = []
        self.rewards = []
        self.actions = []


    def update_mf(self, tau, t, chosen_action):

        # paper has -1, 1 for reward and no reward?!
        # these eqs are according to Otte et al. 2013
        assert(t==self.T-1)
        print("update mf", tau, t)

        Q_mf = self.Q_mf[-1]

        new_Q_mf = [ar.zeros((self.ns, self.na, self.npart, self.nsubs))]

        #second stage
        post_states2 = ar.eye(self.ns)[:,self.observations[-2]]
        post_rewards2 = ar.eye(self.nr)[:,self.rewards[-1]]
        post_past_action2 = ar.eye(self.na)[:,self.actions[-1]]
        curr_state_action_pair2 = post_states2[:,None,None,...]*post_past_action2[None,:,None,...]
        full_rewarded2 = (curr_state_action_pair2*(post_rewards2*self.utility[:,None])[None,None,:,...]).sum(dim=2)

        pred_err2 = (full_rewarded2[:,:,None,:] + curr_state_action_pair2*Q_mf[2])/self.alpha - curr_state_action_pair2*Q_mf[1]#/self.alpha
        update2 = Q_mf[1] + self.alpha*pred_err2
        # also has forgetting factor?! (1-self.alpha)
        new_Q_mf1 = ar.where(curr_state_action_pair2>0, update2, (1-self.alpha)*Q_mf[1])#(1-self.alpha)*
        new_Q_mf.append(new_Q_mf1)

        #first stage
        post_states1 = ar.eye(self.ns)[:,self.observations[-3]]
        post_rewards1 = ar.eye(self.nr)[:,self.rewards[-2]]
        post_past_action1 = ar.eye(self.na)[:,self.actions[-2]]
        curr_state_action_pair1 = post_states1[:,None,None,...]*post_past_action1[None,:,None,...]
        full_rewarded1 = (curr_state_action_pair1*(post_rewards1*self.utility[:,None])[None,None,:,...]).sum(dim=2)

        pred_err1 = (full_rewarded1[:,:,None,:].sum() + (curr_state_action_pair2*new_Q_mf1).sum())/self.alpha - (curr_state_action_pair1*Q_mf[0]).sum()#/self.alpha
        print(pred_err1.shape)
        print("pred err1", pred_err1)
        print("pred err2 sum", pred_err2.sum())
        update = Q_mf[0] + self.alpha*(pred_err1.sum() + self.lamb*pred_err2.sum())
        print(self.lamb)
        print(self.lamb*pred_err2.sum())
        # also has forgetting factor?! (1-self.alpha)
        new_Q_mf.append(ar.where(curr_state_action_pair1>0, update, (1-self.alpha)*Q_mf[0]))#(1-self.alpha)*

        new_Q_mf.reverse()

        print(new_Q_mf)

        self.Q_mf.append(new_Q_mf)


    def update_mb(self, tau, t, chosen_action):

        Q_mb = self.Q_mb[-1]
        Q_mf = self.Q_mf[-1]

        # print("mb update")
        # print(Q_mf[1].shape)

        next_best_Q = ar.amax(Q_mf[1], dim=1)

        # print(Q_mf[1][:,:,0,0])
        # print(next_best_Q[:,:,0])
        # print(self.generative_model_states[:,:,next_best_action,None,None].shape)
        # print(Q_mf[1][None,:,next_best_action,:,:].shape)

        post_states1 = ar.eye(self.ns)[:,self.observations[-3]]
        post_past_action1 = ar.eye(self.na)[:,self.actions[-2]]
        curr_state_action_pair1 = post_states1[:,None,None,...]*post_past_action1[None,:,None,...]

        update_mb1 = (self.generative_model_states[:,:,:,None,None] * next_best_Q[:,None,None,...]).sum(dim=0)
        print("update mb")
        print(update_mb1)
        print((self.generative_model_states * next_best_Q[:,None,None,0,0]).sum(dim=0))

        new_Q_mb1 = ar.where(curr_state_action_pair1>0, update_mb1, (1-self.alpha)*Q_mb[0])
        # print(new_Q_mb1)

        new_Q_mb = [new_Q_mb1, Q_mf[1], ar.zeros((self.ns, self.na, self.npart, self.nsubs))]

        self.Q_mb.append(new_Q_mb)

    def calc_action_probs(self, tau, t):

        Q_mb = ar.stack([self.Q_mb[-1][t][self.observations[-1][i],:,:,i] for i in range(self.nsubs)], dim=-1)
        Q_mf = ar.stack([self.Q_mf[-1][t][self.observations[-1][i],:,:,i] for i in range(self.nsubs)], dim=-1)

        if tau==0:
            rep = ar.eye(self.na)[:,self.prev_first_action[-1]][:,None,:]
        else:
            rep = 0

        exponent = self.beta_mb*Q_mb + self.beta_mf*Q_mf + self.p*rep

        action_probs = ar.softmax(exponent, dim=0)
        print("calc probs")
        print(tau, t, Q_mb[:,0,0])
        print(Q_mf[:,0,0])
        print(exponent[:,0,0])
        print(action_probs)

        self.posterior_actions.append(action_probs)

    def update_beliefs(self, tau, t, observation, reward, chosen_action, possible_policies):

        self.observations.append(observation)
        self.rewards.append(reward)
        self.actions.append(chosen_action)

        if t==1:
            self.prev_first_action.append(chosen_action)

        if t==self.T-1:
            self.update_mf(tau, t, chosen_action)
            self.update_mb(tau, t, chosen_action)
        elif tau>0 and t<self.T-1:
            self.calc_action_probs(tau, t)




class mfmb2Perception(object):
    def __init__(self,
                 generative_model_states,
                 policies,
                 Q_mf_init,
                 Q_mb_init,
                 utility,
                 lamb = ar.tensor([0.9]),
                 alpha = ar.tensor([0.1]),
                 beta_mf = ar.tensor([4.]),
                 beta_mb = ar.tensor([4.]),
                 p = ar.tensor([2.]),
                 mask = None,
                 trials=10,
                 T=3,
                 npart=1, nsubs=1,
                 use_p=True,
                 restrict_alpha=False,
                 max_dt=10, min_alpha=0):

        self.generative_model_states = generative_model_states[:3,:3,...]
        self.alpha = alpha
        self.beta_mb = beta_mb
        self.beta_mf = beta_mf
        self.lamb = lamb
        self.p = p
        self.ns = self.generative_model_states.shape[0]
        self.utility = utility
        self.nr = utility.shape[0]
        self.na = len(ar.unique(policies))
        self.T = T
        self.prev_first_action = []
        self.action_probs = []
        self.nsubs = nsubs

        if mask is None:
            self.mask = ar.ones(trials, nsubs).bool()
        else:
            self.mask = mask

        self.use_p = use_p
        self.restrict_alpha = restrict_alpha
        if self.restrict_alpha:
            self.min_alpha = min_alpha
        self.max_dt = max_dt
        if self.use_p:
            self.npars = 5
        else:
            self.npars = 4
        self.param_names = list(self.locs_to_pars(ar.zeros(self.npars)).keys())

        self.Q_mf_init = Q_mf_init
        self.Q_mb_init = Q_mb_init
        self.Q_mf = [Q_mf_init] #sxa
        self.Q_mb = [Q_mb_init] #sxa

        self.observations = []
        self.rewards = []
        self.actions = []

        self.posterior_actions = [ar.zeros((self.na))+1./self.na]

    def locs_to_pars(self, locs):
        
        if self.restrict_alpha:
            alpha = self.min_alpha + ar.sigmoid(locs[...,1])*(1.-self.min_alpha)
        else:
            alpha = ar.sigmoid(locs[...,1])

        if self.use_p:
            par_dict = {"discount": ar.sigmoid(locs[...,0]),
                        "learning rate": alpha,
                        "mf weight": self.max_dt*ar.sigmoid(locs[...,2]),
                        "mb weight": self.max_dt*ar.sigmoid(locs[...,3]),
                        "repetition": self.max_dt*ar.sigmoid(locs[...,4])}
        else:
            par_dict = {"discount": ar.sigmoid(locs[...,0]),
                        "learning rate": alpha,
                        "mf weight": self.max_dt*ar.sigmoid(locs[...,2]),
                        "mb weight": self.max_dt*ar.sigmoid(locs[...,3])}

        return par_dict

    def set_parameters(self, locs):

        par_dict = self.locs_to_pars(locs)

        if 'discount' in par_dict:
            self.lamb = par_dict['discount']
        if 'learning rate' in par_dict:
            self.alpha = par_dict['learning rate']
        if 'mf weight' in par_dict:
            self.beta_mf = par_dict['mf weight']
        if 'mb weight' in par_dict:
            self.beta_mb = par_dict['mb weight']
        if 'repetition' in par_dict:
            self.p = par_dict['repetition']
        else:
            self.p = ar.zeros_like(self.lamb)

    def reset(self):

        if len(self.alpha.shape) > 1:
            self.npart = self.alpha.shape[0]
            self.nsubs = self.alpha.shape[1]
        else:
            self.nsubs = self.alpha.shape[0]
            self.npart = 1
            #self.alpha_0 = self.alpha_0[None,:]
            self.lamb = self.lamb[None,:]
            self.alpha = self.alpha[None,:]
            self.beta_mf = self.beta_mf[None,:]
            self.beta_mb = self.beta_mb[None,:]
            self.p = self.p[None,:]

        self.prev_first_action = []
        self.action_probs = []

        self.Q_mf = [[ar.stack([ar.stack([self.Q_mf_init[k]]*self.npart)]*self.nsubs).permute(2,3,1,0) for k in range(2)]] #sxa
        self.Q_mb = [[ar.stack([ar.stack([self.Q_mb_init[k]]*self.npart)]*self.nsubs).permute(2,3,1,0) for k in range(2)]] #sxa

        self.posterior_actions = [ar.zeros(self.na,self.npart,self.nsubs)+1./self.na]

        self.observations = []
        self.rewards = []
        self.actions = []


    def update_mf(self, tau, t):

        # paper has -1, 1 for reward and no reward?!
        # these eqs are according to Otte et al. 2013
        # attention! the supplementary material from otto et al seems to be riddled with bugs
        # the MF stage 1 update has now changed to a logical version
        assert(t==self.T-1)
        Q_mf = self.Q_mf[-1]

        Q_mf1 = Q_mf[0]
        Q_mf2 = Q_mf[1]
        # Q_mf3 is 0 anyways according to the paper, but they drag it along so I will too

        action1 = self.actions[-2]
        action2 = self.actions[-1]

        state1 = self.observations[-3]
        state2 = self.observations[-2]

        reward1 = self.utility[self.rewards[-2]]
        reward2 = self.utility[self.rewards[-1]]
        # print(reward2)
        # print(reward1)

        new_Q_mf3 = 0

        # second stage update
        state_action_pair2 = ar.eye(self.ns)[:,state2][:,None,None,...]*ar.eye(self.na)[:,action2][None,:,None,...]

        pred_err2 = (reward2[None,None,None,...] + new_Q_mf3)/self.alpha[None,None,...] - Q_mf2*state_action_pair2
        updated_Q_mf2 = Q_mf2*state_action_pair2 + self.alpha[None,None,...]*pred_err2

        new_Q_mf2 = ar.where(state_action_pair2>0, updated_Q_mf2, (1-self.alpha)[None,None,...]*Q_mf2)

        # mask the participants who didnt do a choice
        new_Q_mf2 = ar.where(self.mask[tau][None,None,:], new_Q_mf2, Q_mf2)

        # first stage update
        state_action_pair1 = ar.eye(self.ns)[:,state1][:,None,None,...]*ar.eye(self.na)[:,action1][None,:,None,...]

        discounted_Q_mf2 = self.lamb*(new_Q_mf2*state_action_pair2).sum(dim=(0,1))
        pred_err1 = (reward1[None,...] + discounted_Q_mf2[None,None,:,:])/self.alpha[None,None,...] - Q_mf1*state_action_pair1
        updated_Q_mf1 = Q_mf1*state_action_pair1 + self.alpha[None,None,...]*(pred_err1)
        # print("updated Q_mf")
        # print(new_Q_mf2)

        new_Q_mf1 = ar.where(state_action_pair1>0, updated_Q_mf1, (1-self.alpha)[None,None,...]*Q_mf1)

        # mask the participants who didnt do a choice
        new_Q_mf1 = ar.where(self.mask[tau][None,None,:], new_Q_mf1, Q_mf1)

        new_Q_mf = [new_Q_mf1, new_Q_mf2]
        self.Q_mf.append(new_Q_mf)


    def update_mb(self, tau, t):

        Q_mb = self.Q_mb[-1]
        Q_mf = self.Q_mf[-1]

        state1 = self.observations[-3]
        action1 = self.actions[-2]

        Q_mf2 = Q_mf[1]

        # best_Q2_s1 = Q_mf2[1].amax(dim=0)
        # best_Q2_s2 = Q_mf2[2].amax(dim=0)

        # Q_mb1_a0 = self.generative_model_states[1,0,0]*best_Q2_s1 + self.generative_model_states[2,0,0]*best_Q2_s2
        # Q_mb1_a1 = self.generative_model_states[1,0,1]*best_Q2_s1 + self.generative_model_states[2,0,1]*best_Q2_s2

        # new_Q_mb1 = ar.stack([ar.stack([Q_mb1_a0, Q_mb1_a1]), ar.zeros(2,self.npart,self.nsubs), ar.zeros(2,self.npart,self.nsubs)])

        best_Q2 = ar.amax(Q_mf2, dim=1)

        new_Q_mb1 = (self.generative_model_states[:,:,:,None,None]*best_Q2[:,None,None,...]).sum(dim=0)

        new_Q_mb = [new_Q_mb1, Q_mf2]

        self.Q_mb.append(new_Q_mb)

    def calc_action_probs(self, tau, t):

        Q_mb = ar.stack([self.Q_mb[-1][t][self.observations[-1][i],:,:,i] for i in range(self.nsubs)], dim=-1)
        Q_mf = ar.stack([self.Q_mf[-1][t][self.observations[-1][i],:,:,i] for i in range(self.nsubs)], dim=-1)

        if t==0:
            rep = ar.eye(self.na)[:,self.prev_first_action[-1]][:,None,:]
        else:
            rep = 0

        exponent = self.beta_mb*Q_mb + self.beta_mf*Q_mf + self.p*rep

        action_probs = ar.softmax(exponent, dim=0)

        self.posterior_actions.append(action_probs)

    def update_beliefs(self, tau, t, observation, reward, chosen_action, possible_policies):

        self.observations.append(observation)
        self.rewards.append(reward)
        self.actions.append(chosen_action)

        if t==1:
            if tau > 0:
                prev_action = ar.where(self.mask[tau], chosen_action, self.prev_first_action[-1])
            else:
                prev_action = chosen_action
            self.prev_first_action.append(prev_action)

        if t==self.T-1:
            # print(reward)
            self.update_mf(tau, t)
            self.update_mb(tau, t)
        elif tau>0 and t<self.T-1:
            self.calc_action_probs(tau, t)
        elif tau==0 and t<self.T-1:
            self.posterior_actions.append(ar.zeros(self.na,self.npart,self.nsubs)+1./self.na)


class mfmb3Perception(object):
    def __init__(self,
                 generative_model_states,
                 policies,
                 Q_mf_init,
                 Q_mb_init,
                 utility,
                 lamb = ar.tensor([0.9]),
                 alpha = ar.tensor([0.1]),
                 beta_mf = ar.tensor([4.]),
                 beta_mb = ar.tensor([4.]),
                 p = ar.tensor([2.]),
                 mask = None,
                 trials=10,
                 T=3,
                 npart=1, nsubs=1,
                 use_p=True,
                 restrict_alpha=False,
                 max_dt=10, min_alpha=0):

        self.generative_model_states = generative_model_states[:3,:3,...]
        self.alpha = alpha
        self.beta_mb = beta_mb
        self.beta_mf = beta_mf
        self.lamb = lamb
        self.p = p
        self.ns = self.generative_model_states.shape[0]
        self.utility = utility
        self.nr = utility.shape[0]
        self.na = len(ar.unique(policies))
        self.T = T
        self.prev_first_action = []
        self.action_probs = []
        self.nsubs = nsubs

        if mask is None:
            self.mask = ar.ones(trials, nsubs).bool()
        else:
            self.mask = mask

        self.use_p = use_p
        self.restrict_alpha = restrict_alpha
        if self.restrict_alpha:
            self.min_alpha = min_alpha
        self.max_dt = max_dt
        if self.use_p:
            self.npars = 5
        else:
            self.npars = 4
        self.param_names = list(self.locs_to_pars(ar.zeros(self.npars)).keys())

        self.Q_mf_init = Q_mf_init
        self.Q_mb_init = Q_mb_init
        self.Q_mf = [Q_mf_init] #sxa
        self.Q_mb = [Q_mb_init] #sxa

        self.observations = []
        self.rewards = []
        self.actions = []

        self.posterior_actions = [ar.zeros((self.na))+1./self.na]

    def locs_to_pars(self, locs):
        
        if self.restrict_alpha:
            alpha = self.min_alpha + ar.sigmoid(locs[...,1])*(1.-self.min_alpha)
        else:
            alpha = ar.sigmoid(locs[...,1])

        if self.use_p:
            par_dict = {"discount": ar.sigmoid(locs[...,0]),
                        "learning rate": alpha,
                        "mf weight": self.max_dt*ar.sigmoid(locs[...,2]),
                        "mb weight": self.max_dt*ar.sigmoid(locs[...,3]),
                        "repetition": ar.sigmoid(locs[...,4])}
        else:
            par_dict = {"discount": ar.sigmoid(locs[...,0]),
                        "learning rate": alpha,
                        "mf weight": self.max_dt*ar.sigmoid(locs[...,2]),
                        "mb weight": self.max_dt*ar.sigmoid(locs[...,3])}

        return par_dict

    def set_parameters(self, locs):

        par_dict = self.locs_to_pars(locs)

        if 'discount' in par_dict:
            self.lamb = par_dict['discount']
        if 'learning rate' in par_dict:
            self.alpha = par_dict['learning rate']
        if 'mf weight' in par_dict:
            self.beta_mf = par_dict['mf weight']
        if 'mb weight' in par_dict:
            self.beta_mb = par_dict['mb weight']
        if 'repetition' in par_dict:
            self.p = par_dict['repetition']
        else:
            self.p = ar.zeros_like(self.lamb)

    def reset(self):

        if len(self.alpha.shape) > 1:
            self.npart = self.alpha.shape[0]
            self.nsubs = self.alpha.shape[1]
        else:
            self.nsubs = self.alpha.shape[0]
            self.npart = 1
            #self.alpha_0 = self.alpha_0[None,:]
            self.lamb = self.lamb[None,:]
            self.alpha = self.alpha[None,:]
            self.beta_mf = self.beta_mf[None,:]
            self.beta_mb = self.beta_mb[None,:]
            self.p = self.p[None,:]

        self.prev_first_action = []
        self.action_probs = []

        self.Q_mf = [[ar.stack([ar.stack([self.Q_mf_init[k]]*self.npart)]*self.nsubs).permute(2,3,1,0) for k in range(2)]] #sxa
        self.Q_mb = [[ar.stack([ar.stack([self.Q_mb_init[k]]*self.npart)]*self.nsubs).permute(2,3,1,0) for k in range(2)]] #sxa

        self.posterior_actions = [ar.zeros(self.na,self.npart,self.nsubs)+1./self.na]

        self.observations = []
        self.rewards = []
        self.actions = []


    def update_mf(self, tau, t):

        # paper has -1, 1 for reward and no reward?!
        # these eqs are according to Otte et al. 2013
        # attention! the supplementary material from otto et al seems to be riddled with bugs
        # the MF stage 1 update has now changed to a logical version
        assert(t==self.T-1)
        Q_mf = self.Q_mf[-1]

        Q_mf1 = Q_mf[0]
        Q_mf2 = Q_mf[1]
        # Q_mf3 is 0 anyways according to the paper, but they drag it along so I will too

        action1 = self.actions[-2]
        action2 = self.actions[-1]

        state1 = self.observations[-3]
        state2 = self.observations[-2]

        reward1 = self.utility[self.rewards[-2]]
        reward2 = self.utility[self.rewards[-1]]
        # print(reward2)
        # print(reward1)

        new_Q_mf3 = 0

        # second stage update
        state_action_pair2 = ar.eye(self.ns)[:,state2][:,None,None,...]*ar.eye(self.na)[:,action2][None,:,None,...]

        pred_err2 = (reward2[None,None,None,...] + new_Q_mf3)/self.alpha[None,None,...] - Q_mf2*state_action_pair2
        updated_Q_mf2 = Q_mf2*state_action_pair2 + self.alpha[None,None,...]*pred_err2

        new_Q_mf2 = ar.where(state_action_pair2>0, updated_Q_mf2, (1-self.alpha)[None,None,...]*Q_mf2)

        # mask the participants who didnt do a choice
        new_Q_mf2 = ar.where(self.mask[tau][None,None,:], new_Q_mf2, Q_mf2)

        # first stage update
        state_action_pair1 = ar.eye(self.ns)[:,state1][:,None,None,...]*ar.eye(self.na)[:,action1][None,:,None,...]

        discounted_Q_mf2 = self.lamb*(new_Q_mf2*state_action_pair2).sum(dim=(0,1))
        pred_err1 = (reward1[None,...] + discounted_Q_mf2[None,None,:,:])/self.alpha[None,None,...] - Q_mf1*state_action_pair1
        updated_Q_mf1 = Q_mf1*state_action_pair1 + self.alpha[None,None,...]*(pred_err1)
        # print("updated Q_mf")
        # print(new_Q_mf2)

        new_Q_mf1 = ar.where(state_action_pair1>0, updated_Q_mf1, (1-self.alpha)[None,None,...]*Q_mf1)

        # mask the participants who didnt do a choice
        new_Q_mf1 = ar.where(self.mask[tau][None,None,:], new_Q_mf1, Q_mf1)

        new_Q_mf = [new_Q_mf1, new_Q_mf2]
        self.Q_mf.append(new_Q_mf)


    def update_mb(self, tau, t):

        Q_mb = self.Q_mb[-1]
        Q_mf = self.Q_mf[-1]

        state1 = self.observations[-3]
        action1 = self.actions[-2]

        Q_mf2 = Q_mf[1]

        # best_Q2_s1 = Q_mf2[1].amax(dim=0)
        # best_Q2_s2 = Q_mf2[2].amax(dim=0)

        # Q_mb1_a0 = self.generative_model_states[1,0,0]*best_Q2_s1 + self.generative_model_states[2,0,0]*best_Q2_s2
        # Q_mb1_a1 = self.generative_model_states[1,0,1]*best_Q2_s1 + self.generative_model_states[2,0,1]*best_Q2_s2

        # new_Q_mb1 = ar.stack([ar.stack([Q_mb1_a0, Q_mb1_a1]), ar.zeros(2,self.npart,self.nsubs), ar.zeros(2,self.npart,self.nsubs)])

        best_Q2 = ar.amax(Q_mf2, dim=1)

        new_Q_mb1 = (self.generative_model_states[:,:,:,None,None]*best_Q2[:,None,None,...]).sum(dim=0)

        new_Q_mb = [new_Q_mb1, Q_mf2]

        self.Q_mb.append(new_Q_mb)

    def calc_action_probs(self, tau, t):

        Q_mb = ar.stack([self.Q_mb[-1][t][self.observations[-1][i],:,:,i] for i in range(self.nsubs)], dim=-1)
        Q_mf = ar.stack([self.Q_mf[-1][t][self.observations[-1][i],:,:,i] for i in range(self.nsubs)], dim=-1)

        if t==0:
            rep = ar.eye(self.na)[:,self.prev_first_action[-1]][:,None,:]#ar.nn.functional.one_hot(self.prev_first_action[-1]).permute(1,0)[:,None,:]
        else:
            rep = ar.zeros(self.na, self.npart, self.nsubs)

        exponent = self.beta_mb[None,...]*Q_mb + self.beta_mf[None,...]*Q_mf + self.p[None,...]*rep

        action_probs = ar.softmax(exponent, dim=0)

        self.posterior_actions.append(action_probs)

    def update_beliefs(self, tau, t, observation, reward, chosen_action, possible_policies):

        self.observations.append(observation)
        self.rewards.append(reward)
        self.actions.append(chosen_action)

        if t==1:
            if tau > 0:
                prev_action = ar.where(self.mask[tau], chosen_action, self.prev_first_action[-1])
            else:
                prev_action = chosen_action
            self.prev_first_action.append(prev_action)

        if t==self.T-1:
            # print(reward)
            self.update_mf(tau, t)
            self.update_mb(tau, t)
        elif tau>0 and t<self.T-1:
            self.calc_action_probs(tau, t)
        elif tau==0 and t<self.T-1:
            self.posterior_actions.append(ar.zeros(self.na,self.npart,self.nsubs)+1./self.na)


class mfmbOrigPerception(object):
    def __init__(self,
                 generative_model_states,
                 policies,
                 Q_mf_init,
                 Q_mb_init,
                 utility,
                 lamb = ar.tensor(0.9),
                 alpha = ar.tensor(0.1),
                 beta = ar.tensor(2.),
                 w = ar.tensor(2.),
                 p = ar.tensor(0.1),
                 mask = None,
                 trials=10,
                 T=3,
                 npart=1, nsubs=1,
                 use_p=True,
                 restrict_alpha=False,
                 max_dt=10, min_alpha=0):

        self.generative_model_states = generative_model_states[:3,:3,...]
        self.alpha = alpha
        self.beta = beta
        self.w = w
        self.lamb = lamb
        self.p = p
        self.ns = self.generative_model_states.shape[0]
        self.utility = utility
        self.nr = utility.shape[0]
        self.na = len(ar.unique(policies))
        self.T = T
        self.prev_first_action = []
        self.action_probs = []
        
        if mask is None:
            self.mask = ar.ones(trials, nsubs).bool()
        else:
            self.mask = mask

        self.use_p = use_p
        if self.use_p:
            self.npars = 5
        else:
            self.npars = 4
        self.restrict_alpha = restrict_alpha
        if self.restrict_alpha:
            self.min_alpha = min_alpha
        self.max_dt = max_dt
        self.param_names = list(self.locs_to_pars(ar.zeros(self.npars)).keys())

        self.Q_mf_init = Q_mf_init
        self.Q_mb_init = Q_mb_init
        self.Q_mf = [Q_mf_init] #sxa
        self.Q_mb = [Q_mb_init] #sxa

        self.observations = []
        self.rewards = []
        self.actions = []

        self.posterior_actions = [ar.zeros((self.na))+1./self.na]

    def locs_to_pars(self, locs):
        
        if self.restrict_alpha:
            alpha = self.min_alpha + ar.sigmoid(locs[...,1])*(1-self.min_alpha)
        else:
            alpha = ar.sigmoid(locs[...,1])

        if self.use_p:
            par_dict = {"discount": ar.sigmoid(locs[...,0]),
                        "learning rate": alpha,
                        "dec temp": self.max_dt*ar.sigmoid(locs[...,2]),
                        "weight": ar.sigmoid(locs[...,3]),
                        "repetition": ar.sigmoid(locs[...,4])}
        else:
            par_dict = {"discount": ar.sigmoid(locs[...,0]),
                        "learning rate": alpha,
                        "dec temp": self.max_dt*ar.sigmoid(locs[...,2]),
                        "weight": ar.sigmoid(locs[...,3])}

        return par_dict

    def set_parameters(self, locs):

        par_dict = self.locs_to_pars(locs)

        if 'discount' in par_dict:
            self.lamb = par_dict['discount']
        if 'learning rate' in par_dict:
            self.alpha = par_dict['learning rate']
        if 'dec temp' in par_dict:
            self.beta = par_dict['dec temp']
        if 'weight' in par_dict:
            self.w = par_dict['weight']
        if 'repetition' in par_dict:
            self.p = par_dict['repetition']
        else:
            self.p = ar.zeros_like(self.lamb)

    def reset(self):

        if len(self.alpha.shape) > 1:
            self.npart = self.alpha.shape[0]
            self.nsubs = self.alpha.shape[1]
        else:
            self.nsubs = self.alpha.shape[0]
            self.npart = 1
            #self.alpha_0 = self.alpha_0[None,:]
            self.lamb = self.lamb[None,:]
            self.alpha = self.alpha[None,:]
            self.beta = self.beta[None,:]
            self.w = self.w[None,:]
            self.p = self.p[None,:]

        self.prev_first_action = []
        self.action_probs = []

        self.Q_mf = [[ar.stack([ar.stack([self.Q_mf_init[k]]*self.npart)]*self.nsubs).permute(2,3,1,0) for k in range(2)]] #sxa
        self.Q_mb = [[ar.stack([ar.stack([self.Q_mb_init[k]]*self.npart)]*self.nsubs).permute(2,3,1,0) for k in range(2)]] #sxa

        self.posterior_actions = [ar.zeros(self.na,self.npart,self.nsubs)+1./self.na]

        self.observations = []
        self.rewards = []
        self.actions = []


    def update_mf(self, tau, t):

        # paper has -1, 1 for reward and no reward?!
        # these eqs are according to Otte et al. 2013
        # attention! the supplementary material from otto et al seems to be riddled with bugs
        # the MF stage 1 update has now changed to a logical version
        assert(t==self.T-1)
        Q_mf = self.Q_mf[-1]

        Q_mf1 = Q_mf[0]
        Q_mf2 = Q_mf[1]
        # Q_mf3 is 0 anyways according to the paper, but they drag it along so I will too

        action1 = self.actions[-2]
        action2 = self.actions[-1]

        state1 = self.observations[-3]
        state2 = self.observations[-2]

        reward1 = self.utility[self.rewards[-2]]
        reward2 = self.utility[self.rewards[-1]]
        # print(reward2)
        # print(reward1)

        new_Q_mf3 = 0

        # second stage update
        state_action_pair2 = ar.eye(self.ns)[:,state2][:,None,None,...]*ar.eye(self.na)[:,action2][None,:,None,...]

        pred_err2 = (reward2[None,None,None,...] + new_Q_mf3)/self.alpha[None,None,...] - Q_mf2*state_action_pair2
        updated_Q_mf2 = Q_mf2*state_action_pair2 + self.alpha[None,None,...]*pred_err2

        new_Q_mf2 = ar.where(state_action_pair2>0, updated_Q_mf2, (1-self.alpha)[None,None,...]*Q_mf2)
        
        # mask the participants who didnt do a choice
        new_Q_mf2 = ar.where(self.mask[tau][None,None,:], new_Q_mf2, Q_mf2)

        # first stage update
        state_action_pair1 = ar.eye(self.ns)[:,state1][:,None,None,...]*ar.eye(self.na)[:,action1][None,:,None,...]

        discounted_Q_mf2 = self.lamb*(new_Q_mf2*state_action_pair2).sum(dim=(0,1))
        pred_err1 = (reward1[None,...] + discounted_Q_mf2[None,None,:,:])/self.alpha[None,None,...] - Q_mf1*state_action_pair1
        updated_Q_mf1 = Q_mf1*state_action_pair1 + self.alpha[None,None,...]*(pred_err1)
        # print("updated Q_mf")
        # print(new_Q_mf2)

        new_Q_mf1 = ar.where(state_action_pair1>0, updated_Q_mf1, (1-self.alpha)[None,None,...]*Q_mf1)
        
        # mask the participants who didnt do a choice
        new_Q_mf1 = ar.where(self.mask[tau][None,None,:], new_Q_mf1, Q_mf1)

        new_Q_mf = [new_Q_mf1, new_Q_mf2]
        self.Q_mf.append(new_Q_mf)


    def update_mb(self, tau, t):

        Q_mb = self.Q_mb[-1]
        Q_mf = self.Q_mf[-1]

        state1 = self.observations[-3]
        action1 = self.actions[-2]

        Q_mf2 = Q_mf[1]

        # best_Q2_s1 = Q_mf2[1].amax(dim=0)
        # best_Q2_s2 = Q_mf2[2].amax(dim=0)

        # Q_mb1_a0 = self.generative_model_states[1,0,0]*best_Q2_s1 + self.generative_model_states[2,0,0]*best_Q2_s2
        # Q_mb1_a1 = self.generative_model_states[1,0,1]*best_Q2_s1 + self.generative_model_states[2,0,1]*best_Q2_s2

        # new_Q_mb1 = ar.stack([ar.stack([Q_mb1_a0, Q_mb1_a1]), ar.zeros(2,self.npart,self.nsubs), ar.zeros(2,self.npart,self.nsubs)])

        best_Q2 = ar.amax(Q_mf2, dim=1)

        new_Q_mb1 = (self.generative_model_states[:,:,:,None,None]*best_Q2[:,None,None,...]).sum(dim=0)

        new_Q_mb = [new_Q_mb1, Q_mf2]

        self.Q_mb.append(new_Q_mb)

    def calc_action_probs(self, tau, t):

        Q_mb = ar.stack([self.Q_mb[-1][t][self.observations[-1][i],:,:,i] for i in range(self.nsubs)], dim=-1)
        Q_mf = ar.stack([self.Q_mf[-1][t][self.observations[-1][i],:,:,i] for i in range(self.nsubs)], dim=-1)

        if t==0:
            rep = ar.eye(self.na)[:,self.prev_first_action[-1]][:,None,:]
        else:
            rep = 0

        exponent = self.beta*(self.w*Q_mb + (1-self.w)*Q_mf + self.p*rep)

        action_probs = ar.softmax(exponent, dim=0)

        self.posterior_actions.append(action_probs)

    def update_beliefs(self, tau, t, observation, reward, chosen_action, possible_policies):

        self.observations.append(observation)
        self.rewards.append(reward)
        self.actions.append(chosen_action)

        if t==1:
            if tau > 0:
                prev_action = ar.where(self.mask[tau], chosen_action, self.prev_first_action[-1])
            else:
                prev_action = chosen_action
            self.prev_first_action.append(prev_action)

        if t==self.T-1:
            # print(reward)
            self.update_mf(tau, t)
            self.update_mb(tau, t)
        elif tau>0 and t<self.T-1:
            self.calc_action_probs(tau, t)
        elif tau==0 and t<self.T-1:
            self.posterior_actions.append(ar.zeros(self.na,self.npart,self.nsubs)+1./self.na)
            
            
class mfmbOrig2Perception(object):
    def __init__(self,
                 generative_model_states,
                 policies,
                 Q_mf_init,
                 Q_mb_init,
                 utility,
                 lamb = ar.tensor(0.9),
                 alpha = ar.tensor(0.1),
                 beta = ar.tensor(2.),
                 w = ar.tensor(2.),
                 p = ar.tensor(0.1),
                 mask = None,
                 trials=10,
                 T=3,
                 npart=1, nsubs=1,
                 use_p=True,
                 restrict_alpha=False,
                 max_dt=10, min_alpha=0):

        self.generative_model_states = generative_model_states[:3,:3,...]
        self.alpha = alpha
        self.beta = beta
        self.w = w
        self.lamb = lamb
        self.p = p
        self.ns = self.generative_model_states.shape[0]
        self.utility = utility
        self.nr = utility.shape[0]
        self.na = len(ar.unique(policies))
        self.T = T
        self.prev_first_action = []
        self.action_probs = []
        
        if mask is None:
            self.mask = ar.ones(trials, nsubs).bool()
        else:
            self.mask = mask

        self.use_p = use_p
        if self.use_p:
            self.npars = 5
        else:
            self.npars = 4
        self.restrict_alpha = restrict_alpha
        if self.restrict_alpha:
            self.min_alpha = min_alpha
        self.max_dt = max_dt
        self.param_names = list(self.locs_to_pars(ar.zeros(self.npars)).keys())

        self.Q_mf_init = Q_mf_init
        self.Q_mb_init = Q_mb_init
        self.Q_mf = [Q_mf_init] #sxa
        self.Q_mb = [Q_mb_init] #sxa

        self.observations = []
        self.rewards = []
        self.actions = []

        self.posterior_actions = [ar.zeros((self.na))+1./self.na]

    def locs_to_pars(self, locs):
        
        if self.restrict_alpha:
            alpha = self.min_alpha + ar.sigmoid(locs[...,1])*(1-self.min_alpha)
        else:
            alpha = ar.sigmoid(locs[...,1])

        if self.use_p:
            par_dict = {"discount": ar.sigmoid(locs[...,0]),
                        "learning rate": alpha,
                        "dec temp": self.max_dt*ar.sigmoid(locs[...,2]),
                        "weight": ar.sigmoid(locs[...,3]),
                        "repetition": ar.sigmoid(locs[...,4])}
        else:
            par_dict = {"discount": ar.sigmoid(locs[...,0]),
                        "learning rate": alpha,
                        "dec temp": self.max_dt*ar.sigmoid(locs[...,2]),
                        "weight": ar.sigmoid(locs[...,3])}

        return par_dict

    def set_parameters(self, locs):

        par_dict = self.locs_to_pars(locs)

        if 'discount' in par_dict:
            self.lamb = par_dict['discount']
        if 'learning rate' in par_dict:
            self.alpha = par_dict['learning rate']
        if 'dec temp' in par_dict:
            self.beta = par_dict['dec temp']
        if 'weight' in par_dict:
            self.w = par_dict['weight']
        if 'repetition' in par_dict:
            self.p = par_dict['repetition']
        else:
            self.p = ar.zeros_like(self.lamb)

    def reset(self):

        if len(self.alpha.shape) > 1:
            self.npart = self.alpha.shape[0]
            self.nsubs = self.alpha.shape[1]
        else:
            self.nsubs = self.alpha.shape[0]
            self.npart = 1
            #self.alpha_0 = self.alpha_0[None,:]
            self.lamb = self.lamb[None,:]
            self.alpha = self.alpha[None,:]
            self.beta = self.beta[None,:]
            self.w = self.w[None,:]
            self.p = self.p[None,:]

        self.prev_first_action = []
        self.action_probs = []

        self.Q_mf = [[ar.stack([ar.stack([self.Q_mf_init[k]]*self.npart)]*self.nsubs).permute(2,3,1,0) for k in range(2)]] #sxa
        self.Q_mb = [[ar.stack([ar.stack([self.Q_mb_init[k]]*self.npart)]*self.nsubs).permute(2,3,1,0) for k in range(2)]] #sxa

        self.posterior_actions = [ar.zeros(self.na,self.npart,self.nsubs)+1./self.na]

        self.observations = []
        self.rewards = []
        self.actions = []


    def update_mf(self, tau, t):

        # paper has -1, 1 for reward and no reward?!
        # these eqs are according to Otte et al. 2013
        # attention! the supplementary material from otto et al seems to be riddled with bugs
        # the MF stage 1 update has now changed to a logical version
        assert(t==self.T-1)
        Q_mf = self.Q_mf[-1]

        Q_mf1 = Q_mf[0]
        Q_mf2 = Q_mf[1]
        # Q_mf3 is 0 anyways according to the paper, but they drag it along so I will too

        action1 = self.actions[-2]
        action2 = self.actions[-1]

        state1 = self.observations[-3]
        state2 = self.observations[-2]

        reward1 = self.utility[self.rewards[-2]]
        reward2 = self.utility[self.rewards[-1]]
        # print(reward2)
        # print(reward1)

        new_Q_mf3 = 0

        # second stage update
        state_action_pair2 = ar.eye(self.ns)[:,state2][:,None,None,...]*ar.eye(self.na)[:,action2][None,:,None,...]

        pred_err2 = (reward2[None,None,None,...] + new_Q_mf3) - Q_mf2*state_action_pair2
        updated_Q_mf2 = Q_mf2*state_action_pair2 + self.alpha[None,None,...]*pred_err2

        new_Q_mf2 = ar.where(state_action_pair2>0, updated_Q_mf2, (1-self.alpha)[None,None,...]*Q_mf2)
        
        # mask the participants who didnt do a choice
        new_Q_mf2 = ar.where(self.mask[tau][None,None,:], new_Q_mf2, Q_mf2)

        # first stage update
        state_action_pair1 = ar.eye(self.ns)[:,state1][:,None,None,...]*ar.eye(self.na)[:,action1][None,:,None,...]

        discounted_Q_mf2 = self.lamb*(new_Q_mf2*state_action_pair2).sum(dim=(0,1))
        pred_err1 = (reward1[None,...] + discounted_Q_mf2[None,None,:,:]) - Q_mf1*state_action_pair1
        updated_Q_mf1 = Q_mf1*state_action_pair1 + self.alpha[None,None,...]*(pred_err1)
        # print("updated Q_mf")
        # print(new_Q_mf2)

        new_Q_mf1 = ar.where(state_action_pair1>0, updated_Q_mf1, (1-self.alpha)[None,None,...]*Q_mf1)
        
        # mask the participants who didnt do a choice
        new_Q_mf1 = ar.where(self.mask[tau][None,None,:], new_Q_mf1, Q_mf1)

        new_Q_mf = [new_Q_mf1, new_Q_mf2]
        self.Q_mf.append(new_Q_mf)


    def update_mb(self, tau, t):

        Q_mb = self.Q_mb[-1]
        Q_mf = self.Q_mf[-1]

        state1 = self.observations[-3]
        action1 = self.actions[-2]

        Q_mf2 = Q_mf[1]

        # best_Q2_s1 = Q_mf2[1].amax(dim=0)
        # best_Q2_s2 = Q_mf2[2].amax(dim=0)

        # Q_mb1_a0 = self.generative_model_states[1,0,0]*best_Q2_s1 + self.generative_model_states[2,0,0]*best_Q2_s2
        # Q_mb1_a1 = self.generative_model_states[1,0,1]*best_Q2_s1 + self.generative_model_states[2,0,1]*best_Q2_s2

        # new_Q_mb1 = ar.stack([ar.stack([Q_mb1_a0, Q_mb1_a1]), ar.zeros(2,self.npart,self.nsubs), ar.zeros(2,self.npart,self.nsubs)])

        best_Q2 = ar.amax(Q_mf2, dim=1)

        new_Q_mb1 = (self.generative_model_states[:,:,:,None,None]*best_Q2[:,None,None,...]).sum(dim=0)

        new_Q_mb = [new_Q_mb1, Q_mf2]

        self.Q_mb.append(new_Q_mb)

    def calc_action_probs(self, tau, t):

        Q_mb = ar.stack([self.Q_mb[-1][t][self.observations[-1][i],:,:,i] for i in range(self.nsubs)], dim=-1)
        Q_mf = ar.stack([self.Q_mf[-1][t][self.observations[-1][i],:,:,i] for i in range(self.nsubs)], dim=-1)

        if t==0:
            rep = ar.eye(self.na)[:,self.prev_first_action[-1]][:,None,:]
        else:
            rep = 0

        exponent = self.beta*(self.w*Q_mb + (1-self.w)*Q_mf + self.p*rep)

        action_probs = ar.softmax(exponent, dim=0)

        self.posterior_actions.append(action_probs)

    def update_beliefs(self, tau, t, observation, reward, chosen_action, possible_policies):

        self.observations.append(observation)
        self.rewards.append(reward)
        self.actions.append(chosen_action)

        if t==1:
            if tau > 0:
                prev_action = ar.where(self.mask[tau], chosen_action, self.prev_first_action[-1])
            else:
                prev_action = chosen_action
            self.prev_first_action.append(prev_action)

        if t==self.T-1:
            # print(reward)
            self.update_mf(tau, t)
            self.update_mb(tau, t)
        elif tau>0 and t<self.T-1:
            self.calc_action_probs(tau, t)
        elif tau==0 and t<self.T-1:
            self.posterior_actions.append(ar.zeros(self.na,self.npart,self.nsubs)+1./self.na)