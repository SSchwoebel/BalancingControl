#%%
import sys
import os
import numpy as np
from misc import normalize

# If running in IPYTHON or as separate cells use this path
# sys.path.append(os.path.join(os.getcwd(),'..','..','code','BalancingControl'))
#If running file normally:
# sys.path.append(os.path.join(os.getcwd(),'code','BalancingControl'))

import action_selection as asl
import agent as agt
import perception as prc
import environment as env
from world import World


def run_single_simulation(pars):
    
    ns = pars["nm"]
    npl = pars["nh"]
    nr = pars["nr"]
    na = pars["na"]
    npi = pars["npi"]
    nc = pars["nc"]
    

    ### set obsevation likelihood p(o|s) 
    A = np.eye(ns)                                           
    pars["generative_model_observations"] = A

    ### set up geneartive model of states (p(s_t|s_t-1,a,c))                                         
    pars["generative_model_states"] = np.repeat(pars["generative_model_states"][:,:,:,None], nc,axis=-1)      # add a trivial context dimension

    ### set reward likelihood p(r|s,phi)
    C_betas = np.ones([nr, npl, nc])      
    C_betas[:,:,:2] = (pars["true_reward_contingencies"][0]*pars["reward_count_bias"]+1)[:,:,None]            # Beta from q(phi|Beta)
    generative_model_rewards = normalize(C_betas)                  # q(r|s,phi)
    pars["generative_model_rewards"] = generative_model_rewards
    pars["dirichlet_rew_params"] = C_betas

    ### set contex transition matrix p(c_t|c_t-1)
    p = pars["context_trans_prob"]                                 
    q = (1-p)/(nc-1)
    transition_matrix_context = np.eye(nc)*p + (np.ones([nc,nc]) - np.eye(nc))*q  
    pars["transition_matrix_context"] = transition_matrix_context

    ### set context prior p(c_1)
    p=0.99
    prior_context = np.zeros((nc)) + (1-p)/(nc-1)
    prior_context[0] = p
    pars["prior_context"] = prior_context

    ### set policy prior p(pi|c)
    C_alphas = np.zeros([npi, nc]) + pars["alpha_0"]
    prior_pi = normalize(C_alphas)
    pars["prior_policies"] = prior_pi
    pars["dirichlet_pol_params"] = C_alphas

    ### set state prior p(s_1)
    state_prior = normalize(np.ones((ns)))
    pars["prior_states"] = state_prior
    
    ### set action selection method
    if pars["averaged_action_selection"]:
        action_selection = asl.AveragedSelector(trials = pars["trials"],
                                      T = pars["T"],
                                      number_of_actions = na)
    else:
        action_selection = asl.MaxSelector(trials = pars["trials"],
                                 T = pars["T"],
                                 number_of_actions = na)


    ### initialize Agent, Environment and World classes
    agent_perception = prc.HierarchicalPerception(**pars)
    agent_perception.pars = pars
    agent = agt.BayesianPlanner(agent_perception,action_selection)
    
    environment = env.PlanetWorld(
                                  pars["generative_model_observations"],
                                  pars["generative_model_states"],
                                  pars["true_reward_contingencies"],
                                  pars["planets"],
                                  pars["starts"],
                                  pars["context_observation"],
                                  pars["context"],
                                  trials = pars["trials"],
                                  T = pars["T"],
                                  all_rewards = pars["all_rewards"]
                                  )

    world = World(environment, agent, trials = pars["trials"], T = pars["T"])

    ### run experiment
    world.simulate_experiment()

    ### save data file
    return world


def create_data_frame(data_path):
    pass
