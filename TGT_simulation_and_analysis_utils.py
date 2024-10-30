#%%
import sys
import os
import numpy as np
import pandas as pd
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

from misc import load_file, save_file, normalize

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
    bias = pars["reward_count_bias"][0]
    ci = pars["reward_count_bias"][1]

    C_betas[:,:,:ci] = (pars["true_reward_contingencies"][0]*bias+1)[:,:,None]            # Beta from q(phi|Beta)
    generative_model_rewards = normalize(C_betas)                  # q(r|s,phi)
    pars["generative_model_rewards"] = generative_model_rewards
    pars["dirichlet_rew_params"] = C_betas

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
    
    # sanity check printing
    vals = ['alpha_0', 'dec_temp', 'context_trans_prob', 'run', 'learn_pol', 'learn_rew', 'learn_context_obs', 'reward_count_bias',  'prior_rewards', 'all_rewards', 'hidden_state_mapping', 'nm', 'nh']
    matrix_vals = ['generative_model_context', 'dirichlet_context_obs_params', 'transition_matrix_context']

    for key in vals:
        print(f"{key}: {pars[key]}")

    for key in matrix_vals:
        print(f"\n{key}: \n{pars[key]}")   
    
    print("\n", "true_reward_contingencies")
    for cont in range(pars["n_reward_contingencies"]):
        print(pars["true_reward_contingencies"][cont],"\n")

    print("generative_model_rewards")
    for cont in range(nc):
        print(pars["generative_model_rewards"][:,:,cont].round(3),"\n")      

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


def create_data_frame(exp_name, current_dir, data_folder="raw_data"):
    fnames = load_file(exp_name +  '_sim_file_names.json')
    dfs = []
    for fi, file in enumerate(fnames):
        world = load_file(os.path.join(current_dir, data_folder,file))
        perc = world.agent.perception
        env = world.environment
        pars = perc.pars
        T = pars["T"]
        n_trials  = pars["trials"]
        factor = T*n_trials
        
        block = pars["block"].repeat(T) + 1
        trial_type = pars["trial_type"].repeat(T)
        optimal_policy = pars["optimal_policy"].repeat(T)
        alpha_0 = np.ones(factor)*pars["alpha_0"]
        dec_temp = np.ones(factor)*pars["dec_temp"]
        true_context =  pars["context"].repeat(T)
        context_cue =  pars["context_observation"].repeat(T)
        context_trans_prob =  np.array(pars["context_trans_prob"]).repeat(factor)
        utility = [list(pars["prior_rewards"])]*factor
        print(fi)
        print(perc.posterior_dirichlet_context_obs[0].round(3))
        # print(perc.learn_context_obs)
        rewards = perc.rewards.flatten("C")
        state = env.state_mapping[np.arange(n_trials)[:,None],perc.observations].flatten("C")
        agent = np.ones(factor)*fi
        t = np.tile(np.arange(T),n_trials)
        trial = np.arange(n_trials).repeat(T) + 1
        actions = perc.actions.flatten('C')
        executed_policy = np.ravel_multi_index(perc.actions[:,1:].T, (2,2,2)).repeat(T)
        inferred_context = np.argmax(perc.posterior_context,axis=-1).flatten('C')
        entropy_context = -(perc.posterior_context*np.log(perc.posterior_context)).sum(axis=-1).flatten('C')
        
        Rho = env.Rho[:,:,:,None] + 1e-15
        post = perc.posterior_dirichlet_rew[:,-1,:,:,:]
        post = (post/post.sum(axis=1)[:,None,:,:]) + 1e-15
        reward_dkl = ((post*np.log(post/Rho)).sum(axis=1)).sum(axis=1) / 3
        
        df = pd.DataFrame.from_dict({
                                     "file":np.array([fi%10]).repeat(factor),
                                     "agent":agent,
                                     "trial_type":trial_type,
                                     "block":block,
                                     "context_cue":context_cue,
                                     "trial":trial,
                                     "t":t,
                                     "step": np.arange(0,factor),
                                     "actions":actions,
                                     "executed_policy":executed_policy,  
                                     "optimal_policy":optimal_policy,
                                     "chose_optimal": executed_policy == optimal_policy,
                                     "true_context":true_context,
                                     "alpha_0": alpha_0,
                                     "dec_temp":dec_temp,
                                     "inferred_context": inferred_context,
                                     "inferred_correct_context": true_context == inferred_context,
                                     "entropy_context": entropy_context,
                                     "context_trans_prob":context_trans_prob,
                                     "dkl_0":reward_dkl[:,0].repeat(T),
                                     "dkl_1":reward_dkl[:,1].repeat(T),
                                     "dkl_2":reward_dkl[:,2].repeat(T),
                                     "dkl_3":reward_dkl[:,3].repeat(T),
                                     "utility": utility,
                                     "reward":rewards
                                    })
        
        dfs.append(df)
        
    df = pd.concat(dfs)
    
    df.to_excel(exp_name + "_data_long_format.xlsx")
    return df
