import torch
import numpy as np
import action_selection as asl
import agent as agt
import perception as prc
import environment as env
import world as wld
import action_selection as asl
import misc

def set_up_Bayesian_agent(pars, n_agents=1):
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

    action_selection = asl.AveragedSelector(trials = pars["trials"],
                                      T = pars["T"],
                                      number_of_actions = na)

    ### initialize Agent, Environment and World classes
    agent_perception = prc.Group2ContextPerception(
        torch.from_numpy(pars["generative_model_observations"]).float(),
        torch.from_numpy(pars["generative_model_states"]).float()[...,0],
        torch.from_numpy(pars["transition_matrix_context"]).float(),
        torch.from_numpy(pars["prior_states"]).float(),
        torch.from_numpy(pars["prior_rewards"]).float(),
        torch.from_numpy(pars["prior_context"]).float(),
        torch.from_numpy(pars["all_policies"]),
        alpha_0=torch.tensor([pars["alpha_0"]]).float(),
        dirichlet_rew_params=torch.from_numpy(pars["dirichlet_rew_params"]).float(),
        dirichlet_context_obs_params=torch.from_numpy(pars["dirichlet_context_obs_params"]).float(),
        learn_habit=pars["learn_habit"],
        learn_rew=pars["learn_rew"],
        infer_context=pars["infer_context"],
        learn_context_obs=pars["learn_context_obs"],
        #to do: make simulation mask!
        mask=pars["mask"],
        hidden_state_mapping=pars["hidden_state_mapping"],
        T=pars["T"],
        trials=pars["trials"],
        use_h=pars["use_h"],
        dec_temp=torch.tensor([pars["dec_temp"]]).float(),
        #now would follow parameters like forgetting rate etc
        pol_lambda=torch.tensor([pars["forgetting_rate_pol"]]).float(),
        r_lambda=torch.tensor([pars["forgetting_rate_rew"]]).float(),
        nsubs=n_agents,
        store_internal_variables = pars["store_internal_variables"],
        infer_alpha_0=pars["infer_alpha_0"],
        infer_decision_temp=pars["infer_decision_temp"],
        infer_policy_rate=pars["infer_policy_rate"],
        infer_reward_rate=pars["infer_reward_rate"]
    )
    agent_perception.pars = pars

    # key_agent_pars = {"dec temp": torch.tensor([pars["dec_temp"]]).float(), "habitual tendency": torch.tensor([pars["alpha_0"]]).float(), 
    #                   "policy rate": torch.tensor([pars["forgetting_rate_pol"]]).float(), "reward rate": torch.tensor([pars["forgetting_rate_rew"]]).float()}

    # agent_perception.set_parameters(par_dict=key_agent_pars)
    agent_perception.reset()

    agent = agt.FittingAgent(agent_perception,action_selection,torch.from_numpy(pars["all_policies"]),
                             trials = pars["trials"], T = pars["T"], number_of_states = pars["nh"],
                             number_of_rewards = pars["nr"],
                             number_of_policies = pars["npi"], nsubs = n_agents)
    
    return agent, agent_perception


def set_up_TMaze(pars):

    TMaze_environment = env.MultiArmedBandid(torch.from_numpy(pars["generative_model_observations"]).float(), 
                                             torch.from_numpy(pars["generative_model_states"]).float(), 
                                             torch.from_numpy(pars["generative_process_rewards"]).float(), 
                                             trials=pars["trials"], T=pars["T"])
    
    return TMaze_environment