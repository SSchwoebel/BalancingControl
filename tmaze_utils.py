import torch
import numpy as np
import action_selection as asl
import agent as agt
import perception as prc
import environment as env
import world as wld
import action_selection as asl
import misc
import os
import glob

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


def set_up_ContextualTMaze(pars):

    TMaze_environment = env.ContextualMultiArmedBandid(torch.from_numpy(pars["generative_model_observations"]).float(), 
                                             torch.from_numpy(pars["generative_model_states"]).float(), 
                                             torch.from_numpy(pars["generative_process_rewards"]).float(), 
                                             torch.from_numpy(pars["generative_process_context_obs"]).float(),
                                             trials=pars["trials"], T=pars["T"])
    
    return TMaze_environment

def run_single_simulation(agent_pars, env_pars, context=False):

    if context:
        TMaze_environment = set_up_ContextualTMaze(env_pars)
    else:
        TMaze_environment = set_up_TMaze(env_pars)

    bayes_agent, bayes_perception = set_up_Bayesian_agent(agent_pars)
    
    w = wld.GroupWorld(TMaze_environment, bayes_agent, trials = agent_pars["trials"], T = agent_pars["T"])

    w.simulate_experiment(range(env_pars["trials"]))

    return w


def restructure_behavioral_data(data, true_vals=None):
    data_obs = torch.stack([d["observations"] for d in data], dim=-1)
    data_rew = torch.stack([d["rewards"] for d in data], dim=-1)
    data_act = torch.stack([d["actions"] for d in data], dim=-1)
    data_val = torch.cat([torch.tensor(d["valid"]) for d in data], dim=-1)
    data_ind = torch.stack([torch.tensor([d["subject"]]) for d in data], dim=-1)

    structured_data = {"subject": data_ind, "observations": data_obs, "rewards": data_rew, "actions": data_act, "valid": data_val}
    
    if true_vals is not None:
        # structure true vals
        
        true_pol_rate = torch.stack([torch.tensor([t["policy rate"]]) for t in true_vals], dim=-1)
        true_rew_rate = torch.stack([torch.tensor([t["reward rate"]]) for t in true_vals], dim=-1)
        true_dec_temp = torch.stack([torch.tensor([t["dec temp"]]) for t in true_vals], dim=-1)
        true_hab_tend = torch.stack([torch.tensor([t["habitual tendency"]]) for t in true_vals], dim=-1)
        true_ind = torch.stack([torch.tensor([t["subject"]]) for t in true_vals], dim=-1)
        
        structured_true_vals = {"subject": true_ind, "policy rate": true_pol_rate, "reward rate": true_rew_rate, "dec temp": true_dec_temp, "habitual tendency": true_hab_tend}
    
        return structured_true_vals, structured_data
    
    else:
        return structured_data

def load_simulation_outputs(base_dir, exp_name, agent_type):
        
    # data 
    fname_data = os.path.join(base_dir, f"{exp_name}_agent_{agent_type}_data.json")
    structured_data = misc.load_file(fname_data)
    # with open(fname_data, 'r') as infile:
    #     loaded_data = json.load(infile)
    # structured_data = pickle.decode(loaded_data)
        
    # true values 
    fname_true_vals = os.path.join(base_dir, f"{exp_name}_agent_"+agent_type+"_true_vals.json")
    structured_true_vals = misc.load_file(fname_true_vals)
    # with open(fname_true_vals, 'r') as infile:
    #     loaded_true_vals = json.load(infile)
    # structured_true_vals = pickle.decode(loaded_true_vals)
    
    return structured_true_vals, structured_data


def set_up_Bayesian_inference_agent(n_agents, pars, base_dir, remove_old=False):

    if remove_old:
        svgs = glob.glob(os.path.join(base_dir,"*.svg"))
        for file in svgs:
            os.remove(file)

        csvs = glob.glob(os.path.join(base_dir,"*.csv"))
        for file in csvs:
            os.remove(file)

        saves = glob.glob(os.path.join(base_dir,"*.save"))
        for file in saves:
            os.remove(file)

        agents = glob.glob(os.path.join(base_dir,"twostage_agent*"))
        for file in agents:
            os.remove(file)

        outputs = glob.glob(os.path.join(base_dir,"*.json"))
        for file in outputs:
            os.remove(file)
        
    agent, agent_perception = set_up_Bayesian_agent(pars, n_agents=n_agents)

    return agent
