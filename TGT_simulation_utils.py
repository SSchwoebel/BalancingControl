#%%
###
import sys
import os
import torch
import copy 

torch.set_num_threads(1)
print("torch threads", torch.get_num_threads())

##

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
    A = torch.eye(ns)                                           
    pars["generative_model_observations"] = A

    ### set up geneartive model of states (p(s_t|s_t-1,a,c))                                         
    pars["generative_model_states"] = torch.repeat(pars["generative_model_states"][:,:,:,None], nc,axis=-1)      # add a trivial context dimension

    ### set reward likelihood p(r|s,phi)
    C_betas = torch.ones([nr, npl, nc])
    bias = pars["reward_count_bias"][0]
    ci = pars["reward_count_bias"][1]

    C_betas[:,:,:ci] = (pars["true_reward_contingencies"][0]*bias+1)[:,:,None]            # Beta from q(phi|Beta)
    generative_model_rewards = C_betas / C_betas.sum(dim=0)[None,...]                 # q(r|s,phi)
    pars["generative_model_rewards"] = generative_model_rewards
    pars["dirichlet_rew_params"] = C_betas

    ### set context prior p(c_1)
    p=0.99
    prior_context = torch.zeros((nc)) + (1-p)/(nc-1)
    prior_context[0] = p
    pars["prior_context"] = prior_context

    ### set policy prior p(pi|c)
    C_alphas = torch.zeros([npi, nc]) + pars["alpha_0"]
    prior_pi = C_alphas / C_alphas.sum(dim=0)[None,...]
    pars["prior_policies"] = prior_pi
    pars["dirichlet_pol_params"] = C_alphas

    ### set state prior p(s_1)
    state_prior = torch.ones((ns)) / ns
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
    
    ### print simulation values for the log
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
    
    # df = load_task_df(experiment_config)

    # fig, axes = plt.subplots(1,2,figsize=(11,3))
    

    # for bi, block in enumerate([0,experiment_config["meta_data"]["training_blocks"]]):
        
    #     ax = axes[bi]
    #     ax.xaxis.set_major_locator(ticker.MultipleLocator(3))
    #     sns.lineplot(ax=ax, data=df.query(f"block == {block}"), x="index", y=df["context"] % 2,color='k',marker="o",label="True context: habit=0 or planing=1")
    #     pivoted = df.query(f"block == {block}").pivot(index="index", columns="context_observation", values="optimal_policy")
    #     pivoted.columns = ["Optimal Policy when cue green","Optimal policy when cue brown"]
    #     pivoted.plot(ax=ax, marker="o",label="optimal_policy")# sns.lineplot(data=df.query("block == 0 "), x="index",y="context")
    #     ax.legend(bbox_to_anchor=(1.1, 0.5))
    #     ax.set_ylabel("Optimal Policy")    
    #     if bi == 0:
    #         ax.get_legend().remove()

    # for title,ax in zip(["Training block","Degradation block"],axes):
    #     ax.set_title(title)
    


    # dataframe = load_task_df(experiment_config)
    # fig, ax = plt.subplots(1,1,figsize=(3,3))
    # df=dataframe.groupby(["block","context_observation"])["exp_reward"].mean().reset_index()
    # print(df.dtypes)
    # sns.lineplot(data=df, ax=ax, x="block",y="exp_reward",hue="context_observation",style="context_observation",marker="o")
    
    # fig,ax = plt.subplots(1,2,figsize=(6,3))
    # df = dataframe.copy()

    # train_blocks = experiment_config["meta_data"]["training_blocks"]
    # block = train_blocks
    # g = sns.countplot(data=df.query(f"trial_type==0 & block == {block-1}"),x="optimal_policy",hue="context_observation",ax=ax[0])
    # g.yaxis.set_major_locator(ticker.MultipleLocator(3))
    # g = sns.countplot(data=df.query(f"trial_type==1 & block == {block}"),x="optimal_policy",hue="context_observation",ax=ax[1])
    # g.yaxis.set_major_locator(ticker.MultipleLocator(3))
# %%
