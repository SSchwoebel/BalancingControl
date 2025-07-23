#%%
###
import sys
import os
import numpy as np
import copy 
import torch
import glob
import pickle
import json
import gc


###
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


from matplotlib.animation import FuncAnimation
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams['animation.embed_limit'] = 400
from IPython.display import HTML

import seaborn as sns

sns.set_style("darkgrid")
plt.rcParams['figure.dpi'] = 100
green ='#095b3a'
brown = '#9c683a'
task_pal= sns.set_palette(['#095b3a', '#9c683a'])
brown_pal = sns.color_palette(["#402e32", "#9c683a","#f59432","#dfe0df"]) 
green_pal = sns.color_palette(["#095b3a", "#637f4f","#7cbc53","#d2e4d6"])

##

# If running in IPYTHON or as separate cells use this path
# sys.path.append(os.path.join(os.getcwd(),'..','..','code','BalancingControl'))
#If running file normally:
# sys.path.append(os.path.join(os.getcwd(),'code','BalancingControl'))


import action_selection as asl
import agent as agt
import perception as prc
import environment as env
from world import GroupWorld

from misc import load_file, save_file, normalize

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
    # C_alphas = np.zeros([npi, nc]) + pars["alpha_0"]
    # prior_pi = normalize(C_alphas)
    # pars["prior_policies"] = prior_pi
    # pars["dirichlet_pol_params"] = C_alphas

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
    
    ### print simulation values for the log
    if False:
        vals = ['alpha_0', 'dec_temp', 'context_trans_prob', 'run', 'learn_habit', 'learn_rew', 'learn_context_obs', 'reward_count_bias',  'prior_rewards', 'all_rewards', 'hidden_state_mapping', 'nm', 'nh', 
                'forgetting_rate_pol', 'forgetting_rate_rew']
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
        state_mapping=torch.from_numpy(pars["planets"]).long(),
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
        infer_reward_rate=pars["infer_reward_rate"],
        infer_cached_rate=pars["infer_cached_rate"],
        infer_cached_weight=pars["infer_cached_weight"]
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
    


def run_single_simulation(pars):
    
    agent, agent_perception = set_up_Bayesian_agent(pars)
    
    environment = env.PlanetSystem(
                                  torch.from_numpy(pars["generative_model_observations"]).float(),
                                  torch.from_numpy(pars["generative_model_states"]).float(),
                                  torch.tensor(pars["true_reward_contingencies"]).float(),
                                  torch.from_numpy(pars["planets"]),
                                  torch.from_numpy(pars["starts"]).int(),
                                  torch.from_numpy(pars["context_observation"]).int(),
                                  torch.from_numpy(pars["context"]).int(),
                                  trials = pars["trials"],
                                  T = pars["T"],
                                  all_rewards = torch.from_numpy(pars["all_rewards"])
                                  )

    world = GroupWorld(environment, agent, trials = pars["trials"], T = pars["T"])

    ### run experiment
    world.simulate_experiment()

    ### save data file
    return world


def restructure_behavioral_data(data, true_vals):
    data_obs = torch.stack([d["observations"] for d in data], dim=-1)
    data_rew = torch.stack([d["rewards"] for d in data], dim=-1)
    data_act = torch.stack([d["actions"] for d in data], dim=-1)
    data_val = torch.cat([torch.tensor(d["valid"]) for d in data], dim=-1)
    data_ind = torch.stack([torch.tensor([d["subject"]]) for d in data], dim=-1)

    structured_data = {"subject": data_ind, "observations": data_obs, "rewards": data_rew, "actions": data_act, "valid": data_val}
    
    if "context" in data[-1].keys():
        data_con = torch.stack([d["context"] for d in data], dim=-1)
        structured_data["context"] = data_con
    
    # structure true vals
    
    true_pol_rate = torch.stack([torch.tensor([t["policy rate"]]) for t in true_vals], dim=-1)
    true_rew_rate = torch.stack([torch.tensor([t["reward rate"]]) for t in true_vals], dim=-1)
    true_dec_temp = torch.stack([torch.tensor([t["dec temp"]]) for t in true_vals], dim=-1)
    true_hab_tend = torch.stack([torch.tensor([t["habitual tendency"]]) for t in true_vals], dim=-1)
    true_cac_wght = torch.stack([torch.tensor([t["cached weight"]]) for t in true_vals], dim=-1)
    true_cac_rate = torch.stack([torch.tensor([t["cached rate"]]) for t in true_vals], dim=-1)
    true_ind = torch.stack([torch.tensor([t["subject"]]) for t in true_vals], dim=-1)
    
    structured_true_vals = {"subject": true_ind, 
                        "dec temp": true_dec_temp, "reward rate": true_rew_rate, 
                        "habitual tendency": true_hab_tend, "policy rate": true_pol_rate,
                        "cached weight": true_cac_wght, "cached rate": true_cac_rate}
    
    return structured_true_vals, structured_data

def load_simulation_outputs(base_dir, exp_name, agent_type):
        
    # data 
    fname_data = os.path.join(base_dir, f"{exp_name}_agent_{agent_type}_data.json")
    structured_data = load_file(fname_data)
    # with open(fname_data, 'r') as infile:
    #     loaded_data = json.load(infile)
    # structured_data = pickle.decode(loaded_data)
        
    # true values 
    fname_true_vals = os.path.join(base_dir, f"{exp_name}_agent_"+agent_type+"_true_vals.json")
    structured_true_vals = load_file(fname_true_vals)
    # with open(fname_true_vals, 'r') as infile:
    #     loaded_true_vals = json.load(infile)
    # structured_true_vals = pickle.decode(loaded_true_vals)
    
    return structured_true_vals, structured_data


def create_data_frame(exp_name, data_folder="raw_data"):
    fnames = load_file(exp_name +  '_sim_file_names.json')
    dfs = []
    for fi, file in enumerate(fnames):
        world = load_file(os.path.join(data_folder,file))
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

        rewards = torch.cat(perc.rewards).numpy()
        #state = env.state_mapping[np.arange(n_trials)[:,None],perc.observations].flatten("C")
        agent = np.ones(factor)*fi
        t = np.tile(np.arange(T),n_trials)
        trial = np.arange(n_trials).repeat(T) + 1
        actions = torch.cat(perc.actions).numpy()
        executed_policy = np.ravel_multi_index(perc.actions_structured.numpy()[:,1:].T, (2,2,2)).repeat(T)
        posterior_context = torch.stack(perc.posterior_context).numpy()[1:,...,0,0]
        inferred_context = np.argmax(posterior_context,axis=-1).flatten('C')
        entropy_context = -(posterior_context*np.log(posterior_context)).sum(axis=-1)
        
        Rho = env.Rho[:,:,:,None].numpy() + 1e-15
        posterior_dirichlet_rew = torch.stack(perc.generative_model_rewards_mb).numpy()[...,0,0]
        post = posterior_dirichlet_rew#perc.posterior_dirichlet_rew[:,-1,:,:,:]
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

def sample_posterior(inferrer, param_names, fname_str, base_dir, n_samples=500, true_vals=None):

    sample_df, locs_sample_df = inferrer.sample_posterior(n_samples=n_samples) #inferrer.plot_posteriors(n_samples=1000)
    # inferrer.plot_posteriors(n_samples=n_samples)
    if true_vals is not None:
        append_trues = True
    else:
        append_trues = False
    
    sample_file = os.path.join(base_dir, fname_str+'_sample_df.csv')
    sample_df.to_csv(sample_file)

    locs_file = os.path.join(base_dir, fname_str+'_locs_sample_df.csv')
    locs_sample_df.to_csv(locs_file)
    
    mean_df = pd.DataFrame()

    for name in param_names:
        means = []
        if append_trues:
            trues = []
        subs = []
        for i in range(inferrer.nsubs):
            means.append(sample_df[sample_df['subject']==i][name].mean())
            if append_trues:
                trues.append(true_vals[name][true_vals['subject']==i])
            subs.append(i)

        mean_df["inferred "+name] = torch.tensor(means)
        if append_trues:
            mean_df["true "+name] = torch.tensor(trues)
        mean_df["subject"] = torch.tensor(subs)
        
    smaller_file = os.path.join(base_dir, fname_str+'_mean_df.csv')
    mean_df.to_csv(smaller_file)

    return mean_df, sample_df, locs_sample_df

def plot_choice_accuracy_mean(dataframe,simulation_params):

    df = dataframe.copy().query(f"t == 0")
    df = df.groupby(["alpha_0","context_trans_prob","agent","block","context_cue"])["chose_optimal"].mean().reset_index()

    plot_pars = {"x":"block","y":"chose_optimal","hue":"context_cue","marker":"o", "palette":task_pal, "errorbar":"sd"}

    fig = plt.plot()
    g = sns.lineplot(data=df, **plot_pars)
    g.vlines(ymin=0, ymax=1,x=simulation_params["training_blocks"]+0.5,ls='--',color='gray')
    g.vlines(ymin=0, ymax=1, x=simulation_params["training_blocks"]+simulation_params["degradation_blocks"]+0.5, ls='--',color='gray')
    g.set_xticks(ticks=np.arange(1,df.block.unique().size+1))
    g.set_ylim([0,1.05])
    plt.show()


def plot_choice_accuracy_alpha_rho(dataframe,simulation_params):

    df = dataframe.copy().query(f"t == 0")
    df = df.groupby(["alpha_0","context_trans_prob","agent","block","context_cue"])["chose_optimal"].mean().reset_index()
    n_rho = int(df.context_trans_prob.unique().size)

    plot_pars = {"x":"block","y":"chose_optimal","hue":"context_cue","marker":"o", "palette":task_pal, "errorbar":"sd"}

    fig, axes = plt.subplots(2,n_rho,figsize=(3*n_rho,4),sharex=True,sharey=True)
    plt.tight_layout()

    if n_rho == 1:
        axes = axes.reshape(2,n_rho) 
    for ri, alpha in enumerate([1,1000]):
        for ci, rho in enumerate(df.context_trans_prob.unique()):
            g = sns.lineplot(ax=axes[ri,ci], data=df.query(f"alpha_0=={alpha} & context_trans_prob == {rho}"),legend= ((ri+1)*(ci+1) == len(axes.flatten())),**plot_pars)
            g.vlines(ymin=0, ymax=1,x=simulation_params["training_blocks"]+0.5,ls='--',color='gray')
            g.vlines(ymin=0, ymax=1, x=simulation_params["training_blocks"]+simulation_params["degradation_blocks"]+0.5, ls='--',color='gray')
            axes[ri,ci].set_xticks(ticks=np.arange(1,df.block.unique().size+1))
            axes[ri,ci].set_ylim([0,1.05])
            axes[ri,ci].set_title(fr"$\rho$ = {rho}, $\alpha_0$ = {alpha}")

    fig.suptitle(fr"Effect of self-transition bias $\rho$ on mean choice accuracy.",y=1.1);

    return fig

def plot_context_inference_mean(dataframe, simulation_params):
    df = dataframe.copy()
    context_df = df.groupby(["alpha_0","dec_temp","context_trans_prob","agent","trial_type","block","context_cue","t"])["inferred_correct_context"].mean().reset_index()
    
    fig = plt.plot()
    g = sns.lineplot(data=context_df, x='block', y='inferred_correct_context', hue="t",palette=task_pal, style='t',marker="o", errorbar="sd")
    g.vlines(ymin=0, ymax=1,x=simulation_params["training_blocks"]+0.5,ls='--',color='gray')
    g.vlines(ymin=0, ymax=1, x=simulation_params["training_blocks"]+simulation_params["degradation_blocks"]+0.5, ls='--',color='gray')
    g.set_xticks(ticks=np.arange(1,df.block.unique().size+1))
    g.set_ylim([0,1.05])
    plt.show()

def plot_context_inference_t_alpha_rho(dataframe, simulation_params):
    df = dataframe.copy()
    context = df.groupby(["alpha_0","dec_temp","context_trans_prob","agent","trial_type","block","context_cue","t"])["inferred_correct_context"].mean().reset_index()
    n_rho = int(df.context_trans_prob.unique().size)

    for alpha in [1,1000]:
    # for alpha,title in zip([100],[r'$\alpha_0=100$']):

        fig, axes = plt.subplots(2,n_rho,figsize=(3*n_rho,4), sharex=True, sharey=True)
        fig.tight_layout()

        if n_rho == 1:
            axes = axes.reshape(2,n_rho) 

        for ci, rho in enumerate(dataframe["context_trans_prob"].unique()):
            for ri, context, palette in zip([0,1],[0,1],[green_pal, brown_pal]):
                g = sns.lineplot(ax=axes[ri,ci], data=df.query(f"alpha_0=={alpha} & context_cue=={context} & context_trans_prob == {rho}"),
                                 x='block', y='inferred_correct_context', hue="t",palette=palette, style='t',marker="o", errorbar="sd", legend = (ci == len(axes[0])-1))
                g.vlines(ymin=0, ymax=2,x=simulation_params["training_blocks"]+0.5,ls='--',color='gray')
                g.vlines(ymin=0, ymax=2,x=simulation_params["training_blocks"]+simulation_params["degradation_blocks"]+0.5,ls='--',color='gray')
                axes[ri,ci].set_xticks(ticks=np.arange(1,df.block.unique().size+1))
                axes[ri,ci].set_ylim([0,1.2])
                axes[ri,ci].set_title(fr"$\rho$ = {rho}")
        plt.suptitle(fr"Mean context inference accuracy as a function $t$, for each trial type and different $\rho$;" + fr"$\alpha_0$ = {alpha}",y=1.1);


def plot_context_entropy_t_alpha_rho(dataframe, simulation_params):
    df = dataframe.copy()
    context = df.groupby(["alpha_0","dec_temp","context_trans_prob", "agent","trial_type","block","context_cue","t"])["entropy_context"].mean().reset_index()
    n_rho = int(df.context_trans_prob.unique().size)


    for alpha,title in zip([1,1000],[r'$\alpha_0=1$', r'$\alpha_0=1000$']):
    # for alpha,title in zip([100],[r'$\alpha_0=100$']):

        fig, axes = plt.subplots(2,n_rho,figsize=(3*n_rho, 4), sharex=True, sharey=True)
        if n_rho == 1:
            axes = axes.reshape(2,n_rho)
        fig.tight_layout()
        for ci, rho in enumerate(dataframe["context_trans_prob"].unique()):
            for ri, context, palette in zip([0,1], [0,1],[green_pal, brown_pal]):
                g = sns.lineplot(ax=axes[ri,ci], data=df.query(f"alpha_0=={alpha} & context_cue=={context} & context_trans_prob=={rho} "),
                                 x='block', y='entropy_context', hue="t",palette=palette, style='t',marker="o", errorbar="sd")
                g.vlines(ymin=0, ymax=1,x=simulation_params["training_blocks"]+0.5,ls='--',color='gray')
                g.vlines(ymin=0, ymax=1,x=simulation_params["training_blocks"]+simulation_params["degradation_blocks"]+0.5,ls='--',color='gray')
                axes[ri, ci].set_xticks(ticks=np.arange(1,df.block.unique().size+1))
                axes[ri, ci].set_ylim([0,1])
                axes[ri,ci].set_title(fr"$\rho$ = {rho}")

        plt.suptitle(fr"Mean context entropy as a function $t$, for each trial type and different $\rho$;" + title,y=1.1);


def plot_individual_agents(dataframe, alpha_0=1, dec_temp=3, context_trans_prob=0.6, t=3):
    
    df = dataframe.copy()
    context = df.groupby(["alpha_0","dec_temp","context_trans_prob","agent","file","trial_type","block","context_cue","t"])[["inferred_correct_context"]].mean().reset_index()

    fig, ax = plt.subplots(1,2,figsize=(7,3));
    
    for cue in range(2):
        df = context.query(f"alpha_0 == {alpha_0} & dec_temp=={dec_temp} & context_trans_prob=={context_trans_prob} & context_cue=={cue} & t=={t}")
        g = sns.lineplot(ax=ax[cue],data=df, x="block", y="inferred_correct_context",hue="file",palette="viridis")
        g.set_xticks(np.arange(13));
        g.set_ylim([0,1.2])
        g.set_title(f"context_cue = {cue}")


def plot_average_DKL(rho, dataframe, simulation_params):
    
    df = dataframe.copy().query(f"t == 0")
    df.head()
    fig,axes = plt.subplots(1,4,figsize=(13,3))
    plt.tight_layout()
    for context,palette in zip([0,1,2,3],["Blues_r","Reds_r"]*2):
        sns.lineplot(ax=axes[context], data=df.query(f"context_trans_prob == {rho}"), x="trial",y=f"dkl_{context}",hue="alpha_0",errorbar="sd", palette=palette)
        # axes[context].set_xticks(np.arange(1,510,5))
        # axes[context].set_xticklabels(np.arange(1,510,5), rotation=90,fontsize=8)  # Rotate x-tick labels
        axes[context].set_xlim([0, (simulation_params["training_blocks"] + simulation_params["degradation_blocks"])*simulation_params["trials_per_block"]])
        axes[context].set_title(fr"$\rho$ = {rho}")


def plot_heatmap(data,title=None,vmin=0,vmax=1):
    
    if not type(data) is list:
        data = [data]
        title = [title]
        
    fig, axes = plt.subplots(1,len(data), figsize=(3*len(data), 3))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    plt.rcParams.update(plt.rcParamsDefault)

    for ai, ax, im in zip(np.arange(len(data)), axes, data):
        sns.heatmap(data=im, annot=True, cmap="viridis", cbar=False, fmt='.2f', ax=ax,vmin=vmin, vmax=vmax)
        
        if title is not None:
            ax.set_title(title[ai])
    
    return fig, axes


def animate_heatmap(data, interval=500,title=None,x_label=None,y_label=None):
        
    fig, ax = plt.subplots(1,1,figsize=(5,5))

    if not title is None:
        fig.suptitle(title)
    
    def update(frame):
        ax.clear()
        sns.heatmap(data[frame], annot=True, cmap="viridis", cbar=False, fmt='.2f',ax=ax,vmin=0, vmax=1)
        ax.set_title(frame+1)
        ax.set_xlabel(x_label,fontsize=12)
        ax.set_ylabel(y_label,fontsize=12)

    animation = FuncAnimation(fig, update, frames=data.shape[0], interval=interval)


    # return HTML(animation.to_jshtml())
    html = HTML(animation.to_jshtml())
    display(html)
    plt.close() # update
    
    return fig    


def animate_multiple_heatmaps(matrices, bins=10, interval=200, titles=None,x_label=None, y_label=None):

    N = matrices[0].shape[0]
    n_matrices = len(matrices)
    
    # Create figure and axes for subplots
    fig, axes = plt.subplots(1, n_matrices, figsize=(3.5*n_matrices, 5),sharey = True)
    
    def update(frame):
        # Clear each axis for the new frame
        for ai, ax, matrix in zip(np.arange(n_matrices), axes, matrices):
            ax.clear()
            # sns.histplot(matrix[frame].flatten(), bins=bins, kde=False, ax=ax, color="blue")
            sns.heatmap(matrix[frame], annot=True, cmap="viridis", cbar=False, fmt='.2f',ax=ax,vmin=-15,vmax=2)
            
            fig.suptitle(f"frame: {frame+1}, context trial: {frame % 6 + 1}")
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(titles[ai])
            
    # Create animation
    anim = FuncAnimation(fig, update, frames=N, interval=interval, repeat=True)

    html = HTML(anim.to_jshtml())
    display(html)
    plt.close()


def plot_task_structure(experiment_config):
    
    df = load_task_df(experiment_config)

    fig, axes = plt.subplots(1,2,figsize=(11,3))
    

    for bi, block in enumerate([0,experiment_config["meta_data"]["training_blocks"]]):
        
        ax = axes[bi]
        ax.xaxis.set_major_locator(ticker.MultipleLocator(3))
        sns.lineplot(ax=ax, data=df.query(f"block == {block}"), x="index", y=df["context"] % 2,color='k',marker="o",label="True context: habit=0 or planing=1")
        pivoted = df.query(f"block == {block}").pivot(index="index", columns="context_observation", values="optimal_policy")
        pivoted.columns = ["Optimal Policy when cue green","Optimal policy when cue brown"]
        pivoted.plot(ax=ax, marker="o",label="optimal_policy")# sns.lineplot(data=df.query("block == 0 "), x="index",y="context")
        ax.legend(bbox_to_anchor=(1.1, 0.5))
        ax.set_ylabel("Optimal Policy")    
        if bi == 0:
            ax.get_legend().remove()

    for title,ax in zip(["Training block","Degradation block"],axes):
        ax.set_title(title)
    

def load_task_df(experiment_config):
    exp_params = copy.deepcopy(experiment_config["experiment_data"])
    exp_params.pop("planets")
    df = pd.DataFrame(exp_params).reset_index()
    
    return df


def plot_reward_probs(contingency_1, contingency_2):
    fig,axes = plot_heatmap([contingency_1, contingency_2])
    fig.suptitle(r"Reward Contingencies during Training and Degradation $p(r|s)$",y=1.05)

    for ax in axes:
        ax.set_xlabel("planets")
        ax.set_ylabel("rewards")


def plot_state_transition_matrix(stm): 
    fig,axes = plot_heatmap([stm[:,:,0].T, stm[:,:,1].T])
    fig.suptitle(r"$p(s_t|s_{t-1},a)$",y=1.05)

    for ax in axes:
        ax.set_xlabel(r"$s_{t-1}$",fontsize=12)
        ax.set_ylabel(r"$s_{t}$",fontsize=14)


def plot_expected_reward_and_optimal_policy(experiment_config):

    dataframe = load_task_df(experiment_config)
    fig, ax = plt.subplots(1,1,figsize=(3,3))
    df=dataframe.groupby(["block","context_observation"])["exp_reward"].mean().reset_index()
    print(df.dtypes)
    sns.lineplot(data=df, ax=ax, x="block",y="exp_reward",hue="context_observation",style="context_observation",marker="o")
    
    fig,ax = plt.subplots(1,2,figsize=(6,3))
    df = dataframe.copy()

    train_blocks = experiment_config["meta_data"]["training_blocks"]
    block = train_blocks
    g = sns.countplot(data=df.query(f"trial_type==0 & block == {block-1}"),x="optimal_policy",hue="context_observation",ax=ax[0])
    g.yaxis.set_major_locator(ticker.MultipleLocator(3))
    g = sns.countplot(data=df.query(f"trial_type==1 & block == {block}"),x="optimal_policy",hue="context_observation",ax=ax[1])
    g.yaxis.set_major_locator(ticker.MultipleLocator(3))


# %%
