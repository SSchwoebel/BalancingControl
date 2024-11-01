#%%
###
import sys
import os
import numpy as np
import copy 


###
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


from matplotlib.animation import FuncAnimation
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
            axes[ri,ci].set_ylim([0,1])
            axes[ri,ci].set_title(fr"$\rho$ = {rho}, $\alpha_0$ = {alpha}")

    fig.suptitle(fr"Effect of self-transition bias $\rho$ on mean choice accuracy.",y=1.1);

    return fig


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

    fig, ax = plt.subplots(1,2,figsize=(7,3))
    
    for cue in range(2):
        df = context.query(f"alpha_0 == {alpha_0} & dec_temp=={dec_temp} & context_trans_prob=={context_trans_prob} & context_cue=={cue} & t=={t}")
        plt.figure()
        g = sns.lineplot(ax=ax[cue],data=df, x="block", y="inferred_correct_context",hue="file",palette="viridis")
        g.set_xticks(np.arange(13));
        g.set_ylim([0,1.2])
        g.set_title(f"context_cue = {cue}")

def plot_average_DKL(rho, dataframe, simulation_params):
    
    df = dataframe.copy().query(f"t == 0")
    
    fig,axes = plt.subplots(1,4,figsize=(13,3))
    plt.tight_layout()
    for context,palette in zip([0,1,2,3],["Blues_r","Reds_r"]*2):
        sns.lineplot(ax=axes[context], data=df.query(f"context_trans_prob == {rho}"), x="trial",y=f"dkl_{context}",hue="alpha_0",errorbar="sd", palette=palette)
        # axes[context].set_xticks(np.arange(1,510,5))
        # axes[context].set_xticklabels(np.arange(1,510,5), rotation=90,fontsize=8)  # Rotate x-tick labels
        axes[context].set_xlim([0, (simulation_params["training_blocks"] + simulation_params["degradation_blocks"])*simulation_params["trials_per_block"]])
        axes[context].set_title(fr"$\rho$ = {rho}")


def animate_histogram(data, interval=500):
        
    fig, ax = plt.subplots(1,1,figsize=(5,5))

    def update(frame):
        ax.clear()
        sns.heatmap(data[frame], annot=True, cmap="viridis", cbar=False, fmt='.2f',ax=ax)
        ax.set_title(frame+1)

    animation = FuncAnimation(fig, update, frames=data.shape[0], interval=interval)


    # return HTML(animation.to_jshtml())
    html = HTML(animation.to_jshtml())
    display(html)
    plt.close() # update
    

def animate_multiple_histograms(matrices, bins=10, interval=200):

    N = matrices[0].shape[0]
    n_matrices = len(matrices)
    
    # Create figure and axes for subplots
    fig, axes = plt.subplots(1, n_matrices, figsize=(3.5*n_matrices, 5))
    
    def update(frame):
        # Clear each axis for the new frame
        for ax, matrix in zip(axes, matrices):
            ax.clear()
            # sns.histplot(matrix[frame].flatten(), bins=bins, kde=False, ax=ax, color="blue")
            sns.heatmap(matrix[frame], annot=True, cmap="viridis", cbar=False, fmt='.2f',ax=ax)
            ax.set_title(frame % 6 + 1)
    # Create animation
    anim = FuncAnimation(fig, update, frames=N, interval=interval, repeat=True)

    html = HTML(anim.to_jshtml())
    display(html)
    plt.close() # u
    
    
def plot_heatmap(data,share_x=True, share_y=True):
    if not isinstance(data,list):
        data = [data]
    n_axes = len(data)
    
    fig, axes = plt.subplots(1,n_axes, figsize=(3*n_axes, 3),sharex=share_x, sharey= share_y)
    plt.tight_layout()
    if n_axes == 1:
        axes = np.array([axes])

    for ai in range(n_axes):
        sns.heatmap(data[ai], annot=True, cmap="viridis", cbar=False, fmt='.2f',ax=axes[ai]);
        
    return fig,axes


def plot_task_structure(experiment_config):
    
    df = load_task_df(experiment_config)

    fig, axes = plt.subplots(1,2,figsize=(11,3))
    

    for bi, block in enumerate([0,experiment_config["meta_data"]["training_blocks"]]):
        
        ax = axes[bi]
        ax.xaxis.set_major_locator(ticker.MultipleLocator(3))
        sns.lineplot(ax=ax, data=df.query(f"block == {block}"), x="index", y=df["context"] % 2,color='k',marker="o",label="habit=0 or planing=1 context")
        pivoted = df.query(f"block == {block}").pivot(index="index", columns="context_observation", values="optimal_policy")
        pivoted.plot(ax=ax, marker="o",label="optimal_policy")# sns.lineplot(data=df.query("block == 0 "), x="index",y="context")
        ax.set_title(f"block {block}")
        ax.legend(bbox_to_anchor=(1.1, 0.5))
        
        if bi == 0:
            ax.get_legend().remove()


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
        ax.set_xlabel(r"$s_{t-1}$")
        ax.set_ylabel(r"$s_{t}$")

def plot_expected_reward_and_optimal_policy(experiment_config):

    dataframe = load_task_df(experiment_config)
    fig, ax = plt.subplots(1,1,figsize=(3,3))
    df=dataframe.groupby(["block","context_observation"])["exp_reward"].mean().reset_index()
    print(df.dtypes)
    sns.lineplot(data=df, ax=ax, x="block",y="exp_reward",hue="context_observation",style="context_observation",marker="o")
    
    fig,ax = plt.subplots(1,2,figsize=(6,3))
    df = dataframe.copy()
    g = sns.countplot(data=df.query("trial_type==0 & block == 2"),x="optimal_policy",hue="context_observation",ax=ax[0])
    g.yaxis.set_major_locator(ticker.MultipleLocator(3))
    g = sns.countplot(data=df.query("trial_type==1 & block == 5"),x="optimal_policy",hue="context_observation",ax=ax[1])
    g.yaxis.set_major_locator(ticker.MultipleLocator(3))


