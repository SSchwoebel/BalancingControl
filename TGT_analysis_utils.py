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
            axes[ri,ci].set_ylim([0,1.05])
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