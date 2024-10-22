"""
This file creates the trial configuration for different variants of the Two Galaxies Task.
Outputs a .json file with following information:

context   : what is the context of the given trial
sequence  : what is the optimal policy for the given trial
starts    : where is the rocket initialized on a given trial
planets   : what is the planet constellation for the given trial
exp_reward: what is the expected reward for the given trial if one executes the optimal policy specified in "sequence"
trial_type: is the trial part of the training, degradation or extinciton phase
block     : which trial block is the trial in 
degradation _blocks: how many degradation blocks there are in the task
training_blocks    : how many training blocks there are
contingency_degradation: whether a contingency switch happened during the degradation phase
switch_cues            : whether context obsevation mapping was switched during contingency degradation
trials_per_block       : how many trials are in an experimental block. The number of trials per context in a block is trials_per_block / number of contexts
blocked                : whether context presentation was blocked or pseudo randomized i.e. if trials from different contexts were presented in ordered "miniblocks", 
(for example 3 trials from context 1, followed by 3 trials from context 2, then by 3 trials from context 1 again etc). Alternative is for trials to be ramdonly presented within a block. 
miniblock_size         : the size of context presentation miniblock (look at description of "blocked", in this example miniblock_size = 3 )
seed                   : seed used for trial generation
nr                     : how many rewards could be received
state_transition_matrix: p(state_t|state_{t-1},action)
reward_contingencies   : p(reward|state) for the training and contingency degradation phases


Experimental setup parameters that can be varied in this script are: 

degradation _blocks: how many degradation blocks there are in the task
training_blocks    : how many training blocks there are
trials_per_block    : how many trials are in an experimental block. The number of trials per context in a block is trials_per_block / number of contexts

contingency_degradation: whether a contingency switch happened during the degradation phase
switch_cues            : whether context obsevation mapping was switched during contingency degradation
blocked                : whether context presentation was done in "miniblocks" or randomly 
miniblock_size         : the size of context presentation miniblock (look at description of "blocked", in this example miniblock_size = 3 )
nr                     : how many rewards could be received
state_transition_matrix: p(state_t|state_{t-1},action)
reward_contingencies   : p(reward|state) for the training and contingency degradation phases
"""

#########  IMPORTS  ################

import numpy as np
from itertools import product
import json  as js
import pandas as pd
import os
from misc import save_file, load_file

############ FUNCTION DEFINITIONS ###############

def create_state_transition_matrix(ns, na, targets):
    ''' 
    creates state transition matrix p( s_t | s_t-1 , a )
    ns = number of possibles states, na = number of possible actions
    targets = p( s_t| s_t-1 = j, a = i) 
    '''
    assert na == targets.shape[0], "length of target planets should match number of possible planets"

    state_transition_matrix = np.zeros([ns,ns,na])

    for action in range(na):
        target = targets[action]
        for r, row in enumerate(state_transition_matrix[:,:,action]):
            row[target[r]] = 1

    return state_transition_matrix

def sequence_of_length_n(r,n):
    ''' 
    creates all permutations of a vector r with length n and k unique digits
    for r = [0,1] and length of sequence n = 3
    000 001 010 011 100 101 etc 
    '''
    r = [r]*n
    return np.asarray(list(product(*r)))


def calculate_trial_policy_expectation(conf, exp_reward, stm, policy):
    '''
    function calculating the expected trial reward for all possible
    starting positions, given a planet constelation and policy.

    conf = planet configuration
    exp_reward = expected reward of planet
    stm = state transition matrix
    policy = action policy such as jump jump move
    '''

    exp_reward = exp_reward[conf]   # extract reward probabilities for planets
    exp_reward = np.repeat(exp_reward.reshape(1, exp_reward.shape[0]), repeats = stm.shape[0], axis=0)
    
    # path holds position of rocket after each action; different rows correspond to different starting positions    
    path = np.zeros([stm.shape[0], stm.shape[1], 3])

    path[:,:,0] = stm[:,:,policy[0]]
    for action in range(len(policy)-1):
        path[:,:,action+1] = path[:,:,action].dot(stm[:,:,policy[action+1]])

    expectation = np.zeros([stm.shape[0], stm.shape[1]])
    for action in range(3):
        expectation += path[:,:,action]*exp_reward            

    expectation = expectation.sum(axis=1)

    return expectation


def create_all_possible_trials(expected_rewards, nr=3, n_planet_positions=6,pol_ind=None, policies=None,state_transition_matrix=None):
    """
    This function creates all possible trials for a given planet-reward mapping,
    number of planets in a configuration and number of actions and time points.
    
    It also removes all trial configurations that have more than one best optimal policy

    @Parameters:
        - expected_rewards: the expected reward of each planet type
    @Output:
        - slices: a list of length equal to the total number of policies. Each list entry
            holds a dataframe with all the trials where that policy is optimal
        - planet_confs: a numpy array that holds all possible trial conformations
    """
    ns = n_planet_positions

    if nr == 3:
        planet_confs = sequence_of_length_n([0,1,2],ns)
    elif nr==2:
        planet_confs = sequence_of_length_n([0,1],ns)

    planet_confs = planet_confs[1:-1]                                        # generate all planet constelations and remove first and last constellation
    expectations = np.zeros([ns, pol_ind.shape[0], planet_confs.shape[0]])   # array holding trial expected reward for every possible trial, starting position and policy
    
    for ci, conf in enumerate(planet_confs):
        for m, policy in enumerate(policies):
            expectations[:, m, ci] = calculate_trial_policy_expectation(conf, expected_rewards, state_transition_matrix, policy)
    
    col_names = ['conf_ind', 'planet_conf_ind', 'planet_conf', 'start', 'policy', 'expected_reward']
    
    s = 0                                                          # unique trial configuration index
    rr = 0                                                         # row index
    nrows = planet_confs.shape[0]*ns*policies.shape[0]             # rows the dataframe will have
    data_array = np.empty([nrows, len(col_names)],dtype=object)    # initialized as object in order to be able to add planet conf as array

    for ci, conf in enumerate(planet_confs):
        for st in np.arange(ns):
            for m, policy in enumerate(policies):
                data_array[rr,:] = [s, ci, (conf), st, m, expectations[st,m,ci]]
                rr += 1
            s += 1

    data = pd.DataFrame(data_array, columns = col_names)
    datatype = {'conf_ind':int, 'start':int, 'planet_conf_ind':int, 'policy':int, 'expected_reward':float}
    data = data.astype(datatype)

    data['max_reward'] = data.groupby('conf_ind')[['expected_reward']].transform('max')   # calculate max reward from all possible policies for a given trial
    data = data.round(3)                                                                  # round the result so that == operation gives correct result
    data['optimal'] = data['max_reward'] == data['expected_reward']                       # define optimal policies
    data['total_optimal'] = data.groupby(['conf_ind'])[['optimal']].transform('sum')      # count optimal policies
    data = data.drop(data[data.total_optimal != 1].index)                                 # drop all configurations that have more than 1 optimal policy

    slices = [None for pi in range(len(policies))]
    for pi in pol_ind:
        slice = data.loc[( data['optimal'] == True) & ( data['policy'] == pi)]
        slices[pi] = slice

    return slices


def create_trials_for_both_contingencies(extend=False, seed = 1,n_planet_positions = 6, n_reward_contingencies=2,
                                         contingency_exp_rewards=None, pol_ind=None,policies=None, nr=3, stm=None):

    '''
    Creates "data" which holds all trial information for a given experiment
    Data is a list of of length number of policies. Each entry holds a list of matrices with all trials where a given policy
    is optimal under a given contingency. If extended is set to true, each matrix holds multiple copies of its trials. 
    This is necessarry when having an experiment with many trials, where all possible trials are used atleast once. 
    
    The rows of the matrix hold info for a single trial and the columns the trial information:
        which policy is optimal in this trial - column 1
        starting position of rocket - column 2
        planet constellation - columns 3 to 8
        expected reward for trial if optimal policy chosen - column 9
    @Parameters:
        extended: whether to copy created trials 5 times in data. Set to True if running out of trials in "create_trials_planning" 
    @Output:
       data
    '''
    
    np.random.seed(seed)
    ns = n_planet_positions

    slices = [None for nn in range(n_reward_contingencies)]          # array that holds dataframes with all trials for all contingencies
    for ri in range(n_reward_contingencies):
        slices[ri] = create_all_possible_trials(contingency_exp_rewards[ri], nr=nr, n_planet_positions=n_planet_positions,
                                                pol_ind=pol_ind, policies=policies, state_transition_matrix=stm)

    data = [[] for i in range(len(pol_ind))]                         # array that holds trial information, separated in terms of which policy is optimal 

    for pol in pol_ind:
        for ci, trials_for_given_contingency in enumerate(slices):

            # create trial matrix dt
            slice = trials_for_given_contingency[pol]                # extact all trials where pol is optimal
            ntrials = slice.shape[0]                                 # count how many there are
            dt = np.zeros([slice.shape[0], 3 + ns])

            # populate trial matrix dt
            plnts = np.array(slice.planet_conf.to_list())            # planets constellations where policy pol is optimal
            strts = slice.start.to_list()                            # starting points where policy pol is optimal
            expected_reward = slice.expected_reward.to_list()        # expected reward for that trial
            
            dt[:,0] = [pol]*ntrials                                  # optimal sequence index
            dt[:,1] = strts                                          # trial starting position
            dt[:,2:-1] = plnts                                       # planets
            dt[:,-1] = expected_reward                               # expected_reward
            np.random.shuffle(dt)                                    # shuffle trials
            
            # this is rewritten and untested!
            if extend:
                dt = np.vstack([dt for i in range(5)])
            data[pol].append(dt)

    return data


def create_trials_planning(data,                                         # all possible trials where a given policy is optimal
                           habit_pol = 3,                                # which policy is being habituated
                           switch_cues= False,                           # will context cue mapping switch
                           training_blocks = 2,                          # how many training blocks (with original reward contingency)
                           degradation_blocks = 1,                       # how many degradation blocks (with switched reward contingency)
                           extinction_blocks = 2,                        # how many extinction blocks
                           trials_per_block = 28,                        # how many trials per experimental block
                           blocked=False,                                # whether context presentation order is blocked
                           block = None,                                 # context block size
                           seed=1,
                           pol_ind=None,
                           habit_pol_1 = 3,
                           habit_pol_2 = 6): 

    np.random.seed(seed)

    assert trials_per_block % 2 == 0, 'Please set an even number of trials per block!'

    nblocks = training_blocks + degradation_blocks + extinction_blocks     # how many blocks of trials in experiment
    half_block = trials_per_block//2                                       # how many trials of each context (habit vs planing)
    ntrials = nblocks*trials_per_block//2                                  # how many trials PER CONTEXT
    ncols = data[0][0].shape[1]                                            # how many columns in final config matrix "trials"

    trials = np.zeros([nblocks*trials_per_block, ncols])                   # final config matrix which holds all experimental data
    trial_type = np.zeros(nblocks*trials_per_block)                        # whether training, degradation or extinction
    context = np.zeros(nblocks*trials_per_block)                           # what is the true context
    context_observation = np.zeros(nblocks*trials_per_block)               # which cue is shown
    blocks = np.zeros(nblocks*trials_per_block)                            # which (experimental) block is this trial in
    
    if block is not None:
        miniblocks = half_block//block                                     # how many context blocks (called miniblocks) are in an experimental trial block


    # split data for each contingency and policy into list for all possible expected rewards
    # this way the expected reward between habit and planning trials can be matched

    unique_exp_rew = np.unique(data[0][0][:,-1])                                        # all possible expected rewards for optimal policy that are shared by all policies
    probs = (np.arange(unique_exp_rew.size)+1)/(np.arange(unique_exp_rew.size)+1).sum() # weight probablity of drawing trial by rewards it can give
    rewards = np.random.choice(np.arange(unique_exp_rew.size), p=probs, size=ntrials)   # draw expected reward for half of the trials, to ensure that the earned points between cotnexts are matched
    planning_seqs = pol_ind[pol_ind != habit_pol]
    planning_seqs = np.tile(planning_seqs, half_block // planning_seqs.size)

    split_data = [[None for contingency in range(2)] for policy in pol_ind]            # list holding data aranged in terms of optimal policy, contingency and exp_reward

    for policy in range(len(data)):
        for contingency in range(len(data[policy])):

            dat = data[policy][contingency]
            split_data[policy][contingency] = [dat[dat[:,-1]==k] for k in unique_exp_rew]
            for k in range(unique_exp_rew.size):
                np.random.shuffle(split_data[policy][contingency][k])


    """
    Populate trial matrix
    THIS ALL STILL NEEDS TO BE DOUBLE CHECKED!
    """
    for i in range(nblocks):                                                    # iterate over all blocks (training, degradation and extinction)

        # set trial type (tt) indicating experimental phases and active contingency
        if i < training_blocks:
            tt = 0
            contingency = 0
        elif i >= training_blocks and i < training_blocks + degradation_blocks:
            tt = 1                                                         
            contingency = 1                                  
        else:
            contingency = 0
            tt = 2
            

        if not blocked:                                                                   # if context presentation NOT grouped
            
            # populate habit seq trials
            for t in range(half_block):                                                   # half-block perspective index (0:half-black)
                tr = int(i*trials_per_block + t)                                          # whole experiment persepctive index (0:total number of trials)
                ri = half_block*i + t                                                     # possible trial perspective index (pulls out elements from split_data)
                trials[tr,:] = split_data[habit_pol][contingency][rewards[ri]][0]
                split_data[habit_pol][contingency][rewards[ri]] = \
                    np.roll(split_data[habit_pol][contingency][rewards[ri]], -1, axis=0)

            # populate planning trials
            for t in range(half_block):
                tr = int(i*trials_per_block + half_block + t)
                ri = half_block*i + t
                trials[tr,:] = split_data[planning_seqs[t]][contingency][rewards[ri]][0]
                split_data[planning_seqs[t]][contingency][rewards[ri]] = \
                    np.roll(split_data[planning_seqs[t]][contingency][rewards[ri]], -1, axis=0)

            #shuffle trials
            np.random.shuffle(trials[i*trials_per_block:(i+1)*trials_per_block,:])

        if blocked:                                                                        # if context presentation blocked


            # pick planing and habit trials with same expected reward
            habit_trials = np.zeros([half_block, trials.shape[1]])
            planning_trials = np.zeros([half_block, trials.shape[1]])
            
            trial_shift = i*trials_per_block + half_block
            reward_shift = half_block*i

            for t in range(half_block):
                tr = trial_shift + t
                ri = reward_shift + t

                habit_trials[t,:] = split_data[habit_pol][contingency][rewards[ri]][0]
                split_data[habit_pol][contingency][rewards[ri]] = \
                    np.roll(split_data[habit_pol][contingency][rewards[ri]], -1, axis=0)
            

                planning_trials[t,:] = split_data[planning_seqs[t]][contingency][rewards[ri]][0]
                split_data[planning_seqs[t]][contingency][rewards[ri]] = \
                    np.roll(split_data[planning_seqs[t]][contingency][rewards[ri]], -1, axis=0)
                

            # shuffle planning trials so that they do not have the same order of optimal sequence within and between blocks
            np.random.shuffle(planning_trials)

            # populate block with grouped context presentation (habit first and then context)
            for mb in range(miniblocks):
                tr = int(i*trials_per_block)
                trials[tr+2*mb*block:tr+(2*mb+1)*block] = habit_trials[mb*block:(mb+1)*block,:]
                trials[tr+(2*mb+1)*block:tr+(2*mb+2)*block] = planning_trials[mb*block:(mb+1)*block,:]
            

        trial_type[i*trials_per_block:(i+1)*trials_per_block] = int(tt)
        blocks[i*trials_per_block:(i+1)*trials_per_block] = int(i)

        context_observation[i*trials_per_block:(i+1)*trials_per_block] = trials[i*trials_per_block:(i+1)*trials_per_block,0] != habit_pol
        
        context[i*trials_per_block:(i+1)*trials_per_block] = context_observation[i*trials_per_block:(i+1)*trials_per_block]
        if tt == 1:
            context[i*trials_per_block:(i+1)*trials_per_block] += 2

        if switch_cues and tt == 1:
            context_observation[i*trials_per_block:(i+1)*trials_per_block] = trials[i*trials_per_block:(i+1)*trials_per_block,0] == habit_pol

    
    # create data frame and save file
    trial_type = trial_type.astype('int32')
    
    config = {
                    "experiment_data":{
                        'context' : context.astype('int32'),
                        'context_observation': context_observation.astype('int32'), 
                        'optimal_policy': trials[:,0].astype('int32'),
                        'starts': trials[:,1].astype('int32'),
                        'planets': trials[:,2:-1].astype('int32'),
                        'exp_reward': trials[:,-1],
                        'trial_type': trial_type,
                        'block': blocks.astype('int32'),
                    }
             }
    
    return config


def generate_experiment(na=2,
                        T=4,
                        n_planet_positions=6,
                        unique_rewards= None,
                        n_reward_contingencies=2,
                        state_transition_matrix=None,
                        planet_reward_probs=None,
                        planet_reward_probs_switched=None,
                        grouped_context_presentation= True,
                        group_size=5,
                        switch_context_cues_during_degradation=False,
                        same_habit_during_training_and_degradation=True,
                        habit_sequence_1=3,
                        habit_sequence_2=None,
                        trials_per_block=42,
                        training_blocks=4,
                        degradation_blocks=4,
                        extinction_blocks=4,
                        seed=3,
                        path=None,
                        fname=None):

    # calculate some important shared quantities
    nr = unique_rewards.size
    policies = sequence_of_length_n(np.arange(2),T-1)
    pol_ind = np.arange(policies.shape[0])
    exp_planet_reward = planet_reward_probs.T.dot(unique_rewards)
    exp_planet_reward_switched = planet_reward_probs_switched.T.dot(unique_rewards)
    contingency_exp_rewards = np.array([exp_planet_reward,exp_planet_reward_switched])

    # create all possible trial for all possible optimal policies
    trials = create_trials_for_both_contingencies(seed=seed,n_planet_positions=n_planet_positions,
                                                  n_reward_contingencies=n_reward_contingencies,
                                                  contingency_exp_rewards=contingency_exp_rewards,
                                                  pol_ind=pol_ind,policies=policies,nr=nr,stm=state_transition_matrix)


    # create specific set of trials for the experiment
    config = create_trials_planning(trials,
                                    switch_cues=switch_context_cues_during_degradation,
                                    training_blocks=training_blocks,
                                    degradation_blocks=degradation_blocks,
                                    extinction_blocks=extinction_blocks,
                                    trials_per_block=trials_per_block,
                                    blocked = grouped_context_presentation,
                                    block = group_size,
                                    habit_pol_1 = habit_sequence_1,
                                    habit_pol_2 = habit_sequence_2,
                                    pol_ind = pol_ind,
                                    seed = seed,
                                    )

    # populate experiment meta data
    config["meta_data"] = {
                        'degradation_blocks': degradation_blocks,
                        'training_blocks': training_blocks,
                        'extinction_blocks':extinction_blocks,
                        'switch_cues': switch_context_cues_during_degradation,
                        'trials_per_block': trials_per_block,
                        'context_presentation_blocked': grouped_context_presentation,
                        'miniblock_size' : group_size,
                        'reward_gen_seed':seed,
                        'contingency_exp_reward': [exp for exp in contingency_exp_rewards],
                        'n_reward_contingencies' : n_reward_contingencies
                    }
    
    # populate agent simulation relevant data
    config["agent_data"]={
                        'true_reward_contingencies': [planet_reward_probs, planet_reward_probs_switched],
                        'generative_model_states':state_transition_matrix.transpose((1,0,2)),
                        'T' : T,
                        'trials': int((training_blocks+degradation_blocks+extinction_blocks)*trials_per_block),
                        'na': na,
                        'npi': policies.shape[0],
                        'nm': n_planet_positions,
                        'nh' : planet_reward_probs.shape[1],
                        'nr': nr,
                        'all_rewards': unique_rewards,
                        'all_policies': policies,
                    } 

    # create config fname
    if fname is not None:
        fname += '_experiment_config.json'
    else:
        fname = (
                f"exp_config_switch{int(switch_context_cues_during_degradation)}"
                f"_context_blocked{int(grouped_context_presentation)}"
                f"_same_habit{int(same_habit_during_training_and_degradation)}"
                f"_{training_blocks}{degradation_blocks}{extinction_blocks}_{trials_per_block}_nr{nr}.json"
        )
    # save config file
    fname = os.path.join(path, fname)
    print('created: ', fname)
    save_file(config,fname)

    return fname

    


                
        






















#%%

# """FILE SANITY CHECK"""


# fname = "/home/terra/Nextcloud2/projects/BayesTwoGalaxiesTask/simulations/experimental_design/config/blocked/planning_config_degradation_1_switch_0_train4_degr4_n70_nr_3.json"

# config = load_file(fname)
# config['experiment_data']['planets'] = config['experiment_data']['planets'].tolist() 
# df = pd.DataFrame.from_dict(config['experiment_data'])

# exp_rewards = df.groupby(['block','context']).mean('exp_reward')['exp_reward']
# print(df.columns)
# print(df.shape)
# print()
# print(exp_rewards)

# # %%
