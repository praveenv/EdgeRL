import numpy as np
import itertools

from collections import defaultdict, OrderedDict
from quality_env import Environment
from quality_statedef import State
from tqdm import tqdm

def epsilon_greedy_policy(Q, epsilon):
    """
    Epsilon-greedy policy based on a given Q-function and epsilon

    Reference: Book RL_2018 Page#81

    Parameters
    ----------
    Q: a dictionary that maps from state to action values. 
    epsilon: probability to select a random action, float num between 0 and 1
    nA: number of actions in the environment

    Returns
    -------
    a policy function that takes state and valid actions as input and
        returns probabilities for each action in the form of an array of length nA

    """

    def policy_eg(state, action_cands):
        """

        Parameters
        ----------
        state: dictionary element that represents the current state {'sensor': {}, 'analytic':{}, 'external':{}}
        action_cands: action candidates - a list of potential actions that can be taken in current state

        Returns
        -------
        normalized action probabilities

        """

        assert len(action_cands) != 0
        ac_len = len(action_cands)

        # we initialize the Action probability vector A with (epsilon / |number of possible actions in this state|) 
        A = np.zeros(ac_len, dtype=float) + epsilon/ac_len
        

        sa_val = np.zeros(ac_len, dtype=float)
        state_Q = Q[state]


        max_v = -np.inf
        # with ties broken arbitrarily / randomly (RL-2018 Book Page#81)
        best_actions = []
        # iterate over possible actions in current state and identify their Q values
        for idx, ac in enumerate(action_cands):
            tmp_v = state_Q[ac]
            sa_val[idx] = tmp_v
        # Find the best possible Q value (i.e.) argmax Q(S,A)
            if tmp_v > max_v:
                max_v = tmp_v
                best_actions = [idx]
            elif tmp_v == max_v:
                best_actions.append(idx)
        # if multiple actions have same max Q value choose a random one to give more probability to 
        assert best_actions is not None
        A[np.random.choice(best_actions)] += (1.0 - epsilon)
        # Return Probability values and the associated Q value (state/action)
        return A, sa_val

    return policy_eg


def convert_to_dict(data):

	current_dict = {}
	current_dict['sensor'] = {}
	current_dict['sensor']['do2'] = 0
	current_dict['sensor']['turbidity'] = 1
	current_dict['sensor']['tds'] = 0
	current_dict['sensor']['conductivity'] = 0
	current_dict['sensor']['ph'] = 0
	current_dict['sensor']['rain_gauge'] = 1

	current_dict['analytic'] = {}
	current_dict['analytic']['coarse'] = 1
	current_dict['analytic']['fine'] = 0

	current_dict['external'] = {}
	current_dict['external']['timeOfDay'] = data[5]

	current_dict['sensor_reading'] = {}
	current_dict['sensor_reading']['do2'] = data[0]
	current_dict['sensor_reading']['turbidity'] = data[1]
	current_dict['sensor_reading']['tds'] = data[2]
	current_dict['sensor_reading']['conductivity'] = data[3]
	current_dict['sensor_reading']['ph'] = data[4]
	current_dict['sensor_reading']['rain_gauge'] = data[6]

	current_truth = data[7]
	return current_dict,current_truth

def parse_train_data(n_ep,data):
    """
    Function to parse the input training data in order to get the tuple that matches the current time step. 
    Parameters
    -----------
    n_ep : integer reflecting current episode number and the training data index that needs to be accessed. 
    data : input training data (one day's worth) of the form {'1':{'sensor':{}, 'analytic':{},'external':{},'truth':{}}} , an ordered Dictionary

    Returns
    --------
    current_actual_environment : State object that reflects the training data[n_ep], has sensor, analytic and external as attributes
    current_true_value : current ground truth value to be compared with the analytics
    """
    
    converted_data, ground_truth = convert_to_dict(data[n_ep])
    current_environment = State(converted_data['sensor'],converted_data['analytic'],converted_data['sensor_reading'],converted_data['external'])

    return current_environment,ground_truth
    # return current_actual_environment,current_true_value


def sarsa(env, n_episodes, data, state_cache, stats=None, alpha=0.5, epsilon=0.1, dis_factor=0.9):
    """
    Sarsa (on-policy TD control): find optimal epsilon-greedy policy

    Reference: RL-book Page#104

    Parameters
    ----------
    env: environment
    n_episodes: number of episodes
    data : Ordered Dictionary (Important!) of training data of the form {'1':{'sensor':{}, 'analytic':{},'external':{},'truth':{}}}
    stats: record episode statistics
    alpha: TD learning step/rate
    epsilon: probability to sample a random action
    dis_factor: discount factor

    Returns
    -------
    Q: action value function
    policy: updated greedy policy

    """

    n_actions = len(env.action_space)

    # initialize action-value function arbitrarily: a nested dictionary maps state -> {action -> action-value}
    Q = defaultdict(lambda: defaultdict(lambda: 0.0))

    # count # of appearing of a (state, action) pair
    sa_count = defaultdict(float)

    # policy we are learning NOTE it will be updated in following loop, since Q will be updated
    # This is an example of nested function in python - from now I will call policy(state,action)
    # and it will use the below definition of Q and epsilon without me having to include them as parameters all the time
    policy = epsilon_greedy_policy(Q, epsilon)

    # Set initial state of the agent using Training data[0] and obtain potential action candidates for that state
    # initial_environment and state are objects of the State class. 
    initial_environment, initial_true_value = parse_train_data(0,data)
    state, action_cands, state_cache = env.set_state(initial_environment,state_cache)

    store_result = []
    store_truth = []
    store_option_chosen = []

     # pick next action based on policy
    probs, _ = policy(state, action_cands)
    # choose next action using the probability values
    action = action_cands[np.random.choice(len(action_cands), p=probs)]

    #iterating over number of episodes
    for n_ep in tqdm(xrange(1,n_episodes)):
    	# print n_ep
        # take a step using above action and get next state, next action candidates and reward
        # first determine the ground truth of next environmental state and the ground truth value for the application
        next_environment , next_true_value = parse_train_data(n_ep,data)
        # then take a step with action and obtain reward by comparing to the ground truth of the next state
        next_state, next_action_cands, reward, state_cache, result, truth, option_chosen = env.step(action,next_environment,next_true_value,state,state_cache,data[n_ep-1])
        store_result.append(result)
        store_truth.append(truth)
        store_option_chosen.append(option_chosen)
        next_probs, _ = policy(next_state,next_action_cands)
        next_action = next_action_cands[np.random.choice(len(next_action_cands), p=next_probs)]

        td_target = reward + dis_factor * Q[next_state][next_action]
        td_error = td_target - Q[state][action]
        Q[state][action] += (alpha * td_error)

        state = next_state
        action = next_action

    return Q, policy, store_result, store_truth, store_option_chosen