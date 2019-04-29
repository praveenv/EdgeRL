import numpy as np
import itertools
import math

from collections import defaultdict, OrderedDict
from video_env import Environment
from video_dqn import DQNAgent
from video_sarsa import sarsa


action_plan_dict = { '1' : {'sensor' : {'motion' : 1, 'camera' : 0, 'rain_gauge' : 1}, 'analytic' : {'coarse' : 1, 'fine' : 0}},
					'2' : {'sensor' : {'motion' : 1, 'camera' : 1, 'rain_gauge' : 1}, 'analytic' : {'coarse' : 1, 'fine' : 1}}
					}

action_plan_rules = { '1' : ['1','2'], '2' : ['1','2']}
action_space = {'1','2'}


# set up the reinforcement learning environment
env = Environment(action_plan_dict,action_plan_rules,action_space)

agent = DQNAgent(1,2)

data_path = "./Data/day1.csv"

data = np.genfromtxt(data_path,delimiter=',',dtype=None,skip_header=1)

state_cache = []
Q, policy, result_stats, truth_stats, option_chosen_stats = sarsa(env,agent,len(data),data,state_cache)