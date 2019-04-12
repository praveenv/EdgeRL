import numpy as np
import itertools

from collections import defaultdict, OrderedDict
from quality_env import Environment
from quality_sarsa import sarsa




# define the action plan IDs and the transition rules
action_plan_dict = { '1' : {'sensor' : {'do2' : 0, 'turbidity' : 1, 'tds' : 0, 'ph' : 0, 'conductivity' : 0, 'rain_gauge' : 1}, 'analytic' : {'coarse' : 1, 'fine' : 0}},
					'2' : {'sensor' : {'do2' : 1, 'turbidity' : 1, 'tds' : 1, 'ph' : 1, 'conductivity' : 1, 'rain_gauge' : 1}, 'analytic' : {'coarse' : 1, 'fine' : 1}}
					}

action_plan_rules = { '1' : ['1','2'], '2' : ['1','2']}
action_space = {'1','2'}

# set up the reinforcement learning environment
env = Environment(action_plan_dict,action_plan_rules,action_space)


# read and setup data for ML model training as well as for the RL environment
data_path = "./Data/water_quality_data.csv"
ml_train_path = "./Data/ML_training_indices.csv"
non_ml_train_path = "./Data/non_ML_training_indices.csv"

data = np.genfromtxt(data_path,delimiter=',').astype(int)
ml_train_indices = np.genfromtxt(ml_train_path,delimiter=',').astype(int)
ml_train_indices = np.delete(ml_train_indices,0)
non_ml_train_indices = np.genfromtxt(non_ml_train_path,delimiter=',').astype(int)
non_ml_train_indices = np.delete(non_ml_train_indices,0)

ml_train_data = data[ml_train_indices]
data = data[non_ml_train_indices]

# Call the RL function

state_cache = []

Q, policy = sarsa(env,len(data),data,state_cache)