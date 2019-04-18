import numpy as np
import itertools
import math

from collections import defaultdict, OrderedDict
from quality_env import Environment
from quality_sarsa import sarsa
from quality_ml_model import generate_ml_models,comparison
from quality_dqn import DQNAgent


# define the action plan IDs and the transition rules
action_plan_dict = { '1' : {'sensor' : {'do2' : 0, 'turbidity' : 1, 'tds' : 0, 'ph' : 0, 'conductivity' : 0, 'rain_gauge' : 1}, 'analytic' : {'coarse' : 1, 'fine' : 0}},
					'2' : {'sensor' : {'do2' : 1, 'turbidity' : 1, 'tds' : 1, 'ph' : 1, 'conductivity' : 1, 'rain_gauge' : 1}, 'analytic' : {'coarse' : 1, 'fine' : 1}}
					}

action_plan_rules = { '1' : ['1','2'], '2' : ['1','2']}
action_space = {'1','2'}

# set up the reinforcement learning environment
env = Environment(action_plan_dict,action_plan_rules,action_space)

agent = DQNAgent(1,2)
# read and setup data for ML model training as well as for the RL environment
data_path = "../Data/water_quality_data.csv"
ml_train_path = "../Data/ML_training_indices.csv"
non_ml_train_path = "../Data/non_ML_training_indices.csv"

data = np.genfromtxt(data_path,delimiter=',').astype(int)
ml_train_indices = np.genfromtxt(ml_train_path,delimiter=',').astype(int)
ml_train_indices = np.delete(ml_train_indices,0)
non_ml_train_indices = np.genfromtxt(non_ml_train_path,delimiter=',').astype(int)
non_ml_train_indices = np.delete(non_ml_train_indices,0)

ml_train_data = data[ml_train_indices]
data = data[non_ml_train_indices]


# Call function to train your ML models on the data

generate_ml_models(ml_train_data)

# Call the RL function
training_limit = int(math.ceil(0.7 * len(data)))
training_data = data[0:training_limit]
test_data = data[(training_limit+1):len(data)]

state_cache = []

Q, policy, result_stats, truth_stats, option_chosen_stats = sarsa(env,agent,len(data),data,state_cache)


# print Q
print "Q TABLE"
for k, v in Q.iteritems():
	for a,b in v.items():
		print k,a,b

print "STATE CACHE"
for idx, v in enumerate(state_cache):
	print idx, v.sensor, v.analytic, v.external

print "ACTION PLAN"
print action_plan_dict

validation_length = len(test_data)
# print "here"
# print result_stats
# print truth_stats
validation_result = result_stats[-validation_length:]
validation_truth = truth_stats[-validation_length:]
validation_option = option_chosen_stats[-validation_length:]
# print validation_result
# print validation_truth

TP_count = 0
TN_count = 0
FP_count = 0
FN_count = 0
validation_cost = 0
# print validation_length
for i in range(0,validation_length):
	# print i
	if validation_result[i] == 1 and validation_truth[i] == 1:
		TP_count += 1
	elif validation_result[i] == 0 and validation_truth[i] == 0:
		TN_count += 1
	elif validation_result[i] == 1 and validation_truth[i] == 0:
		FP_count += 1
	elif validation_result[i] == 0 and validation_truth[i] == 1:
		FN_count += 1
	current_option = validation_option[i]
	if current_option == 1:
		validation_cost += (7 + 1.5)
	elif current_option == 0:
		validation_cost += (2 + 1)

fine_accuracy, coarse_accuracy = comparison(test_data)

print "final"
print TP_count
print TN_count
print FP_count
print FN_count
print validation_length

accuracy = float(float(TP_count + TN_count) / float(validation_length))
print accuracy
print fine_accuracy
print coarse_accuracy

fine_cost = 8.5 * validation_length
coarse_cost = 3 * validation_length
print validation_cost
print fine_cost
print coarse_cost