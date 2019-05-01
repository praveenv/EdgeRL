import numpy as np
import itertools
import math

from collections import defaultdict, OrderedDict
from quality_env import Environment
from quality_sarsa import sarsa
from quality_ml_model import generate_ml_models,comparison
from quality_dqn import DQNAgent

from sklearn.model_selection import KFold

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
data_path = "../Data/water_quality_data_1.csv"
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



kf = KFold(n_splits=10)
split_count = 0
agent_avg_accuracy = []
fine_avg_accuracy = []
coarse_avg_accuracy = []

agent_avg_tp = []
agent_avg_tn = []
agent_avg_fp = []
agent_avg_fn = []

fine_avg_tp = []
fine_avg_tn = []
fine_avg_fp = []
fine_avg_fn = []

coarse_avg_tp = []
coarse_avg_tn = []
coarse_avg_fp = []
coarse_avg_fn = []


result_stats = np.array(result_stats)
truth_stats = np.array(truth_stats)
option_chosen_stats = np.array(option_chosen_stats)


final_stats = np.column_stack((result_stats,truth_stats,option_chosen_stats))

np.savetxt("final_stats.csv",final_stats,delimiter=",")


for train_index,test_index in kf.split(result_stats):
	test_index = np.array(test_index)
	split_count = split_count + 1
	validation_result = result_stats[(test_index)]
	validation_truth = truth_stats[(test_index)]
	# validation_option = option_chosen_stats[test_index]

	test_data = data[(test_index)]
	validation_length = len(test_index)

	TP_count = 0
	TN_count = 0
	FP_count = 0
	FN_count = 0
	for i in range(0,validation_length):
		if validation_result[i] == 1 and validation_truth[i] == 1:
			TP_count += 1
		elif validation_result[i] == 0 and validation_truth[i] == 0:
			TN_count += 1
		elif validation_result[i] == 1 and validation_truth[i] == 0:
			FP_count += 1
		elif validation_result[i] == 0 and validation_truth[i] == 1:
			FN_count += 1
		# current_option = validation_option[i]
		# if current_option == 1:
		# 	validation_cost += ((7 * 0.00412) + 1.5)
		# elif current_option == 0:
		# 	validation_cost += ((2 * 0.00412) + 1)

	fine_accuracy, coarse_accuracy,fine_tp,fine_tn,fine_fp,fine_fn,coarse_tp,coarse_tn,coarse_fp,coarse_fn = comparison(test_data)
	print("ACCURACY STATS FOR SPLIT NUMBER", split_count)
	print TP_count
	print TN_count
	print FP_count
	print FN_count
	print validation_length

	agent_avg_tp.append(TP_count)
	agent_avg_tn.append(TN_count)
	agent_avg_fp.append(FP_count)
	agent_avg_fn.append(FN_count)

	fine_avg_tp.append(fine_tp)
	fine_avg_tn.append(fine_tn)
	fine_avg_fp.append(fine_fp)
	fine_avg_fn.append(fine_fn)

	coarse_avg_tp.append(coarse_tp)
	coarse_avg_tn.append(coarse_tn)
	coarse_avg_fp.append(coarse_fp)
	coarse_avg_fn.append(coarse_fn)

	accuracy = float(float(TP_count + TN_count) / float(validation_length))
	print accuracy
	print fine_accuracy
	print coarse_accuracy
	agent_avg_accuracy.append(accuracy)
	fine_avg_accuracy.append(fine_accuracy)
	coarse_avg_accuracy.append(coarse_accuracy)


print "AVERAGE ACCURACIES AFTER SPLITS"
print np.mean(agent_avg_accuracy)
print np.mean(fine_avg_accuracy)
print np.mean(coarse_avg_accuracy)


print "AVERAGE TRUE POSITIVES"
print np.mean(agent_avg_tp)
print np.mean(fine_avg_tp)
print np.mean(coarse_avg_tp)

print "AVERAGE TRUE NEGATIVES"
print np.mean(agent_avg_tn)
print np.mean(fine_avg_tn)
print np.mean(coarse_avg_tn)

print "AVERAGE FALSE POSITIVES"
print np.mean(agent_avg_fp)
print np.mean(fine_avg_fp)
print np.mean(coarse_avg_fp)

print "AVERAGE FALSE NEGATIVES"
print np.mean(agent_avg_fn)
print np.mean(fine_avg_fn)
print np.mean(coarse_avg_fn)

# Beginning of comments
# # print Q
# print "Q TABLE"
# for k, v in Q.iteritems():
# 	for a,b in v.items():
# 		print k,a,b

# print "STATE CACHE"
# for idx, v in enumerate(state_cache):
# 	print idx, v.sensor, v.analytic, v.external

# print "ACTION PLAN"
# print action_plan_dict

# validation_length = len(test_data)
# # print "here"
# # print result_stats
# # print truth_stats
# validation_result = result_stats[-validation_length:]
# validation_truth = truth_stats[-validation_length:]
# validation_option = option_chosen_stats[-validation_length:]
# # print validation_result
# # print validation_truth

# TP_count = 0
# TN_count = 0
# FP_count = 0
# FN_count = 0
# validation_cost = 0
# # print validation_length
# for i in range(0,validation_length):
# 	# print i
# 	if validation_result[i] == 1 and validation_truth[i] == 1:
# 		TP_count += 1
# 	elif validation_result[i] == 0 and validation_truth[i] == 0:
# 		TN_count += 1
# 	elif validation_result[i] == 1 and validation_truth[i] == 0:
# 		FP_count += 1
# 	elif validation_result[i] == 0 and validation_truth[i] == 1:
# 		FN_count += 1
# 	current_option = validation_option[i]
# 	if current_option == 1:
# 		validation_cost += ((7 * 0.00412) + 1.5)
# 	elif current_option == 0:
# 		validation_cost += ((2 * 0.00412) + 1)

# fine_accuracy, coarse_accuracy = comparison(test_data)

# print "final"
# print TP_count
# print TN_count
# print FP_count
# print FN_count
# print validation_length

# accuracy = float(float(TP_count + TN_count) / float(validation_length))
# print accuracy
# print fine_accuracy
# print coarse_accuracy

# fine_cost = ((7 * 0.00412) + 1.5) * validation_length
# coarse_cost = ((2 * 0.00412) + 1) * validation_length
# print validation_cost
# print fine_cost
# print coarse_cost