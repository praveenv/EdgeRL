import numpy as np
import itertools
import math
import pandas as pd
import collections

from collections import defaultdict, OrderedDict,Counter
from video_env import Environment
from video_dqn import DQNAgent
from video_sarsa import sarsa

from sklearn.metrics import mean_squared_error
from math import sqrt

action_plan_dict = { '1' : {'sensor' : {'motion' : 1, 'camera' : 0, 'rain_gauge' : 1, 'opencv' : 0}, 'analytic' : {'coarse' : 1, 'medium' : 0, 'fine' : 0}},
					'2' : {'sensor' : {'motion' : 1, 'camera' : 0, 'rain_gauge' : 1, 'opencv' : 1}, 'analytic' : {'coarse' : 1, 'medium' : 0, 'fine' : 1}},
					'3' : {'sensor' : {'motion' : 1, 'camera' : 1, 'rain_gauge' : 1, 'opencv' : 0}, 'analytic' : {'coarse' : 1, 'medium' : 1, 'fine' : 0}}
					}

action_plan_rules = { '1' : ['1','2','3'], '2' : ['1','2','3'],'3' : ['1','2','3']}
action_space = {'1','2','3'}


# set up the reinforcement learning environment
env = Environment(action_plan_dict,action_plan_rules,action_space)

agent = DQNAgent(1,3)

data_path = "./Data/combined_days.csv"

data = pd.read_csv(data_path)
fine_grained = np.asarray(data.fine_grained)
coarse_grained = np.asarray(data.coarse_grained)
medium_grained = np.asarray(data.medium_grained)
rain_gauge = np.asarray(data.rain_gauge)
time_of_day = np.asarray(data.time_of_day)
data = np.column_stack((fine_grained,coarse_grained,medium_grained,rain_gauge,time_of_day))

print medium_grained

state_cache = []
Q, policy, result_stats, truth_stats, option_chosen_stats = sarsa(env,agent,len(data),data,state_cache)


training_length = 1440*5

motion_sensor_comparison = data[:,1]
motion_sensor_comparison = np.delete(motion_sensor_comparison,0)
fine_grained_comparison = data[:,0]
fine_grained_comparison = np.delete(fine_grained_comparison,0)
medium_grained_comparison = data[:,2]
medium_grained_comparison = np.delete(medium_grained_comparison,0)


dqn_mean_square = sqrt(mean_squared_error(result_stats,truth_stats))
print "final"
print dqn_mean_square

motion_mean_square = sqrt(mean_squared_error(motion_sensor_comparison,truth_stats))
print motion_mean_square

opencv_mean_square = sqrt(mean_squared_error(medium_grained_comparison,truth_stats))
print opencv_mean_square

# result_stats = np.array(result_stats[(training_length+1):len(result_stats)])
# truth_stats = np.array(truth_stats[(training_length+1):len(truth_stats)])
# motion_sensor_comparison = np.array(motion_sensor_comparison[(training_length+1):len(motion_sensor_comparison)])
# option_chosen_stats = np.array(option_chosen_stats[(training_length+1):len(option_chosen_stats)])
# medium_grained_comparison = np.array(medium_grained_comparison[(training_length+1):len(medium_grained_comparison)])

print motion_sensor_comparison
print medium_grained_comparison
print(len(result_stats))
print(len(truth_stats))
print len(medium_grained_comparison)
print len(motion_sensor_comparison)

final_stats = np.column_stack((result_stats,truth_stats,option_chosen_stats,motion_sensor_comparison,fine_grained_comparison,medium_grained_comparison))
np.savetxt("final_stats_all_days_new_with_medium.csv",final_stats,delimiter=",")

# np.savetxt("result_stats.csv",result_stats,delimiter=",")
# np.savetxt("truth_stats.csv",truth_stats,delimiter=",")
# np.savetxt("motion_sensor_comparison.csv",motion_sensor_comparison,delimiter=",")

result_stats = np.array(result_stats)
truth_stats = np.array(truth_stats)
motion_sensor_comparison = np.array(motion_sensor_comparison)
medium_grained_comparison = np.array(medium_grained_comparison)

total_number_ground_truth = np.sum(truth_stats)
total_number_agent = np.sum(result_stats)
total_number_motion = np.sum(motion_sensor_comparison)
total_number_medium = np.sum(medium_grained_comparison)

dqn_difference = (abs(result_stats - truth_stats))
motion_difference = (abs(motion_sensor_comparison - truth_stats))
medium_difference = abs(medium_grained_comparison - truth_stats)

print "sums"

dqn_difference = np.sum(dqn_difference)
motion_difference = np.sum(motion_difference)
medium_difference = np.sum(medium_difference)

print dqn_difference
print motion_difference
print medium_difference

print "means"

dqn_difference = dqn_difference / len(result_stats)
motion_difference = motion_difference / len(truth_stats)
medium_difference = medium_difference / len(truth_stats)

print dqn_difference
print motion_difference
print medium_difference

print "total number of people"
print total_number_ground_truth
print total_number_agent
print total_number_motion
print total_number_medium

print "Count of option chosen"
option_chosen_stats = np.array(option_chosen_stats)
counter = collections.Counter(option_chosen_stats)
print counter