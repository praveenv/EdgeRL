import numpy as np
import itertools
import math
import pandas as pd

from collections import defaultdict, OrderedDict
from video_env import Environment
from video_dqn import DQNAgent
from video_sarsa import sarsa

from sklearn.metrics import mean_squared_error
from math import sqrt

action_plan_dict = { '1' : {'sensor' : {'motion' : 1, 'camera' : 0, 'rain_gauge' : 1}, 'analytic' : {'coarse' : 1, 'fine' : 0}},
					'2' : {'sensor' : {'motion' : 1, 'camera' : 1, 'rain_gauge' : 1}, 'analytic' : {'coarse' : 1, 'fine' : 1}}
					}

action_plan_rules = { '1' : ['1','2'], '2' : ['1','2']}
action_space = {'1','2'}


# set up the reinforcement learning environment
env = Environment(action_plan_dict,action_plan_rules,action_space)

agent = DQNAgent(1,2)

data_path = "./Data/combined_days.csv"

data = pd.read_csv(data_path)
fine_grained = np.asarray(data.fine_grained)
coarse_grained = np.asarray(data.coarse_grained)
rain_gauge = np.asarray(data.rain_gauge)
time_of_day = np.asarray(data.time_of_day)
data = np.column_stack((fine_grained,coarse_grained,rain_gauge,time_of_day))


state_cache = []
Q, policy, result_stats, truth_stats, option_chosen_stats = sarsa(env,agent,len(data),data,state_cache)


training_length = 1440*5
test_length = 1440*2


motion_sensor_comparison = data[:,1]
motion_sensor_comparison = np.delete(motion_sensor_comparison,0)

dqn_mean_square = sqrt(mean_squared_error(result_stats,truth_stats))
print "final"
print dqn_mean_square

motion_mean_square = sqrt(mean_squared_error(motion_sensor_comparison,truth_stats))


print motion_mean_square

# result_stats = np.array(result_stats[(training_length+1):len(result_stats)])
# truth_stats = np.array(truth_stats[(training_length+1):len(truth_stats)])
# motion_sensor_comparison = np.array(motion_sensor_comparison[(training_length+1):len(motion_sensor_comparison)])

# np.savetxt("result_stats.csv",result_stats,delimiter=",")
# np.savetxt("truth_stats.csv",truth_stats,delimiter=",")
# np.savetxt("motion_sensor_comparison.csv",motion_sensor_comparison,delimiter=",")

dqn_difference = (abs(result_stats - truth_stats))
motion_difference = (abs(motion_sensor_comparison - truth_stats))

print "sums"

dqn_difference = np.sum(dqn_difference)
motion_difference = np.sum(motion_difference)

print dqn_difference
print motion_difference

print "means"

dqn_difference = dqn_difference / len(result_stats)
motion_difference = motion_difference / len(truth_stats)

print dqn_difference
print motion_difference