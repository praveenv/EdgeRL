import numpy as np
import itertools

from collections import defaultdict, OrderedDict
from flood_env import Environment
from flood_sarsa import sarsa
from flood_statedef import State
from flood_Qlearning import qlearning
from flood_nstepSarsa import nstep_sarsa


sensor_dict = {'waterLevel' : [0,1], 'camera' : [0,1]}
analytic_dict = {'waterLevel_A' : [0,1], 'camera_A' : [0,1]}
external_dict = {'timeOfDay' : [0,1], 'rain' : [0,1]}

action_plan_dict = { '1' : { 'sensor' : {'waterLevel' : 1, 'camera' : 0 }, 'analytic' : {'waterLevel_A' : 1, 'camera_A' : 0 }}, 
                    '2' : { 'sensor' : {'waterLevel' : 1, 'camera' : 1 }, 'analytic' : {'waterLevel_A' : 1, 'camera_A' : 1 }}
                    }
action_plan_rules = { '1' : ['1','2'] , '2' : ['1','2'] }
action_space = {'1','2'}

# set up the Reinforcement Learning environment 
env = Environment(action_plan_dict, action_plan_rules,action_space)

# Generate sample training data
data = OrderedDict()
temp = { 'sensor' : {'waterLevel' : 1, 'camera' : 0 }, 'analytic' : {'waterLevel_A' : 1, 'camera_A' : 0 }, 'external' : {'timeOfDay' : 0, 'rain' : 0}, 'truth' : 0}
temp1 = { 'sensor' : {'waterLevel' : 1, 'camera' : 1 }, 'analytic' : {'waterLevel_A' : 1, 'camera_A' : 1 }, 'external' : {'timeOfDay' : 0, 'rain' : 1}, 'truth' : 1}
for i in xrange(0,1000):
	if i < 500:
		data[str(i)] = temp
	else:
		data[str(i)] = temp1

# store the states observed in a cache list and use the index of each state as its key in the Q-table
state_cache = []

# Run the SARSA algorithm
Q, policy = sarsa(env, 1000, data, state_cache)

# Run the Q-Learning algorithm
# Q, policy = qlearning(env, 1000, data, state_cache)

# Run the nstep SARSA algorithm
# Q, policy = nstep_sarsa(env, 1000, 5, data, state_cache)

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
