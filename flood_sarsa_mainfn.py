import numpy as np
import itertools

from collections import defaultdict, OrderedDict

from flood_env import Environment

from flood_sarsa import sarsa

from flood_statedef import State

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




data = OrderedDict()
temp = { 'sensor' : {'waterLevel' : 1, 'camera' : 0 }, 'analytic' : {'waterLevel_A' : 1, 'camera_A' : 0 }, 'external' : {'timeOfDay' : 0, 'rain' : 0}, 'truth' : 0}
temp1 = { 'sensor' : {'waterLevel' : 1, 'camera' : 1 }, 'analytic' : {'waterLevel_A' : 1, 'camera_A' : 1 }, 'external' : {'timeOfDay' : 0, 'rain' : 1}, 'truth' : 1}
for i in xrange(0,30):
	if i < 10:
		data[str(i)] = temp
	else:
		data[str(i)] = temp1

state_cache = []


Q, policy = sarsa(env, 30, data, state_cache)
# print Q
print "Q TABLE"
for k, v in Q.iteritems():
	for a,b in v.items():
		print k,a,b

print "STATE CACHE"
for idx, v in enumerate(state_cache):
	print idx, v.sensor, v.analytic, v.external

print "ACTINO PLAN"
print action_plan_dict
# print Q.items()

# class Foo:
#     def __init__(self):
#         pass

# myinstance = State(temp['sensor'],temp['analytic'],temp['external'])
# yolo = State(temp1['sensor'],temp1['analytic'],temp1['external'])

# myman = myinstance

# Q = defaultdict(lambda: defaultdict(lambda: 0.0))

# Q[myinstance]['1'] += 1
# Q[myinstance]['2'] += 3
# Q[yolo]['2'] +=2
# Q[yolo]['1'] +=5

# state_Q = Q[myinstance]


# print yolo.sensor
# print state_Q
# print Q

# for ac in state_Q:
# 	print ac

