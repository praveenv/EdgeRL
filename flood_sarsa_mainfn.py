import numpy as np
import itertools

from collections import defaultdict, OrderedDict

from flood_env import Environment

from flood_sarsa import sarsa

sensor_dict = {'waterLevel' : [0,1], 'camera' : [0,1]}
analytic_dict = {'waterLevel_A' : [0,1], 'camera_A' : [0,1]}
external_dict = {'timeOfDay' : [0,1], 'rain' : [0,1]}

action_plan_dict = { '1' : { 'sensor' : {'waterLevel' : 1, 'camera' : 0 }, 'analytic' : {'waterLevel_A' : 1, 'camera_A' : 0 }}, 
                    '2' : { 'sensor' : {'waterLevel' : 1, 'camera' : 1 }, 'analytic' : {'waterLevel_A' : 1, 'camera_A' : 1 }}
                    }

action_plan_rules = { '1' : ['1','2'] , '2' : ['1','2'] }


action_space = {'1','2'}

# set up the Reinforcement Learning environment 
env = Environment(sensor_dict, analytic_dict, external_dict, action_plan_dict, action_plan_rules,action_space)




data = OrderedDict()
temp = { 'sensor' : {'waterLevel' : 1, 'camera' : 0 }, 'analytic' : {'waterLevel_A' : 1, 'camera_A' : 0 }, 'external' : {'timeOfDay' : 0, 'rain' : 0}, 'truth' : 0}
temp1 = { 'sensor' : {'waterLevel' : 1, 'camera' : 1 }, 'analytic' : {'waterLevel_A' : 1, 'camera_A' : 1 }, 'external' : {'timeOfDay' : 0, 'rain' : 1}, 'truth' : 1}
for i in xrange(0,30):
	if i < 10:
		data[str(i)] = temp
	else:
		data[str(i)] = temp1


Q, policy = sarsa(env, 30, data)

# for k, v in data.iteritems():
# 	print k, v