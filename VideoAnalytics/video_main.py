import numpy as np
import itertools
import math

from collections import defaultdict, OrderedDict


action_plan_dict = { '1' : {'sensor' : {'motion' : 1, 'camera' : 0, 'rain_gauge' : 1}, 'analytic' : {'coarse' : 1, 'fine' : 0}},
					'2' : {'sensor' : {'motion' : 1, 'camera' : 1, 'rain_gauge' : 1}, 'analytic' : {'coarse' : 1, 'fine' : 1}}
					}

action_plan_rules = { '1' : ['1','2'], '2' : ['1','2']}
action_space = {'1','2'}


# set up the reinforcement learning environment
env = Environment(action_plan_dict,action_plan_rules,action_space)