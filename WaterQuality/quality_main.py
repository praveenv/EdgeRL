import numpy as np
import itertools

from collections import defaultdict, OrderedDict
from quality_env import Environment

# define the action plan IDs and the transition rules
action_plan_dict = { '1' : {'sensor' : {'do2' : 0, 'turbidity' : 1, 'tds' : 0, 'ph' : 0, 'conductivity' : 0, 'rain_gauge' : 1}, 'analytic' : {'coarse' : 1, 'fine' : 0}},
					'2' : {'sensor' : {'do2' : 1, 'turbidity' : 1, 'tds' : 1, 'ph' : 1, 'conductivity' : 1, 'rain_gauge' : 1}, 'analytic' : {'coarse' : 1, 'fine' : 1}}
					}

action_plan_rules = { '1' : ['1','2'], '2' : ['1','2']}
action_space = {'1','2'}

# set up the reinforcement learning environment
env = Environment(action_plan_dict,action_plan_rules,action_space)