import numpy as np

from video_statedef import State

class Environment:

	def __init__(self,action_plan_dict,action_plan_rules,action_space):
		self.action_plan_dict = action_plan_dict
		self.action_plan_rules = action_plan_rules
		self.action_space = list(action_space) # list of unique action plans in this application