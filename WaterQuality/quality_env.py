import numpy as np

from quality_statedef import State

class Environment:

	def __init__(self,action_plan_dict,action_plan_rules,action_space):
		self.action_plan_dict = action_plan_dict
		self.action_plan_rules = action_plan_rules
		self.action_space = list(action_space) # list of unique action plans in this application

    
    def parse_state_action(self,initial_environment):
        '''
        This function is used to parse the current state based on the action plan dictionary
        Returns the Action Plan ID that the current state corresponds to
        State is a dictionary of the form - {'sensor' : {}, 'analytic': {}, 'external' : {}}
        '''

        current_state_sensor = initial_environment.sensor
        current_state_analytic = initial_environment.analytic

        number_of_action_plans = len(self.action_plan_dict)

        #iterate over the action plan dictionary for each possible plan, and match the sensor and analytic key values
        # with current state
        current_action_id = -1
        for key, value in self.action_plan_dict.iteritems():
            if value['sensor'] == current_state_sensor and value['analytic'] == current_state_analytic:
                current_action_id = key

        # make sure current state is present in defined action plans
        assert current_action_id != -1
        
        return current_action_id    

    

    def action_candidates(self,initial_environment):
        '''
        Function to return the valid action plans based on the current state and defined action rules
        Returns a list of indices of valid Action Plan IDs
        The indices are derived from the action space defined in the main script. 
        '''

        current_state_action_plan = self.parse_state_action(initial_environment)
        current_state_action_plan = str(current_state_action_plan)

        current_action_candidates = []
        current_action_candidates = list(self.action_plan_rules[current_state_action_plan])

        return current_action_candidates

    

    def set_state(self,initial_environment,state_cache):
        '''
        Function to set the initial state of the RL agent every episode. 
        Parameter
        ----------
        initial_environment : state object with the sensor, analytic and external information as attributes that are dictionaries. 

        Returns
        -------
        state_hash : index of the current state in the state_cache 
        current_action_candidates : action plan ID's valid for the current state
        state_cache : updated state_cache
        '''

        # Start by checking whether the initial environment is already present in the state_cache. If yes, assign the existing state_hash value , else append the 
        # current state to the state_cache and obtain the new state_hash value. 
        state_hash = 0
        flag = 0
        for idx, cache in enumerate(state_cache):
            if cache.sensor == initial_environment.sensor and cache.analytic == initial_environment.analytic and cache.external == initial_environment.external and cache.sensor_reading = initial_environment.sensor_reading:
                state_hash = idx
                flag = 1

        if flag == 0:
            state_cache.append(initial_environment)
            state_hash = 0

        current_action_candidates = self.action_candidates(initial_environment)
        
        return state_hash, current_action_candidates, state_cache