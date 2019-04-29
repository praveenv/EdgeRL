import numpy as np

from video_statedef import State

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




    def step(self,action,next_environment,next_true_value,state,state_cache,data):
        '''
        This function returns the next state, next action candidates and reward as a result of taking current action in current state
        Parameters
        -----------
        self gives self.state
        action : the action plan ID that needs to be taken in this step
        next_environment : the ground truth environmental conditions for the next time step that need to be incorporated into the new state
        next_true_value : ground truth of the analytic that needs to be compared with the analytic(s) that are turned on to get reward. 

        Return
        ------
        new_state_hash : hash_value of the next state
        action_cands : new action plan IDs that are candidates for the next step
        reward : reward value that is a function of accuracy and cost incurred
        '''
        

        # get the required sensor and analytic states of the next action using the dictionary
        next_action_plan_dict = self.action_plan_dict[action]

        
        # Check whether the next state is already present in the existing state_cache. If yes, then return the existing state_hash value and state value 
        # If not present then append the state to the state_cache and obtain the new hash_value

        new_state_hash = -1
        for idx,state in enumerate(state_cache):
            if state.sensor == next_action_plan_dict['sensor'] and state.analytic == next_action_plan_dict['analytic'] and state.external == next_environment.external and state.sensor_reading == next_environment.sensor_reading:
                new_state_hash = idx
                new_state = state

        if new_state_hash == -1:
            new_state = State(next_action_plan_dict['sensor'],next_action_plan_dict['analytic'],next_environment.sensor_reading,next_environment.external)
            new_state_hash = len(state_cache) # since it is 0 based
            state_cache.append(new_state)


        action_cands = self.action_candidates(new_state)
        
        # Get the reward for performing the current action. In this work, we model the reward as a function of the accuracy and cost incurred of the analytic 
        # compared to the ground truth.         
        reward, result, truth, option_chosen = self.get_reward(action,state,next_true_value,data)

        return new_state_hash, action_cands, reward, state_cache, result, truth, option_chosen