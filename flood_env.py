#!/usr/bin/env python

import numpy as np
from flood_analytics import analytic_controller


class Environment:
    

    def __init__(self,sensor_dict,analytic_dict,external_dict,action_plan_dict,action_plan_rules,action_space):
        
        self.sensor_dict = sensor_dict
        self.analytic_dict = analytic_dict
        self.external_dict = external_dict
        self.action_plan_dict = action_plan_dict
        self.action_plan_rules = action_plan_rules


        self.state = {}

        self.action_space = list(action_space) # list of unique actions plans in this application



    def action_candidates(self):
        '''
        Function to return the valid action plans based on the current state and defined action rules
        Returns a list of indices of valid Action Plan IDs
        The indices are derived from the action space defined in the main script. 
        '''

        current_state_action_plan = self.parse_state_action()
        current_state_action_plan = str(current_state_action_plan)

        current_action_candidates = []
        current_action_candidates = list(self.action_plan_rules[current_state_action_plan])

        return current_action_candidates


    def set_state(self,initial_environment):
        '''
        Function to set the initial state of the RL agent every episode. 
        Parameter
        ----------
        initial_environment : dictionary with the initial sensor, analytic and environmental conditions given by first training data tuple
        '''

        self.state = {}
        self.state['sensor'] = initial_environment['sensor']
        self.state['analytic'] = initial_environment['analytic']
        self.state['external'] = initial_environment['external']

        current_action_candidates = self.action_candidates()
        
        return self.state, current_action_candidates



    def parse_state_action(self):
        '''
        This function is used to parse the current state based on the action plan dictionary
        Returns the Action Plan ID that the current state corresponds to
        State is a dictionary of the form - {'sensor' : {}, 'analytic': {}, 'external' : {}}
        '''

        
        current_state_sensor = self.state['sensor']
        current_state_analytic = self.state['analytic']
        current_state_external = self.state['external']

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

        

    def observe(self):
        '''
        This function returns the current state of the environment
        (i.e.) the sensors and analytics that are currently running and their states
        as well as the current environmental conditions.
        '''
        pass

    

    def get_reward(self,action,next_true_value):
        '''
        This function returns a reward value for the current action taken by comparing the result of the analytics with the ground truth value
        Note : Ideally for reusability, I want to keep this function generic and compute the actual reward value inside the analytic function itself by 
        passing the true_value as a parameter. 
        Parameters
        ----------
        action : Action ID of the current step
        next_true_value : ground truth from data

        Returns
        -------
        reward : numeric value corresponding to the reward. 
        '''

        reward_value = analytic_controller(action,next_true_value)
        return reward_value




    def step(self,action,next_environment,next_true_value):
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
        next_state : new state information
        action_cands : new action plan IDs that are candidates for the next step
        reward : reward value that is a function of accuracy and cost incurred
        '''
        

        # get the required sensor and analytic states of the next action using the dictionary
        next_action_plan_dict = self.action_plan_dict[action]

        #update current state's sensor and analytics with the new one after performing the action
        self.state['sensor'] = next_action_plan_dict['sensor']
        self.state['analytic'] = next_action_plan_dict['analytic']

        # update the new state's environment with the true readings from the data
        self.state['external'] = next_environment['external']

        action_cands = self.action_candidates()
        
        reward = self.get_reward(action,next_true_value)


        return self.state, action_cands, reward


       # self.state, action_cand = self._update_state()
