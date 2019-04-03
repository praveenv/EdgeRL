#!/usr/bin/env python

import numpy as np

class Environment:
    

    def __init__(self):

        self.state = {}


    '''
    This function returns the current state of the environment
    (i.e.) the sensors and analytics that are currently running and their states
    as well as the current environmental conditions.
    '''
    def observe(self):


    '''
    This function returns the action that needs to be taken in this episode
    Need to implement epsilon-greedy here
    
    Parameters
    action : The action plan ID number that was chosen to be performed
    external_dict : Dictionary of external information derived from actual data to update current state
    
    Returns
    state : New state after the performed action as well as incorporating external information
    action_cand : list of allowed Action Plan ID numbers for this new state
    reward : Reward value after taking action
    '''
    def step(self,action):
        
