import numpy as np


class State:

    def __init__(self,sensor_dict,analytic_dict,sensor_reading,external_dict):

        self.sensor = sensor_dict
        self.analytic = analytic_dict
        self.sensor_reading = sensor_reading
        self.external = external_dict

    