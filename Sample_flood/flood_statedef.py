import numpy as np


class State:

    def __init__(self,sensor_dict,analytic_dict,external_dict):

        self.sensor = sensor_dict
        self.analytic = analytic_dict
        self.external = external_dict

    # def __hash__(self):
    #     return hash((self.sensor,self.analytic,self.external))

    # def __eq__(self,other):
    #     return (self.sensor,self.analytic,self.external) == (other.sensor,other.analytic,other.external)

    # def __ne__(self,other):
    #     # Not strictly necessary, but to avoid having both x==y and x!=y
    #     # True at the same time
    #     return not(self == other)