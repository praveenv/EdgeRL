import numpy as np

def analytic_controller(action,state,next_true_value,data):
	if state.analytic['fine'] == 1:
		return fine_grained(action,state,next_true_value,data)
	else:
		return coarse_grained(action,state,next_true_value,data)


def fine_grained(action,state,next_true_value,data):
	option_chosen = 1
	return float(1),next_true_value,next_true_value,option_chosen

def coarse_grained(action,state,next_true_value,data):
	motion_reading = state.sensor_reading['motion']
	actual_reading = next_true_value
	reward = 0
	if next_true_value == 0 and motion_reading == 1:
		reward = float(-10)
	if next_true_value == 1 and motion_reading == 1:
		reward = float(10)
	if next_true_value == 0 and motion_reading == 0:
		reward = float(10)
	if next_true_value > 1 and motion_reading == 1:
		reward = float(float(next_true_value - motion_reading)/float(10))
	option_chosen = 0
	return reward,motion_reading,actual_reading,option_chosen