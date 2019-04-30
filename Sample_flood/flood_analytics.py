import numpy as np

def analytic_controller(action,ground_truth):

	if action == '1':
		return water_level_analytics(ground_truth)

	elif action == '2':
		return camera_analytics(ground_truth)



def water_level_analytics(ground_truth):

	if ground_truth == 0:
		return 5
	elif ground_truth == 1:
		return 0

def camera_analytics(ground_truth):
	if ground_truth == 0:
		return 0

	elif ground_truth == 1:
		return 5