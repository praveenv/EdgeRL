import numpy as np

import pickle

def analytic_controller(action,state,next_true_value,data):
	print "analytic controlelr"
	print state.analytic['fine']
	if state.analytic['fine'] == 1:
		new_data = np.delete(data,7)
		return rf_prediction(new_data,next_true_value)
	else:
		index = [0,2,3,4,7]
		new_data = np.delete(data,index)
		return svm_prediction(new_data,next_true_value)


def rf_prediction(new_data,next_true_value):
	file = open('randomForestmodel.pkl','r')
	rf_model = pickle.load(file)
	file.close()
	result = rf_model.predict(np.array([new_data]))[0]
	store_result = result
	if result != next_true_value:
		# result = -1
		result = -5
	else:
		# result = 1
		result = 5
	# reward = (result * 3) / 5
	reward = float(float(result) / float(1.5 + 7))
	option_chosen = 1
	return reward, store_result, next_true_value,option_chosen



def svm_prediction(new_data,next_true_value):
	file = open('SVMmodel.pkl','r')
	svm_model = pickle.load(file)
	file.close()
	print "inside sVM"
	print new_data
	result = svm_model.predict(np.array([new_data]))[0]
	print result
	print next_true_value
	store_result = result
	if result != next_true_value:
		# result = -3
		result = -1
	else:
		# result = 1
		result = 1
	# reward = (result)
	print result
	reward = float(float(result) / float(1 + 2))
	print reward
	option_chosen = 0
	return reward, store_result, next_true_value,option_chosen