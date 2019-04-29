import numpy as np
import random

from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class DQNAgent:
	def __init__(self,state_size,action_size):
		self.state_size = state_size
		self.action_size = action_size
		self.memory = deque(maxlen = 2000)
		self.gamma = 0.95 # discount rate
		self.epsilon = 0.5 #exploration rate
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.learning_rate = 0.001
		self.model = self._build_model()


	def _build_model(self):
		# Neural net for Deep Q-learning model using Keras
		model = Sequential()
		model.add(Dense(24, input_dim=self.state_size,activation='relu'))
		model.add(Dense(24,activation='relu'))
		model.add(Dense(self.action_size,activation='linear'))
		model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))
		return model

	def remember(self,state,action,reward,next_state):
		self.memory.append((state,int(action),reward,next_state))


	def act(self,state):
		if np.random.rand() <= self.epsilon:
			return random.randrange(1,self.action_size+1)
		act_values = self.model.predict(np.array([state,]))
		return (np.argmax(act_values[0])+1)


	def replay(self,batch_size):
		if len(self.memory) < batch_size:
			batch_size = len(self.memory)
		minibatch = random.sample(self.memory,batch_size)
		for state,action,reward,next_state in minibatch:
			target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state,]))[0])
			target_f = self.model.predict(np.array([state,]))
			target_f[0][action-1] = target

			self.model.fit(np.array([state,]),target_f,verbose = 0)
			if self.epsilon > self.epsilon_min:
				self.epsilon *= self.epsilon_decay
