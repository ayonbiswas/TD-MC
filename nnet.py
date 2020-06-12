import sys, math
import numpy as np
from collections import deque
import random
import copy
import gym
import keras
from keras.models import Sequential
from keras.layers import Dense
from collections import defaultdict
import random
import json




class NeuralNetwork():
    def __init__(self, batchSize = 32, weights=None):
        self.model = Sequential()
        self.model.add(Dense(256, input_dim=8, activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(4, activation='linear'))
        adam = keras.optimizers.adam(lr=0.0001)
        self.model.compile(loss='mse', optimizer=adam)
        if isinstance(weights, str):
            self.model.load_weights(weights)

    def predict(self, state):
        return self.model.predict_on_batch(state)

    def fit(self, X, y):
        self.model.fit(X, y, epochs=1,  verbose=0)

    def save(self, weights):
        self.model.save_weights(weights)

class DeepSarsaAgent():
	def __init__(self,env,  discount, weights, alpha, explorationProb=0.2, exploreProbDecay=0.99, explorationProbMin=0.01,batchSize = 64):
		self.env = env
		self.actions = np.arange(self.env.action_space.n)
		self.discount = discount
		self.explorationProb = explorationProb
		self.exploreProbDecay = exploreProbDecay
		self.explorationProbMin = explorationProbMin
		self.numIters = 0
		self.model = NeuralNetwork(batchSize, weights)
		self.alpha = alpha

	def getAction(self, state):
		if np.random.rand() < self.explorationProb:
			return random.choice(self.actions)
		else:
			return np.argmax(self.model.predict(state))

	def update(self, states, actions, rewards, newStates,newActions, dones):
		states = np.squeeze(states)
		newStates = np.squeeze(newStates)
		states = states.reshape(1, 8)
		newStates = newStates.reshape(1, 8)

		X = states
		y = self.model.predict(states)
		Q = self.model.predict(newStates)
		Q_next = np.take(Q,newActions)
		ind = np.array([i for i in range(len(states))])
		y[[ind], [actions]] = (1-self.alpha)*y[[ind], [actions]] + self.alpha*(rewards + self.discount*(Q_next)*(1-dones))

		self.model.fit(X, y)

	def simulate(self, numTrials=10, train=False, verbose=False,render =False):
		episode_length = []
		totalRewards = []
		max_totalReward = 0
		for trial in range(numTrials):
			state = np.reshape(env.reset(), (1,8))
			action = self.getAction(state)
			totalReward = 0
			iteration = 0

			while True:

				newState, reward, done, info = env.step(action)
				newState = np.reshape(newState, (1,8))
				newAction = self.getAction(newState)

				if render:
					still_open = self.env.render()
					if still_open == False: break

				if train:
					self.update(state, action, reward, newState, newAction, done)
					self.explorationProb = max(self.exploreProbDecay * self.explorationProb, self.explorationProbMin)
				
				totalReward += reward
				state = newState
				action = newAction
				iteration += 1

				if done:
					break

			totalRewards.append(totalReward)
			mean_totalReward = np.mean(totalRewards[-5:])
			if verbose:
				print(('Trial {} Total Reward: {}'.format(trial, totalReward)))
				print(('Mean(last 10 total rewards): {}'.format(np.mean(totalRewards[-10:]))))
			if((mean_totalReward>max_totalReward) and (train==True)):
				# Save Weights
				self.model.save_weights('./weights/deep_sarsa_{}_{}.h5'.format(trial,mean_totalReward))
				max_totalReward = mean_totalReward
				print('The weights are saved with total rewards: ',mean_totalReward)

			print(('Trial {} Total Reward: {}'.format(trial, totalReward)))

		np.save("./rewards/deep_sarsa_{}.npy".format(numTrials), totalRewards)
		return totalRewards



class DeepQAgent(DeepSarsaAgent):
	def __init__(self,env,  discount, weights, alpha, explorationProb=0.2, exploreProbDecay=0.99, explorationProbMin=0.01,render = False):
		super().__init__(env, discount, weights, alpha, explorationProb=0.2, exploreProbDecay=0.99, explorationProbMin=0.01,batchSize=32)
		self.replay = deque(maxlen=65536)
		self.model = NeuralNetwork(batchSize, weights)

	def update(self, states, actions, rewards, newStates, dones):
		# states = np.squeeze(states)
		# newStates = np.squeeze(newStates)

		X = states
		y = self.model.predict(states)
		# calculate gradient
		# print(X.shape,y.shape)
		ind = np.array([i for i in range(states.shape[0])])

		y[[ind], [actions]] =0.5* y[[ind], [actions]]+ 0.5*(rewards + self.discount*(np.amax(self.model.predict(newStates), axis=1))*(1-dones) )
		
		self.model.fit(X, y)

	def updateReplay(self, state, action, reward, newState, done):
		self.replay.append((state, action, reward, newState, done))

	def simulate(self,numTrials=10, train=False, verbose=False,
			 render=False, batchSize=32):

		totalRewards = []  # The rewards we get on each trial
		max_totalReward = 0
		for trial in range(numTrials):
			state = np.reshape(self.env.reset(), (1,8))
			totalReward = 0
			iteration = 0
			while True:
			# while True:
				action = self.getAction(state)
				newState, reward, done, info = self.env.step(action)
				newState = np.reshape(newState, (1,8))
				# Appending the new results to the deque
				self.updateReplay(state, action, reward, newState, done)

				# update
				totalReward += reward
				state = newState
				iteration += 1

				if render == True:
					still_open = self.env.render()
					#if still_open == False: break

				# Conducting memory replay
				if len(self.replay) < batchSize: # Waiting till memory size is larger than batch size
					continue
				else:
					batch = random.sample(self.replay, batchSize)
					states = np.array([sample[0] for sample in batch])
					actions = np.array([sample[1] for sample in batch])
					rewards = np.array([sample[2] for sample in batch])
					newStates = np.array([sample[3] for sample in batch])
					dones = np.array([sample[4] for sample in batch])

					if train:
						self.update(states, actions, rewards, newStates,
											   dones)

						self.explorationProb = max(self.exploreProbDecay * self.explorationProb,
												self.explorationProbMin)

				if done:
					break

			totalRewards.append(totalReward)
			mean_totalReward = np.mean(totalRewards[-5:])
			if verbose:
				print(('Trial {} Total Reward: {}'.format(trial, totalReward)))
				print(('Mean(last 10 total rewards): {}'.format(np.mean(totalRewards[-10:]))))

			if((mean_totalReward>max_totalReward) and (train==True)):
				# Save Weights
				self.model.save_weights('./weights/deep_Q_{}_{}.h5'.format(trial,mean_totalReward))
				max_totalReward = mean_totalReward
				print('The weights are saved with total rewards: ',mean_totalReward)

			print(('Trial {} Total Reward: {}'.format(trial, totalReward)))

		np.save("./rewards/deep_Q_{}.npy".format(numTrials), totalRewards)

		return totalRewards


class DeepMCAgent(DeepSarsaAgent):
	def __init__(self,env,  discount, weights, alpha, explorationProb=0.2, exploreProbDecay=0.99, explorationProbMin=0.01):
		super().__init__(env, discount, weights, alpha, explorationProb=0.2, exploreProbDecay=0.99, explorationProbMin=0.01)

	def update(self, states, actions, rewards):
		# initialize variable
		# states = np.squeeze(states)
		# newStates = np.squeeze(newStates)
		X = states
		y = self.model.predict(states)
		# targets = rewards + self.discount*(np.amax(self.model.predict(newStates), axis=1))*(1-dones)
		ind = np.array([i for i in range(len(states))])
		y[[ind], [actions]] = (1-self.alpha)*y[[ind], [actions]] + self.alpha*(rewards)
		
		# y[[ind], [actions]] = targets
		# update weight
		self.model.fit(X, y)


	def simulate(self, numTrials=10, train=False, verbose=False,render = False):

		totalRewards = []  # The rewards we get on each trial
		max_totalReward = 0
		for trial in range(numTrials):
			state = np.reshape(self.env.reset(), (1,8))
			totalReward = 0
			iteration = 0
			trajectory = []
			rewards = []

			while True:

				action = self.getAction(state)
				newState, reward, done, info = self.env.step(action)
				newState = np.reshape(newState, (1,8))
				trajectory.append((state, action, newState, reward, done))
				rewards.append(reward)

				if render== True:
					still_open = self.env.render()
					if still_open == False: break

				totalReward += reward
				state = newState
				iteration += 1

				if done:
					break

			if train:
				data = []
				for i in range(len(rewards)):
					L = 0
					for r in range(len(rewards)-i):
						L += np.power(self.discount, r)*rewards[r+i]
					# data.append((trajectory[i][0], trajectory[i][1], L))
					self.update(trajectory[i][0], trajectory[i][1], L)

				# for i in range(100):
				# 	batch = random.sample(data, batchSize)
				# 	states = np.array([sample[0] for sample in batch])
				# 	actions = np.array([sample[1] for sample in batch])
				# 	rewards = np.array([sample[2] for sample in batch])
				# 	self.update(states, actions, rewards)


				self.explorationProb = max(self.exploreProbDecay * self.explorationProb, self.explorationProbMin)

			totalRewards.append(totalReward)
			mean_totalReward = np.mean(totalRewards[-5:])
			if verbose:
				print(('Trial {} Total Reward: {}'.format(trial, totalReward)))
				print(('Mean(last 10 total rewards): {}'.format(np.mean(totalRewards[-10:]))))
				
			if((mean_totalReward>max_totalReward) and (train==True)):
				# Save Weights
				self.model.save_weights('./weights/deep_MC_{}_{}.h5'.format(trial,mean_totalReward))
				max_totalReward = mean_totalReward
				print('The weights are saved with total rewards: ',mean_totalReward)

			# print(('Trial {} Total Reward: {}'.format(trial, totalReward)))
		np.save("./rewards/deep_MC_{}.npy".format(numTrials), totalRewards)

		return totalRewards

## Main variables
numEpochs = 1000
numTrials = 1
numTestTrials = 10
trialDemoInterval = numTrials/2
discountFactor = 0.99
explorProbInit = 1.0
exploreProbDecay = 0.999
explorationProbMin = 0.0
batchSize = 2

if __name__ == '__main__':
	# Initiate weights
	# Cold start weights
	weights = None
	# Warm start weights
	#weights = './weights/weights_sarsa.h5'
	env = gym.make('LunarLander-v2')
	# TRAIN
	print('\n++++++++++++ TRAINING +++++++++++++')
	rl = DeepMCAgent(env, discountFactor, weights,
							explorProbInit, exploreProbDecay,
							explorationProbMin, batchSize)
	
	# env.seed(0)

	totalRewards_list = []
	# for i in range(numEpochs):
	totalRewards = rl.simulate( 1000, train=True, verbose=True)
	# totalRewards_list.append(totalRewards)
	# print('Average Total Reward in Trial {}/{}: {}'.format(i, numEpochs, np.mean(totalRewards)))
	env.close()
	# Save Weights
	rl.model.save('weights_deepsarsa.h5')

	# TEST
	print('\n\n++++++++++++++ TESTING +++++++++++++++')
	weights = 'weights_deepsarsa.h5'
	env = gym.make('LunarLander-v2')
	env.seed(3)
	rl = SarsaAlgorithm([0, 1, 2, 3], discountFactor, weights, 0.0, 0.0, 0.0, batchSize)
	totalRewards = simulate(env, rl, numTrials=numTestTrials, train=False, verbose=True, trialDemoInterval=trialDemoInterval)
	env.close()
	print('Average Total Testing Reward: {}'.format(np.mean(totalRewards)))