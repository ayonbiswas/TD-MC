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

class TileCoder:
	def __init__(self, tiles_per_dim, value_limits, tilings, offset=lambda n: 2 * np.arange(n) + 1):
		tiling_dims = np.array(np.ceil(tiles_per_dim), dtype=np.int) + 1
		self._offsets = offset(len(tiles_per_dim)) * \
		  np.repeat([np.arange(tilings)], len(tiles_per_dim), 0).T / float(tilings) % 1
		self._limits = np.array(value_limits)
		self._norm_dims = np.array(tiles_per_dim) / (self._limits[:, 1] - self._limits[:, 0])
		self._tile_base_ind = np.prod(tiling_dims) * np.arange(tilings)
		self._hash_vec = np.array([np.prod(tiling_dims[0:i]) for i in range(len(tiles_per_dim))])
		self._n_tiles = tilings * np.prod(tiling_dims)
  
	def __getitem__(self, x):
		off_coords = ((x - self._limits[:, 0]) * self._norm_dims + self._offsets).astype(int)
		return tuple(self._tile_base_ind + np.dot(off_coords, self._hash_vec))

	@property
	def n_tiles(self):
		return self._n_tiles

class QTilecoder():
	def __init__(self,env, model, discount, alpha = 0.5, explorationProb=0.2, exploreProbDecay=0.99, explorationProbMin=0.01):
		self.env = env
		self.model = model
		self.actions = np.arange(self.env.action_space.n)
		self.discount = discount
		self.explorationProb = explorationProb
		self.exploreProbDecay = exploreProbDecay
		self.explorationProbMin = explorationProbMin
		self.alpha = alpha
		self.numIters = 0
		self.state_bounds = [(-1.5,1.5) for i in range(6)] + [(0,1),(0,1)]
		# self.tiles_per_dim = [5,5,3,3,2,5,1,1]
		self.tiles_per_dim = [3,3,3,3,2,3,1,1]

		self.number_tilings = 3
		self.Tcoder = TileCoder(self.tiles_per_dim, self.state_bounds, self.number_tilings)

	def getAction(self, state):
		self.numIters += 1
		if random.random() < self.explorationProb:
			return random.choice(self.actions)
		else:
			return np.argmax([self.model[(state, action)] for action in self.actions])



	def update(self, state, action, reward, newState):
		# BEGIN_YOUR_CODE (our solution is 9 lines of code, but don't worry if you deviate from this)
		# calculate gradient
		if newState is not None:
			# find maximum Q value from next state
			# print(newState,action)
			Q_next = max([self.model[(newState, possibleAction)] for possibleAction in self.actions])
		else:
			# Q_next of end state is 0
			Q_next = 0.0
		Q = self.model[(state, action)]


		# update weight
		self.model[(state,action)] =(1-self.alpha)* Q + self.alpha*(reward + self.discount * Q_next)

	def simulate(self, numTrials=100, train=False, verbose=False, render=False):
		totalRewards = []  # The rewards we get on each trial
		# tilings = rl.create_tilings(rl.state_bounds,rl.number_tilings,rl.bins,rl.offsets)
		# print(tilings)
		max_totalReward = 0 
		for trial in range(numTrials):
			state = self.env.reset()
			totalReward = 0
			iteration = 0
			state_d = self.Tcoder[tuple(state)]
			# print(state_d)
			# print(state_d)
			# sdaasd
			while True:
				
				action = self.getAction(state_d)
				newState, reward, done, info = self.env.step(action)
				newState_d = self.Tcoder[tuple(newState)]
				if train:
					self.update(state_d, action, reward, newState_d)
				totalReward += reward
				self.explorationProb = max(self.exploreProbDecay * self.explorationProb,
											 self.explorationProbMin)
				state_d = newState_d
				iteration += 1

				if done:
					# print(iteration)
					break
				if verbose == True and render:
					still_open = self.env.render()
					# print(iteration)
					if still_open == False: break

			totalRewards.append(totalReward)
			if verbose:
				print(('Trial {} Total Reward: {}'.format(trial, totalReward)))
				print(('Mean(last 10 total rewards): {}'.format(np.mean(totalRewards[-10:]))))

			mean_totalReward = np.mean(totalRewards[-5:])
			if((mean_totalReward>max_totalReward) and (train==True)):
				# Save Weights
				self.saveF(self.model,'./weights/Q_tile_{}_{}_{}.h5'.format(trial,mean_totalReward,self.number_tilings))
				max_totalReward = mean_totalReward
				print('The weights are saved with total rewards: ',mean_totalReward)

			print(('Trial {} Total Reward: {}'.format(trial, totalReward)))

		np.save("./rewards/Q_tile_{}.npy".format(numTrials), totalRewards)
		return totalRewards


	def saveF(self,obj, name):
		with open('weights/' + name + '.pkl', 'wb') as f:
			pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

	def loadF(self,name):
		with open(self,'weights/' + name + '.pkl', 'rb') as f:
			return pickle.load(f)



class SarsaTilecoder(QTilecoder):
	def __init__(self,env, model,  discount, alpha = 0.5, explorationProb=0.2, exploreProbDecay=0.99, explorationProbMin=0.01):
		super().__init__(env,model,  discount, alpha = 0.5, explorationProb=0.2, exploreProbDecay=0.99, explorationProbMin=0.01)

	def update(self, state, action, reward, newState, newAction):
		# BEGIN_YOUR_CODE (our solution is 9 lines of code, but don't worry if you deviate from this)
		# calculate gradient
		if newState is not None:
			# find maximum Q value from next state
			# print(newState,action)
			Q_next = self.model[(newState, newAction)] 
		else:
			# Q_next of end state is 0
			Q_next = 0.0
		Q = self.model[(state, action)]
		target = reward + self.discount * Q_next
		# update weight
		self.model[(state,action)] =(1-self.alpha)* Q + self.alpha* (reward + self.discount * Q_next)
		# update weight
	def simulate(self, numTrials=100, train=False, verbose=False, render=False):
		totalRewards = []  # The rewards we get on each trial
		# tilings = rl.create_tilings(rl.state_bounds,rl.number_tilings,rl.bins,rl.offsets)
		# print(tilings)
		max_totalReward = 0
		for trial in range(numTrials):
			state = self.env.reset()
			totalReward = 0
			iteration = 0
			state_d = self.Tcoder[tuple(state)]
			# print(state_d)
			# print(state_d)
			# sdaasd
			action = self.getAction(state_d)
			while True:
				
				
				newState, reward, done, info = self.env.step(action)
				newState_d = self.Tcoder[tuple(newState)]
				nextAction = self.getAction(newState_d)
				if train:
					self.update(state_d, action, reward, newState_d,nextAction)
				totalReward += reward
				self.explorationProb = max(self.exploreProbDecay * self.explorationProb,
											 self.explorationProbMin)
				totalReward += reward
				state_d = newState_d
				action = nextAction
				iteration += 1

				if done:
					# print(iteration)
					break
				if verbose == True and render:
					still_open = self.env.render()
					# print(iteration)
					if still_open == False: break

			totalRewards.append(totalReward)
			if verbose and trial % 20 == 0:
				print(('\n---- Trial {} ----'.format(trial)))
				print(('Mean(last 10 total rewards): {}'.format(np.mean(totalRewards[-100:]))))
				# print(('Size(weight vector): {}'.format(len(self.weights))))

			mean_totalReward = np.mean(totalRewards[-5:])
			if((mean_totalReward>max_totalReward) and (train==True)):
				# Save Weights
				self.saveF(self.model,'./weights/sarsa_tile_{}_{}_{}.h5'.format(trial,mean_totalReward,self.number_tilings))
				max_totalReward = mean_totalReward
				print('The weights are saved with total rewards: ',mean_totalReward)

			print(('Trial {} Total Reward: {}'.format(trial, totalReward)))

		np.save("./rewards/sarsa_tile_{}.npy".format(numTrials), totalRewards)
			
		return totalRewards


class MCTilecoder(QTilecoder):
	def __init__(self,env,model, discount, weights, explorationProb=0.2, exploreProbDecay=0.99, explorationProbMin=0.01):
		super().__init__(env, model, discount, alpha = 0.5, explorationProb=0.2, exploreProbDecay=0.99, explorationProbMin=0.01)
		# self.tiles_per_dim = [5,5,5,5,5,5,1,1]
		self.tiles_per_dim = [3,3,3,3,2,3,1,1]

		self.number_tilings = 3
		self.Tcoder = TileCoder(self.tiles_per_dim, self.state_bounds, self.number_tilings)



	def simulate(self, numTrials=10, train=False, verbose=False,
			 render= False):
		totalRewards = []  # The rewards we get on each trial
		max_totalReward = 0
		for trial in range(numTrials):
			state = self.env.reset()
			totalReward = 0
			iteration = 0
			trajectory = []
			rewards = []
			state_d = self.Tcoder[tuple(state)]
			while True:
				# print(state.shape,tuple(state))
				
				action = self.getAction(state_d)
				newState, reward, done, info = self.env.step(action)
				
				
				# newState = np.reshape(newState, (1,8))
				newState_d =self.Tcoder[tuple(newState)] 
				trajectory.append((state_d, action,newState_d , reward, done))
				rewards.append(reward)

				if render == True:
					still_open = self.env.render()
					if still_open == False: break

				totalReward += reward
				state_d = newState_d
				iteration += 1

				if done:
					break
			if train:
				data = []
				for i in range(len(rewards)):
					L = 0
					for r in range(len(rewards)-i):
						L += np.power(self.discount, r)*rewards[r+i]
					self.model[(trajectory[i][0],trajectory[i][1])] = 0.5*self.model[(trajectory[i][0], trajectory[i][1])] + 0.5 * (L)

				self.explorationProb = max(self.exploreProbDecay * self.explorationProb, self.explorationProbMin)

			totalRewards.append(totalReward)
			mean_totalReward = np.mean(totalRewards[-5:])
			if((mean_totalReward>max_totalReward) and (train==True)):
				# Save Weights
				self.saveF(self.model,'./weights/MC_tile_{}_{}_{}.h5'.format(trial,mean_totalReward,self.number_tilings))
				max_totalReward = mean_totalReward
				print('The weights are saved with total rewards: ',mean_totalReward)

			print(('Trial {} Total Reward: {}'.format(trial, totalReward)))

		np.save("./rewards/MC_tile_{}.npy".format(numTrials), totalRewards)

		return totalRewards

numFeatures = 4
numActions = 4
numEpochs = 1
numTrials = 2500 #30000
numTestTrials = 10
trialDemoInterval = 2
discountFactor = 0.99
explorProbInit = 1.0
exploreProbDecay = 0.996
explorationProbMin = 0.01



if __name__ == '__main__':
	# Initiate weights
	# np.random.seed(1)
	# random.seed(1)
	# Cold start weights
	# weights = defaultdict(float)
	# Warm start weights
	# weights = loadF('weights')
	model = defaultdict(float)
	# TRAIN
	# for i in range(numEpochs):

	env = gym.make('LunarLander-v2')
	# # env.seed(0)
	# print('\n++++++++++++ TRAINING +++++++++++++')\
	rl = MCTilecoder(env, model, discountFactor, explorProbInit, exploreProbDecay,
						explorationProbMin)
	totalRewards, model = rl.simulate( numTrials=numTrials, train=True, verbose=True, render=False)
	env.close()
	print('Average Total Training Reward: {}'.format(np.mean(totalRewards)))
	# print(model)
	# print(rl.model)
	# Save Weights
	saveF(model, 'model_tl_q4')
	# with open('weights.json', 'w') as fileOpen:
		# json.dump(model, fileOpen)
	model = loadF('model_tl_q4')
	
	# TEST
	print('\n\n++++++++++++++ TESTING +++++++++++++++')
	rl = MCTilecoder(env, model, discountFactor, 0.0, 1,
							0.00)
	totalRewards, _ = rl.simulate(numTrials=numTestTrials, train=False, verbose=True, render=True)
	print('Average Total Testing Reward: {}'.format(np.mean(totalRewards)))