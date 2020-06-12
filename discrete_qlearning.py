import sys, math
import numpy as np
from collections import deque
import random
import copy
import gym
import keras
from keras.models import Sequential
from keras.layers import Dense


############################################################
'''###########################################
CS221 Final Project: Linear Q-Learning Implementation
Authors:
Kongphop Wongpattananukul (kongw@stanford.edu)
Pouya Rezazadeh Kalehbasti (pouyar@stanford.edu)
Dong Hee Song (dhsong@stanford.edu)
###########################################'''

import sys, math
import numpy as np
from collections import defaultdict
import random

import gym

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
############################################################
class QLearningAlgorithm():
    def __init__(self,model, actions, discount, explorationProb=0.2, exploreProbDecay=0.99, explorationProbMin=0.01):
        self.actions = actions
        self.discount = discount
        self.explorationProb = explorationProb
        self.exploreProbDecay = exploreProbDecay
        self.explorationProbMin = explorationProbMin
        
        self.numIters = 0
        # self.state_bounds = [[-2,2] for i in range(8)]
        self.state_bins = [5 for i in  range(8)]
        # value = d.get(key, "empty")
        # self.feature_ranges = [[-1, 1], [2, 5]]  # 2 features
        self.state_bounds = [(-1.5,1.5) for i in range(6)] + [(0,1),(0,1)]
        # self.tiles_per_dim = [5,5,3,3,2,5,1,1]
        self.tiles_per_dim = [4,4,3,3,2,4,1,1]

        self.number_tilings = 3
        self.Tcoder = TileCoder(self.tiles_per_dim, self.state_bounds, self.number_tilings)

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions)
        else:
            return np.argmax([model[(state, action)] for action in self.actions])
    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        # return 1.0 / math.sqrt(self.numIters)
        return 0.01

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, reward, newState):
        # BEGIN_YOUR_CODE (our solution is 9 lines of code, but don't worry if you deviate from this)
        # calculate gradient
        eta = self.getStepSize()
        if newState is not None:
            # find maximum Q value from next state
            # print(newState,action)
            V_opt = max([model[(newState, possibleAction)] for possibleAction in self.actions])
        else:
            # V_opt of end state is 0
            V_opt = 0.0
        Q_opt = model[(state, action)]

        target = reward + self.discount * V_opt
        # update weight
        model[(state,action)] = Q_opt + 0.5 * (target-Q_opt)
                # y[[ind], [actions]] = y[[ind], [actions]]+ 0.5*(rewards + self.discount*(np.amax(self.model.predict(newStates), axis=1))*(1-dones) - y[[ind], [actions]])
# 
        # for f, v in self.featureExtractor(state, action):
        #     self.weights[f] -=  eta * (Q_opt - target) * v
        # END_YOUR_CODE
    def discretise_state(self,state):
        ratios = [(state[i] + abs(self.state_bounds[i][0])) / (self.state_bounds[i][1] - self.state_bounds[i][0]) for i
          in range(len(state))]
        state_d = [int(round((self.state_bins[i] - 1) * ratios[i])) for i in range(len(state))]
        state_d = [min(self.state_bins[i] - 1, max(0, state_d[i])) for i in range(len(state))]
        return tuple(state_d)
#     return features

    def create_tiling(self,feat_range, bins, offset):
        """
        Create 1 tiling spec of 1 dimension(feature)
        feat_range: feature range; example: [-1, 1]
        bins: number of bins for that feature; example: 10
        offset: offset for that feature; example: 0.2
        """
    
        return np.linspace(feat_range[0], feat_range[1], bins,endpoint=True)[:] + offset

    def create_tilings(self,feature_ranges, number_tilings, bins, offsets):
        """
        feature_ranges: range of each feature; example: x: [-1, 1], y: [2, 5] -> [[-1, 1], [2, 5]]
        number_tilings: number of tilings; example: 3 tilings
        bins: bin size for each tiling and dimension; example: [[10, 10], [10, 10], [10, 10]]: 3 tilings * [x_bin, y_bin]
        offsets: offset for each tiling and dimension; example: [[0, 0], [0.2, 1], [0.4, 1.5]]: 3 tilings * [x_offset, y_offset]
        """
        tilings = []
        # for each tiling
        for tile_i in range(number_tilings):
            tiling_bin = bins[tile_i]
            tiling_offset = offsets[tile_i]

            tiling = []
            # for each feature dimension
            for feat_i in range(len(feature_ranges)):
                feat_range = feature_ranges[feat_i]
                # tiling for 1 feature
                feat_tiling = self.create_tiling(feat_range, tiling_bin[feat_i], tiling_offset[feat_i])
                tiling.append(feat_tiling)
            tilings.append(tiling)
        return np.array(tilings)


    def get_tile_coding(self,feature, tilings):
        """
        feature: sample feature with multiple dimensions that need to be encoded; example: [0.1, 2.5], [-0.3, 2.0]
        tilings: tilings with a few layers
        return: the encoding for the feature on each layer
        """
        num_dims = len(feature)
        feat_codings = []
        for tiling in tilings:
            feat_coding = []
            for i in range(num_dims):
                feat_i = feature[i]
                tiling_i = tiling[i]  # tiling on that dimension
                coding_i = np.digitize(feat_i, tiling_i)
                # print(coding_i)
                feat_coding.append(coding_i)
            feat_codings += feat_coding
        return tuple(feat_codings)
        


# Perform |numTrials| of the following:
# On each trial, take the MDP |mdp| and an RLAlgorithm |rl| and simulates the
# RL algorithm according to the dynamics of the MDP.
# Each trial will run for at most |maxIterations|.
# Return the list of rewards that we get for each trial.

def simulate(env, rl, numTrials=100, train=False, verbose=False, render=False):
    totalRewards = []  # The rewards we get on each trial
    for trial in range(numTrials):
        state = env.reset()
        totalReward = 0
        iteration = 0
        state_d = rl.discretise_state(state)
        while iteration < 3000:#iteration < 1000:
            
            action = rl.getAction(state_d)
            newState, reward, done, info = env.step(action)
            newState_d = rl.discretise_state(newState)
            if train:
                rl.incorporateFeedback(state_d, action, reward, newState_d)
            totalReward += reward
            rl.explorationProb = max(rl.exploreProbDecay * rl.explorationProb,
                                         rl.explorationProbMin)
            state_d = newState_d
            iteration += 1

            if done:
                # print(iteration)
                break
            if verbose == True and render:
                still_open = env.render()
                # print(iteration)
                if still_open == False: break

        totalRewards.append(totalReward)
        if verbose and trial % 20 == 0:
            print(('\n---- Trial {} ----'.format(trial)))
            print(('Mean(last 10 total rewards): {}'.format(np.mean(totalRewards[-100:]))))
            # print(('Size(weight vector): {}'.format(len(rl.weights))))
        
    return totalRewards, model

def simulate_tl(env, rl, numTrials=100, train=False, verbose=False, render=False):
    totalRewards = []  # The rewards we get on each trial
    # tilings = rl.create_tilings(rl.state_bounds,rl.number_tilings,rl.bins,rl.offsets)
    # print(tilings)
    for trial in range(numTrials):
        state = env.reset()
        totalReward = 0
        iteration = 0
        state_d = rl.Tcoder[tuple(state)]
        # print(state_d)
        # print(state_d)
        # sdaasd
        while iteration < 3000:#iteration < 1000:
            
            action = rl.getAction(state_d)
            newState, reward, done, info = env.step(action)
            newState_d = rl.Tcoder[tuple(newState)]
            if train:
                rl.incorporateFeedback(state_d, action, reward, newState_d)
            totalReward += reward
            rl.explorationProb = max(rl.exploreProbDecay * rl.explorationProb,
                                         rl.explorationProbMin)
            state_d = newState_d
            iteration += 1

            if done:
                # print(iteration)
                break
            if verbose == True and render:
                still_open = env.render()
                # print(iteration)
                if still_open == False: break

        totalRewards.append(totalReward)
        if verbose and trial % 20 == 0:
            print(('\n---- Trial {} ----'.format(trial)))
            print(('Mean(last 10 total rewards): {}'.format(np.mean(totalRewards[-100:]))))
            # print(('Size(weight vector): {}'.format(len(rl.weights))))
        
    return totalRewards, model
# Helper functions for storing and loading the weights
import pickle
def saveF(obj, name):
    with open('weights/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def loadF(name):
    with open('weights/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)



# Helper functions for storing and loading the weights
import pickle
def saveF(obj, name):
    with open('weights/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def loadF(name):
    with open('weights/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


## Main variables
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


# Main function
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
    rl = QLearningAlgorithm(model,[0, 1, 2, 3], discountFactor, explorProbInit, exploreProbDecay,
                        explorationProbMin)
    totalRewards, model = simulate_tl(env, rl, numTrials=numTrials, train=True, verbose=True, render=False)
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
    rl = QLearningAlgorithm(model,[0, 1, 2, 3], discountFactor, 0.0, 1,
                            0.00)
    totalRewards, _ = simulate_tl(env, rl, numTrials=numTestTrials, train=False, verbose=True, render=True)
    print('Average Total Testing Reward: {}'.format(np.mean(totalRewards)))

