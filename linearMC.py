import sys, math
import numpy as np
from collections import deque, defaultdict
import random
import copy
import gym
import keras
from keras.models import Sequential
from keras.layers import Dense


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
class SarsaAlgorithm():
    def __init__(self,model, actions, discount, weights, explorationProb=0.2, exploreProbDecay=0.99, explorationProbMin=0.01, batchSize=32):
        self.actions = actions
        self.discount = discount
        self.explorationProb = explorationProb
        self.exploreProbDecay = exploreProbDecay
        self.explorationProbMin = explorationProbMin
        self.weights = weights
        self.numIters = 0
        # self.model = NeuralNetwork(batchSize, weights)
        self.state_bounds = [(-1.5,1.5) for i in range(6)] + [(0,1),(0,1)]
        # self.tiles_per_dim = [5,5,5,5,5,5,1,1]
        self.tiles_per_dim = [3,3,3,3,2,3,1,1]

        self.number_tilings = 3
        self.Tcoder = TileCoder(self.tiles_per_dim, self.state_bounds, self.number_tilings)

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions)
        else:
            return np.argmax([model[(state, action)] for action in self.actions])

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, reward):
        # BEGIN_YOUR_CODE (our solution is 9 lines of code, but don't worry if you deviate from this)
        # calculate gradient
        # eta = self.getStepSize()
        # if newState is not None:
            # find maximum Q value from next state
            # print(newState,action)
            # V_opt = max([model[(newState, possibleAction)] for possibleAction in self.actions])
        # else:
            # V_opt of end state is 0
            # V_opt = 0.0
        # Q_opt = 

        # target = reward + self.discount * V_opt
        # update weight
        model[(state,action)] = 0.5*model[(state, action)] + 0.5 * (reward)

    # def updateCache(self, state, action, reward, newState, newAction, done):
    #     self.cache.append((state, action, reward, newState, newAction, done))

# neural network


# Helper functions for storing and loading the weights
import pickle
def saveF(obj, name):
    with open('weights/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def loadF(name):
    with open('weights/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def simulate(env, rl, numTrials=10, train=False, verbose=False,
             trialDemoInterval=10, batchSize=32):
    totalRewards = []  # The rewards we get on each trial
    max_totalReward = 0
    for trial in range(numTrials):
        state = env.reset()
        totalReward = 0
        iteration = 0
        trajectory = []
        rewards = []
        state = rl.Tcoder[tuple(state)]
        while True:
            # print(state.shape,tuple(state))
            
            action = rl.getAction(state)
            newState, reward, done, info = env.step(action)
            
            
            # newState = np.reshape(newState, (1,8))
            newState =rl.Tcoder[tuple(newState)] 
            trajectory.append((state, action,newState , reward, done))
            rewards.append(reward)

            if verbose == True:
                still_open = env.render()
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
                    L += np.power(rl.discount, r)*rewards[r+i]
                # data.append((trajectory[i][0], trajectory[i][1], L))
                model[(trajectory[i][0],trajectory[i][1])] = 0.5*model[(trajectory[i][0], trajectory[i][1])] + 0.5 * (L)
                # rl.incorporateFeedback(, , L)
            # for i in range(100):
            #     batch = random.sample(data, batchSize)

            # states = np.array([sample[0] for sample in data])
            # # print(states)
            # actions = np.array([sample[1] for sample in data])
            # rewards = np.array([sample[2] for sample in data])
            


            rl.explorationProb = max(rl.exploreProbDecay * rl.explorationProb, rl.explorationProbMin)

        totalRewards.append(totalReward)
        mean_totalReward = np.mean(totalRewards[-5:])
        if((mean_totalReward>max_totalReward) and (train==True)):
            # Save Weights
            saveF(model, 'linear_MC')
            max_totalReward = mean_totalReward
            print('The weights are saved with total rewards: ',mean_totalReward)
        if(trial %50 == 0):
            print(('Trial {} Total Reward: {}'.format(trial, totalReward)))

    return totalRewards

## Main variables
numTrials = 100000
numTestTrials = 5
trialDemoInterval = numTrials/2
discountFactor = 0.99
explorProbInit = 1.0
exploreProbDecay = 0.999
explorationProbMin = 0.01
batchSize = 2

if __name__ == '__main__':
    # Initiate weights
    # Cold start weights
    weights = None
    # # # Warm start weights
    # # #weights = './weights/weights_sarsa.h5'
    # model = defaultdict(float)
    # # TRAIN
    # print('\n++++++++++++ TRAINING +++++++++++++')
    # rl = SarsaAlgorithm(model, [0, 1, 2, 3], discountFactor, weights,
    #                         explorProbInit, exploreProbDecay,
    #                         explorationProbMin, batchSize)
    # env = gym.make('LunarLander-v2')
    # totalRewards_list = []
    # totalRewards = simulate(env, rl, numTrials=numTrials, train=True, verbose=False,
    #                         trialDemoInterval=trialDemoInterval, batchSize=batchSize)
    # env.close()
    # Save Weights

    # TEST
    print('\n\n++++++++++++++ TESTING +++++++++++++++')
    model = loadF('linear_MC')
    # print(weights)
    env = gym.make('LunarLander-v2')
    rl = SarsaAlgorithm(model,[0, 1, 2, 3], discountFactor, weights, 0.0, 0.0, 0.0, batchSize)
    totalRewards = simulate(env, rl, numTrials=numTestTrials, train=False, verbose=True, trialDemoInterval=trialDemoInterval)
    env.close()
    print('Average Total Testing Reward: {}'.format(np.mean(totalRewards)))
