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

############################################################
class QLearningAlgorithm():
    def __init__(self, model, actions, discount, explorationProb=0.2, exploreProbDecay=0.99, explorationProbMin=0.01):
        self.actions = actions
        self.discount = discount
        self.explorationProb = explorationProb
        self.exploreProbDecay = exploreProbDecay
        self.explorationProbMin = explorationProbMin
        # model = defaultdict(float)
        self.numIters = 0
        self.state_bounds = [[-1,1] for i in range(8)]
        self.state_bins = [5 for i in  range(8)]
        # value = d.get(key, "empty")


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
    def incorporateFeedback(self, state, action, reward, newState, newAction):
        # BEGIN_YOUR_CODE (our solution is 9 lines of code, but don't worry if you deviate from this)
        # calculate gradient
        eta = self.getStepSize()
        if newState is not None:
            # find maximum Q value from next state
            # print(newState,action)
            V_opt = model[(newState, newAction)] 
        else:
            # V_opt of end state is 0
            V_opt = 0.0
        Q_opt = model[(state, action)]
        target = reward + self.discount * V_opt
        # update weight
        model[(state,action)] = Q_opt + eta * (target-Q_opt)
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
        while iteration < 1000:#iteration < 1000:
            
            action = rl.getAction(state_d)
            newState, reward, done, info = env.step(action)
            
            newState_d = rl.discretise_state(newState)
            newAction = rl.getAction(newState_d)
            if train:
                rl.incorporateFeedback(state_d, action, reward, newState_d,newAction)
            totalReward += reward
            rl.explorationProb = max(rl.exploreProbDecay * rl.explorationProb,
                                         rl.explorationProbMin)
            state_d = newState_d
            iteration += 1
            if verbose == True and render:
                still_open = env.render()
                # print(iteration)
                if still_open == False: break
            if done:
                # print(iteration)
                break

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
numTrials = 3000 #30000
numTestTrials = 1000
trialDemoInterval = numTrials/2
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
    rl = QLearningAlgorithm(model,[0, 1, 2, 3], discountFactor, explorProbInit, exploreProbDecay,
                            explorationProbMin)
    env = gym.make('LunarLander-v2')
    # env.seed(0)
    print('\n++++++++++++ TRAINING +++++++++++++')
    totalRewards, model = simulate(env, rl, numTrials=numTrials, train=True, verbose=True, render =False)
    env.close()
    print('Average Total Training Reward: {}'.format(np.mean(totalRewards)))
    # print(rl.model)
    # # Save Weights
    saveF(model, 'model_discrete_sarsa')
    # with open('weights.json', 'w') as fileOpen:
        # json.dump(model, fileOpen)
    model = loadF('model_discrete_sarsa')
    
    # TEST
    print('\n\n++++++++++++++ TESTING +++++++++++++++')
    rl = QLearningAlgorithm(model,[0, 1, 2, 3], discountFactor, 0.0, 1,
                            0.00)
    totalRewards, _ = simulate(env, rl, numTrials=numTestTrials, train=False, verbose=True,render  =True)
    print('Average Total Testing Reward: {}'.format(np.mean(totalRewards)))

