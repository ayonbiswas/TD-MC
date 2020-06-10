import sys, math
import numpy as np
from collections import defaultdict
import random

import gym

import json

############################################################
class QLearningAlgorithm():
    def __init__(self, actions, discount, featureExtractor, weights, explorationProb=0.2, exploreProbDecay=0.99, explorationProbMin=0.01):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.exploreProbDecay = exploreProbDecay
        self.explorationProbMin = explorationProbMin
        self.weights = weights
        self.numIters = 0


    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions)
        else:
            return max((self.getQ(state, action), action) for action in self.actions)[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)


    def incorporateFeedback(self, state, action, reward, newState, newAction):
        # calculate gradient
        eta = self.getStepSize()
        if newState is not None:
            V_opt = self.getQ(newState, newAction)
        else:
            V_opt = 0
        Q_opt = self.getQ(state, action)
        target = reward + self.discount * V_opt
        # update weight
        for f, v in self.featureExtractor(state, action):
            self.weights[f] -=  eta * (Q_opt - target) * v


def modFeatureExtractor(state, action):
    # Action: 0: Nop, 1: fire left engine, 2: main engine, 3: right engine
    x, y, Vx, Vy, Th, VTh, LeftC, RightC = state

    #Features
    x_todo = state[0]*0.5 + state[2]*1.0
    angle_todo = state[4]*0.5 + state[5]*1.0
    hover_todo = 0.5*np.abs(state[0]) - state[1]*0.5 -(state[3])*0.5

    if state[6] or state[7]: # legs have contact
        hover_todo = -(state[3])*0.5  # override to reduce fall speed, that's all we need after contact

    features = []
    features.append((('angle1', hover_todo > 0.05, action),1))
    features.append((('hover1', angle_todo < -0.05, action),1))
    features.append((('hover2', angle_todo > +0.05, action),1))
    features.append((('x1', x_todo < -0.1, action),1))
    features.append((('x2', x_todo > +0.1, action),1))


    return features


def simulate(env, rl, numTrials=100, train=False, verbose=False, trialDemoInterval=10):
    totalRewards = []  # The rewards we get on each trial
    max_totalReward = 0.0
    for trial in range(numTrials):
        state = env.reset()
        action = rl.getAction(state)
        totalReward = 0
        iteration = 0
        while True:#iteration < 1000:
            #action = rl.getAction(state)
            newState, reward, done, info = env.step(action)
            newAction = rl.getAction(newState)

            if (train == True):
                rl.incorporateFeedback(state, action, reward, newState, newAction)
                rl.explorationProb = max(rl.exploreProbDecay * rl.explorationProb, rl.explorationProbMin)

            totalReward += reward
            state = newState
            action = newAction
            iteration += 1

            if verbose == True:
                still_open = env.render()
                if still_open == False: break

            if done:
                break

        totalRewards.append(totalReward)
        mean_totalReward = np.mean(totalRewards[-10:])
        if((mean_totalReward>max_totalReward) and (train==True)):
            # Save Weights
            saveF(weights, 'linear_sarsa')
            max_totalReward = mean_totalReward
            print('The weights are saved with total rewards: ',mean_totalReward)

        if trial % 20 == 0:
            print(('\n---- Trial {} ----'.format(trial)))
            print(('Mean(last 20 total rewards): {}'.format(np.mean(totalRewards[-20:]))))
            print(('Size(weight vector): {}'.format(len(rl.weights))))

    return totalRewards, rl.weights


# Helper functions for storing and loading the weights
import pickle
def saveF(obj, name):
    with open('weights/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def loadF(name):
    with open('weights/'+ name + '.pkl', 'rb') as f:
        return pickle.load(f)


## Main variables
numFeatures = 4
numActions = 4
numTrials = 2000
numTestTrials = 30
trialDemoInterval = numTrials/2
discountFactor = 0.99
explorProbInit = 1
exploreProbDecay = 0.99
explorationProbMin = 0.01


# Main function
if __name__ == '__main__':
    # Cold start weights
    weights = defaultdict(float)
    # Warm start weights
    # weights = loadF('weights')

    # TRAIN
    rl = QLearningAlgorithm([0, 1, 2, 3], discountFactor, modFeatureExtractor,
                            weights, explorProbInit, exploreProbDecay,
                            explorationProbMin)
    env = gym.make('LunarLander-v2')
    # env.seed(0)
    print('\n++++++++++++ TRAINING +++++++++++++')
    totalRewards, weights = simulate(env, rl, numTrials=numTrials, train=True, verbose=False, trialDemoInterval=trialDemoInterval)
    env.close()
    print('Average Total Training Reward: {}'.format(np.mean(totalRewards)))



    #TEST
    #load weights
    weights = loadF('linear_sarsa')
    env = gym.make('LunarLander-v2')
    print('\n\n++++++++++++++ TESTING +++++++++++++++')
    rl = QLearningAlgorithm([0, 1, 2, 3], discountFactor, modFeatureExtractor,
                            weights, 0.0, 1,
                            0.00)
    totalRewards, _ = simulate(env, rl, numTrials=numTestTrials, train=False, verbose=True, trialDemoInterval=trialDemoInterval)
    env.close()
    print('Average Total Testing Reward: {}'.format(np.mean(totalRewards)))
