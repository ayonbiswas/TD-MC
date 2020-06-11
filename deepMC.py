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
class SarsaAlgorithm():
    def __init__(self, actions, discount, weights, explorationProb=0.2, exploreProbDecay=0.99, explorationProbMin=0.01, batchSize=32):
        self.actions = actions
        self.discount = discount
        self.explorationProb = explorationProb
        self.exploreProbDecay = exploreProbDecay
        self.explorationProbMin = explorationProbMin
        self.weights = weights
        self.numIters = 0
        self.model = NeuralNetwork(batchSize, weights)


    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        if np.random.rand() < self.explorationProb:
            return random.choice(self.actions)
        else:
            predScores = self.model.predict(state)[0]
            return np.argmax(predScores)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, states, actions, rewards):
        # initialize variable
        states = np.squeeze(states)
        X = states
        y = self.model.predict(states)
        # targets = rewards
        # print(targets.shape)
        alpha = 0.5
        # targets = rewards + self.discount*(np.amax(self.model.predict(newStates), axis=1))*(1-dones)
        ind = np.array([i for i in range(len(states))])
        y[[ind], [actions]] = (1-alpha)*y[[ind], [actions]] + alpha*(rewards)
        
        # y[[ind], [actions]] = targets
        # update weight
        self.model.fit(X, y)

    # def updateCache(self, state, action, reward, newState, newAction, done):
    #     self.cache.append((state, action, reward, newState, newAction, done))

# neural network
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
        state = np.reshape(env.reset(), (1,8))
        totalReward = 0
        iteration = 0
        trajectory = []
        rewards = []

        while True:

            action = rl.getAction(state)
            newState, reward, done, info = env.step(action)
            newState = np.reshape(newState, (1,8))
            trajectory.append((state, action, newState, reward, done))
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
                data.append((trajectory[i][0], trajectory[i][1], L))

            for i in range(100):
                batch = random.sample(data, batchSize)
                states = np.array([sample[0] for sample in batch])
                actions = np.array([sample[1] for sample in batch])
                rewards = np.array([sample[2] for sample in batch])
                rl.incorporateFeedback(states, actions, rewards)


            rl.explorationProb = max(rl.exploreProbDecay * rl.explorationProb, rl.explorationProbMin)

        totalRewards.append(totalReward)
        mean_totalReward = np.mean(totalRewards[-5:])
        if((mean_totalReward>max_totalReward) and (train==True)):
            # Save Weights
            saveF(rl.weights, 'deep_MC')
            max_totalReward = mean_totalReward
            print('The weights are saved with total rewards: ',mean_totalReward)

        print(('Trial {} Total Reward: {}'.format(trial, totalReward)))

    return totalRewards

## Main variables
numTrials = 4000
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
    # # Warm start weights
    # #weights = './weights/weights_sarsa.h5'

    # TRAIN
    # print('\n++++++++++++ TRAINING +++++++++++++')
    # rl = SarsaAlgorithm([0, 1, 2, 3], discountFactor, weights,
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
    weights = loadF('deep_MC')
    print(weights)
    env = gym.make('LunarLander-v2')
    rl = SarsaAlgorithm([0, 1, 2, 3], discountFactor, weights, 0.0, 0.0, 0.0, batchSize)
    totalRewards = simulate(env, rl, numTrials=numTestTrials, train=False, verbose=True, trialDemoInterval=trialDemoInterval)
    env.close()
    print('Average Total Testing Reward: {}'.format(np.mean(totalRewards)))
