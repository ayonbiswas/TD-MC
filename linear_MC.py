import sys, math
import numpy as np
from collections import deque
import random
import copy
import gym
import keras
import itertools
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
            features = modFeatureExtractor(state)
            features = features.reshape((1, 10))
            #print(features.shape)
            predScores = self.model.predict(features)[0]
            return np.argmax(predScores)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, features, actions, rewards):
        # initialize variable
        #states = np.squeeze(states)
        # print(len(features))

        X = features
        y = self.model.predict(features)
        ind = np.array([i for i in range(len(features))])
        y[[ind], [actions]] = rewards
        # update weight
        self.model.fit(X, y)

# neural network
class NeuralNetwork():
    def __init__(self, batchSize = 32, weights=None):
        self.model = Sequential()
        self.model.add(Dense(4, input_dim=10, activation='linear'))
        sgd = keras.optimizers.SGD(learning_rate=0.001)
        self.model.compile(loss='mse', optimizer=sgd)
        if isinstance(weights, str):
            self.model.load_weights(weights)

    def predict(self, state):
        return self.model.predict_on_batch(state)

    def fit(self, X, y):
        self.model.fit(X, y, epochs=1,  verbose=0)

    def save(self, weights):
        self.model.save_weights(weights)


def modFeatureExtractor(state):
    # Action: 0: Nop, 1: fire left engine, 2: main engine, 3: right engine

    state = state[0]

    name = {'angle1', 'hover1', 'hover2', 'x1', 'x2'}
    switch = {True, False}
    features_list = list(itertools.product(name, switch))
    features = dict.fromkeys(features_list, 0)
    #print((features))

    #Features
    x_todo = state[0]*0.5 + state[2]*1.0
    angle_todo = state[4]*0.5 + state[5]*1.0
    hover_todo = 0.5*np.abs(state[0]) - state[1]*0.5 -(state[3])*0.5

    if state[6] or state[7]: # legs have contact
        hover_todo = -(state[3])*0.5  # override to reduce fall speed, that's all we need after contact
    features[('angle1', hover_todo > 0.05)] = 1
    features[('hover1', angle_todo < -0.05)] = 1
    features[('hover2', angle_todo > +0.05)] = 1
    features[('x1', x_todo < -0.1)] = 1
    features[('x2', x_todo > +0.1)] = 1
    f = list(features.values())
    farray = np.array(f)
    return farray


def simulate(env, rl, numTrials=10, train=False, verbose=False,
             trialDemoInterval=10, batchSize=32):
    totalRewards = []  # The rewards we get on each trial
    max_totalReward = 0
    for trial in range(numTrials):
        state = np.reshape(env.reset(), (1, 8))
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
                features = modFeatureExtractor(trajectory[i][0])
                data.append((features, trajectory[i][1], L))
            #print(data)
            for i in range(100):
                batch = random.sample(data, batchSize)
                features = np.array([sample[0] for sample in batch])
                actions = np.array([sample[1] for sample in batch])
                rewards = np.array([sample[2] for sample in batch])
                rl.incorporateFeedback(features, actions, rewards)

            if(trial % 2 == 0):
                rl.explorationProb = max(rl.exploreProbDecay * rl.explorationProb, rl.explorationProbMin)

        totalRewards.append(totalReward)
        mean_totalReward = np.mean(totalRewards[-100:])
        if((mean_totalReward>max_totalReward) and (train==True)):
            # Save Weights
            rl.model.save('linear_MC.h5')
            max_totalReward = mean_totalReward
            print('The weights are saved with total rewards: ',mean_totalReward)

        print(('Trial {} Total Reward: {}'.format(trial, totalReward)))

    return totalRewards

## Main variables
numTrials = 15000
numTestTrials = 10
trialDemoInterval = numTrials/2
discountFactor = 0.99
explorProbInit = 1.0
exploreProbDecay = 0.999
explorationProbMin = 0.01
batchSize = 5

if __name__ == '__main__':
    # Initiate weights
    # Cold start weights
    weights = None
    # Warm start weights
    #weights = './weights/weights_sarsa.h5'

    # TRAIN
    print('\n++++++++++++ TRAINING +++++++++++++')
    rl = SarsaAlgorithm([0, 1, 2, 3], discountFactor, weights,
                            explorProbInit, exploreProbDecay,
                            explorationProbMin, batchSize)
    env = gym.make('LunarLander-v2')
    totalRewards_list = []
    totalRewards = simulate(env, rl, numTrials=numTrials, train=True, verbose=False,
                            trialDemoInterval=trialDemoInterval, batchSize=batchSize)
    env.close()
    # Save Weights

    # TEST
    print('\n\n++++++++++++++ TESTING +++++++++++++++')
    weights = 'linear_MC.h5'
    env = gym.make('LunarLander-v2')
    rl = SarsaAlgorithm([0, 1, 2, 3], discountFactor, weights, 0.0, 0.0, 0.0, batchSize)
    totalRewards = simulate(env, rl, numTrials=numTestTrials, train=False, verbose=True, trialDemoInterval=trialDemoInterval)
    env.close()
    print('Average Total Testing Reward: {}'.format(np.mean(totalRewards)))
