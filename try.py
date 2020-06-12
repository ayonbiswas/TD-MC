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
import itertools


class SarsaLinear():
    def __init__(self,env, actions, discount, weights, explorationProb=0.2, exploreProbDecay=0.99, explorationProbMin=0.01, batchSize = 1):
        self.actions = actions
        self.discount = discount
        self.explorationProb = explorationProb
        self.exploreProbDecay = exploreProbDecay
        self.explorationProbMin = explorationProbMin
        self.weights = weights
        self.numIters = 0
        self.env = env
        self.model = NeuralNetwork(batchSize, weights)

        self.alpha = 0.5
    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for x in self.featureExtractor(state, action):
            score += self.weights[x] 
        return score
    def getAction(self, state):
        if np.random.rand() < self.explorationProb:
            return random.choice(self.actions)
        else:
            features = self.featureExtractor(state)
            features = features.reshape((1, 10))
            #print(features.shape)
            predScores = self.model.predict(features)[0]
            return np.argmax(predScores)


    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)


    def update(self, state, action, reward, newState, newAction,done):
        # calculate gradient
        state = state.reshape(1, 8)
        newState = newState.reshape(1, 8)

        X = state
        f1 = self.featureExtractor(state)
        y = self.model.predict(f1.reshape(1,10))
        f2 = self.featureExtractor(newState)
        Q = self.model.predict(f2.reshape(1,10))
        Q_next = np.take(Q,newAction)
        ind = np.array([i for i in range(len(state))])
        y[[ind], [action]] = (1-self.alpha)*y[[ind], [action]] + self.alpha*(reward + self.discount*(Q_next)*(1-done))

        self.model.fit(f1.reshape(1,10), y)
        # update weight
        # for f, v in self.featureExtractor(state, action):
        #     self.weights[f] -=  eta * (Q_opt - target) 


    # def update(self, features, actions, rewards):
    #     # initialize variable
    #     #state = np.squeeze(states)
    #     # print(len(features))

    #     X = features
    #     y = self.model.predict(features)
    #     ind = np.array([i for i in range(len(features))])
    #     y[[ind], [actions]] = rewards
    #     # update weight
    #     self.model.fit(X, y)


    def featureExtractor(self, state):
    # Action: 0: Nop, 1: fire left engine, 2: main engine, 3: right engine
        # print(state.shape)
        state = state[0]
        # print("aa",state.shape)
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
        # print(farray.shape)
        return farray

    def simulate( self, numTrials=100, train=False, verbose=False, trialDemoInterval=10):
        totalRewards = []  # The rewards we get on each trial
        max_totalReward = 0.0
        for trial in range(numTrials):
            state = self.env.reset()
            state = state.reshape(1,8)
            action = self.getAction(state)

            totalReward = 0
            iteration = 0
            while True:
                
                newState, reward, done, info = self.env.step(action)
                newState = newState.reshape(1,8)
                newAction = self.getAction(newState)

                if (train == True):
                    self.update(state, action, reward, newState, newAction,done)
                    self.explorationProb = max(self.exploreProbDecay * self.explorationProb, self.explorationProbMin)

                totalReward += reward
                state = newState
                action = newAction
                iteration += 1

                if verbose == True:
                    still_open = self.env.render()
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
                # print(('Size(weight vector): {}'.format(len(self.weights))))

        return totalRewards, self.weights


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




## Main variables
numTrials = 15000
numTestTrials = 10
trialDemoInterval = numTrials/2
discountFactor = 0.99
explorProbInit = 1.0
exploreProbDecay = 0.999
explorationProbMin = 0.01
batchSize = 1

if __name__ == '__main__':
    # Initiate weights
    # Cold start weights
    weights = None
    # Warm start weights
    #weights = './weights/weights_sarsa.h5'
    env = gym.make('LunarLander-v2')
    # TRAIN
    print('\n++++++++++++ TRAINING +++++++++++++')
    rl = SarsaLinear(env, [0, 1, 2, 3], discountFactor, weights,
                            explorProbInit, exploreProbDecay,
                            explorationProbMin, batchSize)
    
    totalRewards_list = []
    totalRewards = rl.simulate( numTrials=numTrials, train=True, verbose=False,
                            trialDemoInterval=trialDemoInterval)
    env.close()
    # Save Weights

    # TEST
    print('\n\n++++++++++++++ TESTING +++++++++++++++')
    weights = 'linear_MC.h5'
    env = gym.make('LunarLander-v2')
    rl = SarsaLinear(env, [0, 1, 2, 3], discountFactor, weights, 0.0, 0.0, 0.0, batchSize)
    totalRewards = rl.simulate( numTrials=numTestTrials, train=False, verbose=True, trialDemoInterval=trialDemoInterval)
    env.close()
    print('Average Total Testing Reward: {}'.format(np.mean(totalRewards)))

