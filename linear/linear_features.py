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
import pickle
from gym import wrappers

class SarsaLinear():
    def __init__(self, env,  discount, weights, explorationProb=0.2, exploreProbDecay=0.99, explorationProbMin=0.01):
        self.env = env
        self.actions = np.arange(self.env.action_space.n)
        self.discount = discount
        self.explorationProb = explorationProb
        self.exploreProbDecay = exploreProbDecay
        self.explorationProbMin = explorationProbMin
        self.weights = weights
        self.numIters = 0



    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for x in self.FeatureExtractor(state, action):
            score += self.weights[x] 
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


    def update(self, state, action, reward, newState, newAction):
        # calculate gradient
        eta = self.getStepSize()
        if newState is not None:
            V_opt = self.getQ(newState, newAction)
        else:
            V_opt = 0
        Q_opt = self.getQ(state, action)
        target = reward + self.discount * V_opt
        # update weight
        for f in self.FeatureExtractor(state, action):
            self.weights[f] -=  eta * (Q_opt - target) 


    def FeatureExtractor(self,state, action):
        # Action: 0: Nop, 1: fire left engine, 2: main engine, 3: right engine

        #Features
        x_todo = state[0]*0.5 + state[2]*1.0
        angle_todo = state[4]*0.5 + state[5]*1.0
        hover_todo = 0.5*np.abs(state[0]) - state[1]*0.5 -(state[3])*0.5

        if state[6] or state[7]: # legs have contact
            hover_todo = -(state[3])*0.5  # override to reduce fall speed, that's all we need after contact

        features = []
        features.append(('angle1', hover_todo > 0.05, action))
        features.append(('hover1', angle_todo < -0.05, action))
        features.append(('hover2', angle_todo > +0.05, action))
        features.append(('x1', x_todo < -0.1, action))
        features.append(('x2', x_todo > +0.1, action))


        return features


    def simulate(self, numTrials=100, train=False, verbose=False):
        totalRewards = []  # The rewards we get on each trial
        episode_length = []
        max_totalReward = 0.0

        if not train:   
        # self.env.close()
            self.env = wrappers.Monitor(self.env, './video/Lsarsa', video_callable=lambda episode_id: True,force = True)
        for trial in range(numTrials):
            state = self.env.reset()
            action = self.getAction(state)
            totalReward = 0
            iteration = 0
            while True:

                newState, reward, done, info = self.env.step(action)
                newAction = self.getAction(newState)

                if (train == True):
                    self.update(state, action, reward, newState, newAction)
                    self.explorationProb = max(self.exploreProbDecay * self.explorationProb, self.explorationProbMin)

                totalReward += reward
                state = newState
                action = newAction
                iteration += 1

                if done:
                    break
            totalRewards.append(totalReward)
            episode_length.append(iteration)
            mean_totalReward = np.mean(totalRewards[-10:])
            if((mean_totalReward>max_totalReward) and (train==True)):
                # Save Weights
                self.saveF(self.weights,'linear_sarsa_{}'.format(trial))
                max_totalReward = mean_totalReward
                print('The weights are saved with total rewards: ',mean_totalReward)
            if(not train):
                print(('Trial {} Total Reward: {}'.format(trial, totalReward)))

        

            if trial % 20 == 0:
                print(('\n---- Trial {} ----'.format(trial)))
                print(('Mean(last 20 total rewards): {}'.format(np.mean(totalRewards[-20:]))))
        if(train):
            np.save("./rewards/linear_sarsa_{}.npy".format(numTrials), totalRewards)
            np.save("./episode_length/linear_sarsa_{}.npy".format(numTrials), episode_length)

        return totalRewards, self.weights

    def saveF(self,obj, name):
        with open('./weights/' + name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def loadF(name):
    with open('./weights/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)



class QLearningLinear(SarsaLinear):
    def __init__(self,env,  discount, weights, explorationProb=0.2, exploreProbDecay=0.99, explorationProbMin=0.01):
        super().__init__(env,  discount, weights,  explorationProb=0.2, exploreProbDecay=0.99, explorationProbMin=0.01)

   

    # Return the Q function associated with the weights and features
    # Call this function to get the step size to update the weights.

    def update(self, state, action, reward, newState):
        # calculate gradient
        eta = self.getStepSize()
        if newState is not None:
            # find maximum Q value from next state
            V_opt = max(self.getQ(newState, possibleAction) for possibleAction in self.actions)
        else:
            # V_opt of end state is 0
            V_opt = 0.0
        Q_opt = self.getQ(state, action)
        target = reward + self.discount * V_opt
        # update weight
        for f in self.FeatureExtractor(state, action):
            self.weights[f] -=  eta * (Q_opt - target) 



    def simulate(self, numTrials=100, train=False, verbose=False):
        totalRewards = []  # The rewards we get on each trial
        episode_length = []
        max_totalReward = 0
        if not train:   
        # self.env.close()
            self.env = wrappers.Monitor(self.env, './video/LQlearn', video_callable=lambda episode_id: True,force = True)
        for trial in range(numTrials):
            state = self.env.reset()
            totalReward = 0
            iteration = 0
            while True:#iteration < 1000:
                action = self.getAction(state)
                newState, reward, done, info = self.env.step(action)

                if train:
                    self.update(state, action, reward, newState)
                totalReward += reward
                self.explorationProb = max(self.exploreProbDecay * self.explorationProb,
                                             self.explorationProbMin)
                state = newState
                iteration += 1


                if done:
                    break

            totalRewards.append(totalReward)
            episode_length.append(iteration)
            mean_totalReward = np.mean(totalRewards[-10:])
            if((mean_totalReward>max_totalReward) and (train==True)):
                # Save Weights
                self.saveF(self.weights,'linear_qlearning_{}'.format(trial))
                max_totalReward = mean_totalReward
                print('The weights are saved with total rewards: ',mean_totalReward)

            if(not train):
                print(('Trial {} Total Reward: {}'.format(trial, totalReward)))

            if trial % 20 == 0:
                print(('\n---- Trial {} ----'.format(trial)))
                print(('Mean(last 20 total rewards): {}'.format(np.mean(totalRewards[-20:]))))
        if(train):
            np.save("./rewards/linear_qlearning_{}.npy".format(numTrials), totalRewards)
            np.save("./episode_length/linear_qlearning_{}.npy".format(numTrials),episode_length)

        return totalRewards, self.weights

class Linear():
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


class MCLinear():
    def __init__(self,env, discount, weights, explorationProb=0.2, exploreProbDecay=0.99, explorationProbMin=0.01, batchSize=32):
        self.actions = np.arange(env.action_space.n)
        self.discount = discount
        self.explorationProb = explorationProb
        self.exploreProbDecay = exploreProbDecay
        self.explorationProbMin = explorationProbMin
        self.weights = weights
        self.numIters = 0
        self.env = env
        self.model = Linear(batchSize, weights)


    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        if np.random.rand() < self.explorationProb:
            return random.choice(self.actions)
        else:
            features = self.FeatureExtractor(state)
            features = features.reshape((1, 10))
            #print(features.shape)
            predScores = self.model.predict(features)[0]
            return np.argmax(predScores)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def update(self, features, actions, rewards):
        # initialize variable
        #states = np.squeeze(states)
        # print(len(features))

        X = features
        y = self.model.predict(features)
        ind = np.array([i for i in range(len(features))])
        y[[ind], [actions]] = rewards
        # update weight
        self.model.fit(X, y)




    def FeatureExtractor(self, state):
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


    def simulate(self, numTrials=10, train=False, verbose=False,
                 batchSize=32):
        totalRewards = []  # The rewards we get on each trial
        max_totalReward = 0
        episode_length = []
        if not train:   
        # self.env.close()
            self.env = wrappers.Monitor(self.env, './video/LMC', video_callable=lambda episode_id: True,force = True)
        for trial in range(numTrials):
            state = np.reshape(self.env.reset(), (1, 8))
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
                    features = self.FeatureExtractor(trajectory[i][0])
                    data.append((features, trajectory[i][1], L))
                #print(data)
                for i in range(100):
                    batch = random.sample(data, batchSize)
                    features = np.array([sample[0] for sample in batch])
                    actions = np.array([sample[1] for sample in batch])
                    rewards = np.array([sample[2] for sample in batch])
                    self.update(features, actions, rewards)

                if(trial % 2 == 0):
                    self.explorationProb = max(self.exploreProbDecay * self.explorationProb, self.explorationProbMin)

            totalRewards.append(totalReward)
            mean_totalReward = np.mean(totalRewards[-100:])
            episode_length.append(iteration)
            if((mean_totalReward>max_totalReward) and (train==True)):
                # Save Weights
                self.model.save('./eights/linear_MC_{}.h5'.format(trial))
                max_totalReward = mean_totalReward
                print('The weights are saved with total rewards: ',mean_totalReward)


            if trial % 20 == 0:
                print(('\n---- Trial {} ----'.format(trial)))
                print(('Mean(last 20 total rewards): {}'.format(np.mean(totalRewards[-20:]))))
            if(not train):
                print(('Trial {} Total Reward: {}'.format(trial, totalReward)))
        if(train):
            np.save("./rewards/linear_MC_{}.npy".format(numTrials), totalRewards)
            np.save("./episode_length/linear_MC_{}.npy".format(numTrials), episode_length)


        return totalRewards


## Main variables
numTrials = 10000
numTestTrials = 10
trialDemoInterval = numTrials/2
discountFactor = 0.99
explorProbInit = 1.0
exploreProbDecay = 0.999
explorationProbMin = 0.01
batchSize = 5


if __name__ == '__main__':
    # Cold start weights
    weights = defaultdict(float)
    # Warm start weights
    # weights = loadF('weights')

    # TRAIN
    env = gym.make('LunarLander-v2')

    rl =TileLinear(env, discountFactor,
                            weights, explorProbInit, exploreProbDecay,
                            explorationProbMin)
    # env.seed(0)
    print('\n++++++++++++ TRAINING +++++++++++++')
    totalRewards = rl.simulate( numTrials=numTrials, train=True, verbose=True)
    env.close()
    print('Average Total Training Reward: {}'.format(np.mean(totalRewards)))


    #load weights
    # weights = './weights/linear_MC_7187.h5'
    weights = "./linear_MC_7186.h5"
    #TEST
    print('\n\n++++++++++++++ TESTING +++++++++++++++')
    rl = SarsaLinear(env, discountFactor,
                            weights, 0.0, 1,
                            0.00)
    totalRewards, _ = rl.simulate( numTrials=numTestTrials, train=False, verbose=True)
    print('Average Total Testing Reward: {}'.format(np.mean(totalRewards)))