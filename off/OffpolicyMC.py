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

def random_policy(nA):
 
    
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) / nA
        return A
    return policy_fn

def epsilon_greedy_policy(Q):

    def policy_fn(state):
        A = np.array([0.05,0.05,0.05,0.05])
        best_action = np.argmax(Q[state])
                # print(A[i])
        A[best_action] = 0.85

        return A
    return policy_fn




class MC_offpolicy:

    def __init__(self, env, num_episodes, discount_factor=0.99 ):
        self.env =env
        self.num_episodes = num_episodes 

        self.discount_factor = discount_factor
        self.Q = defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.C = defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.behavior_policy = random_policy(self.env.action_space.n)
        # print(self.behavior_policy)
        self.target_policy = epsilon_greedy_policy(self.Q)
        self.discount_factor = discount_factor
        # print(self.target_policy)
# def target_policy(Q,state,action):

    def update(self,episode):
                    # Sum of discounted returns
        G = 0.0
        # The importance sampling ratio (the weights of the returns)
        W = 1.0
        # For each step in the episode, backwards
        for t in range(len(episode))[::-1]:
            
            state, action, reward = episode[t]
            state = tuple(state)
            # Update the total reward since step t
            G = self.discount_factor * G + reward
            # Update weighted importance sampling formula denominator
            self.C[state][action] += W
            # Update the action-value function using the incremental update formula (5.7)
            # This also improves our target policy which holds a reference to Q
            self.Q[state][action] += (W /( self.C[state][action]+1e-60)) * (G - self.Q[state][action])
            # If the action taken by the behavior policy is not the action 
            # taken by the target policy the probability will be 0 and we can break
            if action ==  np.argmax(self.target_policy(state)):
            #     break
                W = W * (0.85/0.25)
            else:
                W = W * (0.05/0.25)

        # return G

    # def getAction

    def simulate(self,render= False,train = True,verbose= True):

        b_rewards = []   
        t_rewards = []
        max_totalReward = 0
        episode_length = []
        if(train):
            for trial in range(1, self.num_episodes + 1):
                # Print out which episode we're on, useful for debugging.
                if trial % 1000 == 0:
                    print("\rEpisode {}/{}.".format(trial, self.num_episodes), end="")
                    sys.stdout.flush()

                # Generate an episode.
                # An episode is an array of (state, action, reward) tuples
                episode = []
                state = self.env.reset()
                while True:
                    # Sample an action from our policy
                    probs = self.behavior_policy
                    action = np.random.choice(np.arange(len(probs)), p=probs)
                    next_state, reward, done, _ = self.env.step(action)
                    episode.append((state, action, reward))
                    if done:
                        break
                    state = next_state

                total_reward = self.update(episode)
                b_rewards.append(total_reward)

                state = self.env.reset()
                total_reward = 0
                iteration  = 0
                while True:
                    probs = self.target_policy(tuple(state))
                    action = np.random.choice(np.arange(len(probs)), p=probs)
                    newState, reward, done, info = self.env.step(action)
                    
                    
                    # newState = np.reshape(newState, (1,8)          

                    if render == True:
                        still_open = self.env.render()
                        if still_open == False: break

                    total_reward += reward
                    state = newState
                    iteration += 1

                    if done:
                        break
                episode_length.append(iteration)
                t_rewards.append(total_reward)
                if verbose and trial % 20 == 0:
                    print(('\n---- Trial {} ----'.format(trial)))
                    print(('Mean(last 10 total rewards): {}'.format(np.mean(t_rewards[-10:]))))
                # print(('Size(weight vector): {}'.format(len(self.weights))))
                mean_totalReward = np.mean(t_rewards[-10:])
                if((mean_totalReward>max_totalReward) and (train==True)):
                    # Save Weights
                    self.saveF(self.Q,'./weights/linear_MC_{}_{}.h5'.format(trial,mean_totalReward))
                    max_totalReward = mean_totalReward
                    print('The weights are saved with total rewards: ',mean_totalReward)

        print(('Trial {} Total Reward: {}'.format(trial, t_rewards[-1])))
        np.save("./rewards/t_off_MC_{}.npy".format(self.num_episodes ), t_rewards)
        np.save("./rewards/b_off_MC_{}.npy".format(self.num_episodes ), b_rewards)
        np.save("./episode_length.npy".format(self.num_episodes ), episode_length)


        return self.Q, self.target_policy


    def saveF(self,obj, name):
        with open('weights/' + name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def loadF(self,name):
        with open(self,'weights/' + name + '.pkl', 'rb') as f:
            return pickle.load(f)



env =gym.make("LunarLander-v2")
random_policy = random_policy(env.action_space.n)
rl =  MC_offpolicy(env, num_episodes=200)
Q, target_policy = rl.simulate(render= False,train = True,verbose= True)
