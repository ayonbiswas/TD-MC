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

def create_random_policy(nA):
    """
    Creates a random policy function.
    
    Args:
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes an observation as input and returns a vector
        of action probabilities
    """
    A = np.ones(nA, dtype=float) / nA
    def policy_fn(observation):
        return A
    return policy_fn

def create_greedy_policy(Q):
    """
    Creates a greedy policy based on Q values.
    
    Args:
        Q: A dictionary that maps from state -> action values
        
    Returns:
        A function that takes an observation as input and returns a vector
        of action probabilities.
    """
    
    def policy_fn(state):
        A = np.zeros_like(Q[state], dtype=float)
        best_action = np.argmax(Q[state])
        A[best_action] = 1.0
        return A
    return policy_fn
    
def mc_control_importance_sampling(env, num_episodes, behavior_policy, discount_factor=1.0):
# The final action-value function.
# A dictionary that maps state -> action values
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    # The cumulative denominator of the weighted importance sampling formula
    # (across all episodes)
    C = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # Our greedily policy we want to learn
    target_policy = create_greedy_policy(Q)
        
    for i_episode in range(1, num_episodes + 1):
        # Print out which episode we're on, useful for debugging.
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        # Generate an episode.
        # An episode is an array of (state, action, reward) tuples
        episode = []
        state = env.reset()
        for t in range(100):
            # Sample an action from our policy
            probs = behavior_policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
        
        # Sum of discounted returns
        G = 0.0
        # The importance sampling ratio (the weights of the returns)
        W = 1.0
        # For each step in the episode, backwards
        for t in range(len(episode))[::-1]:
            state, action, reward = episode[t]
            state = tuple(state)
            # Update the total reward since step t
            G = discount_factor * G + reward
            # Update weighted importance sampling formula denominator
            C[state][action] += W
            # Update the action-value function using the incremental update formula (5.7)
            # This also improves our target policy which holds a reference to Q
            Q[state][action] += (W / C[state][action]) * (G - Q[state][action])
            # If the action taken by the behavior policy is not the action 
            # taken by the target policy the probability will be 0 and we can break
            if action !=  np.argmax(target_policy(state)):
                break
            W = W * 1./behavior_policy(state)[action]
        if i_episode %250 == 0:

            print(G)
    return Q, target_policy

env =gym.make("LunarLander-v2")
random_policy = create_random_policy(env.action_space.n)
Q, policy = mc_control_importance_sampling(env, num_episodes=500000, behavior_policy=random_policy)
