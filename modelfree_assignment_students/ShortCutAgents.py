import random
import numpy as np
import ShortCutEnvironment as env


class QLearningAgent(object):

    def __init__(self, n_actions, actions, n_states, epsilon=0.25, alpha=0.1):
        self.alpha = alpha
        self.n_actions = n_actions
        self.actions = actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.q_table = np.zeros(shape=(self.n_states, self.n_actions))

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            a = np.random.choice(self.actions, size=1)[0]
        else:
            a = np.argmax(self.q_table[state])
        return a
        
    def update(self, state, next_state, action, reward):
        self.q_table[state, action] = self.q_table[state, action] + self.alpha * (reward + np.max(self.q_table[next_state, :]) - self.q_table[state, action])


class SARSAAgent(object):

    def __init__(self, n_actions, actions, n_states, epsilon=0.25, alpha=0.1):
        self.alpha = alpha
        self.n_actions = n_actions
        self.actions = actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.q_table = np.zeros(shape=(self.n_states, self.n_actions))

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            a = np.random.choice(self.actions, size=1)[0]
        else:
            a = np.argmax(self.q_table[state])
        return a

    def update(self, state, next_state, action, next_action, reward):
        self.q_table[state, action] = self.q_table[state, action] + self.alpha * (reward + self.q_table[next_state, next_action] - self.q_table[state, action])


class ExpectedSARSAAgent(object):

    def __init__(self, n_actions, actions, n_states, epsilon, alpha):
        self.n_actions = n_actions
        self.actions = actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.q_table = np.zeros(shape=(self.n_states, self.n_actions))

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            a = np.random.choice(self.actions, size=1)[0]
        else:
            a = np.argmax(self.q_table[state])
        return a

    def update(self, state, next_state, action, reward):
        expected_value = 0

        for a in range(self.n_actions):
            if self.q_table[next_state][a] == np.max(self.q_table[next_state, :]):
                expected_value += self.q_table[next_state][a] * (1-self.epsilon)
            else:
                expected_value += self.q_table[next_state][a] * (self.epsilon / self.n_actions)

        self.q_table[state, action] = self.q_table[state, action] + self.alpha * (reward + expected_value - self.q_table[state, action])
