from unicodedata import name
from ShortCutEnvironment import *
from ShortCutAgents import *
import numpy as np


def run_episodes(n_episodes=10, alpha=0.1, epsilon=0.25):
    """
    """
    env, n_actions, n_states, actions = load_env()
    for episode in range(n_episodes):
        agent = QLearningAgent(n_actions, actions, n_states, epsilon, alpha)
        while not env.done():
            state = env.state()
            action = agent.select_action(state)
            reward = env.step(action)
            new_state = env.state()
            agent.update(state, new_state, action, reward)
            print('current state:', state)
        env.reset()


def load_env():
    """

    """
    loaded_env = ShortcutEnvironment()
    n_actions = loaded_env.action_size()
    n_states = loaded_env.state_size()
    actions = loaded_env.possible_actions()

    return loaded_env, n_actions, n_states, actions

run_episodes()