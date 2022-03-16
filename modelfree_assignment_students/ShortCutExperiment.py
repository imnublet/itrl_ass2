from unicodedata import name
from ShortCutEnvironment import *
from ShortCutAgents import *
import numpy as np
# TODO: Q_table visualiseren voor de environment

def run_episodes(agent_type='q_learning', n_episodes=30, n_reps=50, alpha=0.1, epsilon=0.1):
    """
    """
    env, n_actions, n_states, actions = load_env()

    if agent_type == 'q_learning':
        return q_learning(env, n_actions, n_states, actions, n_episodes, n_reps, epsilon, alpha)
    if agent_type == 'sarsa':
        sarsa(env, n_actions, n_states, actions, n_episodes, n_reps, epsilon, alpha)

    # print(averaged_cumulative_rewards)


def q_learning(env, n_actions, n_states, actions, n_episodes, n_reps, epsilon, alpha):
    """

    """
    all_cumulative_rewards = np.empty(shape=n_episodes)
    for rep in range(n_reps):
        reward_per_episode = np.empty(0)
        agent = QLearningAgent(n_actions, actions, n_states, epsilon, alpha)
        for episode in range(n_episodes):
            total_reward = 0
            while not env.done():
                state = env.state()
                action = agent.select_action(state)
                reward = env.step(action)
                total_reward += reward
                new_state = env.state()
                agent.update(state, new_state, action, reward)
            reward_per_episode = np.append(reward_per_episode, total_reward)
            # print('total_reward', total_reward)
            # # print(reward_per_episode[episode])
            # # print(np.shape(reward_per_episode))
            # print('ep', episode)
            # print('r_per_e', reward_per_episode)
            env.reset()
        all_cumulative_rewards = np.vstack((all_cumulative_rewards, reward_per_episode))
    return np.mean(all_cumulative_rewards, axis=0)


def sarsa(env, n_actions, n_states, actions, n_episodes, epsilon, alpha):
    """

    """
    for episode in range(n_episodes):
        agent = SARSAAgent(n_actions, actions, n_states, epsilon, alpha)
        while not env.done():
            state = env.state()
            action = agent.select_action(state)
            reward = env.step(action)
            new_state = env.state()
            new_action = agent.select_action(new_state)
            agent.update(state, new_state, action, new_action, reward)
            print('current state:', state)
        env.reset()
    pass


def load_env():
    """

    """
    loaded_env = ShortcutEnvironment()
    n_actions = loaded_env.action_size()
    n_states = loaded_env.state_size()
    actions = loaded_env.possible_actions()

    return loaded_env, n_actions, n_states, actions


all_cumulative_rewards = run_episodes(agent_type='q_learning')
print(all_cumulative_rewards)
# run_episodes(agent_type='sarsa')