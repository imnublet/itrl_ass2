from unicodedata import name
from matplotlib.pyplot import title
from ShortCutEnvironment import *
from ShortCutAgents import *
import numpy as np
import argparse
from Helper import *


def run_episodes(agent_type='q_learning', n_episodes=30, n_reps=50, alpha=0.1, epsilon=0.1, type_env='shortcut'):
    """
    Perform an experiment using a given RL algorithm for n_episodes for n_reps
    :param type_env: Environment type: regular shortcut or windy
    :param agent_type: Type of agent: q_learning, sarsa or expected_sarsa
    :param n_episodes: number of episodes per run
    :param n_reps: number of repetitions for the experiment
    :param alpha: the learning rate for the agent, how much do new values influence the existing q-values?
    :param epsilon: greediness of the agent, how likely is the agent to opt for the maximum value?
    """
    env, n_actions, n_states, actions = load_env(type_env)

    if agent_type == 'q_learning':
        return q_learning(env, n_actions, n_states, actions, n_episodes, n_reps, epsilon, alpha)
    elif agent_type == 'sarsa':
        return sarsa(env, n_actions, n_states, actions, n_episodes, n_reps, epsilon, alpha)
    elif agent_type == "expected_sarsa":
        return expected_sarsa(env, n_actions, n_states, actions, n_episodes, n_reps, epsilon, alpha)
    else:
        raise ValueError('agent_type is invalid')

                          
def q_learning(env, n_actions, n_states, actions, n_episodes, n_reps, epsilon, alpha):
    """
    Perform Q-Learning on the environment until env.done()

    :param env: Environment type: regular shortcut or windy
    :param n_actions: the number of actions the agent can choose from
    :param n_states: the number of states the agent can end up in
    :param actions: the actions the agent can choose from
    :param n_episodes: number of episodes per run
    :param n_reps: number of repetitions for the experiment
    :param epsilon: greediness of the agent, how likely is the agent to opt for the maximum value?
    :param alpha: the learning rate for the agent, how much do new values influence the existing q-values?
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
            env.reset()
        all_cumulative_rewards = np.vstack((all_cumulative_rewards, reward_per_episode))
        q_table = agent.q_table
    return np.mean(all_cumulative_rewards, axis=0), q_table


def sarsa (env, n_actions, n_states, actions, n_episodes, n_reps, epsilon, alpha):
    """
    Perform SARSA on the environment until env.done()

    :param env: Environment type: regular shortcut or windy
    :param n_actions: the number of actions the agent can choose from
    :param n_states: the number of states the agent can end up in
    :param actions: the actions the agent can choose from
    :param n_episodes: number of episodes per run
    :param n_reps: number of repetitions for the experiment
    :param epsilon: greediness of the agent, how likely is the agent to opt for the maximum value?
    :param alpha: the learning rate for the agent, how much do new values influence the existing q-values?
    """
    all_cumulative_rewards = np.empty(shape=n_episodes)
    for rep in range(n_reps):
        # env, n_actions, n_states, actions = load_env()
        reward_per_episode = np.empty(0)
        agent = SARSAAgent(n_actions, actions, n_states, epsilon, alpha)
        for episode in range(n_episodes):
            total_reward = 0
            while not env.done():
                state = env.state()
                action = agent.select_action(state)
                reward = env.step(action)
                total_reward += reward
                new_state = env.state()
                new_action = agent.select_action(new_state)
                agent.update(state, new_state, action, new_action, reward)
            reward_per_episode = np.append(reward_per_episode, total_reward)
            env.reset()
        all_cumulative_rewards = np.vstack((all_cumulative_rewards, reward_per_episode))
        q_table = agent.q_table
    return np.mean(all_cumulative_rewards, axis=0), q_table


def expected_sarsa(env, n_actions, n_states, actions, n_episodes, n_reps, epsilon, alpha):
    """
    Perform Expected SARSA on the environment until env.done()
    :param env: Environment type: regular shortcut or windy
    :param n_actions: the number of actions the agent can choose from
    :param n_states: the number of states the agent can end up in
    :param actions: the actions the agent can choose from
    :param n_episodes: number of episodes per run
    :param n_reps: number of repetitions for the experiment
    :param epsilon: greediness of the agent, how likely is the agent to opt for the maximum value?
    :param alpha: the learning rate for the agent, how much do new values influence the existing q-values?
    """
    all_cumulative_rewards = np.empty(shape=n_episodes)
    for rep in range(n_reps):
        # env, n_actions, n_states, actions = load_env()
        reward_per_episode = np.empty(0)
        agent = ExpectedSARSAAgent(n_actions, actions, n_states, epsilon, alpha)
        for episode in range(n_episodes):
            total_reward = 0
            while not env.done():
                state = env.state()
                action = agent.select_action(state)
                reward = env.step(action)
                total_reward += reward
                new_state = env.state()
                # new_action = agent.select_action(new_state)
                agent.update(state, new_state, action, reward)
            reward_per_episode = np.append(reward_per_episode, total_reward)
            env.reset()
        all_cumulative_rewards = np.vstack((all_cumulative_rewards, reward_per_episode))
        q_table = agent.q_table
    return np.mean(all_cumulative_rewards, axis=0), q_table


def load_env(type_env='shortcut'):
    """
    Load an environment

    :returns loaded_env: the chose environment, regular shortcut or windy
    :returns n_actions: the number of actions the agent can choose from
    :returns n_states: the number of states the agent can end up in
    :returns actions: the actions the agent can choose from
    """
    if type_env == 'shortcut':
        loaded_env = ShortcutEnvironment()
    elif type_env == 'windy':
        loaded_env = WindyShortcutEnvironment()
    else:
        raise(ValueError, 'Invalid environment')
    n_actions = loaded_env.action_size()
    n_states = loaded_env.state_size()
    actions = loaded_env.possible_actions()

    return loaded_env, n_actions, n_states, actions


def make_plot(x, y):
    """
    Make a plot
    :param x: X values for the plot
    :param y: Y values for the plot
    """
    qlearning_plot = ComparisonPlot(title="Learning rate of Q-Learning algorithm")
    qlearning_plot.add_curve(x, y)
    qlearning_plot.save('q_learning plot.png')


def run_experiments(env='shortcut'):
    """
    Run experiments for all agent types and levels of alpha.

    :param env: Environment type: regular shortcut or windy
    """
    episodes = 1000
    repetitions = 100
    if env == "windy":
        ALPHAS = [0.1]
        AGENTS = ['q_learning', 'sarsa']
        episodes = 10000
        repetitions = 1
    else:
        ALPHAS = [0.01, 0.1, 0.5, 0.9]
        AGENTS = ['q_learning', 'sarsa', 'expected_sarsa']
        episodes = 1000
        repetitions = 100
    x = np.arange(episodes)

    for AGENT in AGENTS:
        comparison_plot = ComparisonPlot(title="Agent: %s" % AGENT)
        for ALPHA in ALPHAS:
            q_plot = MatrixPlot(title="%s agent in %s environment, with alpha = %s." % (AGENT, env, ALPHA))
            print("Running:", AGENT, ALPHA)
            all_cumulative_rewards, q_grid = run_episodes(agent_type=AGENT, n_episodes=episodes, n_reps=repetitions, alpha=ALPHA, type_env=env)
            q_plot.plot(q_grid,name="Matrixplot_for_%s_%s_alpha=%s.png" % (env, AGENT, ALPHA))
            comparison_plot.add_curve(x, y=smooth(all_cumulative_rewards, 10), label="Alpha: %s" % ALPHA)
        comparison_plot.save(name="Test_for_%s_%s.png" % (env , AGENT))


def parse_args():
    parser = argparse.ArgumentParser(description='Run three model free reinforcement algorithms')
    parser.add_argument('--env', type=int,
                        help='An optional argument to select which environment you want to run.,'
                             '0 = both environments,'
                             '1 = shortcut environment,'
                             '2 = windy environment')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    env_args = parse_args().env
    if env_args is None or env_args == 1:
        run_experiments(env='shortcut')
    elif env_args is None or env_args == 2:
        run_experiments(env='windy')
