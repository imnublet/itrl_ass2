from unicodedata import name
from ShortCutEnvironment import *
from ShortCutAgents import *
import numpy as np
from Helper import *
# TODO: Q_table visualiseren voor de environment
# TODO: Meerdere alpha values in een plot krijgen


def run_episodes(agent_type='q_learning', n_episodes=30, n_reps=100, alpha=0.1, epsilon=0.1):
    """
    Perform an experiment using a given RL algorithm for n_episodes for n_reps

    :param agent_type:
    :param n_episodes:
    :param n_reps:
    :param alpha:
    :param epsilon:
    """
    # env, n_actions, n_states, actions = load_env()

    if agent_type == 'q_learning':
        # return q_learning(env, n_actions, n_states, actions, n_episodes, n_reps, epsilon, alpha)
        return q_learning(n_episodes, n_reps, epsilon, alpha)
    elif agent_type == 'sarsa':
        return sarsa(n_episodes, n_reps, epsilon, alpha)
    elif agent_type == "expected_sarsa":
        return expected_sarsa(n_episodes, n_reps, epsilon, alpha)


# def q_learning(env, n_actions, n_states, actions, n_episodes, n_reps, epsilon, alpha):
def q_learning(n_episodes, n_reps, epsilon, alpha):
    """
    Perform Q-Learning on the environment until env.done()

    :param env:
    :param n_actions:
    :param n_states:
    :param actions:
    :param n_episodes:
    :param n_reps:
    :param epsilon:
    :param alpha:
    """
    all_cumulative_rewards = np.empty(shape=n_episodes)
    # list_of_q_tables = np.empty(shape=env.state_size())
    for rep in range(n_reps):
        env, n_actions, n_states, actions = load_env()
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
        states_q_table = np.mean(agent.q_table, axis=1)
        # list_of_q_tables = np.vstack((list_of_q_tables, states_q_table))
    # return np.mean(all_cumulative_rewards, axis=0), np.mean(list_of_q_tables, axis=0)
    return np.mean(all_cumulative_rewards, axis=0), states_q_table


def sarsa(n_episodes, n_reps, epsilon, alpha):
    """
     Perform SARSA on the environment until env.done()

     :param env:
     :param n_actions:
     :param n_states:
     :param actions:
     :param n_episodes:
     :param n_reps:
     :param epsilon:
     :param alpha:
     """
    all_cumulative_rewards = np.empty(shape=n_episodes)
    for rep in range(n_reps):
        env, n_actions, n_states, actions = load_env()
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
        states_q_table = np.mean(agent.q_table, axis=1)
    return np.mean(all_cumulative_rewards, axis=0), states_q_table

def expected_sarsa(n_episodes, n_reps, epsilon, alpha):
    """
     Perform Expected SARSA on the environment until env.done()

     :param env:
     :param n_actions:
     :param n_states:
     :param actions:
     :param n_episodes:
     :param n_reps:
     :param epsilon:
     :param alpha:
     """
    all_cumulative_rewards = np.empty(shape=n_episodes)
    for rep in range(n_reps):
        env, n_actions, n_states, actions = load_env()
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
        states_q_table = np.mean(agent.q_table, axis=1)
    return np.mean(all_cumulative_rewards, axis=0), states_q_table



def load_env():
    """
    Load an environment

    :returns loaded_env:
    :returns n_actions:
    :returns n_states:
    :returns actions:
    """
    loaded_env = ShortcutEnvironment()
    n_actions = loaded_env.action_size()
    n_states = loaded_env.state_size()
    actions = loaded_env.possible_actions()

    return loaded_env, n_actions, n_states, actions


def make_plot(x, y):
    """
    Make a plot

    :param x:
    :param y:
    """
    qlearning_plot = ComparisonPlot(title="Learning rate of Q-Learning algorithm")
    qlearning_plot.add_curve(x, y)
    qlearning_plot.save('q_learning plot.png')


def run_experiments():
    """

    """
    episodes = 1000
    repetitions = 100
    ALPHAS = [0.01,0.1,0.5,0.9]
    AGENTS = ['expected_sarsa']
    # AGENTS = ['q_learning','sarsa','expected_sarsa']
    x = np.arange(episodes)

    for AGENT in AGENTS:
        comparison_plot = ComparisonPlot(title="Agent: %s" % AGENT)
        for ALPHA in ALPHAS:
            print("Running:", AGENT, ALPHA)
            all_cumulative_rewards, q_grid = run_episodes(agent_type=AGENT, n_episodes=episodes, n_reps=repetitions, alpha=ALPHA)
            comparison_plot.add_curve(x,all_cumulative_rewards,label="Alpha: %s" % ALPHA)
        comparison_plot.save(name="Test_for_%s.png" % AGENT)
            
    
    print(np.shape(q_grid))
    # all_cumulative_rewards = run_episodes(agent_type='sarsa', n_episodes=episodes)
    print(all_cumulative_rewards)
    q_grid_plot = MatrixPlot()
    q_grid_plot.plot(q_grid)
    make_plot(x=np.arange(episodes), y=all_cumulative_rewards)
    # run_episodes(agent_type='sarsa')


run_experiments()