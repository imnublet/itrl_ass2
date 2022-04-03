#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bandit environment
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
2021
By Thomas Moerland
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


class LearningCurvePlot:

    def __init__(self,title=None):
        self.fig,self.ax = plt.subplots()
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Reward')
        self.ax.set_ylim([0,1.0])
        if title is not None:
            self.ax.set_title(title)

    def add_curve(self,y,label=None):
        ''' y: vector of average reward results
        label: string to appear as label in plot legend '''
        if label is not None:
            self.ax.plot(y,label=label)
        else:
            self.ax.plot(y)

    def save(self,name='test.png'):
        ''' name: string for filename of saved figure '''
        self.ax.legend()
        self.fig.savefig(name,dpi=300)


class ComparisonPlot:

    def __init__(self,title=None):
        self.fig,self.ax = plt.subplots()
        self.ax.set_xlabel('Parameter (exploration)')
        self.ax.set_ylabel('Average cumulative reward')
        self.ax.set_xscale('log')
        if title is not None:
            self.ax.set_title(title)

    def add_curve(self,x,y,label=None):
        ''' x: vector of parameter values
        y: vector of associated mean reward for the parameter values in x
        label: string to appear as label in plot legend '''
        if label is not None:
            self.ax.plot(x,y,label=label)
        else:
            self.ax.plot(x,y)

    def save(self,name='test.png'):
        ''' name: string for filename of saved figure '''
        self.ax.legend()
        self.fig.savefig(name,dpi=300)


def smooth(y, window, poly=1):
    '''
    y: vector to be smoothed
    window: size of the smoothing window '''
    return savgol_filter(y,window,poly)


class MatrixPlot:

    def __init__(self, title=None, env_size=12*12):
        self.fig, self.ax = plt.subplots()
        self.grid_size = env_size
        self.x = int(env_size**0.5)
        self.y = int(env_size**0.5)
        if title is not None:
            self.ax.set_title(title)

            
    def action_to_string(self, action):
        if action == 0:
            return 'v'
        elif action == 1:
            return '^'
        elif action == 2:
            return '<'
        elif action == 3:
            return '>'

    def plot(self, q_table, name="Q.png"):
        grid = np.reshape(np.argmax(q_table, axis=1), (self.x, self.y))
        max_q_table = np.reshape(np.max(q_table, axis=1), (self.x, self.y))

        for i in range(self.x):
            for j in range(self.y):
                pos = max_q_table[i, j]
                if pos != 0:
                    self.ax.text(i, j, self.action_to_string(grid[i, j]), va='center', ha='center', size=10)
                else:
                    self.ax.text(i, j, ' ', va='center', ha='center', size=10)
        self.ax.matshow(grid, cmap=plt.cm.Blues)
        self.ax.axes.xaxis.set_visible(False)
        self.ax.axes.yaxis.set_visible(False)
        self.fig.savefig(name)


if __name__ == '__main__':
    # Test Learning curve plot
    x = np.arange(100)
    y = 0.01*x + np.random.rand(100) - 0.4 # generate some learning curve y
    LCTest = LearningCurvePlot(title="Test Learning Curve")
    LCTest.add_curve(y,label='method 1')
    LCTest.add_curve(smooth(y,window=35),label='method 1 smoothed')
    LCTest.save(name='learning_curve_test.png')

    # Test Performance plot
    PerfTest = ComparisonPlot(title="Test Comparison")
    PerfTest.add_curve(np.arange(5),np.random.rand(5),label='method 1')
    PerfTest.add_curve(np.arange(5),np.random.rand(5),label='method 2')
    PerfTest.save(name='comparison_test.png')