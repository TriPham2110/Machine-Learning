"""
Ref: https://blog.floydhub.com/an-introduction-to-q-learning-reinforcement-learning/
     https://tcnguyen.github.io/reinforcement_learning/sarsa_vs_q_learning.html
"""
import numpy as np


class QLearning:

    def __init__(self, alpha, gamma, actions, rewards, Q):
        if alpha is None:
            self.alpha = 0.5
        self.alpha = alpha  # Learning rate
        if gamma is None:
            self.gamma = 0.75
        self.gamma = gamma  # Discount factor
        self.actions = actions
        self.rewards = rewards
        self.Q = Q

    def train(self, iterations):
        for i in range(iterations):
            current_state = np.random.randint(0, int(len(self.Q)))
            available_actions = []

            for j in range(int(len(self.Q))):
                if self.rewards[current_state, j] >= 0:
                    available_actions.append(j)

            next_state = int(np.random.choice(available_actions, 1))
            temporal_difference = self.rewards[current_state, next_state] + self.gamma * self.Q[next_state, np.argmax(self.Q[next_state, ])] - self.Q[current_state, next_state]
            self.Q[current_state, next_state] += self.alpha * temporal_difference

    def test(self, start_location, end_location):
        current_state = start_location
        route = [current_state]
        while current_state != end_location:
            next_state = np.argmax(self.Q[current_state, ])
            route.append(next_state)
            current_state = next_state
        return route


class SARSA:

    def __init__(self, alpha, gamma, actions, rewards, Q):
        if alpha is None:
            self.alpha = 0.5
        self.alpha = alpha  # Learning rate
        if gamma is None:
            self.gamma = 0.75
        self.gamma = gamma  # Discount factor
        self.actions = actions
        self.rewards = rewards
        self.Q = Q

    def train(self, iterations):
        for i in range(iterations):
            current_state = np.random.randint(0, int(len(self.Q)))
            available_actions = []

            for j in range(int(len(self.Q))):
                if self.rewards[current_state, j] >= 0:
                    available_actions.append(j)

            next_state = int(np.random.choice(available_actions, 1))

            next_actions = []
            for action, Q_val in enumerate(self.Q[next_state, ]):
                if Q_val > 0:
                    next_actions.append(action)

            next_action = int(np.random.choice(next_actions, 1))
            temporal_difference = self.rewards[current_state, next_state] + self.gamma * self.Q[next_state, next_action] - self.Q[current_state, next_state]
            self.Q[current_state, next_state] += self.alpha * temporal_difference

    def test(self, start_location, end_location):
        current_state = start_location
        route = [current_state]
        while current_state != end_location:
            next_actions = []
            for action, Q_val in enumerate(self.Q[current_state, ]):
                if Q_val > 0:
                    next_actions.append(action)

            next_state = int(np.random.choice(next_actions, 1))
            route.append(next_state)
            current_state = next_state
        return route
