"""
Ref: https://blog.floydhub.com/an-introduction-to-q-learning-reinforcement-learning/
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
        new_rewards = np.copy(self.rewards)

        for i in range(iterations):
            current_state = np.random.randint(0, int(len(self.Q)))
            available_actions = []

            for j in range(int(len(self.Q))):
                if new_rewards[current_state, j] >= 0:
                    available_actions.append(j)

            next_state = int(np.random.choice(available_actions, 1))
            temporal_difference = new_rewards[current_state, next_state] + self.gamma * self.Q[next_state, np.argmax(self.Q[next_state, ])] - self.Q[current_state, next_state]
            self.Q[current_state, next_state] += self.alpha * temporal_difference

    def test(self, start_location, end_location):
        current_state = start_location
        route = [current_state]
        while current_state != end_location:
            next_state = np.argmax(self.Q[current_state, ])
            route.append(next_state)
            current_state = next_state
        return route

