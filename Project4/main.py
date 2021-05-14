import numpy as np

import reinforcement_learning

if __name__ == '__main__':
    alpha = 1
    gamma = 0.8
    actions = [0, 1, 2, 3, 4, 5]
    rewards = np.matrix([[-1, -1, -1, -1, 0, -1],
                         [-1, -1, -1, 0, -1, 100],
                         [-1, -1, -1, 0, -1, -1],
                         [-1, 0, 0, -1, 0, -1],
                         [-1, 0, 0, -1, -1, 100],
                         [-1, 0, -1, -1, 0, 100]])

    Q = np.matrix(np.zeros([6, 6]))

    Q_agent = reinforcement_learning.QLearning(alpha, gamma, actions, rewards, Q)
    Q_agent.train(iterations=10000)
    print(Q_agent.Q)
    print(Q_agent.test(2, 1))
