import numpy as np
import reinforcement_learning
np.random.seed(7)

if __name__ == '__main__':
    alpha = 1
    gamma = 0.8
    actions = [0, 1, 2, 3, 4, 5]
    rewards = np.array([[-1, -1, -1, -1, 0, -1],
                        [-1, -1, -1, 0, -1, 100],
                        [-1, -1, -1, 0, -1, -1],
                        [-1, 0, 0, -1, 0, -1],
                        [-1, 0, 0, -1, -1, 100],
                        [-1, 0, -1, -1, 0, 100]])

    Q = np.array(np.zeros([6, 6]))

    start_state = 2
    end_state = 5

    print("Initial state-action matrix\n", Q)
    print("Rewards matrix\n", rewards)

    print("Q-learning")
    Q_agent = reinforcement_learning.QLearning(alpha, gamma, actions, rewards, Q)
    Q_agent.train(iterations=1000)
    print("Updated state-action matrix\n", Q_agent.Q)
    print("Test path with start state", start_state, "and end state", end_state, ":", Q_agent.test(2, 5))

    print("\nSARSA")
    SARSA_agent = reinforcement_learning.SARSA(alpha, gamma, actions, rewards, Q)
    SARSA_agent.train(iterations=1000)
    print("Updated state-action matrix\n", SARSA_agent.Q)
    print("Test path with start state", start_state, "and end state", end_state, ":", SARSA_agent.test(2, 5))
