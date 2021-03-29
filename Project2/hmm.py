'''
https://en.wikipedia.org/wiki/Baum–Welch_algorithm
https://medium.com/analytics-vidhya/hidden-markov-model-part-1-of-the-hmm-series-3f7fea28a08
https://medium.com/analytics-vidhya/baum-welch-algorithm-for-training-a-hidden-markov-model-part-2-of-the-hmm-series-d0e393b4fb86
'''
import numpy as np
import re


class hmm:

    def __init__(self, num_hidden_states, transitions=None, emissions=None, pi=None):
        self.num_hidden_states = num_hidden_states
        self.transitions = transitions
        self.emissions = emissions
        self.pi = pi
        self.alpha = None
        self.beta = None

        if transitions is None:
            self.transitions = np.ones((self.num_hidden_states, self.num_hidden_states))
            self.transitions = self.transitions + np.random.uniform(low=0, high=1,
                                                                    size=(self.num_hidden_states, self.num_hidden_states))
            self.transitions = self.transitions / self.transitions.sum(axis=1, keepdims=1)

        if emissions is None:
            self.emissions = []

        if pi is None:
            self.pi = np.ones((self.num_hidden_states, self.num_hidden_states))
            self.pi = self.pi + np.random.uniform(low=0, high=1, size=(self.num_hidden_states, self.num_hidden_states))
            self.pi = self.pi / self.pi.sum(axis=1, keepdims=1)
            self.pi = self.pi[np.random.randint(0, self.num_hidden_states-1)]

    def build_emissions_matrix(self, observations):
        unique_tokens = []

        for sentence in observations:
            tokens = re.split('\s+', sentence)
            for token in tokens:
                if token not in unique_tokens:
                    unique_tokens.append(token)

        result = np.ones((self.num_hidden_states, len(unique_tokens)))
        result = result + np.random.uniform(low=0, high=1, size=(self.num_hidden_states, len(unique_tokens)))
        result = result / result.sum(axis=1, keepdims=1)

        for i in range(self.num_hidden_states):
            self.emissions.append(dict(zip(unique_tokens, result[i])))

    '''
    The alpha function is defined as the joint probability of the observed data up to time k and the state at time k
    '''
    def forward_procedure(self, observations):
        self.alpha = np.zeros((self.num_hidden_states, len(observations)))

        for i in range(self.num_hidden_states):
            self.alpha[i][0] = self.pi[i] * self.emissions[i][observations[0]]

        for k in range(len(observations) - 1):
            for i in range(self.num_hidden_states):
                sigma = 0
                for j in range(self.num_hidden_states):
                    sigma += self.alpha[j][k] * self.transitions[j][i]
                self.alpha[i][k+1] = self.emissions[i][observations[k+1]] * sigma

    '''
    The beta function is defined as the conditional probability of the observed data from time k+1 given the state at time k
    '''
    def backward_procedure(self, observations):
        self.beta = np.zeros((self.num_hidden_states, len(observations)))
        self.beta[:, len(observations)-1] = 1

        for k in range(len(observations)-2, -1, -1):
            for i in range(self.num_hidden_states):
                for j in range(self.num_hidden_states):
                    self.beta[i][k] += self.beta[j][k+1] * self.transitions[i][j] * self.emissions[j][observations[k+1]]

