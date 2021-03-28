import numpy as np
import re


class hmm:

    def __init__(self, num_hidden_states, transitions=None, emissions=None, pi=None):
        self.num_hidden_states = num_hidden_states
        self.transitions = transitions
        self.emissions = emissions
        self.pi = pi

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
