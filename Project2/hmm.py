"""
https://en.wikipedia.org/wiki/Baum–Welch_algorithm
https://medium.com/analytics-vidhya/hidden-markov-model-part-1-of-the-hmm-series-3f7fea28a08
https://medium.com/analytics-vidhya/baum-welch-algorithm-for-training-a-hidden-markov-model-part-2-of-the-hmm-series-d0e393b4fb86
https://medium.com/analytics-vidhya/viterbi-algorithm-for-prediction-with-hmm-part-3-of-the-hmm-series-6466ce2f5dc6
https://en.wikipedia.org/wiki/Viterbi_algorithm
"""
import numpy as np
import re
import pickle


class hmm:

    def __init__(self, num_hidden_states, transitions=None, emissions=None, pi=None, max_iter=50):
        self.num_hidden_states = num_hidden_states
        self.transitions = transitions
        self.emissions = emissions
        self.pi = pi
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.xi = None
        self.gammas = []
        self.xis = []
        self.iterations = max_iter

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
            tokens = re.split(r'\s+', sentence)
            for token in tokens:
                if token not in unique_tokens:
                    unique_tokens.append(token)

        result = np.ones((self.num_hidden_states, len(unique_tokens)))
        result = result + np.random.uniform(low=0, high=1, size=(self.num_hidden_states, len(unique_tokens)))
        result = result / result.sum(axis=1, keepdims=1)

        for i in range(self.num_hidden_states):
            new_dict = {k: v for k, v in zip(unique_tokens, result[i])}
            self.emissions.append(new_dict)

    '''
    The alpha function is defined as the joint probability of the observed data up to time k and the state at time k
    '''
    def forward_procedure(self, observations):
        self.alpha = np.zeros((self.num_hidden_states, len(observations)))

        for i in range(self.num_hidden_states):
            self.alpha[i][0] = self.pi[i] * self.emissions[i][observations[0]]
        self.alpha[:, 0] = hmm.normalize(self.alpha[:, 0])

        for k in range(len(observations) - 1):
            for i in range(self.num_hidden_states):
                sigma = 0
                for j in range(self.num_hidden_states):
                    sigma = sigma + self.alpha[j][k] * self.transitions[j][i]
                self.alpha[i][k+1] = self.emissions[i][observations[k+1]] * sigma
            self.alpha[:, k+1] = hmm.normalize(self.alpha[:, k+1])

    '''
    The beta function is defined as the conditional probability of the observed data from time k+1 given the state at time k
    '''
    def backward_procedure(self, observations):
        self.beta = np.zeros((self.num_hidden_states, len(observations)))
        self.beta[:, len(observations)-1] = 1

        for k in range(len(observations)-2, -1, -1):
            for i in range(self.num_hidden_states):
                for j in range(self.num_hidden_states):
                    self.beta[i][k] = self.beta[i][k] + self.beta[j][k+1] * self.transitions[i][j] * self.emissions[j][observations[k+1]]

    '''
    gamma is defined as the probability of being in state i at time k given the observed sequence
    xi is defined as the probability of being in state i and j at times k and k+1 respectively given the observed sequence
    '''
    def calculate_temporary_variables(self, observations):
        self.gamma = np.zeros((self.num_hidden_states, len(observations)))
        self.xi = np.zeros((self.num_hidden_states, self.num_hidden_states, len(observations)))

        for k in range(len(observations)):
            sigma = 0
            for i in range(self.num_hidden_states):
                self.gamma[i][k] = self.alpha[i][k] * self.beta[i][k]
                sigma = sigma + self.gamma[i][k]
            self.gamma[:, k] = self.gamma[:, k] / sigma if sigma != 0 else np.zeros(self.gamma[:, k].shape)

        for k in range(len(observations)-1):
            sigma = 0
            for i in range(self.num_hidden_states):
                for j in range(self.num_hidden_states):
                    self.xi[i][j][k] = self.alpha[i][k] * self.transitions[i][j] * self.beta[j][k+1] * self.emissions[j][observations[k+1]]
                    sigma = sigma + self.xi[i][j][k]
            self.xi[:, :, k] = self.xi[:, :, k] / sigma if sigma != 0 else np.zeros(self.xi[:, :, k].shape)

        return self.gamma, self.xi

    def expectation(self, observations):
        for obs in observations:
            sentence = re.split(r'\s+', obs)
            self.forward_procedure(sentence)
            self.backward_procedure(sentence)
            self.gamma, self.xi = self.calculate_temporary_variables(sentence)
            self.gammas.append(self.gamma)
            self.xis.append(self.xi)

    def maximization(self, observations):
        for i in range(self.num_hidden_states):
            tmp = 0
            for k in range(len(observations)):
                tmp = tmp + self.gammas[k][i][0]
            self.pi[i] = tmp
        self.pi = hmm.normalize(self.pi)

        for i in range(self.num_hidden_states):
            sigma_gamma = 0
            for k in range(len(observations)):
                sigma_gamma = sigma_gamma + np.sum(self.gammas[k][i][:-1])
            for j in range(self.num_hidden_states):
                sigma_xi = 0
                for k in range(len(observations)):
                    sigma_xi = sigma_xi + np.sum(self.xis[k][i][j][:-1])
                self.transitions[i][j] = sigma_xi / sigma_gamma
            self.transitions[i] = hmm.normalize(self.transitions[i])

        for i in range(self.num_hidden_states):
            tmp = self.emissions[0].copy()
            for key in tmp:
                tmp[key] = 0.
            sigma = 0
            for k in range(len(observations)):
                sentence = re.split(r'\s+', observations[k])
                sigma = sigma + np.sum(self.gammas[k][i])
                for k_ in range(len(sentence)):
                    token = sentence[k_]
                    update = {token: tmp[token] + self.gammas[k][i][k_]}
                    tmp.update(update)
            for key in tmp:
                tmp[key] = tmp[key] / sigma
            self.emissions[i].update(tmp)

    def save(self, file):
        with open(file+'.pickle', 'wb') as f:
            pickle.dump((self.transitions, self.emissions, self.pi), f)

    def train_model(self, observations, filename='train'):
        self.build_emissions_matrix(observations)
        for i in range(self.iterations):
            self.expectation(observations)
            self.maximization(observations)
        self.save(file=filename)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def normalize(X):
        return [float(i) / np.sum(X) for i in X]

    def generate(self, num):
        start_state = np.random.choice(range(self.num_hidden_states), p=self.pi)
        generated_seq = ''
        cur_state = start_state
        for i in range(num):
            selected_token = np.random.choice(list(self.emissions[cur_state].keys()), p=list(self.emissions[cur_state].values()))
            generated_seq = generated_seq + selected_token + ' '
            cur_state = np.random.choice(range(self.num_hidden_states), p=self.transitions[cur_state])
        print(generated_seq)

    def predict(self, text_line, num):
        seq = re.split(r'\s+', text_line)
        T1 = np.zeros((self.num_hidden_states, len(seq)))
        T2 = np.zeros((self.num_hidden_states, len(seq)))

        for i in range(self.num_hidden_states):
            if seq[0] in self.emissions[i].keys():
                T1[i][0] = self.pi[i] * self.emissions[i][seq[0]]
                T2[i][0] = 0

        for j in range(1, len(seq)):
            for i in range(self.num_hidden_states):
                if seq[j] in self.emissions[i].keys():
                    T1[i][j] = np.max(T1[:, j-1] * self.transitions[i] * self.emissions[i][seq[j]])
                    T2[i][j] = np.argmax(T1[:, j-1] * self.transitions[i] * self.emissions[i][seq[j]])

        zT = np.zeros(len(seq))
        xT = np.zeros(len(seq))

        zT[len(seq)-1] = np.argmax(T1[:, -1])
        xT[len(seq)-1] = zT[len(seq)-1]

        for j in range(len(seq)-1, 0, -1):
            zT[j-1] = T2[int(zT[j]), j]
            xT[j-1] = zT[j-1]

        cur_state = int(xT[len(seq)-1])
        generated_seq = ''
        for i in range(num):
            cur_state = np.random.choice(range(self.num_hidden_states), p=self.transitions[cur_state])
            selected_token = np.random.choice(list(self.emissions[cur_state].keys()), p=list(self.emissions[cur_state].values()))
            generated_seq = generated_seq + selected_token + ' '
        print(text_line + ' ' + generated_seq)
