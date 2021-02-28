"""
Ref: https://www.python-course.eu/expectation_maximization_and_gaussian_mixture_models.php
     http://ethen8181.github.io/machine-learning/clustering/GMM/GMM.html#E-Step
"""

import numpy as np
from scipy.stats import norm, multivariate_normal


class gmm:
    def __init__(self, n_clusters, n_iterations, threshold_lvl):
        self.n_clusters = n_clusters
        self.n_iterations = n_iterations
        self.threshold_lvl = threshold_lvl
        self.mu = None
        self.sigma = None
        self.pi = None
        self.cov = None

    def train(self, X):
        rows, cols = X.shape[0], X.shape[1]

        # initialize mean, weight, and covariance matrix array
        self.mu = X[np.random.randint(0, rows, self.n_clusters)]
        self.pi = np.ones(self.n_clusters) / self.n_clusters
        self.cov = np.zeros((self.n_clusters, cols, cols))
        for dim in range(len(self.cov)):
            np.fill_diagonal(self.cov[dim], 1)

        log_likelihood = 0
        log_likelihoods = []

        # calculate log likelihood to monitor convergence point
        for i in range(self.n_iterations):
            cur_log_likelihood = self.expectation(X)
            self.maximization(X)

            # convergence
            if abs(cur_log_likelihood - log_likelihood) < self.threshold_lvl:
                break

            log_likelihood = cur_log_likelihood
            log_likelihoods.append(log_likelihood)

        return log_likelihoods

    def expectation(self, X):
        return None

    def maximization(self, X):
        return None

