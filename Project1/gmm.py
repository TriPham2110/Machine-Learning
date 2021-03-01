"""
Ref: https://www.python-course.eu/expectation_maximization_and_gaussian_mixture_models.php
     http://ethen8181.github.io/machine-learning/clustering/GMM/GMM.html#E-Step
     http://www.oranlooney.com/post/ml-from-scratch-part-5-gmm/
"""

import numpy as np
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt


class gmm:

    def __init__(self, n_clusters, n_iterations, seed=0):
        self.n_clusters = n_clusters
        self.n_iterations = n_iterations
        self.seed = seed
        self.mu = None
        self.sigma = None
        self.pi = None
        self.responsibility = None

    def train(self, X):
        np.random.seed(self.seed)
        random_row = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.mu = [X[row_index] for row_index in random_row]
        self.sigma = [np.cov(np.transpose(X).astype(float)) for k in range(self.n_clusters)]
        self.pi = np.full(shape=self.n_clusters, fill_value=1 / self.n_clusters)
        self.responsibility = np.full(shape=X.shape, fill_value=1 / self.n_clusters)

        for i in range(self.n_iterations):
            self.expectation(X)
            self.maximization(X)

        return self

    def expectation(self, X):
        likelihood = np.zeros((X.shape[0], self.n_clusters))
        for c in range(self.n_clusters):
            likelihood[:, c] = multivariate_normal.pdf(X, self.mu[c], self.sigma[c])
        self.responsibility = (likelihood * self.pi) / np.sum(likelihood * self.pi, axis=1, keepdims=True)
        self.pi = self.responsibility.mean(axis=0)
        return self

    def maximization(self, X):
        for i in range(self.n_clusters):
            responsibility = self.responsibility[:, [i]]
            total_responsibility = self.responsibility[:, [i]].sum()
            self.mu[i] = (X * responsibility).sum(axis=0) / total_responsibility
            self.sigma[i] = np.cov(np.transpose(X).astype(float),
                                   aweights=(responsibility / total_responsibility).flatten(),
                                   bias=True)
        return self

    def predict(self, X):
        likelihood = np.zeros((X.shape[0], self.n_clusters))
        for c in range(self.n_clusters):
            likelihood[:, c] = multivariate_normal.pdf(X, self.mu[c], self.sigma[c])
        self.responsibility = (likelihood * self.pi) / np.sum(likelihood * self.pi, axis=1, keepdims=True)
        return np.argmax(self.responsibility, axis=1)

    def plot_contours(self, X, mu, sigma, title):
        x, y = np.meshgrid(np.sort(X[:, 0]), np.sort(X[:, 1]))
        XY = np.array([x.flatten(), y.flatten()]).T
        reg_sigma = 1e-6 * np.identity(len(X[0]))
        fig = plt.figure(figsize=(10, 10))
        ax0 = fig.add_subplot(111)
        ax0.scatter(X[:, 0], X[:, 1])
        ax0.set_title(title)
        for m, c in zip(mu, sigma):
            c += reg_sigma
            multi_normal = multivariate_normal(mean=m, cov=c)
            ax0.contour(np.sort(X[:, 0]), np.sort(X[:, 1]),
                        multi_normal.pdf(XY).reshape(len(X), len(X)), colors='black', alpha=0.3)
            ax0.scatter(m[0], m[1], c='grey', zorder=10, s=100)
        plt.show()