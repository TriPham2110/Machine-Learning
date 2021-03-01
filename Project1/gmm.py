"""
Ref: https://www.python-course.eu/expectation_maximization_and_gaussian_mixture_models.php
     http://ethen8181.github.io/machine-learning/clustering/GMM/GMM.html#E-Step
     http://www.oranlooney.com/post/ml-from-scratch-part-5-gmm/
     https://machinelearningmastery.com/scale-machine-learning-data-scratch-python/
"""

import numpy as np
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
import pandas as pd


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
        if isinstance(X, pd.DataFrame):
            X = X.values
        np.random.seed(self.seed)
        random_row = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.mu = [X[row_index] for row_index in random_row]
        self.sigma = [np.cov(np.transpose(X).astype(float)) for k in range(self.n_clusters)]
        self.pi = np.ones(self.n_clusters) / self.n_clusters
        self.responsibility = np.ones(X.shape) / self.n_clusters

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
        fig = plt.figure(figsize=(10, 10))
        ax0 = fig.add_subplot(111)
        ax0.scatter(X[:, 0], X[:, 1])
        ax0.set_title(title)
        col = ['green', 'red', 'magenta', 'indigo', 'black', 'yellow']
        if self.n_clusters <= len(col):
            for i in range(self.n_clusters):
                ax0.contour(np.sort(X[:, 0]), np.sort(X[:, 1]),
                            multivariate_normal.pdf(XY, mean=mu[i], cov=sigma[i]).reshape(len(X), len(X)), colors=col[i], alpha=0.5)
                ax0.scatter(mu[0][0], mu[0][1], c='grey', zorder=10, s=100)
                ax0.scatter(mu[1][0], mu[1][1], c='grey', zorder=10, s=100)
        else:
            for i in range(self.n_clusters):
                ax0.contour(np.sort(X[:, 0]), np.sort(X[:, 1]),
                            multivariate_normal.pdf(XY, mean=mu[i], cov=sigma[i]).reshape(len(X), len(X)), colors='black', alpha=0.5)
                ax0.scatter(mu[0][0], mu[0][1], c='grey', zorder=10, s=100)
                ax0.scatter(mu[1][0], mu[1][1], c='grey', zorder=10, s=100)
        plt.show()

    @staticmethod
    def find_minmax(X):
        minmax = []
        for i in range(len(X[0])):
            col_values = [row[i] for row in X]
            value_min = min(col_values)
            value_max = max(col_values)
            minmax.append([value_min, value_max])
        return minmax

    @staticmethod
    def normalize(X):
        minmax = gmm.find_minmax(X)
        for row in X:
            for i in range(len(row)):
                row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
        return X