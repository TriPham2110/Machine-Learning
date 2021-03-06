"""
Ref: https://www.python-course.eu/expectation_maximization_and_gaussian_mixture_models.php
     http://ethen8181.github.io/machine-learning/clustering/GMM/GMM.html#E-Step
     http://www.oranlooney.com/post/ml-from-scratch-part-5-gmm/
     https://machinelearningmastery.com/scale-machine-learning-data-scratch-python/
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal


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

    def plot_contours(self, X, mu, sigma, x_label='X-axis', y_label='Y-axis', title='GMM contour'):
        x, y = np.meshgrid(np.sort(X[:, 0]), np.sort(X[:, 1]))
        XY = np.array([x.flatten(), y.flatten()]).T
        fig = plt.figure(figsize=(10, 10))
        ax0 = fig.add_subplot(111)
        ax0.scatter(X[:, 0], X[:, 1])
        ax0.set_title(title)
        if (X >= 0).all() and (X <= 1).all():
            ax0.set_xlabel(x_label + '_normalized')
            ax0.set_ylabel(y_label + '_normalized')
        else:
            ax0.set_xlabel(x_label)
            ax0.set_ylabel(y_label)
        col = ['green', 'red', 'magenta', 'indigo', 'black', 'yellow']
        if self.n_clusters <= len(col):
            for i in range(self.n_clusters):
                ax0.contour(np.sort(X[:, 0]), np.sort(X[:, 1]),
                            multivariate_normal.pdf(XY, mean=mu[i], cov=sigma[i]).reshape(len(X), len(X)),
                            colors=col[i], alpha=0.5)
                ax0.scatter(mu[i][0], mu[i][1], c='grey', zorder=10, s=100)
        else:
            for i in range(self.n_clusters):
                ax0.contour(np.sort(X[:, 0]), np.sort(X[:, 1]),
                            multivariate_normal.pdf(XY, mean=mu[i], cov=sigma[i]).reshape(len(X), len(X)),
                            colors='black', alpha=0.5)
                ax0.scatter(mu[i][0], mu[i][1], c='grey', zorder=10, s=100)
        plt.show()

    def plot_distribution(self, X, mu, sigma, x_label='X-axis', y_label='Y-axis', title='GMM distribution'):
        x, y = np.meshgrid(np.sort(X[:, 0]), np.sort(X[:, 1]))

        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x
        pos[:, :, 1] = y

        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca(projection='3d')

        if (X >= 0).all() and (X <= 1).all():
            ax.set_xlabel(x_label + '_normalized')
            ax.set_ylabel(y_label + '_normalized')
        else:
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
        ax.set_zlabel('PDF')

        if self.n_clusters <= 3:
            for i in range(self.n_clusters):
                rv = multivariate_normal(mu[i], sigma[i])
                ax.plot_surface(x, y, rv.pdf(pos), cmap='coolwarm', linewidth=1, antialiased=True)
        else:
            print('Limit number of gaussian distributions to 3')

        ax.set_title(title)
        plt.show()

    @staticmethod
    def normalize(X):
        if isinstance(X, pd.DataFrame):
            X = X.values

        minmax = []
        for i in range(len(X[0])):
            col_values = [row[i] for row in X]
            min = np.min(col_values)
            max = np.max(col_values)
            minmax.append([min, max])

        for row in X:
            for i in range(len(row)):
                row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
        return X