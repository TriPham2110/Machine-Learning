"""
Ref: https://realpython.com/python-histograms/
     https://stackoverflow.com/questions/21619347/creating-a-python-histogram-without-pylab
     https://stackoverflow.com/questions/20011122/fitting-a-normal-distribution-to-1d-data
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


class histogram:
    def __init__(self, list):
        self.list = list

    @staticmethod
    def plot_histogram(x, y, x_label='X-axis', y_label='Y-axis', title='Histogram'):
        bars = plt.bar(x, y)
        for bar in bars:
            y_val = bar.get_height()
            plt.text(bar.get_x() + .1, y_val, y_val)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.show()

    def make_histogram(self, bins, min_value=None, max_value=None):
        histogram_values = np.zeros(bins+1)
        if min_value is None:
            min_value = min(self.list)
        if max_value is None:
            max_value = max(self.list)

        for value in self.list:
            bin_index = int(bins * ((value - min_value) / (max_value - min_value)))
            histogram_values[bin_index] += 1
        bin_lower_bounds = [min_value + i * (max_value - min_value) / len(histogram_values) for i in range(len(histogram_values))]

        return bin_lower_bounds, histogram_values

    def plot_pdf(self, bins):
        plt.hist(self.list, bins=bins, density=True, edgecolor='black')
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        mu, sigma = norm.fit(self.list)
        p = norm.pdf(x, mu, sigma)
        plt.plot(x, p, 'k', linewidth=2)
        title = "Fit results: mu = %.2f,  std = %.2f" % (mu, sigma)
        plt.title(title)
        plt.show()