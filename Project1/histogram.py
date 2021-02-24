"""
Ref: https://realpython.com/python-histograms/
     https://stackoverflow.com/questions/21619347/creating-a-python-histogram-without-pylab
"""
import numpy as np
import matplotlib.pyplot as plt


class histogram:
    def __init__(self, list):
        self.list = list

    @staticmethod
    def plot_histogram(x, y, x_label='X-axis', y_label='Y-axis', title='Histogram'):
        plt.bar(x, y)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.show()

    def make_histogram(self, list, bins, min_value=None, max_value=None):
        histogram_values = np.zeros(bins+1)
        if min_value is None:
            min_value = min(list)
        if max_value is None:
            max_value = max(list)

        for value in list:
            bin_index = int(bins * ((value - min_value) / (max_value - min_value)))
            histogram_values[bin_index] += 1
        bin_lower_bounds = [min_value + i * (max_value - min_value) / len(histogram_values) for i in range(len(histogram_values))]

        return bin_lower_bounds, histogram_values