import numpy as np
import pickle


class NeuralNet:

    def __init__(self, num_input_nodes=None, num_output_nodes=None, hidden_layers=None, learning_rate=0.001,
                 iterations=100):
        self.num_input_nodes = num_input_nodes
        self.num_output_nodes = num_output_nodes
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = []
        self.biases = []

        for i in range(len(self.hidden_layers) + 1):
            if i == 0:
                self.weights.append(np.random.randn(self.num_input_nodes, self.hidden_layers[0]))
                self.biases.append(np.random.rand(self.hidden_layers[0],))
            elif i < len(self.hidden_layers):
                self.weights.append(np.random.randn(self.hidden_layers[i - 1], self.hidden_layers[i]))
                self.biases.append(np.random.rand(self.hidden_layers[i],))
            else:
                self.weights.append(np.random.randn(self.hidden_layers[len(self.hidden_layers) - 1], self.num_output_nodes))
                self.biases.append(np.random.randn(self.num_output_nodes,))

        self.X = None
        self.y = None
        self.activations = [None] * (len(self.hidden_layers) + 1)

    @staticmethod
    def sigmoid(Z):
        return 1 / (1 + np.exp(-Z))

    @staticmethod
    def sigmoid_derivative(Z):
        s = NeuralNet.sigmoid(Z)
        return s * (1 - s)

    def forward_propagation(self):
        self.activations[0] = NeuralNet.sigmoid(self.X.dot(self.weights[0]) + self.biases[0])
        for i in range(len(self.hidden_layers) - 1):
            self.activations[i + 1] = NeuralNet.sigmoid(self.activations[i].dot(self.weights[i + 1]) + self.biases[i + 1])
        self.activations[-1] = NeuralNet.sigmoid(self.activations[-2].dot(self.weights[-1]) + self.biases[-1])

    