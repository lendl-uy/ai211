# Code Author: Jan Lendl R. Uy
# Student Number: 2019-00312
# Class: AI 211 FZZQ

# Coding Challenge 9: Perform Stochastic Gradient Descent

# Useful references:

import numpy as np

np.set_printoptions(precision=2)

class NeuralNetwork:
    
    def __init__(self, input_size, hidden_sizes, output_size):
        self.weights = []
        self.biases = []
        self.layers = []
        
        # Initialize weights and biases for hidden layers
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        layers = len(layer_sizes)
        for i in range(layers-1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0/input_size)
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
        print(f"self.weights = {self.weights}")
        print(f"self.biases = {self.biases}\n")

    def forward_pass(self, input_layer):
        x = input_layer
        self.layers.append(x)

        # Forward pass through hidden layers
        for i in range(len(self.weights)-1):
            z = x @ self.weights[i] + self.biases[i]
            a = self.relu(z)
            self.layers.append(a)
            x = a

        # Forward pass through the output layer
        # Resulting output layer is the prediction
        output_layer = x @ self.weights[-1] + self.biases[-1]
        self.layers.append(output_layer)
        return output_layer

    def backpropagation(self, y_true, learning_rate=0.01):
        
        m = y_true.shape[0]

        # Compute loss and initialize upstream gradient
        y_predicted = self.layers[-1]
        L = self.mean_squared_error(y_predicted, y_true)
        upstream_grad = 2 * (self.layers[-1] - y_true) / m
        
        # Backward pass through output layer
        self.weights[-1] -= learning_rate * np.dot(self.layers[-2].T, upstream_grad)
        self.biases[-1] -= learning_rate * np.sum(upstream_grad, axis=0, keepdims=True)

        # Backward pass through hidden layers
        for i in range(len(self.weights)-2, -1, -1):
            upstream_grad = np.dot(upstream_grad, self.weights[i + 1].T) * self.relu_derivative(self.layers[i + 1])
            self.weights[i] -= learning_rate * np.dot(self.layers[i].T, upstream_grad)
            self.biases[i] -= learning_rate * np.sum(upstream_grad, axis=0, keepdims=True)

    def mean_squared_error(self, predictions, targets):
        return np.mean(np.square(predictions - targets))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
def verify_computational_graph(A, suppress_success_flag=False):
    
    rows, cols = A.shape
        
    if not np.allclose(A, np.zeros((rows, cols), dtype="float"), atol=1e-2):
        raise RuntimeError(f"One or more factors of the decomposition are incorrect!")
    
    if not suppress_success_flag:
        print(f"Computed factors from SVD are correct!")