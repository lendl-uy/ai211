# Code Author: Jan Lendl R. Uy
# Student Number: 2019-00312
# Class: AI 211 FZZQ

# Coding Challenge 8: Compute Gradients

# Useful references:
# [1] https://web.stanford.edu/class/cs224n/readings/gradient-notes.pdf

import numpy as np

np.set_printoptions(precision=2)

class ComputationalGraph:
    
    # Instead of modeling each operation as a node in the graph,
    # these can be represented as vectors for simpler and 
    # faster computation
    def __init__(self, weights, biases):
        self.weights = weights.copy()
        self.biases = biases.copy()
        self.nodes = []

    def forward(self, input_layer):
        x = input_layer
        self.nodes.append(x)

        # Forward pass through the hidden layers
        for i in range(len(self.weights)-1):
            z = x @ self.weights[i] + self.biases[i]
            a = self.relu(z)
            self.nodes.append(a)
            x = a

        # Forward pass through the output layer
        # Resulting output layer is the prediction
        output_layer = x @ self.weights[-1] + self.biases[-1]
        self.nodes.append(output_layer)
        return self.nodes.copy()

    # Computes the gradients of an n-layer neural network
    # Iterative approach to computing the gradients
    def compute_gradients(self, y_true):
            
        m = y_true.shape[0]

        # Compute the loss function using mean squared error
        y_predicted = self.nodes[-1]
        
        # Get the gradient of the loss function, output node
        # d/dy (L) = 2/m (y_predicted-y_true)
        upstream_gradient = 2/m * (y_predicted-y_true)
        
        # Initialize lists for gradients of weights and biases
        weight_gradients = []
        bias_gradients = []
        
        # Compute gradients for the output layer
        weight_gradients.append(np.dot(self.nodes[-2].T, upstream_gradient))
        bias_gradients.append(np.sum(upstream_gradient, axis=0, keepdims=True))

        # Compute gradients for the hidden layers
        for i in range(len(self.weights)-2, -1, -1):
            upstream_gradient = np.dot(upstream_gradient, self.weights[i+1].T) * self.relu_derivative(self.nodes[i+1])
            weight_gradients.append(np.dot(self.nodes[i].T, upstream_gradient))
            bias_gradients.append(np.sum(upstream_gradient, axis=0, keepdims=True))

        # Reverse the gradients to obtain forward-pass order
        weight_gradients.reverse()
        bias_gradients.reverse()
            
        return weight_gradients, bias_gradients
    
    def relu(self, x):
        return np.maximum(0, x)
        
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)