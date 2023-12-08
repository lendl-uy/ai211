# Code Author: Jan Lendl R. Uy
# Student Number: 2019-00312
# Class: AI 211 FZZQ

# Coding Challenge 8: Compute Gradients

# Useful references:
# [1] https://web.stanford.edu/class/cs224n/readings/gradient-notes.pdf

import numpy as np
from NeuralNetwork import relu_derivative

np.set_printoptions(precision=2)

# Computes the gradients of an n-layer neural network
# Iterative approach to computing the gradients
def compute_gradients(layers, weights, y_predicted, y_true):
        
    m = y_true.shape[0]

    # Compute the loss function using mean squared error
    y_predicted = layers[-1]
    
    # Get the gradient of the loss function, output node
    # d/dy (L) = 2/m (y_predicted-y_true)
    initial_gradient = 2/m * (y_predicted-y_true)
    
    # Initialize lists for gradients of weights and biases
    weight_gradients = []
    bias_gradients = []
    
    # Compute gradients for the output layer
    weight_gradients.append(np.dot(layers[-2].T, initial_gradient))
    bias_gradients.append(np.sum(initial_gradient, axis=0, keepdims=True))

    # Compute gradients for the hidden layers
    try:
        for i in range(len(weights)-2, -1, -1):
            initial_gradient = np.dot(initial_gradient, weights[i+1].T) * relu_derivative(layers[i+1])
            weight_gradients.append(np.dot(layers[i].T, initial_gradient))
            bias_gradients.append(np.sum(initial_gradient, axis=0, keepdims=True))
    except:
        print(f"initial_gradient = {initial_gradient}")
        print(f"weights[i+1].T = {weights[i+1].T}")
        print(f"relu_derivative(layers[i+1]) = {relu_derivative(layers[i+1])}")
        raise RuntimeError("Something went wrong!")

    # Reverse the gradients to obtain forward-pass order
    weight_gradients.reverse()
    bias_gradients.reverse()
    
    return weight_gradients, bias_gradients