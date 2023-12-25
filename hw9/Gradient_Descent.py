# Code Author: Jan Lendl R. Uy
# Student Number: 2019-00312
# Class: AI 211 FZZQ

# Coding Challenge 9: Stochasic Gradient Descent

# Useful references:
# [1] https://web.stanford.edu/class/cs224n/readings/gradient-notes.pdf

import numpy as np

np.set_printoptions(precision=2)

EPOCHS = 10000

class ComputationalGraph:
    
    # Instead of modeling each operation as a node in the graph,
    # these can be represented as vectors for simpler and 
    # faster computation
    def __init__(self, input_size, hidden_sizes, output_size, seed=0):
        self.weights = []
        self.biases = []
        self.layers = []
        np.random.seed(seed)
        
        # Initialize weights and biases for hidden layers
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        layers = len(layer_sizes)
        for i in range(layers-1):
            # He initialization for weight matrices w
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0/(layer_sizes[i]))
            # Bias matrices start as zero matrices
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def forward_pass(self, input_layer):
        self.layers = []
        x = input_layer
        self.layers.append(x)

        # Forward pass through the hidden layers
        for i in range(len(self.weights)-1):
            z = x @ self.weights[i] + self.biases[i]
            a = self.relu(z)
            self.layers.append(a)
            x = a

        # Forward pass through the output layer
        # Resulting output layer is the prediction
        output_layer = x @ self.weights[-1] + self.biases[-1]
        self.layers.append(output_layer)
        return self.layers[-1]

    # Performs stochastic gradient descent to select only a sample
    # of the output in performing backpropagation 
    # Utilizes chain rule in calculating gradients
    def stochastic_gradient_descent(self, y_true, λ=0.01):
        
        m, n = y_true.shape
        y_predicted = self.layers[-1]
        
        # Randomly select indices of the input/output which will be
        # used for gradient descent
        # By default batch size is 1 for simplicity
        random_indices = np.random.choice(m, size=1, replace=False)
        # print(f"random_indices = {random_indices}")
        
        # Extract the batch of true labels and predicted outputs
        y_true_batch = y_true[random_indices]
        y_predicted_batch = y_predicted[random_indices]
        
        for index in random_indices:
            # Compute the gradient of the loss function
            # dL/d(yhat) = 2/m * (y_hat - y), m = batch_size
            upstream_gradient = 2 * (y_predicted_batch - y_true_batch)
            
            # Initialize lists for gradients of weights and biases
            weight_gradients = []
            bias_gradients = []

            # Compute gradients for the output layer
            # print(f"upstream_gradient = {upstream_gradient}")
            weight_gradients.append(self.layers[-2][index:index+1, :].T @ upstream_gradient)
            bias_gradients.append(np.sum(upstream_gradient, axis=0, keepdims=True))
            
            # Compute gradients for the hidden layers
            for i in range(len(self.weights)-2, -1, -1):
                upstream_gradient = np.dot(upstream_gradient, self.weights[i+1].T) * self.relu_derivative(self.layers[i+1][index:index+1, :])
                weight_gradient = np.dot(self.layers[i][index:index+1, :].T, upstream_gradient)
                weight_gradients.append(weight_gradient)
                bias_gradient = np.sum(upstream_gradient, axis=0, keepdims=True)
                bias_gradients.append(bias_gradient)
                
                # Update weights and biases for hidden layers
                self.weights[i] -= λ * weight_gradient
                self.biases[i] -= λ * bias_gradient
                
            # Reverse the gradients to obtain forward-pass order
            weight_gradients.reverse()
            bias_gradients.reverse()
            
        return weight_gradients, bias_gradients
    
    def verify_predictions(self, y_true, tolerance=1e-3):
        
        y_predicted = self.layers[-1]
    
        if y_true.shape != y_predicted.shape:
            raise RuntimeError(f"The shape of y_predicted must be {y_true.shape} but was {y_predicted.shape}")
        
        rows, cols = y_predicted.shape

        if not np.allclose(y_predicted-y_true, np.zeros((rows, cols), dtype="float"), atol=tolerance):
            return False
        return True
    
    def relu(self, x):
        return np.maximum(0, x)
        
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def mean_squared_error(self, predictions, targets):
        return np.mean(np.square(predictions - targets))
    
    def print_graph(self):
        print("Computational Graph:")
        for i in range(len(self.layers)):
            print(f"Layer {i+1}: {self.layers[i]}")