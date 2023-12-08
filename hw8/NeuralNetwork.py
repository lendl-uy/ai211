import numpy as np

class NeuralNetwork:
    
    def __init__(self, input_size, hidden_sizes, output_size, seed):
        self.weights = []
        self.biases = []
        self.layers = []
        
        # Initialize weights and biases for hidden layers
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        layers = len(layer_sizes)
        for i in range(layers-1):
            # He initialization for weight matrices w
            np.random.seed(seed)
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0/(layer_sizes[i]))
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def forward_pass(self, input_layer):
        x = input_layer
        self.layers.append(x)

        # Forward pass through the hidden layers
        for i in range(len(self.weights)-1):
            z = x @ self.weights[i] + self.biases[i]
            a = relu(z)
            self.layers.append(a)
            x = a

        # Forward pass through the output layer
        # Resulting output layer is the prediction
        output_layer = x @ self.weights[-1] + self.biases[-1]
        self.layers.append(output_layer)
        #print(f"layers = {self.layers}")
        return output_layer
    
    def backpropagation(self, weight_gradients, bias_gradients, y_true, λ=0.03):

        # Update weights and biases using computed gradients
        for i in range(len(self.weights)):
            self.weights[i] -= λ * weight_gradients[i]
            self.biases[i] -= λ * bias_gradients[i]
            
    def verify_predictions(self, y_true, tolerance=1e-3, suppress_success_flag=False):
        
        y_predicted = self.layers[-1]
    
        if y_true.shape != y_predicted.shape:
            raise RuntimeError(f"The shape of y_predicted must be {y_true.shape} but was {y_predicted.shape}")
        
        rows, cols = y_predicted.shape

        if not np.allclose(y_predicted-y_true, np.zeros((rows, cols), dtype="float"), atol=tolerance):
            #raise RuntimeError(f"y_predicted must be ≈ {y_true} but was {y_predicted}!")
            return False
                
        #print(f"y_predicted = {y_predicted}")
        #print(f"Computed gradients are correct! Gradient descent converges to a sufficient minimum!")
        return True
    
    def get_layers(self):
        return self.layers.copy() # Return a defensive copy
    
    def get_weights(self):
        return self.weights.copy() # Return a defensive copy
    
    def get_biases(self):
        return self.biases.copy() # Return a defensive copy
    
def relu(x):
    return np.maximum(0, x)
    
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def mean_squared_error(predictions, targets):
    return np.mean(np.square(predictions - targets))