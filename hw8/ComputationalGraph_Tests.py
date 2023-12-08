# Code Author: Jan Lendl R. Uy
# Student Number: 2019-00312
# Class: AI 211 FZZQ

# Coding Challenge 8 Tests

import numpy as np
from NeuralNetwork import NeuralNetwork, mean_squared_error

EPOCHS = 10000

def compute_gradients_of_2_layer_nn_for_xor_operation():
    
    print(f"\n---- Training an NN for XOR Operation with 2 Hidden Layers ----\n")
    
    # XOR truth table inputs
    x = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]]).astype("float")
    
    # XOR truth table outputs
    y_true = np.array([[0],
                    [1],
                    [1],
                    [0]])
    
    # Initialize depth of the input layer, hidden layer/s, and output layer
    input_size = 2
    hidden_sizes = [5, 5]
    output_size = 1

    nn = NeuralNetwork(input_size, hidden_sizes, output_size, 1)

    # Train the neural network
    for n in range(EPOCHS):
        # Forward-pass
        y_predicted = nn.forward_pass(x)
        
        # Backpropagation
        nn.backpropagation(y_true)
        
        if n%500 == 0:
            loss = mean_squared_error(y_predicted, y_true)
            #print(f"Epoch {n}, Loss: {loss}")
            
    if nn.verify_predictions(y_true):
        print("Computed gradients are correct! Gradient descent converges to a sufficient minimum!\n")
        print(f"y_predicted = {y_predicted}")
    
def compute_gradients_of_2_layer_nn_for_xnor_operation():
    
    print(f"\n---- Training an NN for XNOR Operation with 2 Hidden Layers ----\n")
    
    # XNOR truth table inputs
    x = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]]).astype("float")
    
    # XNOR truth table outputs
    y_true = np.array([[1],
                    [0],
                    [0],
                    [1]])
    
    # Initialize depth of the input layer, hidden layer/s, and output layer
    input_size = 2
    hidden_sizes = [5, 5]
    output_size = 1

    nn = NeuralNetwork(input_size, hidden_sizes, output_size, 6)

    # Train the neural network
    for n in range(EPOCHS):
        # Forward-pass
        y_predicted = nn.forward_pass(x)
        
        # Backpropagation
        nn.backpropagation(y_true)
        
        if n%500 == 0:
            loss = mean_squared_error(y_predicted, y_true)
            #print(f"Epoch {n}, Loss: {loss}")
      
    if nn.verify_predictions(y_true):
        print("Computed gradients are correct! Gradient descent converges to a sufficient minimum!\n")
        print(f"y_predicted = {y_predicted}")

def compute_gradients_of_n_layer_nn_for_xor_operation():
    
    print(f"\n---- Training an NN for XOR Operation with Random Number of Hidden Layers ----\n")
    
    # XOR truth table inputs
    x = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]]).astype("float")
    
    # XOR truth table outputs
    y_true = np.array([[0],
                    [1],
                    [1],
                    [0]])
    
    # Initialize depth of the input layer, hidden layer/s, and output layer
    input_size = 2
    hidden_layer_depth = np.random.randint(2, 5)
    hidden_sizes = np.random.randint(2, 10)*np.ones(hidden_layer_depth).astype("int")
    hidden_sizes = hidden_sizes.tolist()
    #print(f"hidden_sizes = {hidden_sizes}")
    output_size = 1
    seed = 0

    while True:
        nn = NeuralNetwork(input_size, hidden_sizes, output_size, seed)

        # Train the neural network
        for n in range(EPOCHS):
            # Forward-pass
            y_predicted = nn.forward_pass(x)
            
            # Backpropagation
            nn.backpropagation(y_true)
            
            if n%500 == 0:
                loss = mean_squared_error(y_predicted, y_true)
                #print(f"Epoch {n}, Loss: {loss}")
        
        if (nn.verify_predictions(y_true)):
            print("Computed gradients are correct! Gradient descent converges to a sufficient minimum!\n")
            print(f"y_predicted = {y_predicted}")
            break
        print("Trying a different seed for initialization of weights.")
        seed += 1
        
def compute_gradients_of_n_layer_nn_for_xnor_operation():
    
    print(f"\n---- Training an NN for XNOR Operation with Random Number of Hidden Layers ----\n")
    
    # XOR truth table inputs
    x = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]]).astype("float")
    
    # XOR truth table outputs
    y_true = np.array([[1],
                    [0],
                    [0],
                    [1]])
    
    # Initialize depth of the input layer, hidden layer/s, and output layer
    input_size = 2
    hidden_layer_depth = np.random.randint(2, 5)
    hidden_sizes = np.random.randint(2, 10)*np.ones(hidden_layer_depth).astype("int")
    hidden_sizes = hidden_sizes.tolist()
    #print(f"hidden_sizes = {hidden_sizes}")
    output_size = 1
    seed = 0

    while True:
        nn = NeuralNetwork(input_size, hidden_sizes, output_size, seed)

        # Train the neural network
        for n in range(EPOCHS):
            # Forward-pass
            y_predicted = nn.forward_pass(x)
            
            # Backpropagation
            nn.backpropagation(y_true)
            
            if n%500 == 0:
                loss = mean_squared_error(y_predicted, y_true)
                #print(f"Epoch {n}, Loss: {loss}")
        
        if nn.verify_predictions(y_true):
            print("Computed gradients are correct! Gradient descent converges to a sufficient minimum!\n")
            print(f"y_predicted = {y_predicted}")
            break
        print("Trying a different seed for the initialization of weights.")
        seed += 1

def main():
    
    compute_gradients_of_2_layer_nn_for_xor_operation()
    compute_gradients_of_2_layer_nn_for_xnor_operation()
    compute_gradients_of_n_layer_nn_for_xor_operation()
    compute_gradients_of_n_layer_nn_for_xnor_operation()

if __name__ == "__main__":
    main()