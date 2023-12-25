# Code Author: Jan Lendl R. Uy
# Student Number: 2019-00312
# Class: AI 211 FZZQ

# Coding Challenge 9 Tests

from Gradient_Descent import *
import matplotlib.pyplot as plt

EPOCHS = 10000
SAMPLING_RATE = 100
MAX_SEED_ATTEMPTS = 25

def plot_mse(epochs, errors, title):

    plt.plot(epochs, errors)
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.title(title)
    plt.show()
    
def train_2_layer_nn_with_xor_operation_using_stochastic_gradient_descent():
    
    print(f"\n---- Testing Stochastic Gradient Descent for a 2-Layer NN Trained with XOR ----\n")
    
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
    hidden_sizes = [3, 3]
    output_size = 1
    
    epoch_samples = np.linspace(0, EPOCHS, SAMPLING_RATE)

    nn = ComputationalGraph(input_size, hidden_sizes, output_size)

    # Train the neural network
    for seed in range(MAX_SEED_ATTEMPTS):
        
        # Reinitialize neural network with new randomizer seed
        nn = ComputationalGraph(input_size, hidden_sizes, output_size, seed)
        mse_samples = []
        
        for n in range(EPOCHS):
            # Forward-pass
            y_predicted = nn.forward_pass(x)
            
            # Backpropagation
            nn.stochastic_gradient_descent(y_true)
            
            if n%SAMPLING_RATE == 0:
                loss = nn.mean_squared_error(y_predicted, y_true)
                mse_samples.append(loss)
                # print(f"Epoch {n}, Loss: {loss}")
            
        if nn.verify_predictions(y_true):
            break
        print(f"Trying a different seed for randomization.")
    
    y_predicted = nn.forward_pass(x)      
    print(f"y_predicted = {y_predicted}")
    plot_mse(epoch_samples, mse_samples, f"Mean Squared Error vs. Epoch for 2-Layer NN")
    
def train_n_layer_nn_with_xor_operation_using_stochastic_gradient_descent():
    
    print(f"\n---- Testing Stochastic Gradient Descent for an N-Layer NN Trained with XOR ----\n")
    
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
    hidden_layer_depth = np.random.randint(2, 10)
    hidden_sizes = np.random.randint(2, 10)*np.ones(hidden_layer_depth).astype("int")
    hidden_sizes = hidden_sizes.tolist()
    output_size = 1
    
    epoch_samples = np.linspace(0, EPOCHS, SAMPLING_RATE)

    nn = ComputationalGraph(input_size, hidden_sizes, output_size)

    # Train the neural network
    for seed in range(MAX_SEED_ATTEMPTS):
        
        # Reinitialize neural network with new randomizer seed
        nn = ComputationalGraph(input_size, hidden_sizes, output_size, seed)
        mse_samples = []
        
        for n in range(EPOCHS):
            # Forward-pass
            y_predicted = nn.forward_pass(x)
            
            # Backpropagation
            nn.stochastic_gradient_descent(y_true)
            
            if n%SAMPLING_RATE == 0:
                loss = nn.mean_squared_error(y_predicted, y_true)
                mse_samples.append(loss)
                # print(f"Epoch {n}, Loss: {loss}")
            
        if nn.verify_predictions(y_true):
            break
        print(f"Trying a different seed for randomization.")
    
    y_predicted = nn.forward_pass(x)      
    print(f"y_predicted = {y_predicted}")
    plot_mse(epoch_samples, mse_samples, f"Mean Squared Error vs. Epoch for {hidden_layer_depth}-Layer NN")

def main():
    
    train_2_layer_nn_with_xor_operation_using_stochastic_gradient_descent()
    train_n_layer_nn_with_xor_operation_using_stochastic_gradient_descent()

if __name__ == "__main__":
    main()