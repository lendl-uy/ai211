# Performing Stochastic Gradient Descent
## AI 211 Coding Challenge 8

This code submission by Jan Lendl Uy comprises two files:

- NeuralNetwork.py
- Compute_Gradients.py
- Compute_Gradients_Tests.py

`NeuralNetwork.py` contains a class definition of a neural network (NN) with n-hidden layers. Its architecture involves fully-connected neurons with zero bias and a ReLU activation function to introduce nonlinearities to the network. Tweaking the seed when initializing the NN is necessary to determine the best set of weights for training.

In `Compute_Gradients.py`, it contains the implementation for computing the gradients of an n-layer neural network. You may check this file to inspect the implementation of the algorithm.

On the other hand, `Compute_Gradients_Tests.py` contains the tests performed in verifying the correctness of the computed gradients. Correctness is checked by training a neural network and determining if it arrives to a sufficiently low minimum. RUN this file to check the results for different tests which include:

- Training a 2-layer neural network for XOR operation 
- Training a n-layer neural network for XOR operation
- Training a 2-layer neural network for XNOR operation
- Training a n-layer neural network for XNOR operation