# Performing Stochastic Gradient Descent
## AI 211 Coding Challenge 8

This code submission by Jan Lendl Uy comprises two files:

- NeuralNetwork.py
- ComputationalGraph.py
- ComputationalGraph_Tests.py

`NeuralNetwork.py` contains a class definition of a neural network (NN) with n-hidden layers. Its architecture involves fully-connected neurons with zero bias and a ReLU activation function to introduce nonlinearities to the network. The calculations are carried out by the ComputationalGraph module to abstract the details of NN training.

In `ComputationalGraph.py`, it contains the implementation for computing the gradients of an n-layer neural network using a computational graph. All the calculation-heavy aspects of training a neural network are performed in this module.

On the other hand, `ComputationalGraph_Tests.py` contains the tests performed in verifying the correctness of the computed gradients. Correctness is checked by training a neural network and determining if it arrives to a sufficiently low minimum. RUN this file to check the results for different tests which include:

- Training a 2-layer neural network for XOR operation 
- Training a n-layer neural network for XOR operation
- Training a 2-layer neural network for XNOR operation
- Training a n-layer neural network for XNOR operation