# MNIST Neural Network From Scratch

A neural network framework implemented entirely using NumPy.

## Features

- Forward propagation
- Backpropagation
- Multiple activation functions
- Multiple loss functions
- Multiple initialization methods
- Multiple optimizers

## Activations

- ReLU
- LeakyReLU
- PReLU
- ELU
- Swish
- GELU
- Mish
- Sigmoid
- Tanh

## Optimizers

- SGD
- Momentum
- Nesterov
- AdaGrad
- RMSProp
- Adam

## Experimental Results

864 combinations were tested on MNIST.

Best configuration:

- Optimizer: AdaGrad
- Activation: GELU
- Initializer: LeCun
- Loss: CrossEntropy

Accuracy: 69.33% after 2 epochs.