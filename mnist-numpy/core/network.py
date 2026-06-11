import numpy as np
from .forward import forward
from .backward import backward

class NeuralNetworks:
    def __init__(self, layer_sizes, hidden_activation, output_activation, initializer):
        self.layer_sizes = layer_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.parameters = (initializer.initialize(layer_sizes))

    def forward(self, x):
        return forward(x, self.parameters, self.hidden_activation, self.output_activation)
    
    def backward(self, X, y, cache, loss):
        return backward(X, y, self.parameters, cache, self.hidden_activation, self.output_activation, loss)
    