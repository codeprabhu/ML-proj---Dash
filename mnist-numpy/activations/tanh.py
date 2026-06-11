import numpy as np
from .activations import Activation

class Tanh(Activation):
    def forward(self,x):
        return np.tanh(x)

    def backward(self,Z, dA):
        return (1-np.tanh(Z)**2)*dA