import numpy as np
from .activations import Activation

class Sigmoid(Activation):
    def forward(self, x):
        return 1/(1+np.exp(-x))
    
    def backward(self, Z, dA):
        s = self.forward(Z)
        return  dA*s*(1-s)