import numpy as np
from .activations import Activation

class Swish(Activation):
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def forward(self,x):
        return x*self.sigmoid(x)
    
    def backward(self,Z,dA):
        s = self.sigmoid(Z)
        return (s+Z*Z*(1-s))*dA