import numpy as np
from .activations import Activation

class ReLU(Activation):
    def forward(self,x):
        return np.maximum(0,x)
    
    def backward(self,Z, dA):
        return dA*(Z>0)
