import numpy as np
from .loss import Loss
class MSE(Loss):
    def forward(self, y_true, y_pred):
        return np.mean((y_true-y_pred)**2)
    
    def backward(self, y_true, y_pred):
        return (2*(y_pred-y_true)/y_true.size)
    