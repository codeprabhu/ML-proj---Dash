import numpy as np
from .loss import Loss
class Huber(Loss):
    def __init__(self, delta=1.0):
        self.delta = delta

    def forward(self, y_true, y_pred):
        error = y_pred-y_true
        small = (np.abs(error) <= self.delta)
        loss = np.where(small, 0.5*error**2, self.delta*(np.abs(error) - 0.5*self.delta))
        return np.mean(loss)
    
    def backward(self, y_true, y_pred):
        error = y_pred - y_true
        small = np.abs(error)<= self.delta
        return np.where(small, error, self.delta*np.sign(error))