import numpy as np
from .optmizer import Optimizer
class Nesterov(Optimizer):

    def __init__(
        self,
        parameters,
        learning_rate=0.01,
        beta=0.9
    ):
        self.learning_rate = learning_rate
        self.beta = beta
        self.v = {}

        L = len(parameters)//2
        for l in range(1,L+1):
            self.v[f"W{l}"] = np.zeros_like(parameters[f"W{l}"])
            self.v[f"b{l}"] = np.zeros_like(parameters[f"b{l}"])

    def step(self, parameters, grads):
        L = len(parameters)//2
        for l in range(1,L+1):
            self.v[f"W{l}"] = self.beta*self.v[f"W{l}"] + (1-self.beta)*grads[f"dW{l}"]
            self.v[f"b{l}"] = self.beta*self.v[f"b{l}"] + (1-self.beta)*grads[f"db{l}"]

            parameters[f"W{l}"] -= self.learning_rate*(self.beta*self.v[f"W{l}"] + (1-self.beta)*grads[f"dW{l}"])
            parameters[f"b{l}"] -= self.learning_rate*(self.beta*self.v[f"b{l}"] + (1-self.beta)*grads[f"db{l}"])

        return parameters