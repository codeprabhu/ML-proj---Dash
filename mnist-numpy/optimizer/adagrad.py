import numpy as np
from .optmizer import Optimizer
class AdaGrad(Optimizer):

    def __init__(
        self,
        parameters,
        learning_rate=0.01,
        epsilon=1e-8
    ): 
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.cache = {}
        L = len(parameters)//2

        for l in range(1,L+1):
            self.cache[f"W{l}"] = np.zeros_like(parameters[f"W{l}"])
            self.cache[f"b{l}"] = np.zeros_like(parameters[f"b{l}"])

    def step(self, parameters, grads):
        L = len(parameters)//2
        for l in range(1,L+1):
            self.cache[f"W{l}"] += grads[f"dW{l}"] ** 2
            self.cache[f"b{l}"] += grads[f"db{l}"] ** 2

            parameters[f"W{l}"] -= self.learning_rate*grads[f"dW{l}"] /(np.sqrt(self.cache[f"W{l}"]) + self.epsilon)
            parameters[f"b{l}"] -= self.learning_rate*grads[f"db{l}"] /(np.sqrt(self.cache[f"b{l}"]) + self.epsilon)