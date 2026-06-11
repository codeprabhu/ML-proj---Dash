import numpy as np
from .optmizer import Optimizer

class Momentum(Optimizer):

    def __init__(
        self,
        parameters,
        learning_rate=0.01,
        beta=0.9
    ):

        super().__init__(
            parameters,
            learning_rate
        )

        self.beta = beta

        self.velocity = {}

        L = len(parameters)//2

        for l in range(1, L+1):

            self.velocity[f"W{l}"] = (
                np.zeros_like(
                    parameters[f"W{l}"]
                )
            )

            self.velocity[f"b{l}"] = (
                np.zeros_like(
                    parameters[f"b{l}"]
                )
            )

    def step(self, parameters, grads):
        L = len(parameters)//2
        for l in range(1, L+1):
            self.velocity[f"W{l}"] = (self.beta*self.velocity[f"W{l}"]+(1-self.beta)*grads[f"dW{l}"])
            self.velocity[f"b{l}"] = (self.beta*self.velocity[f"b{l}"]+(1-self.beta)*grads[f"db{l}"])
            parameters[f"W{l}"] -= (self.learning_rate*self.velocity[f"W{l}"])
            parameters[f"b{l}"] -= (self.learning_rate*self.velocity[f"b{l}"])

        return parameters