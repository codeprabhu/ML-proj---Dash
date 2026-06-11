from .optmizer import Optimizer

class SGD(Optimizer):

    def __init__(
        self,
        parameters,
        learning_rate=0.01
    ):
        super().__init__(
            parameters,
            learning_rate
        )

    def step(
        self,
        parameters,
        grads
    ):

        L = len(parameters)//2

        for l in range(1, L+1):

            parameters[f"W{l}"] -= (
                self.learning_rate
                *
                grads[f"dW{l}"]
            )

            parameters[f"b{l}"] -= (
                self.learning_rate
                *
                grads[f"db{l}"]
            )

        return parameters