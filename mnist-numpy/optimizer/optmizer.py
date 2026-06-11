from abc import ABC, abstractmethod

class Optimizer(ABC):

    def __init__(
        self,
        parameters,
        learning_rate=0.01
    ):
        self.learning_rate = learning_rate

    @abstractmethod
    def step(
        self,
        parameters,
        grads
    ):
        pass