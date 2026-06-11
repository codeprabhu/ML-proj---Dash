from abc import ABC, abstractmethod
import numpy as np
class Initializer(ABC):
    @abstractmethod
    def initialize(self, layer_sizes):
        pass


class RandomNormal(Initializer):
    def initialize(self, layer_sizes):
        parameters =  {}
        for l in range(1, len(layer_sizes)):
            parameters[f"W{l}"] = (np.random.randn(layer_sizes[l-1],layer_sizes[l]))*0.01
            parameters[f"b{l}"] = np.zeros((1, layer_sizes[l]))

        return parameters
    
class Xavier(Initializer):
    def initialize(self, layer_sizes):
        parameters = {}
        for l in range(1, len(layer_sizes)):
            scale = np.sqrt(1/layer_sizes[l-1])
            parameters[f"W{l}"] = (np.random.randn(layer_sizes[l-1], layer_sizes[l]))*scale
            parameters[f"b{l}"] = np.zeros((1,layer_sizes[l]))

        return parameters
    
class He(Initializer):
    def initialize(self, layer_sizes):
        parameters = {}
        for l in range(1, len(layer_sizes)):
            scale = np.sqrt(2/layer_sizes[l-1])
            parameters[f"W{l}"] = (np.random.randn(layer_sizes[l-1], layer_sizes[l])* scale)
            parameters[f"b{l}"] = np.zeros((1, layer_sizes[l]))

        return parameters
    
class LeCun(Initializer):
    def initialize(self, layer_sizes):
        parameters = {}
        for l in range(1, len(layer_sizes)):
            scale = np.sqrt(1/layer_sizes[l-1])
            parameters[f"W{l}"] = np.random.randn(layer_sizes[l-1], layer_sizes[l])*scale
            parameters[f"b{l}"] = np.zeros((1, layer_sizes[l]))
        
        return parameters
    
INITIALIZERS = {

    "random": RandomNormal,

    "xavier": Xavier,

    "he": He,

    "lecun": LeCun
}