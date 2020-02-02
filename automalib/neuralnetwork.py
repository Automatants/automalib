import numpy as np
from automalib.utils import batch_wrapper_object

class NeuralNetwork ():
    """
    Class used to handle NN. It is initialized with
    a list of (weight, bias, activ. function). It
    can be treated as a function.
    """
    
    def __init__ (self, layers):
        entry = np.array(layers)
        self.weights    = entry[:,0]
        self.biases     = entry[:,1]
        self.funcs      = entry[:,2]
    
    @batch_wrapper_object()
    def __call__ (self, batch):
        out = batch
        for i in range(len(self.weights)):
            tmp = self.weights[i].dot(out.T).T
            out = self.funcs[i](tmp + self.biases[i])
        
        return out