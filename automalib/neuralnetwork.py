import numpy as np

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
    
    def __f (self, batch):
        out = batch.copy()
        for i in range(len(self.weights)):
            # self.weights[i].dot(out.T).T
            # With help from http://ajcr.net/Basic-guide-to-einsum/
            tmp = np.einsum('ij,kj->ki', self.weights[i], out)
            out = self.funcs[i](tmp + self.biases[i])
        
        return out
    
    def __call__ (self, batch):
        input = np.array(batch)
        if len(input.shape) == 1: return self.__f(np.array([input]))[0]
        else: return self.__f(input)