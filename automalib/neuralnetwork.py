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
    
    def __call__ (self, batch):        
        layer_output = np.array(batch)
        if len(layer_output.shape) == 1: return self(np.array([layer_output]))[0]
        
        for i in range(len(self.weights)):
            # self.weights[i].dot(layer_output.T).T
            # With help from http://ajcr.net/Basic-guide-to-einsum/
            tmp = np.einsum('ij,kj->ki', self.weights[i], layer_output)
            layer_output = self.funcs[i](tmp + self.biases[i])
        
        return layer_output