import numpy as np
from automalib.utils import batch_wrapper

# --- Non-zero constant ---
EPSILON = 10**-10

# --- Cost function class ---
class CostFunction ():
    """
    Class that represents a cost function. Base function is
    mandatory but you can miss the derivative. Here, the function
    is derived along the first vector. Both functions must handle
    2D args (lists of vectors).
    """
    
    def __derivative_not_implemented (self, *args, **kwargs):
        raise NotImplementedError("Derivative is not implemented.")
    
    def __init__ (self, function, derivative = None):
        self.__f = batch_wrapper(function)
        self.derivative = batch_wrapper(derivative) if derivative else self.__derivative_not_implemented
    
    def __call__ (self, h, y): return self.__f(h, y)

# --- norms ---
def L (pow):
    """
    L<pow> cost / loss
    """
    return CostFunction(
        lambda h, y: 1/pow * np.mean(np.power(np.abs(h - y), pow), axis = 1),
        lambda h, y: np.mean(np.power(np.abs(h - y), pow - 1), axis = 1)
    )

L1 = L(1)
L2 = L(2)

# --- cross-entropy ---
cross_entropy = CostFunction(
    lambda h, y: - np.sum(y*np.log(h + EPSILON) + (1 - y)*np.log(1 - h - EPSILON), axis = 1),
    lambda h, y: - np.sum(y/(h + EPSILON) - (1 - y)/(1 - h - EPSILON), axis = 1)
)