import numpy as np

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
        self.__f = function
        self.derivative = derivative or self.__derivative_not_implemented
    
    def __call__ (self, h, y):
        h_ev = np.array(h)
        y_ev = np.array(y)
        if len(h_ev.shape) == 1: return self.__f(np.array([h_ev]), np.array([y_ev]))[0]
        else: return self.__f(h_ev, y_ev)

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