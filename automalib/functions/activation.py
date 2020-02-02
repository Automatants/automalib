import numpy as np
from automalib.utils import batch_wrapper

# --- Activation function class ---
class ActivationFunction ():
    """
    Class that represents an activation function. Base function is
    mandatory but you can miss the derivative. Here, the function
    is derived along the vector. Both functions must handle
    2D arg (list of vectors).
    """
    
    def __derivative_not_implemented (self, *args, **kwargs):
        raise NotImplementedError("Derivative is not implemented.")
    
    def __init__ (self, function, derivative = None):
        self.__f = batch_wrapper(function)
        self.derivative = batch_wrapper(derivative) if derivative else self.__derivative_not_implemented
    
    def __call__ (self, x): return self.__f(x)

# --- identity ---
identity = ActivationFunction(
    lambda x: x,
    lambda x: np.full_like(x, 1.)
)

# --- sigmoid ---
def __sigm (x): return 1. / (1. + np.exp(-x))

sigmoid = ActivationFunction(
    __sigm,
    lambda x: __sigm(x) * (1. - __sigm(x))
)

# --- leaky relu ---
def leaky_relu (slope):
    """
    Returns a leaky relu with a custom slope.
    """
    
    return ActivationFunction(
        lambda x: np.where(x > 0., 1., slope) * x,
        lambda x: np.where(x > 0., 1., slope)
    )

# --- relu ---
relu = leaky_relu(0.)

# --- step ---
step = ActivationFunction(
        lambda x: np.where(x > 0., 1., 0.),
        lambda x: np.full_like(x, 0.)
)

# --- tanh ---
tanh = ActivationFunction(
    lambda x: np.tanh(x),
    lambda x: 1./np.power(np.cosh(x), 2.)
)