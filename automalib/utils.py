import numpy as np

def batch_wrapper (f, dim = 1):
    """
    Wrapper for functions working with batches.
    Does not work for object's methods.
    f(A)        - f(A)
    f([A, B])   - [f(A), f(B)]
    """
    
    def wrap (*args, **kwargs):
        ev = np.array(args)
        if len(ev.shape) == dim + 1: return f(*ev.reshape(ev.shape[:1] + (1,) + ev.shape[1:]), **kwargs)[0]
        else: return f(*ev, **kwargs)
    return wrap

def batch_wrapper_object (dim = 1):
    """
    Wrapper for functions working with batches.
    Works for object's methods.
    f(A)        - f(A)
    f([A, B])   - [f(A), f(B)]
    """
    
    def wrapper (f):
        def wrap (self, *args, **kwargs):
            ev = np.array(args)
            if len(ev.shape) == dim + 1: return f(self, *ev.reshape(ev.shape[:1] + (1,) + ev.shape[1:]), **kwargs)[0]
            else: return f(self, *ev, **kwargs)
        return wrap
    return wrapper