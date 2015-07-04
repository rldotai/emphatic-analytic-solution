"""
Simple feature functions.
"""
import numpy as np


class Feature:
    """Feature function template/base class."""
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __len__(self):
        return self._length

    @property 
    def length(self):
        return self._length


class Wrap(Feature):
    """
    Wrap a sequence of features and ensure terminal states map to the zero 
    vector of appropriate length.
    """
    def __init__(self, *features, terminals=list()):
        self.terminals = set(map(as_tuple, terminals))
        self._features = features
        self._length = sum(len(x) for x in features)

    def __call__(self, x):
        if as_tuple(x) in self.terminals:
            return np.zeros(self._length)
        else:
            return np.concatenate([f(x) for f in self._features])


class FeatureFunction(Feature):
    def __init__(self, n, func):
        self._length = n 
        self._func = func 

    def __call__(self, x):
        ret = np.array(self._func(func))
        assert(len(ret) == self._length)
        return ret 


class Identity(Feature):
    def __init__(self, n):
        self._length = n

    def __call__(self, x):
        x = np.atleast_1d(x)
        assert(len(x) == self._length)
        return x


class Bias(Feature):
    def __init__(self, val=1.0):
        self.val = float(val)
        self._length = 1
        self._array = np.atleast_1d(val)

    def __call__(self, x):
        return np.copy(self._array)


class Unary2Int(Feature):
    def __init__(self, n, offset=1):
        self._length = 1
        self.offset = offset

    def __call__(self, x):
        x = np.array(x)
        assert(x.ndim <= 1)
        assert(np.all((x == 0) | (x == 1)))
        return np.array(self.offset + np.flatnonzero(x == 1))


# These don't really work well with states as basis vectors...
# class Int2Binary(Feature):
#     def __init__(self, n, offset=-1):
#         self._length = n 
#         self.offset = offset 
#         self._array = (1 << np.arange(n))

#     def __call__(self, x: int):
#         x = np.array(x) - self.offset
#         return (x & self._array) > 0


class Int2Unary(Feature):
    def __init__(self, n, offset=0):
        self._length = n 
        self.offset = offset 
        self._array = np.eye(n)

    def __call__(self, x: int):
        return self._array[x - self.offset]


def as_tuple(x):
    """Convert arrays, iterables, or numbers to tuples."""
    if isinstance(x, np.ndarray):
        return tuple(x.flat)
    elif hasattr(x, '__iter__'):
        return tuple(x)
    else:
        return (x,)