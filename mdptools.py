import numpy as np 
from numpy.linalg import pinv
from functools import reduce, wraps


def as_tuple(x):
    """Convert arrays, iterables, or numbers to tuples."""
    if isinstance(x, np.ndarray):
        return tuple(x.flat)
    elif hasattr(x, '__iter__'):
        return tuple(x)
    else:
        return (x,)

def all_equal(seq):
    """Check that all elements of a sequence are the same. """
    it = iter(seq)
    s0 = next(it)
    return all(i == s0 for i in it)

def mult(*arrays):
    """Array multiplication for a sequence of arrays."""
    return reduce(np.dot, arrays)

# Functions for MDPs (oriented towards a matrix representation)
def find_terminals(mat):
    return [row for ix, row in enumerate(mat) if row[ix] == 1]

def find_nonterminals(mat):
    return [row for ix, row in enumerate(mat) if row[ix] != 1]

def int2basis(x, n, dtype=np.float):
    ret = np.zeros(n, dtype=dtype)
    ret[x] = 1
    return ret 

def basis2int(vec):
    vec = np.array(vec)
    assert(vec.ndim <= 1)
    assert(np.all((vec == 0) | (vec == 1)))
    return int(np.flatnonzero(vec == 1))

def state_indices(pmat):
    assert(pmat.ndim == 2)
    assert(all_equal(pmat.shape))
    return [x for x in range(len(pmat))]

def state_vectors(pmat):
    assert(pmat.ndim == 2)
    assert(all_equal(pmat.shape))
    return [row for row in np.eye(len(pmat))]

def state_matrix(states):
    return np.eye(len(states))

def state_mapping(states):
    n = len(states)
    return {ix: s for ix, s in enumerate(states)}

def feature_matrix(states, phi):
    """Compute the feature matrix for the given states & feature function."""
    return np.array([phi(s) for s in states])

def feature_mapping(states, phi):
    return {s: phi(s) for s in states}

# TODO: dct2pmat, dct2rvec

def canonical(pmat):
    pass

def discount(mat, val=1.0):
    """-->  identity - dot(mat, diag(val))"""
    di = np.diag_indices(n, ndim=2)
    dmat = np.zeros((n,n))
    dmat[di] = val
    return np.eye(n) - np.dot(mat, dmat)

def trace_warp(pmat, gm=1.0, lm=0.0):
    """(I - P_\pi \Gamma \Lambda)"""
    n = len(np.diag(pmat))
    di = np.diag_indices(n, ndim=2)
    gmat = np.zeros((n,n))
    lmat = np.zeros((n,n))
    gmat[di] = gm
    lmat[di] = lm
    return np.eye(n) - np.dot(pmat, np.dot(gmat, lmat))

def discount_warp(pmat, gm=1.0, lm=0.0):
    """I - (I - PGL)^-1 (I - PG)"""
    n = len(np.diag(pmat))
    # (I - PGL)^-1
    a = pinv(trace_warp(pmat, gm, lm))
    # (I - PG)
    b = discount(pmat, gm)
    return np.eye(n) - np.dot(a, b)


def calc_values(states, phi_func, weights):
    X = feature_matrix(states, phi_func)
    return np.dot(X, weights)


# Parameters as functions of state
class Parameter:
    """Parameter template"""
    def __init__(self, func, terminals=list()):
        self.func = func
        self.terminals = set(map(as_tuple, terminals))

    def __call__(self, x, *args, **kwargs):
        if as_tuple(x) in self.terminals:
            return 0.0
        else:
            return self.func(x, *args, **kwargs)

class Constant:
    def __init__(self, val, terminals=list()):
        self.terminals = set(map(as_tuple, terminals))
        self.val = val

    def func(self, x):
        return self.val

    def __call__(self, x):
        if as_tuple(x) in self.terminals:
            return 0.0
        else:
            return self.func(x)