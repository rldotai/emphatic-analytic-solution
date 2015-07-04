import numpy as np 
from mdptools import *

def td_solution(P, R, s0, phi_func, gm_func, lm_func):
    """
    The TD-Solution to the MDP defined by `P`, `R`, and `gm_func` starting in
    state `s0`, under function approximation according to `phi_func` and using 
    eligibility traces/bootstrapping according to to `lm_func`.

    Assumes the matrix `P` and vector `R` are in canonical form.
    """
    # TODO: check parameters
    ns = len(P)
    indices = state_indices(P)
    states = state_vectors(P)
    I = np.eye(ns)
    X = feature_matrix(states, phi_func)
    G = np.diag([gm_func(s) for s in states])
    L = np.diag([lm_func(s) for s in states])
    
    # compute intermediate values
    d = distribution(P, s0)              # distribution vector
    D = np.diag(d)                       # distribution matrix
    P_trace = pinv(I - mult(P, G, L))    # trace reweighting matrix
    
    # Solve the system of equations
    b = mult(X.T, D, P_trace, R)
    A = mult(X.T, D, P_trace, (I - np.dot(P, G)), X)
    
    return np.dot(pinv(A), b).astype(np.float)


def etd_solution(P, R, s0, phi_func, gm_func, lm_func, i_func):
    ns = len(P)
    indices = state_indices(P)
    states = state_vectors(P)
    
    # Compute matrices/vectors for state-dependent parameter functions
    I = np.eye(ns)
    X = feature_matrix(states, phi_func)
    G = np.diag([gm_func(s) for s in states])
    L = np.diag([lm_func(s) for s in states])
    ivec = np.array([i_func(s) for s in states])

    # Compute intermediate values
    d = distribution(P, s0)                 # distribution vector
    D = np.diag(d)                          # distribution matrix
    d_i = np.dot(D, ivec)                   # interest-weighted distribution 
    P_trace = pinv(I - mult(P, G, L))       # trace reweighting matrix
    P_gm = I - np.dot(P, G)                 # gamma-discounted occupancy 
    P_disc = I - np.dot(P_trace, P_gm)      # trace-weighted distribution
    mvec = np.dot(pinv(I - P_disc.T), d_i)  # emphasis vector
    M = np.diag(mvec)                       # emphasis matrix

    # Solve the system of equations
    b = mult(X.T, M, P_trace, R)
    A = mult(X.T, M, P_trace, P_gm, X)
    
    return np.dot(pinv(A), b).astype(np.float) 


def exact_solution(P, R, gm_func):
    """The true returns for each state."""
    ns = len(P)
    states = state_vectors(P)
    I = np.eye(ns)
    G = np.diag([gm_func(s) for s in states])
    # astype is hacky, unsure why it's returning objects... type coercion?
    return np.dot(pinv(I - np.dot(P, G)), R).astype(np.float)