import numpy as np 
import pandas as pd
from collections import defaultdict
from numpy.linalg import pinv

from mdptools import * 
from solvers import td_solution, etd_solution, exact_solution


def simulate(pmat, s0, num_episodes=10):
    """Simulate a number of episodes"""
    assert(valid_pmat(pmat))
    ns = len(pmat)
    states = state_vectors(pmat)
    indices = state_indices(pmat)
    terminals = set(map(as_tuple, find_terminals(pmat)))
    ret = []
    for i in range(num_episodes):
        ix = basis2int(s0)
        s = states[ix]
        ret.append(ix)
        while as_tuple(s) not  in terminals:
            prob = np.dot(s, pmat)
            ix = np.random.choice(indices, p=prob)
            s = states[ix]
            ret.append(ix)
    return ret 
    

def make_episodes(num_episodes, env, policy):
    return [make_episode(env, policy) for i in range(num_episodes)]
    

def make_episode(env, policy):
    env.reset()
    ret = []
    while not env.is_terminal():
        # Observe, take action, get next observation, and compute reward
        s  = env.observe()
        a  = policy(s)
        r  = env.do(a)
        sp = env.observe()

        # Append step to episode trajectory
        ret.append((s, a, r, sp))
    return ret

def apply_fa(episodes, phi_func):
    """Apply function approximation to a series of episodes."""
    ret = []
    for episode in episodes:
        tmp = []
        for step in episode[:-1]:
            s, a, r, sp = step
            fvec   = phi_func(s)
            fvec_p = phi_func(sp)
            tmp.append((fvec, a, r, fvec_p))
        
        # Account for final step of the episode
        s, a, r, sp = episode[-1]
        fvec   = phi_func(s)
        fvec_p = np.zeros(phi_func.length, dtype=np.float)
        tmp.append((fvec, a, r, fvec_p))
        ret.append(tmp)
    return ret

def apply_rfunc(episodes, rfunc):
    ret = []
    for episode in episodes:
        tmp = []
        for step in episode:
            s, a, r, sp = step
            new_r = rfunc(s, a, sp)
            tmp.append((s, a, new_r, sp))
        
        # Add episode to list of episodes
        ret.append(tmp)
    return ret


def expected_return(episodes, gamma=1.0):
    """Calculate the expected return for each state in a series of episodes."""
    value  = defaultdict(list)
    visits = defaultdict(int)
    for episode in episodes:
        ret = 0
        for step in reversed(episode):
            s, a, r, sp = step
            s = as_tuple(s)
            ret = ret*gamma + r
            visits[s] += 1
            value[s].append(ret)
    return {s: sum(value[s])/visits[s] for s in value.keys()}


def learn(algo, episodes, fixed_params=dict(), param_funcs=dict(), repeats=1):
    """
    Have `agent` perform learning on `episodes`, with parameters that may 
    be fixed or *state dependent*. Optionally repeat over the episodes.
    """
    for i in range(repeats):
        for episode in episodes:
            algo.reset()
            for step in episode:
                s, a, r, sp = step 

                # Parameters for update
                params = {"s": s, "a": a, "r": r, "sp": sp}
                params.update(**fixed_params)
                for name, func in param_funcs.items():
                    params[name] = func(s)

                # Update
                algo.update(**params)
    return algo 
