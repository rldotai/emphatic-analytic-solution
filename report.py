import numpy as np 
import pandas as pd
from numpy.linalg import pinv

from mdptools import * 
from solvers import td_solution, etd_solution, exact_solution


def report(P, R, s0, phi_func, gm_func, lm_func, i_func):
    """Function for collating and comparing the various solutions."""
    states = state_vectors(P)
    nn = len(find_nonterminals(P))
    X = feature_matrix(states, phi_func)
    V = exact_solution(P, R, gm_func)
    
    # Best approximation (least squares)
    w_approx, *_ = np.linalg.lstsq(X, V)
    V_approx = np.dot(X, w_approx)
    E_approx = np.sum((V - V_approx)**2)/nn
    
    # TD
    w_td = td_solution(P, R, s0, phi_func, gm_func, lm_func)
    V_td = np.dot(X, w_td)
    E_td = np.sum((V - V_td)**2)/nn
    
    # Emphatic TD fixed point
    w_etd = etd_solution(P, R, s0, phi_func, gm_func, lm_func, i_func)
    V_etd = np.dot(X, w_etd)
    E_etd = np.sum((V - V_etd)**2)/nn
        
    dct = {"weights": 
            {
              "Least-Squares": w_approx, "TD": w_td, "ETD": w_etd,
            }, 
           "MSE": 
            {
              "Least-Squares": E_approx, "TD": E_td, "ETD": E_etd,
            },
           "values":
            {
                "Least-Squares": V_approx, "TD": V_td, "ETD": V_etd,
            },
           }
           
             
    
    df = pd.DataFrame(dct, 
                      index=["Least-Squares", "TD", "ETD"],
                      columns=["weights", "MSE", "values"],)
    
    # Additional Information
    print("Expected Reward:")
    print(R)
    
    print("Feature Matrix:")
    print(X)
    
    print("True Values:")
    print(V)
    
    print("Emphasis As Good or Better?:", (E_etd <= E_td))
    return df