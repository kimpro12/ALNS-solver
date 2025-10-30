import numpy as np

def accept_solution(curr_cost, new_cost, temp, rng):
    delta = new_cost - curr_cost
    if delta <= 0.0:
        return True
    if temp <= 1e-12:
        return False
    p = np.exp(-delta / temp)
    return rng.random() < p
