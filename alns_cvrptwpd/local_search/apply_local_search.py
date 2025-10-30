import numpy as np
from .two_opt import two_opt_route

def apply_local_search(routes, lens, dist, budget=50):
    """Very small LS: try 2-opt on random routes up to `budget` attempts.
    Stops on first improvement."""
    m = routes.shape[0]
    if m == 0: return False
    tries = 0
    rng = np.random.default_rng()
    while tries < budget:
        r = int(rng.integers(0, m))
        L = int(lens[r])
        if L >= 3:
            improved, _ = two_opt_route(dist, routes[r], L)
            if improved:
                return True
        tries += 1
    return False
