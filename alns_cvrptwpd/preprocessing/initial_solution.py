import numpy as np
from ..config.enums import *
from ..operators.repair.greedy_insertion import greedy_insertion

def build_initial(data):
    # Start with no customers in any route; unrouted = all customers 1..n-1
    routes = data["routes"].copy()
    lens   = data["lens"].copy()
    loads  = data["loads"].copy()

    unrouted = np.arange(1, data["n"], dtype=np.int64)  # all customers

    # Greedy tail insert (place each customer at best end position)
    inserted = greedy_insertion(unrouted, routes, lens, loads,
                                data["node_f"], data["veh_f"], data["dist"],
                                tail_only=True)
    unrouted = unrouted[inserted:] if inserted < len(unrouted) else np.empty(0, dtype=np.int64)

    # Pack into a solution dict
    return {
        "routes": routes,
        "lens": lens,
        "loads": loads,
        "unrouted": unrouted,
        "best_cost": np.inf,
        "total_dist": 0.0,
    }
