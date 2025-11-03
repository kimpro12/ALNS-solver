import numpy as np
from ..config.enums import *
from ..engine.state_update import refresh_route_loads
from ..operators.repair import best_insertion

def build_initial(data):
    # Start with no customers in any route; unrouted = all customers 1..n-1
    routes = data["routes"].copy()
    lens   = data["lens"].copy()
    loads  = data["loads"].copy()

    unrouted = np.arange(1, data["n"], dtype=np.int32)  # all customers

    inserted, used_mask, changed = best_insertion(
        unrouted,
        routes,
        lens,
        loads,
        data["node_f"],
        data.get("node_i"),
        data["veh_f"],
        data["dist"],
        edge_vec=data.get("edge_vec"),
        cost_w=data.get("cost_w"),
    )
    if inserted > 0:
        unrouted = unrouted[~used_mask]
        if changed.any():
            refresh_route_loads(routes, lens, data["node_f"], loads=loads, changed=changed)
    else:
        unrouted = unrouted.copy()

    # Pack into a solution dict
    return {
        "routes": routes,
        "lens": lens,
        "loads": loads,
        "unrouted": unrouted,
        "best_cost": np.inf,
        "total_dist": 0.0,
        "total_cost": 0.0,
    }
