import numpy as np
from numba import njit
import numba

from ...config.enums import (
    NODE_DEMAND,
    NODE_PD_PAIR,
    NODE_PD_ROLE,
    NODE_SERVICE,
    NODE_TW_CLOSE,
    NODE_TW_OPEN,
    VEH_CAPACITY,
)

def edge_cost(dist, edge_vec, cost_w, a, b):
    if edge_vec is not None:
        return float(np.dot(edge_vec[a, b], cost_w))
    return float(dist[a, b])

def simulate_route(sequence, dist, node_f, node_i, veh_cap, edge_vec=None, cost_w=None):
    """Evaluate a full route sequence for feasibility and travel metrics."""

    if edge_vec is not None and cost_w is None:
        cost_w = np.ones(edge_vec.shape[2], dtype=np.float32)
    elif edge_vec is not None:
        cost_w = np.asarray(cost_w, dtype=np.float32)
        if cost_w.shape[0] != edge_vec.shape[2]:
            raise ValueError("cost_w dimension mismatch with edge_vec")

    demand = node_f[:, NODE_DEMAND]
    service = node_f[:, NODE_SERVICE]
    tw_open = node_f[:, NODE_TW_OPEN]
    tw_close = node_f[:, NODE_TW_CLOSE]
    pd_pair = node_i[:, NODE_PD_PAIR] if node_i is not None else None
    pd_role = node_i[:, NODE_PD_ROLE] if node_i is not None else None

    load = 0.0
    time = 0.0
    total_cost = 0.0
    total_dist = 0.0
    prev = 0
    mx = 2 # Default maximum cust_id
    for cust in sequence:
        mx = max(mx, cust)
    is_visited_pickups = np.zeros(mx + 1, dtype = np.bool_)

    for cust in sequence:
        cust = int(cust)
        if cust <= 0:
            continue
        load += demand[cust]
        if load > veh_cap + 1e-9 or load < -1e-9:
            return False, np.inf, np.inf

        role = 0
        pair = -1
        if pd_role is not None:
            role = pd_role[cust]
            pair = pd_pair[cust]
        if role == 2 and pair >= 0 and is_visited_pickups[pair] == False:
            return False, np.inf, np.inf

        travel = float(dist[prev, cust])
        total_dist += travel
        if edge_vec is not None:
            total_cost += float(np.dot(edge_vec[prev, cust], cost_w))
        else:
            total_cost += travel

        arrival = time + travel
        start = arrival if arrival >= tw_open[cust] else float(tw_open[cust])
        if start - tw_close[cust] > 1e-9:
            return False, np.inf, np.inf
        time = start + service[cust]
        prev = cust
        if role == 1 and pair >= 0:
            is_visited_pickups[cust] = True

    if sequence:
        travel = float(dist[prev, 0])
        total_dist += travel
        if edge_vec is not None:
            total_cost += float(np.dot(edge_vec[prev, 0], cost_w))
        else:
            total_cost += travel
        time += travel

    return True, total_cost, total_dist

def extract_route(routes, lens, r):
    L = int(lens[r])
    if L == 0:
        return []
    return [int(x) for x in routes[r, :L] if x > 0]

def vehicle_capacities(veh_f):
    veh_f = np.asarray(veh_f)
    if veh_f.ndim == 1:
        return veh_f
    return veh_f[:, VEH_CAPACITY]

def apply_move(routes, lens, loads, node_f, route_idx, pos, customer):
    L = int(lens[route_idx])
    L_max = routes.shape[1]
    if L >= L_max:
        return False
    if pos < L:
        routes[route_idx, pos + 1 : L + 1] = routes[route_idx, pos:L]
    routes[route_idx, pos] = customer
    lens[route_idx] = L + 1
    if loads is not None:
        loads[route_idx] += node_f[customer, NODE_DEMAND]
    return True
