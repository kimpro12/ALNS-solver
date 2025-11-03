import numpy as np
import numba
from numba import njit

from .utils import apply_move, extract_route, simulate_route, vehicle_capacities

def _compute_moves_for_customer(
    customer,
    routes,
    lens,
    node_f,
    node_i,
    veh_caps,
    dist,
    edge_vec=None,
    cost_w=None,
):
    moves = []
    m, L_max = routes.shape
    for r in range(m):
        L = int(lens[r])
        if L >= L_max:
            continue
        base_seq = extract_route(routes, lens, r)
        feasible, base_cost, base_dist = simulate_route(
            base_seq, dist, node_f, node_i, veh_caps[r], edge_vec=edge_vec, cost_w=cost_w
        )
        if not feasible:
            continue
        for pos in range(L + 1):
            new_seq = base_seq.copy()
            new_seq.insert(pos, customer)
            feas, new_cost, new_dist = simulate_route(
                new_seq, dist, node_f, node_i, veh_caps[r], edge_vec=edge_vec, cost_w=cost_w
            )
            if not feas:
                continue
            delta_cost = new_cost - base_cost
            delta_dist = new_dist - base_dist
            moves.append((delta_cost, delta_dist, r, pos))
    moves.sort(key=lambda x: (x[0], x[1]))
    return moves

def best_insertion(
    unrouted,
    routes,
    lens,
    loads,
    node_f,
    node_i,
    veh_f,
    dist,
    edge_vec=None,
    cost_w=None,
    rng=None,
):
    """Classical best insertion: always choose the move with the lowest Δcost."""

    if unrouted.size == 0:
        return 0, np.zeros(0, dtype=bool), np.zeros(routes.shape[0], dtype=np.int32)

    veh_caps = vehicle_capacities(veh_f)
    n = len(unrouted)
    used = np.zeros(n, dtype=bool)
    blocked = np.zeros(n, dtype=bool)
    changed = np.zeros(routes.shape[0], dtype=np.int32)
    inserted = 0

    while True:
        best = None
        best_idx = -1
        for idx in range(n):
            if used[idx] or blocked[idx]:
                continue
            customer = int(unrouted[idx])
            moves = _compute_moves_for_customer(
                customer, routes, lens, node_f, node_i, veh_caps, dist, edge_vec=edge_vec, cost_w=cost_w
            )
            if not moves:
                blocked[idx] = True
                continue
            move = moves[0]
            if best is None or move[0] < best[0] - 1e-9 or (
                abs(move[0] - best[0]) <= 1e-9 and move[1] < best[1]
            ):
                best = move + (customer,)
                best_idx = idx
        if best is None:
            break
        delta_cost, delta_dist, route_idx, pos, customer = best
        if apply_move(routes, lens, loads, node_f, route_idx, pos, customer):
            used[best_idx] = True
            changed[route_idx] = 1
            inserted += 1
        else:
            blocked[best_idx] = True

    return inserted, used, changed
