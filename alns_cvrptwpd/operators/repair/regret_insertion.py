import numpy as np

from .best_insertion import _compute_moves_for_customer
from .utils import apply_move, vehicle_capacities


def regret_insertion(
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
    k=2,
    rng=None,
):
    """Regret-k insertion (default k=2).

    Customers with the largest regret (Δ₂ - Δ₁) are inserted first."""

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
        best_move = None
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
            best_cost = moves[0][0]
            second_cost = moves[min(k - 1, len(moves) - 1)][0] if len(moves) >= k else moves[-1][0]
            regret = second_cost - best_cost
            if best is None or regret > best[0] + 1e-9 or (
                abs(regret - best[0]) <= 1e-9 and best_cost < best[1]
            ):
                best = (regret, best_cost)
                best_move = moves[0] + (customer,)
                best_idx = idx
        if best_move is None:
            break
        delta_cost, delta_dist, route_idx, pos, customer = best_move
        if apply_move(routes, lens, loads, node_f, route_idx, pos, customer):
            used[best_idx] = True
            changed[route_idx] = 1
            inserted += 1
        else:
            blocked[best_idx] = True

    return inserted, used, changed
