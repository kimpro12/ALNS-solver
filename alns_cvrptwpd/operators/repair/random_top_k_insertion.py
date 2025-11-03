import numpy as np

from .best_insertion import _compute_moves_for_customer
from .utils import apply_move, vehicle_capacities


def random_top_k_insertion(
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
    top_k=3,
    rng=None,
    weights=None,
    focus=1.0,
):
    """Insert customers by sampling among the top-k lowest Δcost moves."""

    if unrouted.size == 0:
        return 0, np.zeros(0, dtype=bool), np.zeros(routes.shape[0], dtype=np.int32)

    if rng is None:
        rng = np.random.default_rng()

    veh_caps = vehicle_capacities(veh_f)
    n = len(unrouted)
    used = np.zeros(n, dtype=bool)
    blocked = np.zeros(n, dtype=bool)
    changed = np.zeros(routes.shape[0], dtype=np.int32)
    inserted = 0

    while True:
        candidates = []
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
            candidates.append((move, idx, customer))
        if not candidates:
            break
        candidates.sort(key=lambda x: (x[0][0], x[0][1]))
        limit = min(top_k, len(candidates))

        if weights is not None and focus > 0:
            raw = np.array(
                [
                    1.0
                    + focus
                    * (weights[int(c[2])] if int(c[2]) < len(weights) else 0.0)
                    for c in candidates[:limit]
                ],
                dtype=np.float32,
            )
            total = raw.sum()
            if total > 0:
                probs = raw / total
                choice = int(rng.choice(limit, p=probs))
            else:
                choice = int(rng.integers(limit))
        else:
            choice = int(rng.integers(limit))

        move, idx, customer = candidates[choice]
        delta_cost, delta_dist, route_idx, pos = move
        if apply_move(routes, lens, loads, node_f, route_idx, pos, customer):
            used[idx] = True
            changed[route_idx] = 1
            inserted += 1
        else:
            blocked[idx] = True

    return inserted, used, changed
