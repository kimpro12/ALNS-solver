import numpy as np

from ._numba_utils import collect_route_positions


def shaw_removal(routes, lens, coords, remove_k, rng, weights=None):
    """Simple Shaw-like removal: pick a random seed, remove its nearest neighbors.
    Returns (removed_customers, changed_routes_mask)."""
    if rng is None:
        rng = np.random.default_rng()

    m, Lmax = routes.shape
    capacity = routes.size
    route_indices = np.empty(capacity, dtype=np.int32)
    pos_indices = np.empty(capacity, dtype=np.int32)
    cust_indices = np.empty(capacity, dtype=np.int32)

    total_candidates = collect_route_positions(
        routes, lens, route_indices, pos_indices, cust_indices
    )
    if total_candidates == 0:
        return np.empty(0, dtype=np.int32), np.zeros(m, dtype=np.int32)

    route_indices = route_indices[:total_candidates]
    pos_indices = pos_indices[:total_candidates]
    cust_indices = cust_indices[:total_candidates]

    probs = None
    if weights is not None:
        weights_arr = np.asarray(weights, dtype=np.float32)
        if weights_arr.size > 0:
            probs = np.zeros(total_candidates, dtype=np.float32)
            max_idx = weights_arr.shape[0]
            valid_mask = cust_indices < max_idx
            probs[valid_mask] = weights_arr[cust_indices[valid_mask]]
            if probs.sum() <= 0:
                probs = None

    if probs is None:
        seed_idx = int(rng.integers(total_candidates))
    else:
        probs = probs / probs.sum()
        seed_idx = int(rng.choice(total_candidates, p=probs))

    seed_c = int(cust_indices[seed_idx])
    seed_coords = coords[seed_c]

    customer_coords = coords[cust_indices]
    diff = customer_coords - seed_coords
    dists = np.sum(diff * diff, axis=1)

    count = min(remove_k, total_candidates)
    order = np.argsort(dists)[:count]

    chosen_routes = route_indices[order]
    chosen_pos = pos_indices[order]
    chosen_cust = cust_indices[order]

    order2 = np.lexsort((-chosen_pos, chosen_routes))
    chosen_routes = chosen_routes[order2]
    chosen_pos = chosen_pos[order2]
    chosen_cust = chosen_cust[order2]

    changed = np.zeros(m, dtype=np.int32)
    removed = np.empty(order2.size, dtype=np.int32)

    for idx in range(order2.size):
        r = int(chosen_routes[idx])
        pos = int(chosen_pos[idx])
        cust = int(chosen_cust[idx])
        L = int(lens[r])
        if pos < L - 1:
            routes[r, pos : L - 1] = routes[r, pos + 1 : L]
        routes[r, L - 1] = 0
        lens[r] = L - 1
        removed[idx] = cust
        changed[r] = 1

    return removed, changed
