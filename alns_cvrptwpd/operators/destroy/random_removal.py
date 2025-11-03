import numpy as np

from ._numba_utils import collect_route_positions


def random_removal(routes, lens, remove_k, rng, weights=None):
    """Remove up to k random customers from the current routes.
    Returns (removed_customers, changed_routes_mask).
    """
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

    cust_indices = cust_indices[:total_candidates]
    route_indices = route_indices[:total_candidates]
    pos_indices = pos_indices[:total_candidates]

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

    count = min(remove_k, total_candidates)
    if probs is None:
        idxs = rng.choice(total_candidates, size=count, replace=False)
    else:
        probs = probs / probs.sum()
        idxs = rng.choice(total_candidates, size=count, replace=False, p=probs)

    idxs = np.asarray(idxs, dtype=np.int32)
    chosen_routes = route_indices[idxs]
    chosen_pos = pos_indices[idxs]
    chosen_cust = cust_indices[idxs]

    order = np.lexsort((-chosen_pos, chosen_routes))
    chosen_routes = chosen_routes[order]
    chosen_pos = chosen_pos[order]
    chosen_cust = chosen_cust[order]

    changed = np.zeros(m, dtype=np.int32)
    removed = np.empty(count, dtype=np.int32)

    for idx in range(count):
        r = int(chosen_routes[idx])
        i = int(chosen_pos[idx])
        c = int(chosen_cust[idx])
        L = int(lens[r])
        if i < L - 1:
            routes[r, i : L - 1] = routes[r, i + 1 : L]
        routes[r, L - 1] = 0
        lens[r] = L - 1
        removed[idx] = c
        changed[r] = 1

    return removed, changed
