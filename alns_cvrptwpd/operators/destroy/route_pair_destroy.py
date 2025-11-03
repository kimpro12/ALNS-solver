import numpy as np


def route_pair_destroy(routes, lens, k_block, rng, route_weights=None):
    """Pick two longest routes and remove a small contiguous block from each.

    Returns (removed_customers, changed_routes_mask)."""
    m, Lmax = routes.shape
    if route_weights is not None:
        weights = route_weights.copy()
        order = np.argsort(weights)[::-1]
    else:
        order = np.argsort(lens)[::-1]
    removed = []
    changed = np.zeros(m, dtype=np.int32)

    for ridx in order[:2]:
        L = int(lens[ridx])
        if L == 0:
            continue
        block = max(1, min(k_block, L//2))
        start = rng.integers(0, L - block + 1)
        removed.extend(list(routes[ridx, start:start+block]))
        # shift left
        if start + block < L:
            routes[ridx, start:L-block] = routes[ridx, start+block:L]
        # zero tail
        routes[ridx, L-block:L] = 0
        lens[ridx] = L - block
        changed[ridx] = 1

    if removed:
        removed = np.array([c for c in removed if c > 0], dtype=np.int32)
    else:
        removed = np.empty(0, dtype=np.int32)
    return removed, changed
