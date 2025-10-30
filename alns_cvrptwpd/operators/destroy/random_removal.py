import numpy as np

def random_removal(routes, lens, remove_k, rng):
    """Remove up to k random customers from the current routes.
    Returns (removed_customers, changed_routes_mask).
    """
    m, Lmax = routes.shape
    all_pos = []
    for r in range(m):
        L = int(lens[r])
        for i in range(L):
            c = int(routes[r, i])
            if c > 0:
                all_pos.append((r, i, c))
    if not all_pos:
        return np.empty(0, dtype=np.int64), np.zeros(m, dtype=np.int64)

    rng.shuffle(all_pos)
    removed = []
    changed = np.zeros(m, dtype=np.int64)

    for (r, i, c) in all_pos[:remove_k]:
        # delete position i in route r (left shift)
        L = int(lens[r])
        if i < L-1:
            routes[r, i:L-1] = routes[r, i+1:L]
        routes[r, L-1] = 0
        lens[r] = L - 1
        removed.append(c)
        changed[r] = 1
    return np.array(removed, dtype=np.int64), changed
