import numpy as np

def shaw_removal(routes, lens, coords, remove_k, rng):
    """Simple Shaw-like removal: pick a random seed, remove its nearest neighbors.
    Returns (removed_customers, changed_routes_mask)."""
    m, Lmax = routes.shape
    # Collect all (r,i,c)
    all_pos = []
    for r in range(m):
        L = int(lens[r])
        for i in range(L):
            c = int(routes[r, i])
            if c > 0:
                all_pos.append((r, i, c))
    if not all_pos:
        return np.empty(0, dtype=np.int64), np.zeros(m, dtype=np.int64)

    seed_r, seed_i, seed_c = all_pos[rng.integers(len(all_pos))]
    # distances to seed
    dists = []
    sx, sy = coords[seed_c]
    for (r, i, c) in all_pos:
        dx = coords[c,0] - sx
        dy = coords[c,1] - sy
        dists.append((dx*dx + dy*dy, r, i, c))
    dists.sort(key=lambda x: x[0])

    removed = []
    changed = np.zeros(routes.shape[0], dtype=np.int64)
    for _, r, i, c in dists[:remove_k]:
        if c <= 0: continue
        L = int(lens[r])
        # delete (r,i)
        if i < L-1:
            routes[r, i:L-1] = routes[r, i+1:L]
        routes[r, L-1] = 0
        lens[r] = L - 1
        removed.append(c)
        changed[r] = 1
    return np.array(removed, dtype=np.int64), changed
