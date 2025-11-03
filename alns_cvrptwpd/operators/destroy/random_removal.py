import numpy as np


def random_removal(routes, lens, remove_k, rng, weights=None):
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
    total_candidates = len(all_pos)
    if total_candidates == 0:
        return np.empty(0, dtype=np.int32), np.zeros(m, dtype=np.int32)

    changed = np.zeros(m, dtype=np.int32)
    removed = []

    probs = None
    if weights is not None:
        probs = np.array(
            [weights[int(c)] if int(c) < len(weights) else 0.0 for (_, _, c) in all_pos],
            dtype=np.float32,
        )
        if probs.sum() <= 0:
            probs = None

    count = min(remove_k, total_candidates)
    if probs is None:
        idxs = rng.choice(total_candidates, size=count, replace=False)
    else:
        probs = probs / probs.sum()
        idxs = rng.choice(total_candidates, size=count, replace=False, p=probs)

    chosen = [all_pos[int(idx)] for idx in idxs]
    chosen.sort(key=lambda x: (x[0], -x[1]))

    for r, i, c in chosen:
        L = int(lens[r])
        if i < L - 1:
            routes[r, i : L - 1] = routes[r, i + 1 : L]
        routes[r, L - 1] = 0
        lens[r] = L - 1
        removed.append(c)
        changed[r] = 1

    return np.array(removed, dtype=np.int32), changed
