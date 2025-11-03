import numpy as np


def _edge_cost(dist, edge_vec, cost_w, a, b):
    if edge_vec is not None:
        return float(np.dot(edge_vec[a, b], cost_w))
    return float(dist[a, b])


def worst_removal(
    routes,
    lens,
    dist,
    remove_k,
    rng,
    edge_vec=None,
    cost_w=None,
    weights=None,
    focus=1.0,
):
    """Remove customers that contribute the largest marginal travel cost.

    Parameters
    ----------
    routes : ndarray
        Route matrix of shape ``(m, L_max)``.
    lens : ndarray
        Current route lengths per vehicle.
    dist : ndarray
        Symmetric distance matrix.
    remove_k : int
        Number of customers to remove.
    rng : numpy.random.Generator
        Random generator used for tie-breaking.
    edge_vec : ndarray, optional
        Optional vectorised edge costs ``(n, n, d)``.
    cost_w : ndarray, optional
        Optional weighting vector that matches ``edge_vec``'s last dimension.

    Returns
    -------
    tuple
        ``(removed_customers, changed_routes_mask)``
    """

    if remove_k <= 0:
        return np.empty(0, dtype=np.int32), np.zeros(routes.shape[0], dtype=np.int32)

    m, L_max = routes.shape
    changed = np.zeros(m, dtype=np.int32)
    removed = []

    # Pre-normalise weight vector if present
    if edge_vec is not None and cost_w is None:
        cost_w = np.ones(edge_vec.shape[2], dtype=np.float32)
    elif edge_vec is None:
        cost_w = None
    else:
        cost_w = np.asarray(cost_w, dtype=np.float32)
        if cost_w.shape[0] != edge_vec.shape[2]:
            raise ValueError("cost_w dimension mismatch with edge_vec")

    for _ in range(remove_k):
        best = None
        best_entry = None

        for r in range(m):
            L = int(lens[r])
            if L == 0:
                continue
            prev = 0
            for i in range(L):
                v = int(routes[r, i])
                if v <= 0:
                    continue
                nxt = int(routes[r, i + 1]) if i < L - 1 else 0
                delta = (
                    _edge_cost(dist, edge_vec, cost_w, prev, v)
                    + _edge_cost(dist, edge_vec, cost_w, v, nxt)
                    - _edge_cost(dist, edge_vec, cost_w, prev, nxt)
                )
                if weights is not None:
                    bias = 1.0 + focus * weights[int(v)] if int(v) < len(weights) else 1.0
                else:
                    bias = 1.0
                # store with small random noise for stochastic tie-breaks
                noise = rng.random() * 1e-9 if rng is not None else 0.0
                score = (delta * bias) + noise
                if best is None or score > best:
                    best = score
                    best_entry = (r, i, v)
                prev = v

        if best_entry is None:
            break

        r, idx, customer = best_entry
        L = int(lens[r])
        if idx < L - 1:
            routes[r, idx:L - 1] = routes[r, idx + 1 : L]
        routes[r, L - 1] = 0
        lens[r] = L - 1
        changed[r] = 1
        removed.append(customer)

    if removed:
        return np.array(removed, dtype=np.int32), changed
    return np.empty(0, dtype=np.int32), changed
