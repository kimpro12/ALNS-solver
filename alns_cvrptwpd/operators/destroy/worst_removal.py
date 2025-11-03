import numpy as np
from numba import njit


@njit(cache=True)
def _edge_cost_nb(dist, edge_vec, cost_w, has_edge_vec, a, b):
    if has_edge_vec:
        acc = 0.0
        dim = edge_vec.shape[2]
        for k in range(dim):
            acc += edge_vec[a, b, k] * cost_w[k]
        return acc
    return dist[a, b]


@njit(cache=True)
def _collect_candidates(
    routes,
    lens,
    dist,
    edge_vec,
    cost_w,
    weights,
    focus,
    has_edge_vec,
    has_weights,
    scores,
    route_indices,
    pos_indices,
    cust_indices,
):
    count = 0
    m, _ = routes.shape
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
                _edge_cost_nb(dist, edge_vec, cost_w, has_edge_vec, prev, v)
                + _edge_cost_nb(dist, edge_vec, cost_w, has_edge_vec, v, nxt)
                - _edge_cost_nb(dist, edge_vec, cost_w, has_edge_vec, prev, nxt)
            )
            bias = 1.0
            if has_weights and v < weights.shape[0]:
                bias += focus * weights[v]
            scores[count] = delta * bias
            route_indices[count] = r
            pos_indices[count] = i
            cust_indices[count] = v
            count += 1
            prev = v
    return count
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

    m, _ = routes.shape
    changed = np.zeros(m, dtype=np.int32)

    if edge_vec is not None:
        if cost_w is None:
            cost_w_arr = np.ones(edge_vec.shape[2], dtype=np.float32)
        else:
            cost_w_arr = np.asarray(cost_w, dtype=np.float32)
            if cost_w_arr.shape[0] != edge_vec.shape[2]:
                raise ValueError("cost_w dimension mismatch with edge_vec")
        edge_vec_arr = np.asarray(edge_vec, dtype=np.float32)
        has_edge_vec = True
    else:
        edge_vec_arr = np.empty((1, 1, 1), dtype=np.float32)
        cost_w_arr = np.empty(0, dtype=np.float32)
        has_edge_vec = False

    if weights is not None:
        weights_arr = np.asarray(weights, dtype=np.float32)
        has_weights = weights_arr.size > 0
    else:
        weights_arr = np.empty(0, dtype=np.float32)
        has_weights = False

    capacity = routes.size
    scores = np.empty(capacity, dtype=np.float64)
    route_indices = np.empty(capacity, dtype=np.int32)
    pos_indices = np.empty(capacity, dtype=np.int32)
    cust_indices = np.empty(capacity, dtype=np.int32)
    removed = np.empty(remove_k, dtype=np.int32)
    removed_count = 0

    for _ in range(remove_k):
        candidate_count = _collect_candidates(
            routes,
            lens,
            dist,
            edge_vec_arr,
            cost_w_arr,
            weights_arr,
            float(focus),
            has_edge_vec,
            has_weights,
            scores,
            route_indices,
            pos_indices,
            cust_indices,
        )

        if candidate_count == 0:
            break

        score_view = scores[:candidate_count]
        if rng is not None:
            noise = rng.random(candidate_count) * 1e-9
            score_view = score_view + noise

        best_idx = int(np.argmax(score_view))
        r = int(route_indices[best_idx])
        idx = int(pos_indices[best_idx])
        customer = int(cust_indices[best_idx])
        L = int(lens[r])
        if L == 0:
            continue
        if idx < L - 1:
            routes[r, idx : L - 1] = routes[r, idx + 1 : L]
        routes[r, L - 1] = 0
        lens[r] = L - 1
        changed[r] = 1
        removed[removed_count] = customer
        removed_count += 1

    if removed_count > 0:
        return removed[:removed_count], changed
    return np.empty(0, dtype=np.int32), changed
