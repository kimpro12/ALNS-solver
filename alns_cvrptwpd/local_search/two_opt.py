import numpy as np

def _route_cost(dist, route, L):
    if L <= 0: return 0.0
    prev = 0
    s = 0.0
    for i in range(L):
        v = int(route[i])
        s += dist[prev, v]
        prev = v
    s += dist[prev, 0]
    return s

def two_opt_route(dist, route, L):
    """First-improvement 2-opt on a single route. Returns (improved, new_L)."""
    if L < 3: return False, L
    best_gain = 0.0
    best_i = best_j = -1

    # Precompute cumulative? For simplicity, use O(L^2) checks (L small).
    for i in range(L-1):
        a = 0 if i == 0 else int(route[i-1])
        i_node = int(route[i])
        for j in range(i+1, L):
            j_node = int(route[j])
            b = 0 if j == L-1 else int(route[j+1])
            # Old edges: (a -> i), (j -> b)
            # New edges: (a -> j), (i -> b)
            old = dist[a, i_node] + dist[j_node, b]
            new = dist[a, j_node] + dist[i_node, b]
            gain = old - new
            if gain > 1e-9:
                best_gain = gain
                best_i, best_j = i, j
                # first-improvement: apply immediately
                route[best_i:best_j+1] = route[best_i:best_j+1][::-1]
                return True, L
    return False, L
