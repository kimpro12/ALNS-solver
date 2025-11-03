import numpy as np

def _delta_dist_insert(dist, route, L, pos, v):
    a = 0 if pos == 0 else int(route[pos-1])
    b = 0 if pos == L else int(route[pos])
    return dist[a, v] + dist[v, b] - dist[a, b]

def greedy_insertion(unrouted, routes, lens, loads, node_f, veh_f, dist, tail_only=False):
    """Insert unrouted customers greedily (min delta distance). Mutates routes/lens/loads.

    If tail_only=True, only insert at the end of routes (fast constructor)."""
    cap = veh_f[0, 0] if veh_f.ndim == 2 else veh_f[0]
    m, Lmax = routes.shape
    inserted = 0

    for idx in range(len(unrouted)):
        v = int(unrouted[idx])
        best = (1e30, -1, -1)  # (delta, r, pos)

        for r in range(m):
            L = int(lens[r])
            # capacity check (quick prefix approximation)
            if loads[r] + node_f[v, 0] > cap:  # NODE_DEMAND=0
                continue
            if tail_only:
                pos = L
                delta = _delta_dist_insert(dist, routes[r], L, pos, v)
                if delta < best[0]:
                    best = (delta, r, pos)
            else:
                # try all positions 0..L
                for pos in range(L+1):
                    delta = _delta_dist_insert(dist, routes[r], L, pos, v)
                    if delta < best[0]:
                        best = (delta, r, pos)

        if best[1] >= 0:
            _, r, pos = best
            L = int(lens[r])
            if L < Lmax:
                if pos < L:
                    routes[r, pos+1:L+1] = routes[r, pos:L]
                routes[r, pos] = v
                lens[r] = L + 1
                loads[r] += node_f[v, 0]
                inserted += 1
            else:
                break  # route full, stop early

    return inserted
