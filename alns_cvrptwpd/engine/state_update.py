import numpy as np

def route_distance(dist, route, L):
    if L <= 0: return 0.0
    s = dist[0, route[0]]
    for i in range(1, L):
        s += dist[route[i-1], route[i]]
    s += dist[route[L-1], 0]
    return float(s)

def total_distance(dist, routes, lens):
    m = routes.shape[0]
    s = 0.0
    for r in range(m):
        s += route_distance(dist, routes[r], int(lens[r]))
    return float(s)
