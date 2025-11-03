import numpy as np
from ..config.enums import *

def _euclid(a, b):
    dx = a[:,None,0] - b[None,:,0]
    dy = a[:,None,1] - b[None,:,1]
    return np.sqrt(dx*dx + dy*dy, dtype=np.float32)

def generate_data(n_customers=60, n_vehicles=6, L_max=40, seed=0):
    np.random.seed(seed)
    # Nodes: 0 is depot, 1..n_customers are customers
    n = n_customers + 1
    m = n_vehicles

    # coords
    coords = np.zeros((n, 2), dtype=np.float32)
    coords[0] = np.array([50.0, 50.0])  # depot at center
    coords[1:] = np.random.uniform(0, 100, size=(n_customers, 2)).astype(np.float32)

    dist = _euclid(coords, coords).astype(np.float32, copy=False)
    ttime = dist.copy()  # unit speed; TW scaffolding only

    # vectorized cost chunk (dim=1 for distance; extend as needed)
    edge_vec = dist[..., None].copy()
    cost_w = np.ones(edge_vec.shape[2], dtype=np.float32)

    # node features
    node_f = np.zeros((n, F_NODE_F), dtype=np.float32)
    node_i = np.zeros((n, F_NODE_I), dtype=np.int32)

    # random demand 1..10
    node_f[1:, NODE_DEMAND] = np.random.randint(1, 11, size=n_customers).astype(np.float32)
    node_f[0, NODE_DEMAND] = 0.0

    # service times small (2..5)
    node_f[1:, NODE_SERVICE] = np.random.randint(2, 6, size=n_customers).astype(np.float32)

    # simple TW: wide windows (open=0, close=1e9) so they don't bind in the demo
    node_f[:, NODE_TW_OPEN] = 0.0
    node_f[:, NODE_TW_CLOSE] = 1e9

    # PD pairs: minimal demo (no PD). Extend as needed.
    node_i[:, NODE_PD_PAIR] = -1
    node_i[:, NODE_PD_ROLE] = 0
    node_i[:, NODE_REQUIRED] = 1

    # vehicles
    veh_f = np.zeros((m, F_VEH_F), dtype=np.float32)
    veh_i = np.zeros((m, 2), dtype=np.int32)
    veh_i[:, 0] = 0  # start depot
    veh_i[:, 1] = 0  # end depot
    # capacity roughly scales with total demand / m * 1.4
    avg_load = node_f[1:, NODE_DEMAND].sum() / m
    veh_f[:, VEH_CAPACITY] = max(15.0, 1.4 * avg_load)
    veh_f[:, VEH_FIXED_COST] = 0.0
    veh_f[:, VEH_MAX_DUR] = 1e12

    # routes & helpers
    routes = np.zeros((m, L_max), dtype=np.int32)
    lens   = np.zeros(m, dtype=np.int32)
    loads  = np.zeros(m, dtype=np.float32)

    return {
        "n": n,
        "m": m,
        "L_max": L_max,
        "coords": coords,
        "dist": dist,
        "ttime": ttime,
        "edge_vec": edge_vec,
        "cost_w": cost_w,
        "node_f": node_f,
        "node_i": node_i,
        "veh_f": veh_f,
        "veh_i": veh_i,
        "routes": routes,
        "lens": lens,
        "loads": loads,
    }
