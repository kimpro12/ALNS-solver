import numpy as np

from ..config.enums import (
    F_COST,
    F_DIST,
    F_LOAD,
    F_VEC0,
    F_TIME_F,
    DIM_MODE_ADD,
    NODE_DEMAND,
    NODE_SERVICE,
    NODE_TW_CLOSE,
    NODE_TW_OPEN,
    T_ARR,
    T_LATE,
    T_LEAVE,
    T_SLACK,
    T_START,
    T_WAIT,
    VEH_CAPACITY,
    VEH_MAX_DUR,
)
from .dimension import apply_dimension_behavior, cost_vehicle


def _ensure_vector_inputs(edge_vec, cost_w):
    if edge_vec is None or edge_vec.ndim != 3:
        return None, None, 0
    vec_dim = edge_vec.shape[2]
    if cost_w is None:
        cost_w = np.ones(vec_dim, dtype=np.float32)
    else:
        cost_w = np.asarray(cost_w, dtype=np.float32)
        if cost_w.shape[0] != vec_dim:
            raise ValueError("cost_w dimension mismatch with edge_vec")
    return edge_vec, cost_w, vec_dim


def compute_route_states(
    routes,
    lens,
    dist,
    node_f,
    veh_f,
    edge_vec=None,
    cost_w=None,
    dim_mode=DIM_MODE_ADD,
):
    """Compute per-route prefix states, cost aggregates and feasibility flags."""

    edge_vec, cost_w, vec_dim = _ensure_vector_inputs(edge_vec, cost_w)

    m, L_max = routes.shape
    core_dim = F_VEC0 + vec_dim
    core_f = np.zeros((m, L_max + 1, core_dim), dtype=np.float32)
    time_f = np.zeros((m, L_max + 1, F_TIME_F), dtype=np.float32)
    feasible = np.ones(m, dtype=np.bool_)
    route_dist = np.zeros(m, dtype=np.float32)
    route_cost = np.zeros(m, dtype=np.float32)
    route_duration = np.zeros(m, dtype=np.float32)
    route_late = np.zeros(m, dtype=np.float32)
    route_overload = np.zeros(m, dtype=np.float32)
    route_dur_excess = np.zeros(m, dtype=np.float32)

    demand = node_f[:, NODE_DEMAND]
    service = node_f[:, NODE_SERVICE]
    tw_open = node_f[:, NODE_TW_OPEN]
    tw_close = node_f[:, NODE_TW_CLOSE]
    if veh_f.ndim == 2:
        capacities = veh_f[:, VEH_CAPACITY]
        max_dur = veh_f[:, VEH_MAX_DUR]
    else:
        capacities = veh_f
        max_dur = np.full(m, np.inf, dtype=np.float32)

    for r in range(m):
        L = int(lens[r])
        prev = 0
        load_acc = 0.0
        dist_acc = 0.0
        cost_acc = 0.0
        vec_acc = np.zeros(vec_dim, dtype=np.float32) if vec_dim else None
        late_violation = False
        overload = False
        dur_violation = False
        leave_time = 0.0
        late_sum = 0.0

        for i in range(1, L + 1):
            v = int(routes[r, i - 1])
            load_acc += demand[v]
            core_f[r, i, F_LOAD] = load_acc
            if load_acc - capacities[r] > 1e-9:
                overload = True

            leg_dist = float(dist[prev, v])
            dist_acc += leg_dist
            core_f[r, i, F_DIST] = dist_acc

            if vec_dim:
                step_vec = edge_vec[prev, v]
                for f in range(vec_dim):
                    prev_val = vec_acc[f]
                    vec_acc[f] = apply_dimension_behavior(prev_val, 0.0, step_vec[f], dim_mode)
                core_f[r, i, F_VEC0 : F_VEC0 + vec_dim] = vec_acc
                cost_acc += float(cost_vehicle(step_vec, cost_w))
            else:
                cost_acc += leg_dist
            core_f[r, i, F_COST] = cost_acc

            arrival = leave_time + leg_dist
            start = arrival if arrival >= tw_open[v] else float(tw_open[v])
            wait = start - arrival if start > arrival else 0.0
            leave = start + service[v]
            late = start - tw_close[v] if start > tw_close[v] else 0.0
            slack = tw_close[v] - start if tw_close[v] > start else 0.0

            time_f[r, i, T_ARR] = arrival
            time_f[r, i, T_START] = start
            time_f[r, i, T_LEAVE] = leave
            time_f[r, i, T_WAIT] = wait
            time_f[r, i, T_LATE] = late
            time_f[r, i, T_SLACK] = slack

            if late > 1e-9:
                late_violation = True
            late_sum += max(late, 0.0)

            leave_time = leave
            prev = v

        if L > 0:
            leg_dist = float(dist[prev, 0])
            dist_acc += leg_dist
            if vec_dim:
                step_vec = edge_vec[prev, 0]
                for f in range(vec_dim):
                    vec_acc[f] = apply_dimension_behavior(vec_acc[f], 0.0, step_vec[f], dim_mode)
                cost_acc += float(cost_vehicle(step_vec, cost_w))
            else:
                cost_acc += leg_dist
            route_duration[r] = leave_time + leg_dist
            dur_excess = max(route_duration[r] - max_dur[r], 0.0)
            if dur_excess > 1e-9:
                dur_violation = True
            route_dur_excess[r] = dur_excess
        else:
            route_duration[r] = 0.0
            route_dur_excess[r] = 0.0

        route_overload[r] = max(load_acc - capacities[r], 0.0)
        route_late[r] = late_sum

        route_dist[r] = dist_acc
        route_cost[r] = cost_acc
        if overload or late_violation or dur_violation:
            feasible[r] = False

    total_dist = float(route_dist.sum())
    total_cost = float(route_cost.sum())

    return {
        "total_dist": total_dist,
        "total_cost": total_cost,
        "core_f": core_f,
        "time_f": time_f,
        "feasible": feasible,
        "route_dist": route_dist,
        "route_cost": route_cost,
        "route_duration": route_duration,
        "route_late": route_late,
        "route_overload": route_overload,
        "route_dur_excess": route_dur_excess,
    }


def refresh_route_loads(routes, lens, node_f, loads=None, changed=None):
    """Recompute cumulative route loads after structural changes.

    Parameters
    ----------
    routes : ndarray (m, L_max)
        Route matrix storing customer indices per position.
    lens : ndarray (m,)
        Current route lengths.
    node_f : ndarray (n, F_NODE_F)
        Node feature matrix that includes demands.
    loads : ndarray (m,), optional
        Buffer to update in-place. When ``None`` a new array is allocated.
    changed : ndarray (m,), optional
        Boolean/int mask of routes whose structure changed. When omitted all
        routes are recomputed.

    Returns
    -------
    ndarray
        Array of route loads corresponding to ``lens``.
    """

    demand = node_f[:, NODE_DEMAND]
    m = routes.shape[0]

    if loads is None:
        loads = np.zeros(m, dtype=np.float32)

    if changed is None:
        idxs = np.arange(m, dtype=np.int32)
    else:
        idxs = np.nonzero(changed)[0]
        if idxs.size == 0:
            return loads

    for r in idxs:
        L = int(lens[r])
        if L:
            segment = routes[r, :L]
            loads[r] = float(demand[segment].sum())
        else:
            loads[r] = 0.0

    return loads
