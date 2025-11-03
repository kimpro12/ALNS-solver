"""Numba-backed helpers for vector dimension updates and feasibility checks."""

from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover - fallback when numba is unavailable.
    def njit(*args, **kwargs):  # type: ignore
        def wrapper(func):
            return func

        if args and callable(args[0]):
            return args[0]
        return wrapper

from ..config.enums import DIM_MODE_ADD, DIM_MODE_FUNC, DIM_MODE_MAX


@njit
def vector_state_update_delta(vehicle_state: np.ndarray, uav: np.ndarray, bav: np.ndarray) -> None:
    """Apply a delta update between an old (``uav``) and new (``bav``) vector state.

    The helper is intended for cached route states where ``vehicle_state`` already
    stores the accumulated prefix vector. By subtracting the previous arc
    contribution (``uav``) and adding the new one (``bav``) the route metrics can be
    refreshed without recomputing the whole prefix tensor.
    """

    for f in range(bav.shape[0]):
        old_val = uav[f]
        new_val = bav[f]
        if old_val == new_val:
            continue
        vehicle_state[f] += new_val - old_val


@njit
def apply_dimension_behavior(prev_val: float, uav: float, bav: float, mode: int) -> float:
    """Apply a dimension behaviour update.

    ``mode`` follows the constants defined in :mod:`alns_cvrptwpd.config.enums`.

    The fallback (``DIM_MODE_FUNC``) behaves like a tempered additive update which
    prefers the larger of the two contributions before adding the remaining delta.
    This keeps the helper robust for custom dimension behaviours while staying
    JIT-friendly.
    """

    if mode == DIM_MODE_ADD:
        return prev_val + uav + bav
    if mode == DIM_MODE_MAX:
        return uav if uav >= bav else bav

    # ``DIM_MODE_FUNC`` (and any other custom extension) defaults to an additive
    # update with a soft maximum emphasis.
    dominant = uav if uav >= bav else bav
    residual = uav + bav - dominant
    return prev_val + dominant + residual


@njit
def vector_state_update(
    vehicle_state: np.ndarray,
    uav: np.ndarray,
    bav: np.ndarray,
    dim_mode: int,
) -> None:
    """Update a vector dimension state using the configured behaviour mode."""

    for f in range(bav.shape[0]):
        prev = vehicle_state[f]
        vehicle_state[f] = apply_dimension_behavior(prev, uav[f], bav[f], dim_mode)


@njit
def recompute_route_state(
    route_state: np.ndarray,
    route_seq: np.ndarray,
    route_len: int,
    edge_vec: np.ndarray,
    dim_mode: int,
) -> None:
    """Rebuild the prefix tensor for a route from scratch."""

    dims = route_state.shape[1]
    for f in range(dims):
        route_state[0, f] = 0.0

    prev = 0
    for i in range(route_len):
        nxt = int(route_seq[i])
        step = edge_vec[prev, nxt]
        for f in range(dims):
            prev_val = route_state[i, f]
            route_state[i + 1, f] = apply_dimension_behavior(prev_val, 0.0, step[f], dim_mode)
        prev = nxt


@njit
def vector_feasible_check_fast_masked(
    vehicle_state: np.ndarray,
    vehicle_limit: np.ndarray,
    mask: np.ndarray,
) -> bool:
    """Return ``True`` if the masked dimensions all satisfy ``state <= limit``."""

    for idx in range(mask.shape[0]):
        if not mask[idx]:
            continue
        if vehicle_state[idx] > vehicle_limit[idx] + 1e-9:
            return False
    return True


@njit
def dim_eval(dim_state: float, dim_limit: float, dim_mode: int) -> tuple[bool, float]:
    """Evaluate feasibility and residual slack for a single dimension state."""

    if dim_mode == DIM_MODE_MAX:
        feasible = dim_state <= dim_limit + 1e-9
        slack = dim_limit - dim_state
        if slack < 0.0:
            slack = 0.0
        return feasible, slack

    # ``DIM_MODE_ADD`` and ``DIM_MODE_FUNC`` share the same feasibility semantics
    # (accumulated value must not exceed the limit).
    feasible = dim_state <= dim_limit + 1e-9
    slack = dim_limit - dim_state
    if slack < 0.0:
        slack = 0.0
    return feasible, slack


@njit
def cost_vehicle(vehicle_state: np.ndarray, cost_weight: np.ndarray) -> float:
    """Compute the weighted cost contribution for a vehicle state vector."""

    total = 0.0
    for i in range(cost_weight.shape[0]):
        total += vehicle_state[i] * cost_weight[i]
    return total


@njit
def spatial_penalty(
    coords: np.ndarray, customer: int, centroid: np.ndarray, alpha_t: float
) -> float:
    """Compute the regional concentration penalty for a customer."""

    dx = coords[customer, 0] - centroid[0]
    dy = coords[customer, 1] - centroid[1]
    dist = math.sqrt(dx * dx + dy * dy)
    return dist / (alpha_t + 1e-9)


@njit
def _find_root(parent: np.ndarray, idx: int) -> int:
    while parent[idx] != idx:
        parent[idx] = parent[parent[idx]]
        idx = parent[idx]
    return idx


@njit
def build_giant_tour_savings(dist: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Construct a giant tour ordering using a Clarke–Wright style scan."""

    n = nodes.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int32)
    if n == 1:
        out = np.empty(1, dtype=np.int32)
        out[0] = nodes[0]
        return out

    size = n * (n - 1) // 2
    savings = np.empty(size, dtype=np.float32)
    pair_i = np.empty(size, dtype=np.int32)
    pair_j = np.empty(size, dtype=np.int32)

    idx = 0
    for i in range(n):
        a = nodes[i]
        for j in range(i + 1, n):
            b = nodes[j]
            savings[idx] = dist[0, a] + dist[b, 0] - dist[a, b]
            pair_i[idx] = i
            pair_j[idx] = j
            idx += 1

    order = np.argsort(savings)[::-1]
    parent = np.empty(n, dtype=np.int32)
    head = np.empty(n, dtype=np.int32)
    tail = np.empty(n, dtype=np.int32)
    next_idx = np.empty(n, dtype=np.int32)
    prev_idx = np.empty(n, dtype=np.int32)

    for i in range(n):
        parent[i] = i
        head[i] = i
        tail[i] = i
        next_idx[i] = -1
        prev_idx[i] = -1

    for ord_idx in range(order.shape[0]):
        pos = order[ord_idx]
        i = pair_i[pos]
        j = pair_j[pos]
        ri = _find_root(parent, i)
        rj = _find_root(parent, j)
        if ri == rj:
            continue
        if tail[ri] != i or head[rj] != j:
            continue
        next_idx[i] = j
        prev_idx[j] = i
        parent[rj] = ri
        head[ri] = head[ri]
        tail[ri] = tail[rj]
        head[rj] = head[ri]
        tail[rj] = tail[ri]

    out = np.empty(n, dtype=np.int32)
    used = np.zeros(n, dtype=np.uint8)
    cnt = 0
    for start in range(n):
        if prev_idx[start] != -1 or used[start] == 1:
            continue
        curr = start
        while curr != -1 and used[curr] == 0:
            out[cnt] = nodes[curr]
            used[curr] = 1
            cnt += 1
            curr = next_idx[curr]

    for leftover in range(n):
        if used[leftover] == 0:
            out[cnt] = nodes[leftover]
            cnt += 1

    return out[:cnt]


@njit
def repair_giant_tour_focus_njit(
    dist: np.ndarray,
    nodes: np.ndarray,
    coords: np.ndarray,
    centroid: np.ndarray,
    alpha_t: float,
) -> np.ndarray:
    """Blend giant-tour savings with spatial penalty ordering for repairs."""

    if nodes.shape[0] <= 1:
        return nodes.copy()

    savings_order = build_giant_tour_savings(dist, nodes)
    penalties = np.empty(savings_order.shape[0], dtype=np.float32)
    for i in range(savings_order.shape[0]):
        penalties[i] = spatial_penalty(coords, int(savings_order[i]), centroid, alpha_t)
    ranked = np.argsort(penalties)
    out = np.empty_like(savings_order)
    for i in range(ranked.shape[0]):
        out[i] = savings_order[ranked[i]]
    return out

