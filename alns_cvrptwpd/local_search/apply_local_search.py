"""Intra-route local search primitives with feasibility awareness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..operators.repair.utils import (
    extract_route,
    simulate_route,
    vehicle_capacities,
)


@dataclass
class _RouteContext:
    """Lightweight cache describing a single route."""

    index: int
    sequence: List[int]
    base_cost: float


def _build_route_context(
    routes: np.ndarray,
    lens: np.ndarray,
    r: int,
    dist: np.ndarray,
    node_f: np.ndarray,
    node_i: Optional[np.ndarray],
    veh_caps: np.ndarray,
    edge_vec: Optional[np.ndarray],
    cost_w: Optional[np.ndarray],
) -> Optional[_RouteContext]:
    seq = extract_route(routes, lens, r)
    if len(seq) <= 1:
        return None

    feasible, base_cost, _ = simulate_route(
        seq,
        dist,
        node_f,
        node_i,
        float(veh_caps[r]),
        edge_vec=edge_vec,
        cost_w=cost_w,
    )

    if not feasible:
        base_cost = float("inf")

    return _RouteContext(index=r, sequence=seq, base_cost=base_cost)


def _apply_sequence(
    routes: np.ndarray,
    lens: np.ndarray,
    ctx: _RouteContext,
    new_seq: Sequence[int],
) -> None:
    r = ctx.index
    L = len(new_seq)
    routes[r, :L] = np.asarray(new_seq, dtype=np.int32)
    if L < routes.shape[1]:
        routes[r, L:] = 0
    lens[r] = L
    ctx.sequence = list(new_seq)
    ctx.base_cost = float("nan")


def _improved(cost: float, base_cost: float, tol: float = 1e-9) -> bool:
    if np.isinf(base_cost):
        return np.isfinite(cost)
    return cost + tol < base_cost


def _relocate_moves(seq: Sequence[int]) -> Iterable[Tuple[int, int]]:
    L = len(seq)
    for i in range(L):
        for j in range(L):
            if i == j:
                continue
            yield i, j


def _swap_moves(seq: Sequence[int]) -> Iterable[Tuple[int, int]]:
    L = len(seq)
    for i in range(L - 1):
        for j in range(i + 1, L):
            yield i, j


def _two_opt_moves(seq: Sequence[int]) -> Iterable[Tuple[int, int]]:
    L = len(seq)
    for i in range(L - 2):
        for j in range(i + 2, L + 1):
            yield i, j


def _evaluate_sequence(
    seq: Sequence[int],
    dist: np.ndarray,
    node_f: np.ndarray,
    node_i: Optional[np.ndarray],
    veh_cap: float,
    edge_vec: Optional[np.ndarray],
    cost_w: Optional[np.ndarray],
) -> Tuple[bool, float]:
    feas, cost, _ = simulate_route(
        list(seq),
        dist,
        node_f,
        node_i,
        veh_cap,
        edge_vec=edge_vec,
        cost_w=cost_w,
    )
    return feas, cost


def _first_improvement(
    routes: np.ndarray,
    lens: np.ndarray,
    ctx: _RouteContext,
    dist: np.ndarray,
    node_f: np.ndarray,
    node_i: Optional[np.ndarray],
    veh_cap: float,
    edge_vec: Optional[np.ndarray],
    cost_w: Optional[np.ndarray],
    rng: np.random.Generator,
) -> bool:
    if len(ctx.sequence) <= 1:
        return False

    operations = ["relocate", "swap", "two_opt"]
    rng.shuffle(operations)

    seq = ctx.sequence
    base_cost = ctx.base_cost

    for op in operations:
        if op == "relocate":
            for i, j in _relocate_moves(seq):
                candidate = seq.copy()
                node = candidate.pop(i)
                candidate.insert(j, node)
                feas, cost = _evaluate_sequence(
                    candidate,
                    dist,
                    node_f,
                    node_i,
                    veh_cap,
                    edge_vec,
                    cost_w,
                )
                if feas and _improved(cost, base_cost):
                    _apply_sequence(routes, lens, ctx, candidate)
                    return True
        elif op == "swap":
            for i, j in _swap_moves(seq):
                candidate = seq.copy()
                candidate[i], candidate[j] = candidate[j], candidate[i]
                feas, cost = _evaluate_sequence(
                    candidate,
                    dist,
                    node_f,
                    node_i,
                    veh_cap,
                    edge_vec,
                    cost_w,
                )
                if feas and _improved(cost, base_cost):
                    _apply_sequence(routes, lens, ctx, candidate)
                    return True
        else:  # two_opt
            for i, j in _two_opt_moves(seq):
                candidate = seq[:i] + seq[i:j][::-1] + seq[j:]
                feas, cost = _evaluate_sequence(
                    candidate,
                    dist,
                    node_f,
                    node_i,
                    veh_cap,
                    edge_vec,
                    cost_w,
                )
                if feas and _improved(cost, base_cost):
                    _apply_sequence(routes, lens, ctx, candidate)
                    return True

    return False


def _best_improvement(
    routes: np.ndarray,
    lens: np.ndarray,
    ctx: _RouteContext,
    dist: np.ndarray,
    node_f: np.ndarray,
    node_i: Optional[np.ndarray],
    veh_cap: float,
    edge_vec: Optional[np.ndarray],
    cost_w: Optional[np.ndarray],
) -> bool:
    if len(ctx.sequence) <= 1:
        return False

    seq = ctx.sequence
    base_cost = ctx.base_cost
    best_cost = base_cost
    best_seq: Optional[List[int]] = None

    for i, j in _relocate_moves(seq):
        candidate = seq.copy()
        node = candidate.pop(i)
        candidate.insert(j, node)
        feas, cost = _evaluate_sequence(
            candidate,
            dist,
            node_f,
            node_i,
            veh_cap,
            edge_vec,
            cost_w,
        )
        if feas and cost < best_cost - 1e-9:
            best_cost = cost
            best_seq = candidate

    for i, j in _swap_moves(seq):
        candidate = seq.copy()
        candidate[i], candidate[j] = candidate[j], candidate[i]
        feas, cost = _evaluate_sequence(
            candidate,
            dist,
            node_f,
            node_i,
            veh_cap,
            edge_vec,
            cost_w,
        )
        if feas and cost < best_cost - 1e-9:
            best_cost = cost
            best_seq = candidate

    for i, j in _two_opt_moves(seq):
        candidate = seq[:i] + seq[i:j][::-1] + seq[j:]
        feas, cost = _evaluate_sequence(
            candidate,
            dist,
            node_f,
            node_i,
            veh_cap,
            edge_vec,
            cost_w,
        )
        if feas and cost < best_cost - 1e-9:
            best_cost = cost
            best_seq = candidate

    if best_seq is not None:
        _apply_sequence(routes, lens, ctx, best_seq)
        return True

    return False


def _select_route(
    candidate_routes: np.ndarray,
    route_weights: Optional[np.ndarray],
    rng: np.random.Generator,
) -> Optional[int]:
    if candidate_routes.size == 0:
        return None

    if route_weights is None:
        choice = int(rng.integers(candidate_routes.size))
        return int(candidate_routes[choice])

    weights = route_weights[candidate_routes]
    weights = np.asarray(weights, dtype=np.float32)
    total = float(weights.sum())
    if total <= 0:
        choice = int(rng.integers(candidate_routes.size))
        return int(candidate_routes[choice])

    probs = weights / total
    idx = int(rng.choice(candidate_routes.size, p=probs))
    return int(candidate_routes[idx])


def _route_priority(candidate_routes: np.ndarray, route_weights: Optional[np.ndarray]) -> np.ndarray:
    if candidate_routes.size == 0:
        return candidate_routes

    if route_weights is None:
        return candidate_routes

    weights = route_weights[candidate_routes]
    order = np.argsort(weights)[::-1]
    return candidate_routes[order]


def apply_local_search(
    routes: np.ndarray,
    lens: np.ndarray,
    dist: np.ndarray,
    node_f: np.ndarray,
    node_i: Optional[np.ndarray],
    veh_f: np.ndarray,
    *,
    edge_vec: Optional[np.ndarray] = None,
    cost_w: Optional[np.ndarray] = None,
    budget: int = 50,
    heavy: bool = False,
    rng: Optional[np.random.Generator] = None,
    route_weights: Optional[np.ndarray] = None,
) -> bool:
    """Apply intra-route local search moves.

    Parameters
    ----------
    routes, lens
        Current solution representation.
    dist, node_f, node_i, veh_f
        Problem data used for feasibility checks.
    edge_vec, cost_w
        Optional vectorised edge cost representation.
    budget
        Maximum number of route attempts in light search or processed routes in
        heavy search.
    heavy
        When ``True`` executes best-improvement passes, otherwise
        first-improvement with early exit.
    rng
        Random generator used for route sampling.
    route_weights
        Optional per-route focus weights (typically produced by SFR).

    Returns
    -------
    bool
        ``True`` when the procedure applied at least one improving move.
    """

    if budget <= 0:
        return False

    if rng is None:
        rng = np.random.default_rng()

    veh_caps = vehicle_capacities(veh_f)
    candidate_routes = np.nonzero(lens > 1)[0]
    if candidate_routes.size == 0:
        return False

    improved_any = False

    if heavy:
        ordered_routes = _route_priority(candidate_routes, route_weights)
        processed = 0
        for r in ordered_routes:
            if processed >= budget:
                break
            ctx = _build_route_context(
                routes,
                lens,
                int(r),
                dist,
                node_f,
                node_i,
                veh_caps,
                edge_vec,
                cost_w,
            )
            processed += 1
            if ctx is None:
                continue
            if _best_improvement(
                routes,
                lens,
                ctx,
                dist,
                node_f,
                node_i,
                float(veh_caps[int(r)]),
                edge_vec,
                cost_w,
            ):
                improved_any = True
        return improved_any

    tries = 0
    while tries < budget:
        ridx = _select_route(candidate_routes, route_weights, rng)
        if ridx is None:
            break
        ctx = _build_route_context(
            routes,
            lens,
            ridx,
            dist,
            node_f,
            node_i,
            veh_caps,
            edge_vec,
            cost_w,
        )
        tries += 1
        if ctx is None:
            continue
        if _first_improvement(
            routes,
            lens,
            ctx,
            dist,
            node_f,
            node_i,
            float(veh_caps[ridx]),
            edge_vec,
            cost_w,
            rng,
        ):
            return True

    return improved_any


