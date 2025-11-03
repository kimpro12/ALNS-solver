import numpy as np
from .acceptance import accept_solution
from .state_update import compute_route_states, refresh_route_loads
from .penalty import (
    init_penalty_state,
    penalty_cost,
    snapshot_penalty_state,
    update_penalty_state,
)
from .sfr import SpatialFocus
from ..config.enums import DIM_MODE_ADD, F_LOAD, F_PEN, P_CAP, P_DUR, P_TW, P_UNS
from ..operators.destroy import (
    random_removal,
    route_pair_destroy,
    shaw_removal,
    worst_removal,
)
from ..operators.repair import (
    best_insertion,
    random_top_k_insertion,
    regret_insertion,
)
from ..local_search.apply_local_search import apply_local_search


def _normalise_weights(weights):
    total = float(weights.sum())
    if total <= 0:
        return np.full_like(weights, 1.0 / len(weights), dtype=np.float32)
    return weights / total


def _update_operator_weight(weights, idx, rho, reward):
    weights[idx] = (1.0 - rho) * weights[idx] + rho * reward
    if weights[idx] <= 0.0:
        weights[idx] = 1e-6


def _blend_weights(weights, focus):
    if weights is None:
        return None
    weights = np.asarray(weights, dtype=np.float32)
    if weights.size == 0:
        return None
    focus = float(np.clip(focus, 0.0, 1.0))
    if focus >= 1.0:
        total = float(weights.sum())
        if total <= 0:
            return None
        return weights / total
    if focus <= 0.0:
        uniform = np.full(weights.shape, 1.0 / weights.size, dtype=np.float32)
        return uniform
    total = float(weights.sum())
    if total <= 0:
        base = np.full(weights.shape, 1.0 / weights.size, dtype=np.float32)
        return base
    base = weights / total
    uniform = np.full(weights.shape, 1.0 / weights.size, dtype=np.float32)
    blend = focus * base + (1.0 - focus) * uniform
    blend_sum = float(blend.sum())
    if blend_sum <= 0:
        return uniform
    return blend / blend_sum


def _initial_temperature(dist, temp0):
    if temp0 is not None:
        try:
            val = float(temp0)
        except (TypeError, ValueError):
            val = None
        else:
            if val > 0.0:
                return val
    if dist.size == 0:
        return 1.0
    tri_idx = np.triu_indices(dist.shape[0], k=1)
    candidates = dist[tri_idx]
    candidates = candidates[candidates > 0]
    if candidates.size == 0:
        return 1.0
    return float(np.median(candidates))

def _clone_solution(sol):
    clone = {
        "routes": sol["routes"].copy(),
        "lens": sol["lens"].copy(),
        "loads": sol["loads"].copy(),
        "unrouted": sol["unrouted"].copy(),
        "best_cost": sol.get("best_cost", np.inf),
        "total_dist": sol.get("total_dist", 0.0),
        "total_cost": sol.get("total_cost", 0.0),
    }
    if "core_f" in sol:
        clone["core_f"] = sol["core_f"].copy()
    if "time_f" in sol:
        clone["time_f"] = sol["time_f"].copy()
    if "route_dist" in sol:
        clone["route_dist"] = sol["route_dist"].copy()
    if "route_cost" in sol:
        clone["route_cost"] = sol["route_cost"].copy()
    if "penalty_vec" in sol:
        clone["penalty_vec"] = sol["penalty_vec"].copy()
    if "penalty_cost" in sol:
        clone["penalty_cost"] = float(sol["penalty_cost"])
    if "penalty_weights" in sol:
        clone["penalty_weights"] = sol["penalty_weights"].copy()
    if "penalty_scale" in sol:
        clone["penalty_scale"] = sol["penalty_scale"].copy()
    if "penalty_violations" in sol:
        clone["penalty_violations"] = sol["penalty_violations"].copy()
    clone["feasible"] = sol.get(
        "feasible", np.ones(sol["routes"].shape[0], dtype=bool)
    ).copy()
    return clone

def _evaluate(sol, data, penalty_state):
    stats = compute_route_states(
        sol["routes"],
        sol["lens"],
        data["dist"],
        data["node_f"],
        data["veh_f"],
        edge_vec=data.get("edge_vec"),
        cost_w=data.get("cost_w"),
        dim_mode=data.get("dim_mode", DIM_MODE_ADD),
    )
    sol["total_dist"] = stats["total_dist"]
    sol["total_cost"] = stats["total_cost"]
    sol["core_f"] = stats["core_f"]
    sol["time_f"] = stats["time_f"]
    sol["feasible"] = stats["feasible"]
    sol["route_dist"] = stats["route_dist"]
    sol["route_cost"] = stats["route_cost"]
    sol["route_duration"] = stats["route_duration"]
    sol["route_late"] = stats["route_late"]
    sol["route_overload"] = stats["route_overload"]
    sol["route_dur_excess"] = stats["route_dur_excess"]

    for r in range(sol["routes"].shape[0]):
        L = int(sol["lens"][r])
        sol["loads"][r] = stats["core_f"][r, L, F_LOAD] if L > 0 else 0.0

    penalty_vec = np.zeros(F_PEN, dtype=np.float32)
    penalty_vec[P_CAP] = float(stats["route_overload"].sum())
    penalty_vec[P_DUR] = float(stats["route_dur_excess"].sum())
    penalty_vec[P_TW] = float(stats["route_late"].sum())
    penalty_vec[P_UNS] = float(sol["unrouted"].size)

    penalty_val = penalty_cost(penalty_state, penalty_vec)
    sol["penalty_vec"] = penalty_vec
    sol["penalty_cost"] = penalty_val

    snap = snapshot_penalty_state(penalty_state)
    sol["penalty_weights"] = snap["weights"]
    sol["penalty_scale"] = snap["scale"]
    sol["penalty_violations"] = snap["violations"]

    return stats["total_cost"] + penalty_val

def run_alns(solution, data, params, metrics):
    rng = np.random.default_rng()
    curr = _clone_solution(solution)
    penalty_state = init_penalty_state(params)
    curr_cost = _evaluate(curr, data, penalty_state)
    update_penalty_state(penalty_state, curr["penalty_vec"])
    best = _clone_solution(curr)
    best["best_cost"] = curr_cost

    temp = _initial_temperature(data["dist"], params.get("sa_temp0"))
    cooling = float(params.get("sa_cooling", 0.995))
    temp_min = float(params.get("sa_temp_min", 1e-6))
    iters = int(params.get("iters", 3000))
    log_period = int(params.get("log_period", 500))
    k_remove = int(params.get("destroy_remove_k", 10))
    route_pair_k = int(params.get("destroy_route_pair_k", max(1, k_remove // 2)))
    worst_k = int(params.get("destroy_worst_k", k_remove))
    repair_top_k = int(params.get("repair_top_k", 3))
    regret_k = int(params.get("regret_k", 3))
    ls_budget = int(params.get("ls_budget_per_iter", 50))
    ls_heavy_period = int(params.get("ls_heavy_period", 0))
    ls_heavy_budget = int(params.get("ls_heavy_budget", max(ls_budget, 1)))
    rho = float(params.get("adapt_rho", 0.2))
    reward_best = float(params.get("adapt_reward_best", 6.0))
    reward_accept = float(params.get("adapt_reward_accept", 3.0))
    reward_curr = float(params.get("adapt_reward_curr", 1.0))
    reward_reject = float(params.get("adapt_reward_reject", 0.1))

    sfr = None
    sfr_destroy_focus = float(params.get("sfr_destroy_focus", 1.0))
    sfr_repair_focus = float(params.get("sfr_repair_focus", 1.0))
    sfr_ls_focus = float(params.get("sfr_ls_focus", 1.0))
    if params.get("sfr_enabled", False) and "coords" in data:
        sfr = SpatialFocus(
            data["coords"],
            alpha0=float(params.get("sfr_alpha0", 25.0)),
            alpha_min=float(params.get("sfr_alpha_min", 5.0)),
            gamma=float(params.get("sfr_gamma", 1.0)),
            beta=float(params.get("sfr_beta", 0.2)),
            destroy_bias=float(params.get("sfr_destroy_bias", 1.0)),
            ls_bias=float(params.get("sfr_ls_bias", 1.0)),
        )
        sfr.update(curr["routes"], curr["lens"], iteration=0, total_iters=iters)

    D_ops = ["random", "shaw", "route_pair", "worst"]
    R_ops = ["best", "random_top_k", "regret"]
    Wd = np.ones(len(D_ops), dtype=np.float32)
    Wr = np.ones(len(R_ops), dtype=np.float32)

    for it in range(1, iters + 1):
        # Sample ops
        d_idx = int(rng.choice(len(D_ops), p=_normalise_weights(Wd)))
        r_idx = int(rng.choice(len(R_ops), p=_normalise_weights(Wr)))

        cand = _clone_solution(curr)
        destroy_weights = _blend_weights(sfr.customer_weights, sfr_destroy_focus) if sfr else None
        route_focus = sfr.route_weights(curr["routes"], curr["lens"]) if sfr else None
        route_destroy_weights = _blend_weights(route_focus, sfr_destroy_focus)
        # Destroy
        if D_ops[d_idx] == "random":
            removed, changed = random_removal(
                cand["routes"], cand["lens"], k_remove, rng, weights=destroy_weights
            )
        elif D_ops[d_idx] == "shaw":
            removed, changed = shaw_removal(
                cand["routes"],
                cand["lens"],
                data["coords"],
                k_remove,
                rng,
                weights=destroy_weights,
            )
        elif D_ops[d_idx] == "route_pair":
            removed, changed = route_pair_destroy(
                cand["routes"],
                cand["lens"],
                max(1, route_pair_k),
                rng,
                route_weights=route_destroy_weights,
            )
        else:
            removed, changed = worst_removal(
                cand["routes"],
                cand["lens"],
                data["dist"],
                max(1, worst_k),
                rng,
                edge_vec=data.get("edge_vec"),
                cost_w=data.get("cost_w"),
                weights=destroy_weights,
                focus=sfr_destroy_focus,
            )

        if changed.any():
            refresh_route_loads(
                cand["routes"],
                cand["lens"],
                data["node_f"],
                loads=cand["loads"],
                changed=changed,
            )

        if len(removed) > 0:
            # Repair (greedy best insertion)
            unrouted = (
                np.concatenate([cand["unrouted"], removed])
                if cand["unrouted"].size
                else removed.copy()
            )
        else:
            unrouted = cand["unrouted"].copy()

        if unrouted.size > 0:
            if sfr is not None:
                unrouted = sfr.repair_order(unrouted, data["dist"])
            repair_weights = _blend_weights(sfr.customer_weights, sfr_repair_focus) if sfr else None
            if R_ops[r_idx] == "best":
                inserted, used_mask, repair_changed = best_insertion(
                    unrouted,
                    cand["routes"],
                    cand["lens"],
                    cand["loads"],
                    data["node_f"],
                    data.get("node_i"),
                    data["veh_f"],
                    data["dist"],
                    edge_vec=data.get("edge_vec"),
                    cost_w=data.get("cost_w"),
                )
            elif R_ops[r_idx] == "random_top_k":
                inserted, used_mask, repair_changed = random_top_k_insertion(
                    unrouted,
                    cand["routes"],
                    cand["lens"],
                    cand["loads"],
                    data["node_f"],
                    data.get("node_i"),
                    data["veh_f"],
                    data["dist"],
                    edge_vec=data.get("edge_vec"),
                    cost_w=data.get("cost_w"),
                    top_k=max(1, repair_top_k),
                    rng=rng,
                    weights=repair_weights,
                    focus=sfr_repair_focus,
                )
            else:
                inserted, used_mask, repair_changed = regret_insertion(
                    unrouted,
                    cand["routes"],
                    cand["lens"],
                    cand["loads"],
                    data["node_f"],
                    data.get("node_i"),
                    data["veh_f"],
                    data["dist"],
                    edge_vec=data.get("edge_vec"),
                    cost_w=data.get("cost_w"),
                    k=max(2, regret_k),
                    rng=rng,
                )

            if inserted > 0:
                cand["unrouted"] = unrouted[~used_mask]
                if repair_changed.any():
                    refresh_route_loads(
                        cand["routes"],
                        cand["lens"],
                        data["node_f"],
                        loads=cand["loads"],
                        changed=repair_changed,
                    )
            else:
                cand["unrouted"] = unrouted
        else:
            cand["unrouted"] = unrouted

        ls_route_weights = None
        if sfr is not None:
            ls_raw = sfr.route_weights(cand["routes"], cand["lens"])
            if ls_raw is not None:
                ls_scaled = np.array([sfr.ls_route_weight(v) for v in ls_raw], dtype=np.float32)
                total_ls = ls_scaled.sum()
                if total_ls > 0:
                    ls_scaled /= total_ls
                    ls_route_weights = _blend_weights(ls_scaled, sfr_ls_focus)

        if ls_heavy_period > 0 and (it % ls_heavy_period) == 0:
            apply_local_search(
                cand["routes"],
                cand["lens"],
                data["dist"],
                data["node_f"],
                data.get("node_i"),
                data["veh_f"],
                edge_vec=data.get("edge_vec"),
                cost_w=data.get("cost_w"),
                budget=ls_heavy_budget,
                heavy=True,
                rng=rng,
                route_weights=ls_route_weights,
            )
        else:
            apply_local_search(
                cand["routes"],
                cand["lens"],
                data["dist"],
                data["node_f"],
                data.get("node_i"),
                data["veh_f"],
                edge_vec=data.get("edge_vec"),
                cost_w=data.get("cost_w"),
                budget=ls_budget,
                heavy=False,
                rng=rng,
                route_weights=ls_route_weights,
            )

        new_cost = _evaluate(cand, data, penalty_state)
        prev_cost = curr_cost
        accepted = accept_solution(prev_cost, new_cost, temp, rng)
        status = "REJECT"

        if accepted:
            curr, curr_cost = cand, new_cost
            if new_cost < best["best_cost"]:
                best = _clone_solution(curr)
                best["best_cost"] = new_cost
                status = "BEST"
            elif new_cost < prev_cost:
                status = "IMPROVE"
            else:
                status = "ACCEPT"
        else:
            status = "REJECT"

        update_penalty_state(penalty_state, curr["penalty_vec"])

        if status == "BEST":
            reward = reward_best
        elif status == "IMPROVE":
            reward = reward_accept
        elif status == "ACCEPT":
            reward = reward_curr
        else:
            reward = reward_reject

        _update_operator_weight(Wd, d_idx, rho, reward)
        _update_operator_weight(Wr, r_idx, rho, reward)

        # cooling
        temp = max(temp_min, temp * cooling)

        # logging
        if (it % log_period) == 0 or it == 1:
            metrics.append(
                it,
                curr_cost,
                best["best_cost"],
                temp,
                d_op=D_ops[d_idx],
                r_op=R_ops[r_idx],
                status=status,
                penalty_cost=curr.get("penalty_cost", 0.0),
                penalty_vec=curr.get("penalty_vec"),
                penalty_weights=penalty_state["weights"],
                penalty_violations=penalty_state["viol_ema"],
            )

        if sfr is not None:
            sfr.update(curr["routes"], curr["lens"], it, iters)

    return best
