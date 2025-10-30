import numpy as np
from .acceptance import accept_solution
from .state_update import compute_route_states, refresh_route_loads
from ..config.enums import F_LOAD
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
        return np.full_like(weights, 1.0 / len(weights), dtype=np.float64)
    return weights / total


def _update_operator_weight(weights, idx, rho, reward):
    weights[idx] = (1.0 - rho) * weights[idx] + rho * reward
    if weights[idx] <= 0.0:
        weights[idx] = 1e-6


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
    clone["feasible"] = sol.get(
        "feasible", np.ones(sol["routes"].shape[0], dtype=bool)
    ).copy()
    return clone

def _evaluate(sol, data):
    stats = compute_route_states(
        sol["routes"],
        sol["lens"],
        data["dist"],
        data["node_f"],
        data["veh_f"],
        edge_vec=data.get("edge_vec"),
        cost_w=data.get("cost_w"),
    )
    sol["total_dist"] = stats["total_dist"]
    sol["total_cost"] = stats["total_cost"]
    sol["core_f"] = stats["core_f"]
    sol["time_f"] = stats["time_f"]
    sol["feasible"] = stats["feasible"]
    sol["route_dist"] = stats["route_dist"]
    sol["route_cost"] = stats["route_cost"]

    for r in range(sol["routes"].shape[0]):
        L = int(sol["lens"][r])
        sol["loads"][r] = stats["core_f"][r, L, F_LOAD] if L > 0 else 0.0

    return stats["total_cost"]

def run_alns(solution, data, params, metrics):
    rng = np.random.default_rng()
    curr = _clone_solution(solution)
    curr_cost = _evaluate(curr, data)
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

    D_ops = ["random", "shaw", "route_pair", "worst"]
    R_ops = ["best", "random_top_k", "regret"]
    Wd = np.ones(len(D_ops), dtype=np.float64)
    Wr = np.ones(len(R_ops), dtype=np.float64)

    for it in range(1, iters + 1):
        # Sample ops
        d_idx = int(rng.choice(len(D_ops), p=_normalise_weights(Wd)))
        r_idx = int(rng.choice(len(R_ops), p=_normalise_weights(Wr)))

        cand = _clone_solution(curr)
        # Destroy
        if D_ops[d_idx] == "random":
            removed, changed = random_removal(cand["routes"], cand["lens"], k_remove, rng)
        elif D_ops[d_idx] == "shaw":
            removed, changed = shaw_removal(
                cand["routes"], cand["lens"], data["coords"], k_remove, rng
            )
        elif D_ops[d_idx] == "route_pair":
            removed, changed = route_pair_destroy(
                cand["routes"], cand["lens"], max(1, route_pair_k), rng
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

        # Light LS (2-opt first-improvement)
        if ls_heavy_period > 0 and (it % ls_heavy_period) == 0:
            apply_local_search(cand["routes"], cand["lens"], data["dist"], budget=ls_heavy_budget)
        else:
            apply_local_search(cand["routes"], cand["lens"], data["dist"], budget=ls_budget)

        new_cost = _evaluate(cand, data)
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
            )

    return best
