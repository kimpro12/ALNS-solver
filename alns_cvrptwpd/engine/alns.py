import numpy as np
from .acceptance import accept_solution
from .state_update import total_distance
from ..operators.destroy.random_removal import random_removal
from ..operators.destroy.shaw_removal import shaw_removal
from ..operators.destroy.route_pair_destroy import route_pair_destroy
from ..operators.repair.greedy_insertion import greedy_insertion
from ..local_search.apply_local_search import apply_local_search

def _clone_solution(sol):
    return {
        "routes": sol["routes"].copy(),
        "lens": sol["lens"].copy(),
        "loads": sol["loads"].copy(),
        "unrouted": sol["unrouted"].copy(),
        "best_cost": sol.get("best_cost", np.inf),
        "total_dist": sol.get("total_dist", 0.0),
    }

def _evaluate(sol, data):
    dist = total_distance(data["dist"], sol["routes"], sol["lens"])
    sol["total_dist"] = dist
    return dist

def run_alns(solution, data, params, metrics):
    rng = np.random.default_rng()
    curr = _clone_solution(solution)
    curr_cost = _evaluate(curr, data)
    best = _clone_solution(curr)
    best["best_cost"] = curr_cost

    temp = float(params.get("sa_temp0", 100.0))
    cooling = float(params.get("sa_cooling", 0.995))
    iters = int(params.get("iters", 3000))
    log_period = int(params.get("log_period", 500))
    k_remove = int(params.get("destroy_remove_k", 10))
    ls_budget = int(params.get("ls_budget_per_iter", 50))

    D_ops = ["random", "shaw", "route_pair"]
    R_ops = ["greedy"]  # extend later
    Wd = np.ones(len(D_ops), dtype=np.float64)
    Wr = np.ones(len(R_ops), dtype=np.float64)

    for it in range(1, iters+1):
        # Sample ops
        d_idx = int(rng.choice(len(D_ops), p=Wd / Wd.sum()))
        r_idx = int(rng.choice(len(R_ops), p=Wr / Wr.sum()))

        cand = _clone_solution(curr)
        # Destroy
        if D_ops[d_idx] == "random":
            removed, changed = random_removal(cand["routes"], cand["lens"], k_remove, rng)
        elif D_ops[d_idx] == "shaw":
            removed, changed = shaw_removal(cand["routes"], cand["lens"], data["coords"], k_remove, rng)
        else:
            removed, changed = route_pair_destroy(cand["routes"], cand["lens"], max(1, k_remove//2), rng)

        if len(removed) > 0:
            # Repair (greedy best insertion)
            unrouted = np.concatenate([cand["unrouted"], removed]) if cand["unrouted"].size else removed.copy()
            inserted = greedy_insertion(unrouted, cand["routes"], cand["lens"], cand["loads"],
                                        data["node_f"], data["veh_f"], data["dist"], tail_only=False)
            # any left remain unrouted
            cand["unrouted"] = unrouted[inserted:] if inserted < len(unrouted) else np.empty(0, dtype=np.int64)

        # Light LS (2-opt first-improvement)
        apply_local_search(cand["routes"], cand["lens"], data["dist"], budget=ls_budget)

        new_cost = _evaluate(cand, data)
        accepted = accept_solution(curr_cost, new_cost, temp, rng)

        if accepted:
            curr, curr_cost = cand, new_cost
            if new_cost < best["best_cost"]:
                best = _clone_solution(curr)
                best["best_cost"] = new_cost

        # cooling
        temp *= cooling

        # logging
        if (it % log_period) == 0 or it == 1:
            metrics.append(it, curr_cost, best["best_cost"], temp, d_op=D_ops[d_idx], r_op=R_ops[r_idx])

    return best
