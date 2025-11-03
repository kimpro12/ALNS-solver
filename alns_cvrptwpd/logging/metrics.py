import csv
import json
import numpy as np

from ..config.enums import F_PEN, P_CAP, P_DUR, P_TW, P_UNS


class Metrics:
    def __init__(self):
        self.rows = []

    def append(
        self,
        it,
        curr,
        best,
        temp,
        d_op="-",
        r_op="-",
        status="",
        penalty_cost=0.0,
        penalty_vec=None,
        penalty_weights=None,
        penalty_violations=None,
    ):
        if penalty_vec is None:
            penalty_vec = np.zeros(F_PEN, dtype=np.float32)
        else:
            penalty_vec = np.asarray(penalty_vec, dtype=np.float32)
        if penalty_weights is None:
            penalty_weights = np.zeros(F_PEN, dtype=np.float32)
        else:
            penalty_weights = np.asarray(penalty_weights, dtype=np.float32)
        if penalty_violations is None:
            penalty_violations = np.zeros(F_PEN, dtype=np.float32)
        else:
            penalty_violations = np.asarray(penalty_violations, dtype=np.float32)

        self.rows.append(
            (
                it,
                float(curr),
                float(best),
                float(temp),
                d_op,
                r_op,
                status,
                float(penalty_cost),
                penalty_vec.copy(),
                penalty_weights.copy(),
                penalty_violations.copy(),
            )
        )

    def save_csv(self, path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "iter",
                    "curr_cost",
                    "best_cost",
                    "temp",
                    "destroy_op",
                    "repair_op",
                    "status",
                    "penalty_cost",
                    "pen_cap",
                    "pen_dur",
                    "pen_tw",
                    "pen_uns",
                    "w_cap",
                    "w_dur",
                    "w_tw",
                    "w_uns",
                    "v_cap",
                    "v_dur",
                    "v_tw",
                    "v_uns",
                ]
            )
            for row in self.rows:
                it, curr, best, temp, d_op, r_op, status, pen_cost, pen_vec, pen_w, pen_v = row
                w.writerow(
                    [
                        it,
                        curr,
                        best,
                        temp,
                        d_op,
                        r_op,
                        status,
                        pen_cost,
                        float(pen_vec[P_CAP]),
                        float(pen_vec[P_DUR]),
                        float(pen_vec[P_TW]),
                        float(pen_vec[P_UNS]),
                        float(pen_w[P_CAP]),
                        float(pen_w[P_DUR]),
                        float(pen_w[P_TW]),
                        float(pen_w[P_UNS]),
                        float(pen_v[P_CAP]),
                        float(pen_v[P_DUR]),
                        float(pen_v[P_TW]),
                        float(pen_v[P_UNS]),
                    ]
                )


def save_metrics_json(path, metrics, best, params, *, extra=None):
    data = {
        "final_best_cost": float(best["best_cost"]),
        "final_total_dist": float(best["total_dist"]),
        "iters_logged": len(metrics.rows),
        "params": params,
    }
    if extra:
        data.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_routes_csv(path, routes, lens):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["route_id", "pos", "customer_id"])
        for r in range(routes.shape[0]):
            L = int(lens[r])
            for i in range(L):
                w.writerow([r, i + 1, int(routes[r, i])])
