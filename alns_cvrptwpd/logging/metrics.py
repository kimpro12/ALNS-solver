import csv, json

class Metrics:
    def __init__(self):
        self.rows = []  # (iter, curr, best, temp, d_op, r_op)

    def append(self, it, curr, best, temp, d_op="-", r_op="-"):
        self.rows.append((it, float(curr), float(best), float(temp), d_op, r_op))

    def save_csv(self, path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["iter", "curr_cost", "best_cost", "temp", "destroy_op", "repair_op"])
            for row in self.rows:
                w.writerow(row)

def save_metrics_json(path, metrics, best, params):
    data = {
        "final_best_cost": float(best["best_cost"]),
        "final_total_dist": float(best["total_dist"]),
        "iters_logged": len(metrics.rows),
        "params": params,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def save_routes_csv(path, routes, lens):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["route_id", "pos", "customer_id"])
        for r in range(routes.shape[0]):
            L = int(lens[r])
            for i in range(L):
                w.writerow([r, i+1, int(routes[r, i])])
