# Simple parameter defaults (extend freely)
DEFAULTS = {
    "iters": 3000,
    "log_period": 500,
    "sa_temp0": 100.0,
    "sa_cooling": 0.995,
    "ls_budget_per_iter": 50,   # tries in light LS (first-improvement)
    "ls_heavy_period": 1000,    # not used in the tiny demo but left for extension
    "destroy_remove_k": 10,     # customers removed by destroy
    "regret_k": 3,              # for regret insertion (stub)
}
