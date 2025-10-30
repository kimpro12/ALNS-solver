# Simple parameter defaults (extend freely)
DEFAULTS = {
    "iters": 3000,
    "log_period": 500,
    "sa_temp0": 100.0,
    "sa_temp_min": 1e-6,
    "sa_cooling": 0.995,
    "ls_budget_per_iter": 50,   # tries in light LS (first-improvement)
    "ls_heavy_period": 1000,    # not used in the tiny demo but left for extension
    "ls_heavy_budget": 200,
    "destroy_remove_k": 10,     # customers removed by destroy
    "destroy_route_pair_k": 2,  # block size for route pair destroy
    "destroy_worst_k": 5,       # number of removals for worst removal
    "repair_top_k": 3,          # candidates considered by random top-k
    "regret_k": 3,              # for regret insertion
    "adapt_rho": 0.2,
    "adapt_reward_best": 6.0,
    "adapt_reward_accept": 3.0,
    "adapt_reward_curr": 1.0,
    "adapt_reward_reject": 0.1,
}
