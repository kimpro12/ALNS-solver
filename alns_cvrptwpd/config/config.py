# Simple parameter defaults (extend freely)
DEFAULTS = {
    "iters": 50000,
    "log_period": 1000,
    "sa_temp0": 100.0,
    "sa_temp_min": 1e-6,
    "sa_cooling": 0.999,
    "ls_budget_per_iter": 50,   # light LS first-improvement attempts per iteration
    "ls_heavy_period": 750,     # perform heavy LS every N iterations
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
    "sfr_enabled": True,
    "sfr_alpha0": 30.0,
    "sfr_alpha_min": 5.0,
    "sfr_gamma": 1.2,
    "sfr_beta": 0.25,
    "sfr_destroy_focus": 0.7,
    "sfr_repair_focus": 0.6,
    "sfr_ls_focus": 0.5,
    "sfr_destroy_bias": 1.0,
    "sfr_ls_bias": 0.5,
    "penalty_weight_cap": 25.0,
    "penalty_weight_dur": 10.0,
    "penalty_weight_tw": 5.0,
    "penalty_weight_uns": 500.0,
    "penalty_target_rate": 0.05,
    "penalty_update_rate": 0.1,
    "penalty_violation_alpha": 0.05,
    "penalty_weight_min": 1e-4,
    "penalty_weight_max": 1e6,
    "penalty_history_size": 128,
    "penalty_scale_floor": 1.0,
    "penalty_tolerance": 1e-9,
}
