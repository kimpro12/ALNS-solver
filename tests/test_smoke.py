import numpy as np
from alns_cvrptwpd.data.generate_data import generate_data
from alns_cvrptwpd.preprocessing.initial_solution import build_initial
from alns_cvrptwpd.engine.alns import run_alns
from alns_cvrptwpd.logging.metrics import Metrics
from alns_cvrptwpd.config.config import DEFAULTS

def test_pipeline_smoke():
    data = generate_data(n_customers=30, n_vehicles=4, L_max=30, seed=0)
    sol = build_initial(data)
    params = DEFAULTS.copy()
    params["iters"] = 200
    metrics = Metrics()
    best = run_alns(sol, data, params, metrics)
    assert best["best_cost"] < 1e12
