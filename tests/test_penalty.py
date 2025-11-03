import numpy as np

from alns_cvrptwpd.config.config import DEFAULTS
from alns_cvrptwpd.config.enums import F_PEN, P_CAP
from alns_cvrptwpd.engine.penalty import init_penalty_state, penalty_cost, update_penalty_state


def test_penalty_cost_scales_with_weights():
    params = DEFAULTS.copy()
    params["penalty_weight_cap"] = 10.0
    state = init_penalty_state(params)
    vec = np.zeros(F_PEN, dtype=np.float32)
    vec[P_CAP] = 3.0
    cost = penalty_cost(state, vec)
    assert np.isclose(cost, 30.0)


def test_penalty_update_increases_weight_for_persistent_violation():
    params = DEFAULTS.copy()
    params["penalty_target_rate"] = 0.01
    params["penalty_update_rate"] = 0.5
    params["penalty_violation_alpha"] = 0.5
    state = init_penalty_state(params)
    vec = np.zeros(F_PEN, dtype=np.float32)
    vec[P_CAP] = 10.0
    base_cost = penalty_cost(state, vec)
    for _ in range(20):
        update_penalty_state(state, vec)
    updated_cost = penalty_cost(state, vec)
    assert not np.isclose(state["weights"][P_CAP], params["penalty_weight_cap"])
    assert state["viol_ema"][P_CAP] > params["penalty_target_rate"]
    assert updated_cost > 0.0
