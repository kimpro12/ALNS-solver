import math

import numpy as np

from alns_cvrptwpd.config.enums import DIM_MODE_ADD, DIM_MODE_FUNC, DIM_MODE_MAX
from alns_cvrptwpd.engine.dimension import (
    build_giant_tour_savings,
    cost_vehicle,
    dim_eval,
    repair_giant_tour_focus_njit,
    spatial_penalty,
    vector_feasible_check_fast_masked,
    vector_state_update,
    vector_state_update_delta,
)


def test_vector_state_update_delta():
    state = np.array([1.0, 2.0], dtype=np.float32)
    old = np.array([0.5, 0.5], dtype=np.float32)
    new = np.array([1.5, -0.5], dtype=np.float32)
    vector_state_update_delta(state, old, new)
    np.testing.assert_allclose(state, [2.0, 1.0])


def test_vector_state_update_modes():
    state = np.zeros(3, dtype=np.float32)
    uav = np.array([0.5, 1.0, 0.0], dtype=np.float32)
    bav = np.array([1.0, 0.5, 2.0], dtype=np.float32)

    temp = state.copy()
    vector_state_update(temp, uav, bav, DIM_MODE_ADD)
    np.testing.assert_allclose(temp, [1.5, 1.5, 2.0])

    temp = state.copy()
    vector_state_update(temp, uav, bav, DIM_MODE_MAX)
    np.testing.assert_allclose(temp, [1.0, 1.0, 2.0])

    temp = state.copy()
    vector_state_update(temp, uav, bav, DIM_MODE_FUNC)
    # FUNC falls back to additive with a soft maximum emphasis. The result must
    # stay greater or equal to the pure max behaviour for each dimension.
    assert np.all(temp >= np.array([1.0, 1.0, 2.0]))


def test_vector_feasible_check_fast_masked():
    state = np.array([5.0, 2.0, 7.0], dtype=np.float32)
    limit = np.array([6.0, 2.5, 6.5], dtype=np.float32)
    mask = np.array([True, False, True])
    assert not vector_feasible_check_fast_masked(state, limit, mask)
    state[2] = 6.0
    assert vector_feasible_check_fast_masked(state, limit, mask)


def test_dim_eval_returns_feasibility_and_slack():
    feasible, slack = dim_eval(5.0, 6.0, DIM_MODE_ADD)
    assert feasible and math.isclose(slack, 1.0)

    feasible, slack = dim_eval(7.0, 6.0, DIM_MODE_MAX)
    assert not feasible and math.isclose(slack, 0.0)


def test_cost_vehicle_matches_dot_product():
    vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    weights = np.array([0.5, 1.5, -0.5], dtype=np.float32)
    assert math.isclose(cost_vehicle(vec, weights), float(vec.dot(weights)))


def test_spatial_penalty_scales_distance():
    coords = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
    centroid = np.array([0.0, 0.0], dtype=np.float32)
    penalty = spatial_penalty(coords, 1, centroid, 2.0)
    assert math.isclose(penalty, 5.0 / 2.0)


def test_build_giant_tour_savings_orders_nodes():
    dist = np.array(
        [
            [0.0, 2.0, 3.0, 4.0],
            [2.0, 0.0, 1.0, 5.0],
            [3.0, 1.0, 0.0, 2.0],
            [4.0, 5.0, 2.0, 0.0],
        ],
        dtype=np.float32,
    )
    nodes = np.array([1, 2, 3], dtype=np.int32)
    order = build_giant_tour_savings(dist, nodes)
    assert set(order.tolist()) == {1, 2, 3}
    assert order.shape[0] == 3


def test_repair_giant_tour_focus_prefers_central_nodes():
    dist = np.array(
        [
            [0.0, 2.0, 3.0, 5.0],
            [2.0, 0.0, 1.0, 4.0],
            [3.0, 1.0, 0.0, 2.5],
            [5.0, 4.0, 2.5, 0.0],
        ],
        dtype=np.float32,
    )
    coords = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ],
        dtype=np.float32,
    )
    centroid = np.array([0.0, 0.0], dtype=np.float32)
    nodes = np.array([1, 2, 3], dtype=np.int32)
    order = repair_giant_tour_focus_njit(dist, nodes, coords, centroid, 1.0)
    assert set(order.tolist()) == {1, 2, 3}
    assert order[0] == 1
