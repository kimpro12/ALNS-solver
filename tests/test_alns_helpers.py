import numpy as np

from alns_cvrptwpd.engine.alns import _initial_temperature, _update_operator_weight
from alns_cvrptwpd.engine.sfr import SpatialFocus


def test_initial_temperature_uses_median_when_missing():
    dist = np.array(
        [
            [0.0, 5.0, 7.0],
            [5.0, 0.0, 9.0],
            [7.0, 9.0, 0.0],
        ]
    )
    # median of strictly positive upper-triangular entries: [5,7,9] -> 7
    temp = _initial_temperature(dist, None)
    assert np.isclose(temp, 7.0)


def test_update_operator_weight_moves_towards_reward():
    weights = np.array([1.0, 2.0, 3.0])
    _update_operator_weight(weights, 1, rho=0.25, reward=10.0)
    expected = (1 - 0.25) * 2.0 + 0.25 * 10.0
    assert np.isclose(weights[1], expected)
    # other indices untouched
    assert weights[0] == 1.0
    assert weights[2] == 3.0


def test_initial_temperature_respects_positive_override():
    dist = np.zeros((2, 2))
    temp = _initial_temperature(dist, 42.5)
    assert temp == 42.5


def test_spatial_focus_updates_weights_towards_active_region():
    coords = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [5.0, 0.0],
            [10.0, 0.0],
        ],
        dtype=np.float32,
    )
    routes = np.array([[1, 2, 0]], dtype=np.int32)
    lens = np.array([2], dtype=np.int32)

    sfr = SpatialFocus(coords, alpha0=5.0, alpha_min=1.0, gamma=1.0, beta=0.5)
    sfr.update(routes, lens, iteration=0, total_iters=10)

    weights = sfr.customer_weights
    assert weights[1] > weights[3]
