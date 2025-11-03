import numpy as np

from alns_cvrptwpd.config.enums import (
    F_NODE_F,
    F_NODE_I,
    F_VEH_F,
    NODE_DEMAND,
    NODE_SERVICE,
    NODE_TW_CLOSE,
    NODE_TW_OPEN,
    VEH_CAPACITY,
)
from alns_cvrptwpd.operators.destroy.random_removal import random_removal
from alns_cvrptwpd.operators.destroy.worst_removal import worst_removal
from alns_cvrptwpd.operators.repair import (
    best_insertion,
    random_top_k_insertion,
    regret_insertion,
)
from alns_cvrptwpd.operators.repair.best_insertion import _compute_moves_for_customer
from alns_cvrptwpd.operators.repair.utils import extract_route, vehicle_capacities


def _make_node_features(n):
    node_f = np.zeros((n, F_NODE_F), dtype=np.float32)
    node_f[:, NODE_TW_OPEN] = 0.0
    node_f[:, NODE_TW_CLOSE] = 100.0
    node_f[:, NODE_SERVICE] = 0.0
    return node_f


def _make_node_indices(n):
    return np.zeros((n, F_NODE_I), dtype=np.int32)


def _make_vehicle_features(m, capacity):
    veh_f = np.zeros((m, F_VEH_F), dtype=np.float32)
    veh_f[:, VEH_CAPACITY] = capacity
    return veh_f


def test_worst_removal_prefers_high_cost_customer():
    routes = np.array([[1, 2, 3, 0]], dtype=np.int32)
    lens = np.array([3], dtype=np.int32)
    dist = np.array(
        [
            [0.0, 10.0, 5.0, 5.0],
            [10.0, 0.0, 20.0, 5.0],
            [5.0, 20.0, 0.0, 5.0],
            [5.0, 5.0, 5.0, 0.0],
        ],
        dtype=np.float32,
    )
    rng = np.random.default_rng(42)

    removed, changed = worst_removal(routes, lens, dist, remove_k=1, rng=rng)

    assert removed.tolist() == [1]
    assert changed.tolist() == [1]
    assert lens[0] == 2
    np.testing.assert_array_equal(routes[0, :2], np.array([2, 3]))


def test_best_insertion_selects_lowest_delta_route():
    routes = np.array([[1, 0, 0], [2, 0, 0]], dtype=np.int32)
    lens = np.array([1, 1], dtype=np.int32)
    loads = np.array([1.0, 1.0], dtype=np.float32)
    dist = np.array(
        [
            [0.0, 1.0, 3.0, 1.0],
            [1.0, 0.0, 10.0, 10.0],
            [3.0, 10.0, 0.0, 1.0],
            [1.0, 10.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    node_f = _make_node_features(4)
    node_f[1:, NODE_DEMAND] = 1.0
    node_i = _make_node_indices(4)
    veh_f = _make_vehicle_features(2, capacity=4.0)

    unrouted = np.array([3], dtype=np.int32)
    veh_caps = vehicle_capacities(veh_f)
    expected_moves = _compute_moves_for_customer(
        3, routes.copy(), lens.copy(), node_f, node_i, veh_caps, dist
    )

    inserted, used_mask, changed = best_insertion(
        unrouted,
        routes,
        lens,
        loads,
        node_f,
        node_i,
        veh_f,
        dist,
    )

    assert inserted == 1
    assert used_mask.tolist() == [True]
    assert changed.tolist() == [0, 1]
    best_move = expected_moves[0]
    route_idx, pos = best_move[2], best_move[3]
    sequence = extract_route(routes, lens, route_idx)
    assert sequence[pos] == 3
    assert loads[1] == 2.0


def test_best_insertion_respects_time_windows():
    routes = np.array([[1, 0, 0]], dtype=np.int32)
    lens = np.array([1], dtype=np.int32)
    loads = np.array([1.0], dtype=np.float32)
    dist = np.array(
        [
            [0.0, 2.0, 2.0],
            [2.0, 0.0, 10.0],
            [2.0, 10.0, 0.0],
        ],
        dtype=np.float32,
    )
    node_f = _make_node_features(3)
    node_f[1:, NODE_DEMAND] = 1.0
    node_f[2, NODE_TW_CLOSE] = 3.0
    node_i = _make_node_indices(3)
    veh_f = _make_vehicle_features(1, capacity=5.0)

    unrouted = np.array([2], dtype=np.int32)
    inserted, used_mask, changed = best_insertion(
        unrouted,
        routes,
        lens,
        loads,
        node_f,
        node_i,
        veh_f,
        dist,
    )

    assert inserted == 1
    np.testing.assert_array_equal(routes[0, :2], np.array([2, 1]))
    assert used_mask.tolist() == [True]
    assert changed.tolist() == [1]


def test_random_top_k_matches_best_when_k_is_one():
    routes = np.array([[1, 0, 0], [2, 0, 0]], dtype=np.int32)
    lens = np.array([1, 1], dtype=np.int32)
    loads = np.array([1.0, 1.0], dtype=np.float32)
    dist = np.array(
        [
            [0.0, 1.0, 3.0, 1.0],
            [1.0, 0.0, 10.0, 10.0],
            [3.0, 10.0, 0.0, 1.0],
            [1.0, 10.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    node_f = _make_node_features(4)
    node_f[1:, NODE_DEMAND] = 1.0
    node_i = _make_node_indices(4)
    veh_f = _make_vehicle_features(2, capacity=4.0)

    unrouted = np.array([3], dtype=np.int32)
    veh_caps = vehicle_capacities(veh_f)
    expected_moves = _compute_moves_for_customer(
        3, routes.copy(), lens.copy(), node_f, node_i, veh_caps, dist
    )

    inserted, used_mask, changed = random_top_k_insertion(
        unrouted,
        routes,
        lens,
        loads,
        node_f,
        node_i,
        veh_f,
        dist,
        top_k=1,
        rng=np.random.default_rng(0),
    )

    assert inserted == 1
    assert used_mask.tolist() == [True]
    best_move = expected_moves[0]
    sequence = extract_route(routes, lens, best_move[2])
    assert sequence[best_move[3]] == 3


def test_regret_insertion_prefers_high_regret_customer():
    coords = np.array(
        [
            [0.0, 0.0],
            [0.0, 4.0],
            [1.0, 0.0],
            [10.0, 0.0],
            [4.0, 4.0],
        ]
    )
    dx = coords[:, None, 0] - coords[None, :, 0]
    dy = coords[:, None, 1] - coords[None, :, 1]
    dist = np.sqrt(dx * dx + dy * dy)

    routes = np.array([[1, 4, 0]], dtype=np.int32)
    lens = np.array([2], dtype=np.int32)
    loads = np.array([2.0], dtype=np.float32)
    node_f = _make_node_features(5)
    node_f[1:, NODE_DEMAND] = 1.0
    node_i = _make_node_indices(5)
    veh_f = _make_vehicle_features(1, capacity=6.0)

    unrouted = np.array([2, 3], dtype=np.int32)
    inserted, used_mask, changed = regret_insertion(
        unrouted,
        routes,
        lens,
        loads,
        node_f,
        node_i,
        veh_f,
        dist,
        k=2,
    )

    # Only one slot available; the higher-regret customer (3) should be inserted.
    assert inserted == 1
    assert used_mask.tolist() == [False, True]
    np.testing.assert_array_equal(routes[0, :3], np.array([1, 4, 3]))

    veh_caps = vehicle_capacities(veh_f)
    moves_c2 = _compute_moves_for_customer(2, routes, lens, node_f, node_i, veh_caps, dist)
    assert moves_c2 == []  # route is full; remaining customer stays unrouted.


def test_random_removal_uses_weights():
    routes = np.array([[1, 2, 3, 0]], dtype=np.int32)
    lens = np.array([3], dtype=np.int32)
    rng = np.random.default_rng(123)
    weights = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    removed, changed = random_removal(routes, lens, 1, rng, weights=weights)

    assert removed.tolist() == [1]
    assert changed.tolist() == [1]
