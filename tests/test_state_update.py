import numpy as np

from alns_cvrptwpd.config.enums import (
    NODE_DEMAND,
    NODE_SERVICE,
    NODE_TW_CLOSE,
    NODE_TW_OPEN,
    VEH_CAPACITY,
    VEH_MAX_DUR,
)
from alns_cvrptwpd.data.generate_data import generate_data
from alns_cvrptwpd.engine.state_update import compute_route_states, refresh_route_loads
from alns_cvrptwpd.operators.destroy.random_removal import random_removal
from alns_cvrptwpd.preprocessing.initial_solution import build_initial


def _expected_loads(routes, lens, node_f):
    demand = node_f[:, NODE_DEMAND]
    loads = np.zeros(routes.shape[0], dtype=np.float32)
    for r in range(routes.shape[0]):
        L = int(lens[r])
        if L:
            loads[r] = float(demand[routes[r, :L]].sum())
    return loads


def test_refresh_route_loads_updates_changed_routes():
    data = generate_data(n_customers=12, n_vehicles=3, L_max=10, seed=1)
    sol = build_initial(data)

    # Force a baseline load recompute to populate the array fully
    refresh_route_loads(sol["routes"], sol["lens"], data["node_f"], loads=sol["loads"])
    baseline = sol["loads"].copy()

    removed, changed = random_removal(sol["routes"], sol["lens"], remove_k=4, rng=np.random.default_rng(0))
    assert removed.size > 0

    refresh_route_loads(sol["routes"], sol["lens"], data["node_f"], loads=sol["loads"], changed=changed)

    expected = _expected_loads(sol["routes"], sol["lens"], data["node_f"])
    np.testing.assert_allclose(sol["loads"], expected)

    # Routes that did not change keep their baseline value
    for r in range(len(changed)):
        if not changed[r]:
            assert sol["loads"][r] == baseline[r]


def test_refresh_route_loads_allocates_when_missing():
    data = generate_data(n_customers=8, n_vehicles=2, L_max=8, seed=2)
    loads = refresh_route_loads(data["routes"], data["lens"], data["node_f"])
    assert isinstance(loads, np.ndarray)
    np.testing.assert_array_equal(loads, np.zeros_like(loads))


def test_compute_route_states_penalty_vectors():
    dist = np.array(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    node_f = np.zeros((3, 4), dtype=np.float32)
    node_f[1, NODE_DEMAND] = 3.0
    node_f[2, NODE_DEMAND] = 4.0
    node_f[1, NODE_SERVICE] = 0.0
    node_f[2, NODE_SERVICE] = 0.0
    node_f[1, NODE_TW_OPEN] = 0.0
    node_f[2, NODE_TW_OPEN] = 0.0
    node_f[1, NODE_TW_CLOSE] = 0.5
    node_f[2, NODE_TW_CLOSE] = 1.0

    veh_f = np.zeros((1, 3), dtype=np.float32)
    veh_f[0, VEH_CAPACITY] = 5.0
    veh_f[0, VEH_MAX_DUR] = 2.5

    routes = np.array([[1, 2]], dtype=np.int32)
    lens = np.array([2], dtype=np.int32)

    stats = compute_route_states(
        routes,
        lens,
        dist,
        node_f,
        veh_f,
    )

    assert stats["route_overload"].shape == (1,)
    assert stats["route_dur_excess"].shape == (1,)
    assert stats["route_late"].shape == (1,)
    np.testing.assert_allclose(stats["route_overload"], [2.0])
    np.testing.assert_allclose(stats["route_dur_excess"], [0.5])
    np.testing.assert_allclose(stats["route_late"], [1.5])
    assert not stats["feasible"].all()
