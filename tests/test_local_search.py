import numpy as np

from alns_cvrptwpd.local_search.apply_local_search import apply_local_search


def _build_node_features(n, tw_open=None, tw_close=None):
    from alns_cvrptwpd.config.enums import F_NODE_F

    node_f = np.zeros((n, F_NODE_F), dtype=np.float32)
    node_f[:, 2] = 0.0 if tw_open is None else tw_open
    node_f[:, 3] = 100.0 if tw_close is None else tw_close
    return node_f


def _empty_node_i(n):
    from alns_cvrptwpd.config.enums import F_NODE_I

    return np.zeros((n, F_NODE_I), dtype=np.int32)


def test_light_local_search_improves_route():
    dist = np.array(
        [
            [0.0, 1.0, 5.0, 5.0],
            [1.0, 0.0, 1.0, 2.0],
            [5.0, 1.0, 0.0, 1.0],
            [5.0, 2.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    node_f = _build_node_features(4)
    node_i = _empty_node_i(4)
    veh_f = np.array([[10.0, 0.0, 100.0]], dtype=np.float32)

    routes = np.array([[1, 3, 2, 0]], dtype=np.int32)
    lens = np.array([3], dtype=np.int32)

    improved = apply_local_search(
        routes,
        lens,
        dist,
        node_f,
        node_i,
        veh_f,
        budget=10,
        heavy=False,
        rng=np.random.default_rng(42),
    )

    assert improved
    assert list(routes[0, : lens[0]]) == [1, 2, 3]


def test_local_search_respects_time_windows():
    dist = np.array(
        [
            [0.0, 1.0, 10.0, 2.0],
            [1.0, 0.0, 2.0, 2.0],
            [10.0, 2.0, 0.0, 10.0],
            [2.0, 2.0, 10.0, 0.0],
        ],
        dtype=np.float32,
    )

    node_f = _build_node_features(4)
    node_f[1, 3] = 100.0
    node_f[2, 3] = 4.0
    node_f[3, 2] = 0.0
    node_f[3, 3] = 100.0
    node_i = _empty_node_i(4)
    veh_f = np.array([[10.0, 0.0, 100.0]], dtype=np.float32)

    routes = np.array([[1, 2, 3, 0]], dtype=np.int32)
    lens = np.array([3], dtype=np.int32)

    improved = apply_local_search(
        routes,
        lens,
        dist,
        node_f,
        node_i,
        veh_f,
        budget=5,
        heavy=True,
        rng=np.random.default_rng(7),
    )

    assert not improved
    assert list(routes[0, : lens[0]]) == [1, 2, 3]
