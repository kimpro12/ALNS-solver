import json
from pathlib import Path

import numpy as np
import pytest

from alns_cvrptwpd.glue.io import (
    compute_euclid,
    load_config,
    load_matrices,
    load_nodes,
    load_vehicles,
    validate_inputs,
)


def test_load_config_yaml(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("seed: 7\nparams:\n  iters: 12\n", encoding="utf-8")

    cfg = load_config(cfg_path)
    assert cfg["seed"] == 7
    assert cfg["params"]["iters"] == 12


def test_load_config_json(tmp_path):
    cfg_path = tmp_path / "cfg.json"
    cfg = {"seed": 5}
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    loaded = load_config(cfg_path)
    assert loaded == cfg


def _write_nodes(tmp_path: Path):
    path = tmp_path / "nodes.csv"
    path.write_text(
        """x,y,demand,service,tw_open,tw_close,pd_pair,pd_role,required
0,0,0,0,0,10,-1,0,1
1,0,1,2,0,10,-1,0,1
""",
        encoding="utf-8",
    )
    return path


def _write_vehicles(tmp_path: Path):
    path = tmp_path / "vehicles.csv"
    path.write_text(
        """start_depot,end_depot,capacity,fixed_cost,max_dur
0,0,5,0,100
""",
        encoding="utf-8",
    )
    return path


def _write_matrices(tmp_path: Path):
    path = tmp_path / "matrices.npz"
    dist = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    ttime = dist + 1.0
    edge_vec = dist[..., None]
    cost_w = np.array([1.0], dtype=np.float32)
    np.savez(path, dist=dist, ttime=ttime, edge_vec=edge_vec, cost_w=cost_w)
    return path


def test_load_tables(tmp_path):
    n_path = _write_nodes(tmp_path)
    v_path = _write_vehicles(tmp_path)

    coords, node_f, node_i = load_nodes(n_path)
    veh_f, veh_i = load_vehicles(v_path)

    assert coords.shape == (2, 2)
    assert node_f.shape[0] == 2
    assert node_i.shape[0] == 2
    assert veh_f.shape == (1, 3)
    assert veh_i.shape == (1, 2)
    assert node_f[1, 0] == pytest.approx(1.0)
    assert veh_f[0, 0] == pytest.approx(5.0)


def test_load_matrices(tmp_path):
    m_path = _write_matrices(tmp_path)
    dist, ttime, edge_vec, cost_w = load_matrices(m_path)

    assert dist.shape == (2, 2)
    assert ttime[0, 1] == pytest.approx(2.0)
    assert edge_vec.shape == (2, 2, 1)
    assert cost_w.shape == (1,)


def test_validate_inputs_success(tmp_path):
    n_path = _write_nodes(tmp_path)
    v_path = _write_vehicles(tmp_path)
    m_path = _write_matrices(tmp_path)

    coords, node_f, node_i = load_nodes(n_path)
    veh_f, veh_i = load_vehicles(v_path)
    dist, ttime, edge_vec, cost_w = load_matrices(m_path)

    data = {
        "coords": coords,
        "node_f": node_f,
        "node_i": node_i,
        "veh_f": veh_f,
        "veh_i": veh_i,
        "dist": dist,
        "ttime": ttime,
        "edge_vec": edge_vec,
        "cost_w": cost_w,
    }

    validate_inputs(data)


def test_validate_inputs_failure(tmp_path):
    n_path = _write_nodes(tmp_path)
    v_path = _write_vehicles(tmp_path)

    coords, node_f, node_i = load_nodes(n_path)
    veh_f, veh_i = load_vehicles(v_path)

    bad_dist = np.zeros((3, 3), dtype=np.float32)
    data = {
        "coords": coords,
        "node_f": node_f,
        "node_i": node_i,
        "veh_f": veh_f,
        "veh_i": veh_i,
        "dist": bad_dist,
        "ttime": bad_dist,
        "edge_vec": bad_dist[..., None],
        "cost_w": np.ones(1, dtype=np.float32),
    }

    with pytest.raises(ValueError):
        validate_inputs(data)


def test_compute_euclid():
    coords = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
    dist = compute_euclid(coords)
    assert dist[0, 1] == pytest.approx(5.0)
