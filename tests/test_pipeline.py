import json
from pathlib import Path

import numpy as np

from alns_cvrptwpd.glue.pipeline import assemble_data, build_params, run_pipeline


def _write_dataset(tmp_path: Path):
    (tmp_path / "dataset").mkdir()
    nodes = tmp_path / "dataset" / "nodes.csv"
    nodes.write_text(
        """x,y,demand,service,tw_open,tw_close,pd_pair,pd_role,required
0,0,0,0,0,10,-1,0,1
1,0,1,2,0,10,-1,0,1
""",
        encoding="utf-8",
    )
    vehicles = tmp_path / "dataset" / "vehicles.csv"
    vehicles.write_text(
        """start_depot,end_depot,capacity,fixed_cost,max_dur
0,0,5,0,100
""",
        encoding="utf-8",
    )
    dist = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    np.savez(tmp_path / "dataset" / "matrices.npz", dist=dist, ttime=dist)
    return nodes, vehicles, tmp_path / "dataset" / "matrices.npz"


def test_assemble_data(tmp_path):
    nodes, vehicles, matrices = _write_dataset(tmp_path)
    cfg = {
        "dataset": {
            "nodes": str(nodes.relative_to(tmp_path)),
            "vehicles": str(vehicles.relative_to(tmp_path)),
            "matrices": str(matrices.relative_to(tmp_path)),
        },
        "L_max": 5,
    }

    data = assemble_data(cfg, base_dir=tmp_path)
    assert data["routes"].shape == (1, 5)
    assert data["lens"].shape == (1,)
    assert data["dist"].shape == (2, 2)


def test_build_params_override():
    cfg = {"iters": 10, "params": {"log_period": 1}}
    params = build_params(cfg)
    assert params["iters"] == 10
    assert params["log_period"] == 1


def test_run_pipeline(tmp_path):
    nodes, vehicles, matrices = _write_dataset(tmp_path)
    outdir = tmp_path / "out"
    cfg = {
        "seed": 0,
        "params": {"iters": 5, "log_period": 1},
        "dataset": {
            "nodes": str(nodes.relative_to(tmp_path)),
            "vehicles": str(vehicles.relative_to(tmp_path)),
            "matrices": str(matrices.relative_to(tmp_path)),
        },
        "L_max": 5,
    }

    result = run_pipeline(cfg, base_dir=tmp_path, outdir=outdir)
    assert "best" in result
    assert (outdir / "metrics.json").exists()
    assert (outdir / "routes.csv").exists()
    assert (outdir / "metrics_log.csv").exists()

    with open(outdir / "metrics.json", encoding="utf-8") as f:
        metrics_data = json.load(f)
    assert metrics_data["seed"] == 0
    assert metrics_data["iters_logged"] >= 1
