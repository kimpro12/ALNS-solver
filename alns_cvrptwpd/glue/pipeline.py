"""Command line pipeline orchestrating dataset loading and ALNS execution."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ..config.config import DEFAULTS
from ..engine.alns import run_alns
from ..logging.metrics import Metrics, save_metrics_json, save_routes_csv
from ..preprocessing.initial_solution import build_initial
from ..engine.state_update import refresh_route_loads
from .io import (
    compute_euclid,
    load_config,
    load_initial_npz,
    load_matrices,
    load_nodes,
    load_vehicles,
    validate_inputs,
)


def _resolve(base: Path, maybe_path: Optional[str]) -> Optional[Path]:
    if maybe_path is None:
        return None
    return (base / maybe_path).resolve()


def assemble_data(cfg: Dict[str, Any], base_dir: Path) -> Dict[str, np.ndarray]:
    """Load dataset artifacts following the configuration contract."""

    dataset = cfg.get("dataset", {})

    nodes_path = dataset.get("nodes")
    vehicles_path = dataset.get("vehicles")
    matrices_path = dataset.get("matrices")
    initial_path = dataset.get("initial_solution")

    if nodes_path is None or vehicles_path is None:
        raise ValueError("dataset.nodes and dataset.vehicles must be provided")

    coords, node_f, node_i = load_nodes(_resolve(base_dir, nodes_path))
    veh_f, veh_i = load_vehicles(_resolve(base_dir, vehicles_path))

    if matrices_path is not None:
        dist, ttime, edge_vec, cost_w = load_matrices(_resolve(base_dir, matrices_path))
    else:
        dist = compute_euclid(coords)
        ttime = dist.copy()
        edge_vec = dist[..., None]
        cost_w = np.ones(edge_vec.shape[-1], dtype=np.float32)

    data = {
        "coords": coords,
        "dist": dist,
        "ttime": ttime,
        "edge_vec": edge_vec,
        "cost_w": cost_w,
        "node_f": node_f,
        "node_i": node_i,
        "veh_f": veh_f,
        "veh_i": veh_i,
        "n": coords.shape[0],
        "m": veh_f.shape[0],
    }

    validate_inputs(data)

    m = veh_f.shape[0]
    L_max = int(cfg.get("L_max", dataset.get("L_max", coords.shape[0] + 5)))
    routes = np.zeros((m, L_max), dtype=np.int32)
    lens = np.zeros(m, dtype=np.int32)

    if initial_path is not None:
        r_path = _resolve(base_dir, initial_path)
        init_routes, init_lens = load_initial_npz(r_path)
        if init_routes.shape[0] != m:
            raise ValueError("initial routes vehicle dimension mismatch")
        if init_routes.shape[1] > L_max:
            raise ValueError("initial routes length exceeds configured L_max")
        routes[:, : init_routes.shape[1]] = init_routes
        lens[:] = init_lens

    loads = np.zeros(m, dtype=np.float32)

    data.update({
        "routes": routes,
        "lens": lens,
        "loads": loads,
        "L_max": L_max,
    })

    if initial_path is not None:
        refresh_route_loads(
            routes,
            lens,
            node_f,
            loads=data["loads"],
        )

    return data

def build_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    params = DEFAULTS.copy()
    params.update(cfg.get("params", {}))
    if "iters" in cfg:
        params["iters"] = int(cfg["iters"])
    if "log_period" in cfg:
        params["log_period"] = int(cfg["log_period"])
    return params


def run_pipeline(
    cfg: Dict[str, Any],
    *,
    base_dir: Path,
    outdir: Path,
    export_trace: bool = False,
) -> Dict[str, Any]:
    """Execute the ALNS solver according to ``cfg`` and return best solution."""

    outdir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("seed", 0))
    np.random.seed(seed)
    random.seed(seed)

    data = assemble_data(cfg, base_dir)

    initial = build_initial(data)
    params = build_params(cfg)
    metrics = Metrics()

    best = run_alns(initial, data, params, metrics)

    meta = {
        "seed": seed,
        "config_version": cfg.get("version", "dev"),
        "iters_logged": len(metrics.rows),
    }

    if export_trace:
        trace_path = outdir / "trace.npz"
        arr_iters = np.array([row[0] for row in metrics.rows], dtype=np.int32)
        arr_curr = np.array([row[1] for row in metrics.rows], dtype=np.float32)
        arr_best = np.array([row[2] for row in metrics.rows], dtype=np.float32)
        arr_temp = np.array([row[3] for row in metrics.rows], dtype=np.float32)
        np.savez(
            trace_path,
            iters=arr_iters,
            curr=arr_curr,
            best=arr_best,
            temp=arr_temp,
            best_routes=best["routes"],
            best_lens=best["lens"],
        )
        meta["trace"] = str(trace_path)

    save_metrics_json(outdir / "metrics.json", metrics, best, params, extra=meta)
    save_routes_csv(outdir / "routes.csv", best["routes"], best["lens"])
    metrics.save_csv(outdir / "metrics_log.csv")

    return {
        "best": best,
        "metrics": metrics,
        "params": params,
        "meta": meta,
    }


def load_and_run(
    config_path: Path,
    outdir: Path,
    *,
    seed_override: Optional[int] = None,
    export_trace: bool = False,
) -> Dict[str, Any]:
    """Convenience wrapper combining ``load_config`` and :func:`run_pipeline`."""

    cfg = load_config(config_path)
    if seed_override is not None:
        cfg["seed"] = int(seed_override)

    base_dir = Path(config_path).resolve().parent
    return run_pipeline(cfg, base_dir=base_dir, outdir=outdir, export_trace=export_trace)


def build_arg_parser():
    import argparse

    ap = argparse.ArgumentParser(description="ALNS-CVRPTWPD solver")
    ap.add_argument("--config", required=True, help="Path to YAML/JSON configuration")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--seed", type=int, default=None, help="Optional RNG seed override")
    ap.add_argument(
        "--trace",
        action="store_true",
        help="Export compact trace.npz alongside metrics",
    )
    return ap


def main(argv: Optional[list[str]] = None) -> Dict[str, Any]:
    ap = build_arg_parser()
    args = ap.parse_args(argv)

    outdir = Path(args.outdir).resolve()
    cfg_path = Path(args.config).resolve()

    result = load_and_run(
        cfg_path,
        outdir,
        seed_override=args.seed,
        export_trace=args.trace,
    )

    best = result["best"]
    used = int((best["lens"] > 0).sum())
    summary = {
        "best_cost": float(best["best_cost"]),
        "total_distance": float(best["total_dist"]),
        "vehicles_used": used,
        "vehicles_total": int(result["best"]["routes"].shape[0]),
        "unserved": len(best["unrouted"]),
    }

    print("\n[DONE]")
    print(json.dumps(summary, indent=2))
    return result


__all__ = [
    "assemble_data",
    "build_params",
    "build_arg_parser",
    "load_and_run",
    "main",
    "run_pipeline",
]
