"""Dataset and configuration helpers for the command-line glue layer.

The chapter 10 specification calls for light-weight, NumPy/Pandas friendly
loading utilities that are able to ingest YAML configuration files alongside
CSV/Parquet node & vehicle tables as well as NPZ cost matrices.  The helpers in
this module intentionally avoid class wrappers so they remain Numba-friendly
and easy to compose inside tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np

from ..config.enums import (
    F_NODE_F,
    F_NODE_I,
    F_VEH_F,
    NODE_DEMAND,
    NODE_PD_PAIR,
    NODE_PD_ROLE,
    NODE_REQUIRED,
    NODE_SERVICE,
    NODE_TW_CLOSE,
    NODE_TW_OPEN,
    VEH_CAPACITY,
    VEH_FIXED_COST,
    VEH_MAX_DUR,
)


def load_config(path_yaml: Path) -> Dict:
    """Read a YAML (or JSON) configuration file.

    Parameters
    ----------
    path_yaml:
        Path to the configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary.  Empty files resolve to ``{}``.
    """

    import json

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - yaml is available in tests
        raise RuntimeError("pyyaml is required to load configuration files") from exc

    path = Path(path_yaml)
    if not path.exists():
        raise FileNotFoundError(path)

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}

    if path.suffix.lower() == ".json":
        return json.loads(text)

    cfg = yaml.safe_load(text)
    return cfg or {}


def _read_frame(path_like: Path):
    """Return a Pandas ``DataFrame`` from CSV or Parquet input."""

    import pandas as pd

    path = Path(path_like)
    if path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    return df.fillna(0)


def _read_structured_csv(path_like: Path) -> np.ndarray:
    arr = np.genfromtxt(
        path_like,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    if arr.size == 0:
        raise ValueError(f"empty table: {path_like}")
    if arr.ndim == 0:
        arr = arr.reshape((1,))
    return arr


def load_nodes(path_table: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load node coordinates and attributes from a CSV/Parquet table."""

    try:
        df = _read_frame(path_table)
        if not {"x", "y"}.issubset(df.columns):
            raise ValueError("node table must contain 'x' and 'y' columns")

        n = len(df.index)
        coords = df[["x", "y"]].to_numpy(dtype=np.float32, copy=True)

        node_f = np.zeros((n, F_NODE_F), dtype=np.float32)
        node_i = np.zeros((n, F_NODE_I), dtype=np.int32)

        node_f[:, NODE_DEMAND] = df.get("demand", 0.0).to_numpy(dtype=np.float32, copy=True)
        node_f[:, NODE_SERVICE] = df.get("service", 0.0).to_numpy(dtype=np.float32, copy=True)
        node_f[:, NODE_TW_OPEN] = df.get("tw_open", 0.0).to_numpy(dtype=np.float32, copy=True)
        node_f[:, NODE_TW_CLOSE] = df.get("tw_close", 0.0).to_numpy(dtype=np.float32, copy=True)

        node_i[:, NODE_PD_PAIR] = df.get("pd_pair", -1).to_numpy(dtype=np.int32, copy=True)
        node_i[:, NODE_PD_ROLE] = df.get("pd_role", 0).to_numpy(dtype=np.int32, copy=True)
        node_i[:, NODE_REQUIRED] = df.get("required", 1).to_numpy(dtype=np.int32, copy=True)

        return coords, node_f, node_i
    except ModuleNotFoundError:
        arr = _read_structured_csv(path_table)
        names = set(arr.dtype.names or ())
        if "x" not in names or "y" not in names:
            raise ValueError("node table must contain 'x' and 'y' columns")

        n = arr.shape[0]
        coords = np.column_stack((arr["x"], arr["y"])).astype(np.float32, copy=False)

        node_f = np.zeros((n, F_NODE_F), dtype=np.float32)
        node_i = np.zeros((n, F_NODE_I), dtype=np.int32)

        def _get(name: str, default: float, dtype) -> np.ndarray:
            if name in names:
                return np.asarray(arr[name], dtype=dtype)
            return np.full(n, default, dtype=dtype)

        node_f[:, NODE_DEMAND] = _get("demand", 0.0, np.float32)
        node_f[:, NODE_SERVICE] = _get("service", 0.0, np.float32)
        node_f[:, NODE_TW_OPEN] = _get("tw_open", 0.0, np.float32)
        node_f[:, NODE_TW_CLOSE] = _get("tw_close", 0.0, np.float32)

        node_i[:, NODE_PD_PAIR] = _get("pd_pair", -1, np.int32)
        node_i[:, NODE_PD_ROLE] = _get("pd_role", 0, np.int32)
        node_i[:, NODE_REQUIRED] = _get("required", 1, np.int32)

        return coords, node_f, node_i


def load_vehicles(path_table: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load vehicle capacity & depot indices from a CSV/Parquet table."""

    try:
        df = _read_frame(path_table)

        if not {"start_depot", "end_depot"}.issubset(df.columns):
            raise ValueError("vehicle table must contain 'start_depot' and 'end_depot'")

        m = len(df.index)
        veh_f = np.zeros((m, F_VEH_F), dtype=np.float32)
        veh_i = np.zeros((m, 2), dtype=np.int32)

        veh_f[:, VEH_CAPACITY] = df.get("capacity", 0.0).to_numpy(dtype=np.float32, copy=True)
        veh_f[:, VEH_FIXED_COST] = df.get("fixed_cost", 0.0).to_numpy(dtype=np.float32, copy=True)
        veh_f[:, VEH_MAX_DUR] = df.get("max_dur", 0.0).to_numpy(dtype=np.float32, copy=True)

        veh_i[:, 0] = df.get("start_depot", 0).to_numpy(dtype=np.int32, copy=True)
        veh_i[:, 1] = df.get("end_depot", 0).to_numpy(dtype=np.int32, copy=True)

        return veh_f, veh_i
    except ModuleNotFoundError:
        arr = _read_structured_csv(path_table)
        names = set(arr.dtype.names or ())
        if "start_depot" not in names or "end_depot" not in names:
            raise ValueError("vehicle table must contain 'start_depot' and 'end_depot'")

        m = arr.shape[0]
        veh_f = np.zeros((m, F_VEH_F), dtype=np.float32)
        veh_i = np.zeros((m, 2), dtype=np.int32)

        def _get(name: str, default: float, dtype) -> np.ndarray:
            if name in names:
                return np.asarray(arr[name], dtype=dtype)
            return np.full(m, default, dtype=dtype)

        veh_f[:, VEH_CAPACITY] = _get("capacity", 0.0, np.float32)
        veh_f[:, VEH_FIXED_COST] = _get("fixed_cost", 0.0, np.float32)
        veh_f[:, VEH_MAX_DUR] = _get("max_dur", 0.0, np.float32)

        veh_i[:, 0] = _get("start_depot", 0, np.int32)
        veh_i[:, 1] = _get("end_depot", 0, np.int32)

        return veh_f, veh_i


def load_matrices(path_npz: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load distance/time/cost matrices from an ``.npz`` archive."""

    path = Path(path_npz)
    with np.load(path) as data:
        dist = np.array(data["dist"], dtype=np.float32, copy=False)
        ttime = np.array(data.get("ttime", dist), dtype=np.float32, copy=False)

        if "edge_vec" in data:
            edge_vec = np.array(data["edge_vec"], dtype=np.float32, copy=False)
        else:
            edge_vec = dist[..., None]

        if "cost_w" in data:
            cost_w = np.array(data["cost_w"], dtype=np.float32, copy=False)
        else:
            cost_w = np.ones(edge_vec.shape[-1], dtype=np.float32)

    return dist, ttime, edge_vec, cost_w


def load_initial_npz(path_npz: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load optional initial routes/lens arrays from an ``.npz`` archive."""

    path = Path(path_npz)
    with np.load(path) as data:
        routes = np.array(data["routes"], dtype=np.int32, copy=False)
        lens = np.array(data["lens"], dtype=np.int32, copy=False)
    return routes, lens


def validate_inputs(data: Mapping[str, np.ndarray]) -> None:
    """Run lightweight shape and dtype checks on the assembled dataset."""

    coords = np.asarray(data["coords"])
    dist = np.asarray(data["dist"])
    ttime = np.asarray(data["ttime"])
    edge_vec = np.asarray(data["edge_vec"])
    cost_w = np.asarray(data["cost_w"])
    node_f = np.asarray(data["node_f"])
    node_i = np.asarray(data["node_i"])
    veh_f = np.asarray(data["veh_f"])
    veh_i = np.asarray(data["veh_i"])

    n = coords.shape[0]
    m = veh_f.shape[0]

    if coords.shape[1] != 2:
        raise ValueError("coords must have shape (n, 2)")
    if dist.shape != (n, n):
        raise ValueError("dist must have shape (n, n)")
    if ttime.shape != (n, n):
        raise ValueError("ttime must match dist dimensions")
    if edge_vec.shape[:2] != (n, n):
        raise ValueError("edge_vec must align with dist dimensions")
    if edge_vec.shape[2] != cost_w.shape[0]:
        raise ValueError("edge_vec third dim must match cost_w length")
    if node_f.shape != (n, F_NODE_F):
        raise ValueError("node_f must have shape (n, F_NODE_F)")
    if node_i.shape != (n, F_NODE_I):
        raise ValueError("node_i must have shape (n, F_NODE_I)")
    if veh_f.shape != (m, F_VEH_F):
        raise ValueError("veh_f must have shape (m, F_VEH_F)")
    if veh_i.shape != (m, 2):
        raise ValueError("veh_i must have shape (m, 2)")

    if np.any(node_i[:, NODE_REQUIRED] < 0):
        raise ValueError("node required flags must be >= 0")
    if np.any(node_i[:, NODE_PD_ROLE] < 0):
        raise ValueError("node PD roles must be >= 0")
    if np.any(node_i[:, NODE_PD_PAIR] < -1):
        raise ValueError("node PD pair indices must be >= -1")


def compute_euclid(coords: np.ndarray) -> np.ndarray:
    """Helper to compute Euclidean pairwise distances for fallback datasets."""

    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1), dtype=np.float32)


__all__ = [
    "load_config",
    "load_nodes",
    "load_vehicles",
    "load_matrices",
    "load_initial_npz",
    "validate_inputs",
    "compute_euclid",
]
