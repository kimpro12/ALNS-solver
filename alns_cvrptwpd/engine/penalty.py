import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - fallback when numba is unavailable
    def njit(*args, **kwargs):
        if args and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(func):
            return func

        return decorator

from ..config.enums import F_PEN


@njit(cache=True)
def _violation_flags(vec, tol):
    flags = np.zeros(vec.shape[0], dtype=np.float32)
    for i in range(vec.shape[0]):
        flags[i] = 1.0 if vec[i] > tol[i] else 0.0
    return flags


def _history_values(state, idx):
    size = state["history"].shape[1]
    count = int(state["hist_count"][idx])
    if count <= 0:
        return np.empty(0, dtype=np.float32)
    buf = state["history"][idx]
    if count < size:
        return buf[:count]
    start = int(state["hist_idx"][idx])
    if start == 0:
        return buf.copy()
    return np.concatenate((buf[start:], buf[:start]))


def init_penalty_state(params):
    base = np.array(
        [
            float(params.get("penalty_weight_cap", 25.0)),
            float(params.get("penalty_weight_dur", 10.0)),
            float(params.get("penalty_weight_tw", 5.0)),
            float(params.get("penalty_weight_uns", 500.0)),
        ],
        dtype=np.float32,
    )
    min_w = float(params.get("penalty_weight_min", 1e-4))
    max_w = float(params.get("penalty_weight_max", 1e6))
    history_size = int(params.get("penalty_history_size", 128))
    violation_alpha = float(params.get("penalty_violation_alpha", 0.05))
    update_rate = float(params.get("penalty_update_rate", 0.1))
    target = float(params.get("penalty_target_rate", 0.05))
    scale_floor = float(params.get("penalty_scale_floor", 1.0))
    tol = float(params.get("penalty_tolerance", 1e-9))

    state = {
        "weights": base.copy(),
        "base": base,
        "min": min_w,
        "max": max_w,
        "history": np.zeros((F_PEN, history_size), dtype=np.float32),
        "hist_idx": np.zeros(F_PEN, dtype=np.int32),
        "hist_count": np.zeros(F_PEN, dtype=np.int32),
        "viol_ema": np.zeros(F_PEN, dtype=np.float32),
        "violation_alpha": violation_alpha,
        "update_rate": update_rate,
        "target": np.full(F_PEN, target, dtype=np.float32),
        "scale": np.full(F_PEN, scale_floor, dtype=np.float32),
        "scale_floor": scale_floor,
        "tol": np.full(F_PEN, tol, dtype=np.float32),
    }
    return state


def penalty_cost(state, pen_vec):
    pen_vec = np.asarray(pen_vec, dtype=np.float32)
    return float(np.dot(state["weights"], pen_vec))


def update_penalty_state(state, pen_vec):
    pen_vec = np.asarray(pen_vec, dtype=np.float32)
    if pen_vec.shape[0] != F_PEN:
        raise ValueError("penalty vector dimension mismatch")

    history = state["history"]
    hist_idx = state["hist_idx"]
    hist_count = state["hist_count"]
    size = history.shape[1]
    for i in range(F_PEN):
        pos = int(hist_idx[i])
        history[i, pos] = pen_vec[i]
        hist_idx[i] = (pos + 1) % size
        if hist_count[i] < size:
            hist_count[i] += 1

    flags = _violation_flags(pen_vec, state["tol"])  # type: ignore[arg-type]
    alpha = state["violation_alpha"]
    state["viol_ema"] = (1.0 - alpha) * state["viol_ema"] + alpha * flags

    update_rate = state["update_rate"]
    scale_floor = state["scale_floor"]
    min_w = state["min"]
    max_w = state["max"]

    for i in range(F_PEN):
        values = _history_values(state, i)
        if values.size == 0:
            scale = scale_floor
        else:
            q95 = float(np.quantile(values, 0.95))
            q50 = float(np.quantile(values, 0.5))
            scale = max(scale_floor, q95 if q95 > 0.0 else q50 if q50 > 0.0 else scale_floor)
        state["scale"][i] = scale
        adjust = 1.0 + update_rate * (state["viol_ema"][i] - state["target"][i])
        if adjust < 0.1:
            adjust = 0.1
        weight = state["base"][i] * adjust / max(scale, scale_floor)
        if weight < min_w:
            weight = min_w
        if weight > max_w:
            weight = max_w
        state["weights"][i] = weight


def snapshot_penalty_state(state):
    return {
        "weights": state["weights"].copy(),
        "violations": state["viol_ema"].copy(),
        "scale": state["scale"].copy(),
    }
