"""Spatial focus regularisation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .dimension import repair_giant_tour_focus_njit, spatial_penalty


@dataclass
class SpatialFocus:
    """Maintain soft spatial focus weights for destroy/repair/LS operators."""

    coords: np.ndarray
    alpha0: float
    alpha_min: float
    gamma: float
    beta: float
    destroy_bias: float = 1.0
    ls_bias: float = 1.0

    def __post_init__(self) -> None:
        self.coords = np.asarray(self.coords, dtype=np.float32)
        if self.coords.ndim != 2 or self.coords.shape[1] != 2:
            raise ValueError("coords must have shape (n, 2)")
        self.centroid = self.coords[0].astype(np.float32, copy=True)
        self.alpha = float(self.alpha0)
        self._weights = np.zeros(self.coords.shape[0], dtype=np.float32)
        self._weights[1:] = 1.0 / max(1, self.coords.shape[0] - 1)

    @property
    def customer_weights(self) -> np.ndarray:
        return self._weights

    def route_weights(self, routes: np.ndarray, lens: np.ndarray) -> Optional[np.ndarray]:
        m = routes.shape[0]
        weights = np.zeros(m, dtype=np.float32)
        for r in range(m):
            L = int(lens[r])
            if L == 0:
                continue
            seq = routes[r, :L]
            mask = seq > 0
            if not np.any(mask):
                continue
            weights[r] = float(self._weights[seq[mask]].sum())
        if weights.sum() <= 0:
            return None
        weights /= weights.sum()
        return weights

    def _alpha_schedule(self, progress: float) -> float:
        progress = min(max(progress, 0.0), 1.0)
        scaled = (1.0 - progress) ** max(self.gamma, 1e-6)
        return max(self.alpha_min, self.alpha0 * scaled)

    def update(
        self,
        routes: np.ndarray,
        lens: np.ndarray,
        iteration: int,
        total_iters: int,
    ) -> None:
        progress = 0.0 if total_iters <= 0 else iteration / float(total_iters)
        self.alpha = self._alpha_schedule(progress)

        customers = []
        for r in range(routes.shape[0]):
            L = int(lens[r])
            if L == 0:
                continue
            seq = routes[r, :L]
            customers.extend(int(c) for c in seq if c > 0)

        if customers:
            target = self.coords[customers].mean(axis=0)
            self.centroid = (1.0 - self.beta) * self.centroid + self.beta * target

        diffs = self.coords - self.centroid
        dists = np.linalg.norm(diffs, axis=1)
        scale = max(self.alpha, 1e-6)
        weights = np.exp(-((dists / scale) ** 2))
        weights[0] = 0.0  # never focus the depot directly
        total = weights.sum()
        if total <= 0:
            weights[1:] = 1.0
            total = weights.sum()
        self._weights = weights / total

    def destroy_weight(self, customer: int) -> float:
        if customer <= 0 or customer >= self._weights.shape[0]:
            return 0.0
        return 1.0 + self.destroy_bias * self._weights[customer]

    def ls_route_weight(self, value: float) -> float:
        return (1.0 - self.ls_bias) + self.ls_bias * value

    def penalty(self, customer: int, alpha: Optional[float] = None) -> float:
        if customer <= 0 or customer >= self.coords.shape[0]:
            return 0.0
        alpha_val = self.alpha if alpha is None else float(alpha)
        return spatial_penalty(self.coords, customer, self.centroid, alpha_val)

    def repair_order(self, unrouted: np.ndarray, dist: np.ndarray) -> np.ndarray:
        if unrouted.size <= 1:
            return unrouted.copy()
        order = repair_giant_tour_focus_njit(dist, unrouted.astype(np.int32), self.coords, self.centroid, self.alpha)
        return order
