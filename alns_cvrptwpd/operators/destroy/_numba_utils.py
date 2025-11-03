"""Shared Numba-accelerated utilities for destroy operators."""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def collect_route_positions(routes, lens, route_indices, pos_indices, cust_indices):
    """Populate arrays with the positions of all active customers.

    Parameters
    ----------
    routes : ndarray
        Route matrix ``(m, L_max)``.
    lens : ndarray
        Current route lengths per vehicle.
    route_indices, pos_indices, cust_indices : ndarray
        Output buffers that must be large enough to hold every active customer.

    Returns
    -------
    int
        Number of populated entries in the output buffers.
    """

    count = 0
    m = routes.shape[0]
    for r in range(m):
        L = int(lens[r])
        for i in range(L):
            c = int(routes[r, i])
            if c > 0:
                route_indices[count] = r
                pos_indices[count] = i
                cust_indices[count] = c
                count += 1
    return count
