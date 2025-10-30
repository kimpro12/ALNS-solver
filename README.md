# ALNS–CVRPTWPD Project (Skeleton, Numba‑friendly)

A compact, **Numba‑friendly** project skeleton for solving CVRPTWPD with Adaptive Large Neighborhood Search (ALNS).
Core code avoids OOP; glue/config may use Python conveniences. Designed to match the structure in your spec:
- Destroy/Repair operators
- Local Search (2‑opt, relocate, swap; lightweight demo implemented)
- ALNS loop with SA‑acceptance and simple adaptive weights scaffold
- Dimension/State update hooks ready for vector cost (distance is dim 0)
- Logging & simple metrics

> This is a runnable demo that generates a toy dataset and performs a few thousand ALNS iterations with
> **Random Removal + Greedy Insertion** and **2‑opt local search**. You can extend operators and constraints
> following the provided stubs.

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
python main.py --seed 0 --n_customers 80 --n_vehicles 6 --iters 5000
```

You’ll see a progress print every 500 iters and a final summary.
Artifacts:
- `out/metrics.json` — best/current cost & stats
- `out/routes.csv` — best solution routes
- `out/log.csv`     — iteration logs (subsampled)

## Notes

- Time Windows (TW) & Pickup‑Delivery (PD) are scaffolded (columns exist) but enforcement is **minimal** in this demo.
- Core is written in a way that you can add `@njit` later (Numba optional). When you do, pass pre‑allocated buffers.
- Distances are Euclidean; travel time is distance / 1.0 (unit speed) for simplicity.
