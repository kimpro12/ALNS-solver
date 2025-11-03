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

# Run the solver on the bundled toy dataset (outputs are written to out/toy_run)
python main.py \
  --config examples/toy_config.yaml \
  --outdir out/toy_run \
  --seed 0 \
  --trace
```

You will receive a JSON summary plus three artifact files in the output directory:

- `metrics.json` – best/current cost & parameters used for the run
- `routes.csv` – best solution routes (vehicle x stop indices)
- `metrics_log.csv` – iteration log sampled every `log_period`
- `trace.npz` – optional lightweight arrays (iterations, costs, temperature)

> The configuration file references the sample CSV dataset stored under
> `examples/toy_dataset/`. You can copy the folder, adjust coordinates/demands or
> vehicle capacities, and point a new config file at your modified tables. If you
> omit the optional distance matrix in your own configs, the pipeline will compute
> Euclidean distances from the coordinates on the fly.

## Notes

- Time Windows (TW) & Pickup‑Delivery (PD) are scaffolded (columns exist) but enforcement is **minimal** in this demo.
- Core is written in a way that you can add `@njit` later (Numba optional). When you do, pass pre‑allocated buffers.
- Distances are Euclidean; travel time is distance / 1.0 (unit speed) for simplicity.
