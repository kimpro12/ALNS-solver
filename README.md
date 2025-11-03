# ALNS–CVRPTWPD Solver (Numba‑friendly skeleton)

A compact, **Numba‑friendly** implementation scaffold for solving the Capacitated Vehicle Routing Problem with Time Windows and Pickup‑Delivery (CVRPTWPD) using Adaptive Large Neighborhood Search (ALNS). The core logic is structured around contiguous NumPy buffers so that key routines can be JIT‑compiled with Numba when additional performance is required.

## Highlights

- Modular destroy/repair operators with lightweight local search moves (2‑opt, relocate, swap) ready for experimentation.
- Simulated annealing acceptance and adaptive operator weighting to balance exploration/exploitation.
- Minimal‑OOP design that favours array operations and Numba compatibility; pure Python conveniences are isolated to configuration/loading code.
- Example dataset and configuration for quick trials, including hooks for Euclidean distance generation when a distance matrix is not provided.

> The repository ships with a runnable demo that executes a few thousand ALNS iterations using **Random Removal + Greedy Insertion** and **2‑opt** improvement. Extend the stubs to plug in richer constraints, scoring, or more advanced operators.

## Project layout

```
alns_cvrptwpd/      Core solver modules (operators, local search, ALNS loop, utilities)
examples/           Sample dataset and YAML configuration demonstrating inputs
tests/              Pytest-based unit tests covering operators and helpers
main.py             CLI entry point for running experiments
out/                Default location for run artifacts (created on demand)
```

## Installation

```bash
python -m venv .venv && . .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The only runtime dependencies are NumPy, Numba, and a few light utilities. GPU acceleration is not required.

## Running the toy example

```bash
# Execute the solver on the bundled toy dataset (artifacts written to out/toy_run)
python main.py \
  --config examples/toy_config.yaml \
  --outdir out/toy_run \
  --seed 0 \
  --trace
```

### Output artifacts

Each run produces a JSON summary plus several helper files inside `out/<run_name>`:

- `metrics.json` – best/current objective values and run metadata.
- `routes.csv` – flattened vehicle routes indexed by stop position.
- `metrics_log.csv` – periodic log of solution quality, temperature, and operator statistics.
- `trace.npz` – optional compressed arrays used for plotting or offline analysis (iterations, costs, temperatures).

## Configuration tips

The default configuration (`examples/toy_config.yaml`) points to CSV tables under `examples/toy_dataset/`. To run your own instances:

1. Duplicate the dataset folder and edit the coordinates, demands, time windows, and vehicle capacities as needed.
2. Update the YAML file to reference your new CSV paths and tweak operator weights, iteration counts, or penalty coefficients.
3. If you omit a precomputed distance matrix, the loader generates Euclidean distances from coordinates on the fly.

## Extending the solver

- **Operators:** Implement additional destroy/repair heuristics under `alns_cvrptwpd/operators/`. Existing modules illustrate how to keep data in contiguous arrays for `@njit` compatibility.
- **Local search:** Drop new moves in `alns_cvrptwpd/local_search/` and register them in the ALNS loop. Moves operate on shared buffers so they can be accelerated with Numba.
- **Constraints:** Use the dimension/state hooks to track custom metrics (e.g., load, ride time). Each dimension has space reserved for vector costs, with distance stored in index 0 by default.
- **Performance:** Many helper routines are already structured for Numba. When adding new kernels, prefer NumPy arrays, `numba.typed.List`, or other Numba‑supported containers and pass preallocated buffers to avoid Python overhead.

## Testing

Run the automated checks before committing changes:

```bash
pytest
```

The tests cover core operator behaviour and serve as a sanity check when experimenting with new heuristics.

## Status & next steps

- Time Window and Pickup‑Delivery columns are present in the dataset, but only lightly enforced in the demo.
- Travel times equal Euclidean distance at unit speed; plug in richer travel models by extending the distance loader.
- Contributions, issue reports, and performance tips are welcome!
