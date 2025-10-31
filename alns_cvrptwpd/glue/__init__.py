"""Glue helpers exposed for CLI and integration harnesses."""

from .io import (
    compute_euclid,
    load_config,
    load_initial_npz,
    load_matrices,
    load_nodes,
    load_vehicles,
    validate_inputs,
)
from .pipeline import (
    assemble_data,
    build_arg_parser,
    build_params,
    load_and_run,
    main,
    run_pipeline,
)

__all__ = [
    "assemble_data",
    "build_arg_parser",
    "build_params",
    "compute_euclid",
    "load_and_run",
    "load_config",
    "load_initial_npz",
    "load_matrices",
    "load_nodes",
    "load_vehicles",
    "main",
    "run_pipeline",
    "validate_inputs",
]