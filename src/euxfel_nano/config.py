"""Loading the YAML configuration in ``config/``.

Analysis parameters (calibration constants, selection windows, field bins,
classifier hyperparameters) live in ``config/params.yaml`` and the run lists in
``config/runs.yaml``, so a reader can see every number a figure depends on in
one place.
"""

import os
from functools import lru_cache
from pathlib import Path

import yaml


def config_dir():
    """Directory holding ``params.yaml`` and ``runs.yaml``.

    Override with the ``EUXFEL_NANO_CONFIG`` environment variable; otherwise
    resolved relative to the installed package.
    """
    env = os.environ.get("EUXFEL_NANO_CONFIG")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "config"


@lru_cache(maxsize=None)
def params():
    """Contents of ``config/params.yaml``."""
    with open(config_dir() / "params.yaml") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=None)
def runs():
    """Contents of ``config/runs.yaml``."""
    with open(config_dir() / "runs.yaml") as f:
        return yaml.safe_load(f)


def run_ids(key, transmission="1.0"):
    """Run IDs for a sample key ("500", "300", "100", "cs").

    ``transmission`` names a run group. Relative attenuator settings are given
    as bare numbers ("1.0", "0.5"); groups that are not transmission scans,
    such as the 7594 V runs behind Fig. 1c, are named in full
    ("setting1_7594V").
    """
    cfg = runs()
    groups = cfg["samples"][cfg["keys"][key]]["runs"]
    for name in (f"transmission_{transmission}", transmission):
        if name in groups:
            return groups[name]
    raise KeyError(
        f"no run group {transmission!r} for sample {key!r}; "
        f"available: {sorted(groups)}")


def measured_diameter_m(key):
    """TEM-measured diameter in metres for a sample key."""
    cfg = runs()
    sample = cfg["samples"][cfg["keys"][key]]
    return sample["measured_diameter_nm"] * 1e-9


def measured_diameters_m():
    """TEM-measured diameters for all sample keys."""
    return {k: measured_diameter_m(k) for k in runs()["keys"]}


def field_bins():
    """Surface electric field bins as a list of ``(low, high)`` tuples."""
    return [tuple(b) for b in params()["field_bins"]]


def species_windows_ns():
    """Species flight-time windows in nanoseconds, keyed by species label."""
    return {k: tuple(v) for k, v in params()["species_windows_ns"].items()}
