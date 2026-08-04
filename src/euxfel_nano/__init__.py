"""Analysis code for *X-ray Driven Trihydrogen Formation on Silica Nanosurfaces*.

European XFEL proposal 3408, SQS instrument, NQS endstation.

Raw data: https://doi.org/10.22003/XFEL.EU-DATA-003408-00

Modules
-------
constants    Machine, detector and sample constants fixed for the whole dataset.
io           Reading the preprocessed per-run HDF5 files.
selection    Run, pulse and spatial event selection.
calibration  Time-of-flight to mass-over-charge calibration.
fields       Per-shot surface electric field from the electron yield.
yields       Field-binned hydrogen ion yields, bootstrap errors and fits.
binning      Binning of spectra by per-pulse ion count; waterfalls, heatmaps.
classifier   Monomer/dimer CNN for the diffraction patterns.
covariance   Covariance and partial-covariance maps.
pnccd        pnCCD dark correction, display and integration.
plotting     Diagnostic plots.
"""

__version__ = "1.0.0"

from . import (binning, calibration, config, constants, covariance, fields,
               io, plotting, pnccd, seeds, selection, yields)

# classifier imports tensorflow, so it is not pulled in by default:
#     from euxfel_nano import classifier

from .constants import (CHANNELS_PER_PULSE, TEM_DIAMETERS_M,
                        TIME_BETWEEN_PULSES, channel_time)
from .io import read, read_ion, read_ions_electrons, read_ions_photons, read_pnccd
from .selection import (ion_selection, events_selection,
                        events_selection_ions_electrons,
                        events_selection_ions_photons, select_left_of_line,
                        select_right_of_line, fish_plot_ion_selection)
from .calibration import apply_calibration, mq_of, tof_of
from .fields import build_charge_map, build_charge_maps, field_from_electron_count
from .yields import (FieldBinnedData, dec_func, get_yields, sat_func,
                     significance, significance_from_arrays)

__all__ = [
    "binning", "calibration", "config", "constants", "covariance", "fields",
    "io", "plotting", "pnccd", "seeds", "selection", "yields",
    "TIME_BETWEEN_PULSES", "CHANNELS_PER_PULSE", "channel_time",
    "TEM_DIAMETERS_M",
    "read", "read_ion", "read_ions_electrons", "read_ions_photons", "read_pnccd",
    "ion_selection", "events_selection", "events_selection_ions_electrons",
    "events_selection_ions_photons", "select_right_of_line", "select_left_of_line",
    "fish_plot_ion_selection",
    "apply_calibration", "tof_of", "mq_of",
    "build_charge_map", "build_charge_maps", "field_from_electron_count",
    "FieldBinnedData", "get_yields", "significance", "significance_from_arrays",
    "sat_func", "dec_func",
]
