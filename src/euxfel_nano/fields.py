"""Per-shot surface electric field from the electron time-of-flight yield.

The eTOF spectrometer is used as an electron counter. Integrating its trace
over the 100-200 ns photoelectron window gives, for each shot, a signal
proportional to the number of electrons N that escaped the nanoparticle. The
particle is then treated as a uniformly charged sphere of radius R,

    E(R) = N e / (4 pi eps0 R^2)                                (SI Eq. 2)

which, with R in nm, is E = 1.44 N / R^2 V/nm.

Two systematics are absorbed into a single multiplicative constant and do not
affect the ordering of fields or the shape of the field-dependent yields: the
single-electron detector response and the eTOF collection efficiency. The
absolute field scale is instead fixed by matching to the SIMION calibration of
the Si+ kinetic energy over the common 0-2 V/nm range.

Particle diameters
------------------
:data:`euxfel_nano.constants.TEM_DIAMETERS_M` holds the TEM-measured diameters
(504, 312, 126 and 147 nm). The Supplementary Information states that the
nominal sizes are labels only and that every field is computed from these
measured values, so they are the default here. See the "TOF calibration and
particle diameters" note in the README before changing this.
"""

import numpy as np
import pandas as pd
from scipy import constants

from .constants import TEM_DIAMETERS_M, channel_time

# Flight-time window containing the photoelectron feature (Fig. S9b). The
# feature rises at ~115 ns and peaks near 140 ns; structure beyond ~400 ns is
# backscatter off the ion-detector MCP and is excluded.
ELECTRON_WINDOW_S = (1e-7, 2e-7)


def electron_window_channels(window=ELECTRON_WINDOW_S):
    """Digitiser channel bounds of the photoelectron integration window."""
    return int(window[0] / channel_time), int(window[1] / channel_time)


def field_from_electron_count(num_e, diameter_m):
    """Surface field in V/nm for ``num_e`` escaped electrons on a sphere.

    ``num_e`` is the integrated eTOF signal in digitiser units; see the module
    docstring on why the unknown detector gain only rescales the axis.
    """
    radius = diameter_m / 2
    return (np.asarray(num_e) * constants.e
            / (4 * constants.pi * constants.epsilon_0 * radius ** 2) * 1e-9)


def build_charge_map(selected_etof, bounded_dfevent, diameter_m,
                     window=ELECTRON_WINDOW_S):
    """Map each pulseId to its per-shot surface field in V/nm.

    Parameters
    ----------
    selected_etof
        eTOF DataArray for the selected shots, with a ``pulseId`` coordinate.
    bounded_dfevent
        Ion event table restricted to the analysis selection; only its
        pulseIds are used, to intersect with the eTOF shots.
    diameter_m
        Particle diameter in metres (use the TEM-measured value).
    """
    lo, hi = electron_window_channels(window)
    short = selected_etof.isel(data=slice(lo, hi))
    etof_sums = -np.sum(short, axis=1)
    counts = bounded_dfevent.pulseId.value_counts()
    ids = np.intersect1d(counts.index, etof_sums.pulseId.values)
    num_e = etof_sums.sel(pulseId=ids).values
    return pd.Series(field_from_electron_count(num_e, diameter_m), index=ids)


def build_charge_maps(etofs, event_dfs, diameters_m=None,
                      window=ELECTRON_WINDOW_S):
    """Build charge maps for every particle type at once.

    ``etofs`` and ``event_dfs`` are dicts keyed by particle type
    ("500", "300", "100", "cs"); ``diameters_m`` defaults to the TEM values.
    """
    diameters_m = TEM_DIAMETERS_M if diameters_m is None else diameters_m
    return {k: build_charge_map(etofs[k], event_dfs[k], diameters_m[k], window)
            for k in etofs}


def dimer_effective_field(field_each, separation_m, diameter_m):
    """Effective field on a dimer particle, including its neighbour.

    The two particles are treated as independent point-charge sources and the
    neighbour's field is added by linear superposition at the interparticle
    separation measured from the two-particle interference fringes. This
    neglects mutual shadowing, dielectric screening, and the altered geometry
    of fused or necked aggregates, which do not present as two-sphere
    diffraction patterns.

    Parameters
    ----------
    field_each
        Surface field of one particle, V/nm, from its own charge.
    separation_m
        Centre-to-centre separation from the diffraction fringes.
    diameter_m
        Particle diameter in metres.
    """
    radius = diameter_m / 2
    # Field of the neighbour evaluated at this particle's surface, scaled from
    # its own surface field by the inverse-square falloff.
    neighbour = np.asarray(field_each) * (radius / np.asarray(separation_m)) ** 2
    return np.asarray(field_each) + neighbour


def separation_from_fringes(n_fringes, fringe_extent_pixels,
                            wavelength_m=None, detector_distance_m=None,
                            pixel_size_m=None):
    """Centre-to-centre dimer separation from the azimuthal fringe spacing.

    Uses the small-angle far-field relation s = m*lambda*L / y, where y is the
    distance on the detector spanned by ``m`` fringes.
    """
    from .constants import (WAVELENGTH_M, PNCCD_DETECTOR_DISTANCE_M,
                            PNCCD_PIXEL_SIZE_M)
    wavelength_m = WAVELENGTH_M if wavelength_m is None else wavelength_m
    detector_distance_m = (PNCCD_DETECTOR_DISTANCE_M if detector_distance_m is None
                           else detector_distance_m)
    pixel_size_m = PNCCD_PIXEL_SIZE_M if pixel_size_m is None else pixel_size_m
    y = np.asarray(fringe_extent_pixels) * pixel_size_m
    return n_fringes * wavelength_m * detector_distance_m / y
