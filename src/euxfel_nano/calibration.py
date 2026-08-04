"""Time-of-flight to mass-over-charge calibration.

Two calibrations appear in this analysis; see ``config/params.yaml`` and the
note in the README.

* :func:`tof_of` / :func:`mq_of` implement Eq. 1 of the Supplementary
  Information, ``tof = t0 + k*sqrt(m/q)``, with the exponent held at its
  physical value of 2. Parameters are tabulated per voltage setting.
* :func:`power_law` / :func:`apply_calibration` implement the earlier
  unconstrained fit ``m/q = a * tof**b``, in which the nonzero ``t0`` is
  absorbed into a free exponent (returning b = 2.013 and 2.018 for the two
  voltage settings). Retained because it produced the published Fig. 1c m/q
  axis.
"""

import numpy as np
from scipy.optimize import curve_fit

from .plotting import big_ion_tof, zoomed_ion_tof

# Eq. 1 parameters per voltage setting (Supplementary Information).
VOLTAGE_SETTINGS = {
    1: {"t0": 7.45113e-9, "k": 2.06479e-7},   # Fig. 1c
    2: {"t0": 6.82e-9, "k": 2.450e-7},        # all remaining figures
}

# m/q of the argon charge states used as calibration anchors (Fig. S4).
ARGON_LINES_MQ = [40, 20, 40 / 3, 40 / 4, 40 / 5]


def tof_of(mq, setting=1):
    """Flight time (s) for a given m/q, via SI Eq. 1."""
    p = VOLTAGE_SETTINGS[setting]
    return p["t0"] + p["k"] * np.sqrt(mq)


def mq_of(tof, setting=1):
    """m/q for a given flight time (s), the inverse of :func:`tof_of`."""
    p = VOLTAGE_SETTINGS[setting]
    return ((np.asarray(tof) - p["t0"]) / p["k"]) ** 2


def fit_eq1(calibration_lines, mq_lines=None):
    """Fit t0 and k of SI Eq. 1 to measured argon anchor flight times."""
    mq_lines = ARGON_LINES_MQ if mq_lines is None else mq_lines
    (t0, k), _ = curve_fit(
        lambda mq, t0, k: t0 + k * np.sqrt(mq),
        np.asarray(mq_lines, dtype=float),
        np.asarray(calibration_lines, dtype=float),
        p0=[7e-9, 2e-7],
        maxfev=10000,
    )
    return t0, k


def apply_calibration_eq1(dfevents, setting=1):
    """Add an ``mq`` column to each dataframe using SI Eq. 1."""
    out = []
    for dfevent in dfevents:
        dfevent["mq"] = mq_of(dfevent.tof.to_numpy(), setting)
        out.append(dfevent)
    return out


def power_law(x, a, b):
    'Calibration fit power law'
    return a * x**b


def compute_calibration(calibration_lines):

    # Corresponding m/q argon values
    mq_lines = [40,20,40/3,40/4,40/5]
    
    # Initial guesses for parameters a and b
    initial_guess = [1.6e13, 2]#[1.9e20, 3]
    
    # Perform the curve fitting
    params, covariance = curve_fit(power_law, calibration_lines, mq_lines, p0=initial_guess, maxfev=10000)

    # Extract the fitted values for a and b
    a_fit, b_fit = params

    print(f"The fit looks as follows: m/q = {a_fit:.2e} * tof^{b_fit:.2f}")
    
    return a_fit, b_fit


def calibrate(backgrd_dfevent):
    'Computes calibration by least mean squares using backgrd_dfevent'
    'Uses user input to compute fit based on displayed plots'
    
    # Show a large widget ion tof
    big_ion_tof(backgrd_dfevent)

    done = False
    while not done:

        # Ask for five numbers input
        anchors = []
        for i in range(5):
            value = input(f"Enter value Ar{i + 1}: ")
            try:
                anchors.append(float(value))
            except ValueError:
                print("Invalid input. Please enter a number.")

        # Show five additional plots based on inputs
        for anchor in anchors:
            zoomed_ion_tof(backgrd_dfevent,anchor)

        # Ask if the user is done
        done_response = input("Are you done? (y/n): ").strip().lower()
        if done_response == 'y':
            done = True
    
    # Compute calibration fit
    a_fit, b_fit = compute_calibration(anchors)
    
    return a_fit, b_fit


def apply_calibration(dfevents,a_fit,b_fit):
    'Applies calibration to each dfevent of the list of dfevents and outputs calibrated_dfevents list of dataframes with m/q column'
        
    calibrated_dfevents = list()
        
    for dfevent in dfevents:
        dfevent['mq'] = a_fit * dfevent.tof ** b_fit
        calibrated_dfevents.append(dfevent)
        
    return calibrated_dfevents
