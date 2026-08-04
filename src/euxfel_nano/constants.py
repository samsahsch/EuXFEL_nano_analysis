"""Experimental constants for European XFEL proposal 3408 (SQS/NQS instrument).

Values here describe the machine and detector configuration and are fixed for
the whole dataset. Analysis choices that vary per figure (ion-count windows,
field bins, integration ranges) live in ``config/params.yaml`` instead.
"""

# European XFEL ran 92 pulses per train at this separation. Heavy ions reach the
# detector within microseconds, so the spacing is chosen to avoid wrap-around
# where late ions would be recorded at early times in the following pulse.
TIME_BETWEEN_PULSES = 3.54462e-6  # s

# Timepix3 digitiser channels per pulse.
CHANNELS_PER_PULSE = 14080

# Duration of one digitiser channel.
channel_time = TIME_BETWEEN_PULSES / CHANNELS_PER_PULSE

# Photon energy, just above the Si K-edge of 1.84 keV.
PHOTON_ENERGY_EV = 1880.0

# Corresponding wavelength, hc / E, used for the CDI fringe analysis.
WAVELENGTH_M = 0.66e-9

# pnCCD geometry (Kuster et al. 2021).
PNCCD_DETECTOR_DISTANCE_M = 0.346
PNCCD_PIXEL_SIZE_M = 75e-6
PNCCD_SHAPE = (1024, 1024)

# Particle diameters measured by TEM (Fig. S16), in metres. The nominal names
# (500, 300, 100 nm SiO2 and 150 nm Au@SiO2) are labels only; every electric
# field in the analysis is computed from the measured diameter.
TEM_DIAMETERS_M = {
    "500": 504e-9,
    "300": 312e-9,
    "100": 126e-9,
    "cs": 147e-9,
}

TEM_DIAMETER_ERRORS_M = {
    "500": 32e-9,
    "300": 14e-9,
    "100": 7e-9,
    "cs": 9e-9,
}
