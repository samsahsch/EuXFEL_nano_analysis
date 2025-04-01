### Preprocessing script



# Disclaimers:
# - Need approximately 300 GB of memory per run
# - Will not work if timepix crashed at some point, that is if two or more trains are missing at the beginning or end



# Run as follows:

# RUNID = 390
# %run ./preprocessing.ipynb



# For multiple runs:

# for i in range(390,395):
#     RUNID = i
#     %run ./preprocessing.ipynb
    
    
    
# If not installed yet, need to install saving methods the first time:

# pip install --user tables   ### to save dataframe
# pip install h5netcdf        ### to save xarrays



# To read preprocessed files :

# dfevent, dfpulse, etof, pnccd = read(RUNID)

# dfevent and dfpulse are dataframes per event and per pulse respectively
# etof and pnccd are xarrays for etof data and pnccd data respectively



# Identifiers specified as follows:

# trainid = aaa0bbbb
# pulseid = aaa0bbbb0cc
# aaa is runid, bbbb is train number, cc is pulse number




### Variables


#RUNID = 405 # given when running file

T0 = 7.899e-5                       # time before the first ions hits
TIME_BETWEEN_PULSES = 3.54462e-6    # timing between two pulses
DIGITIZER_OFFSET = 22115            # number of channels before actual data, found with photon peak runs 351,352
DIGITIZER_END = int(1.6e6)          # number of channels after which only noise is present
DIGITIZER_NOISE_PERIODICITY = 16    # noise periodicity?


#pip install --user tables ### to save dataframe

import numpy as np
import pandas as pd
import xarray as xr
import math
from matplotlib import pyplot as plt
from extra_data import open_run
import h5py
import os
import time

print('Opening RUN' + str(RUNID) + '...')
start_time = time.time()

run = open_run(proposal=3408, run=RUNID)