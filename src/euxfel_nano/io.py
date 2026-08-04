"""Reading preprocessed per-run HDF5 files.

Each run is preprocessed by ``scripts/preprocess.py`` into a single HDF5 file
containing the per-event and per-pulse tables and the eTOF / pnCCD arrays.
The directory holding those files is resolved by :func:`data_root`.
"""

import os
from pathlib import Path

import pandas as pd
import xarray as xr

from .constants import channel_time

_DEFAULT_DATA_ROOT = "../preprocess"


def data_root():
    """Directory holding the preprocessed ``datarun<ID>.h5`` files.

    Override with the ``EUXFEL_NANO_DATA`` environment variable. The default
    reproduces the layout used for the published analysis, where the notebooks
    were run one directory below the preprocessed files.
    """
    return Path(os.environ.get("EUXFEL_NANO_DATA", _DEFAULT_DATA_ROOT))


def run_path(runid):
    """Path to the preprocessed HDF5 file for ``runid``."""
    return data_root() / f"datarun{runid}.h5"


def read(runid):
    'Read the preprocessed data of run with ID runid saved in the h5 file with a corresponding name'
    'Outputs dataframes per event, per pulse, and xarrays etof, pnccd in that order'
    
    filename = run_path(runid)
    
    dfevent = pd.read_hdf(filename, 'dfevent')
    dfpulse = pd.read_hdf(filename, 'dfpulse')
    
    etof = xr.open_dataarray(filename, group="etof")
    pnccd = xr.open_dataarray(filename, group="pnccd")
    
    return dfevent, dfpulse, etof, pnccd


def read_ions_electrons(runid,tof_limit=None):
    'Read the preprocessed data of run with ID runid saved in the h5 file with a corresponding name'
    'Outputs dataframes per event, per pulse, and xarrays etof in that order'
    'tof_limit limiting electron time of flight loaded in'
    
    filename = run_path(runid)
    
    dfevent = pd.read_hdf(filename, 'dfevent')
    dfpulse = pd.read_hdf(filename, 'dfpulse')
    
    etof = xr.open_dataarray(filename, group="etof")

    if type(tof_limit) == int:
        max_coord = int(tof_limit/channel_time)
        etof = etof[:max_coord]
    
    return dfevent, dfpulse, etof


def read_ions_photons(runid):
    'Read the preprocessed data of run with ID runid saved in the h5 file with a corresponding name'
    'Outputs dataframes per event, per pulse, and xarrays pnccd in that order'
    
    filename = run_path(runid)
    
    dfevent = pd.read_hdf(filename, 'dfevent')
    dfpulse = pd.read_hdf(filename, 'dfpulse')
    
    pnccd = xr.open_dataarray(filename, group="pnccd")
    
    return dfevent, dfpulse, pnccd


def read_ion(runid):
    'Read the preprocessed data of run with ID runid saved in the h5 file with a corresponding name'
    'Outputs dataframes per event and per pulse'
    
    filename = run_path(runid)
    
    dfevent = pd.read_hdf(filename, 'dfevent')
    dfpulse = pd.read_hdf(filename, 'dfpulse')
    
    return dfevent, dfpulse


def read_pnccd(runid):
    'Read the preprocessed data of run with ID runid saved in the h5 file with a corresponding name'
    'Outputs dataframes per event, per pulse, and xarrays etof, pnccd in that order'
    
    filename = run_path(runid)
    pnccd = xr.open_dataarray(filename, group="pnccd")
    
    return pnccd
