"""Run, pulse and spatial event selection on the ion VMI detector.

``ion_selection`` and its variants read one or more runs and keep the pulses
whose total ion count falls inside a threshold window; the spatial helpers cut
on detector coordinates, including the tilted regions and the dividing line
used to separate the two reaction mechanisms (Fig. S5).
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from .io import read, read_ion, read_ions_electrons, read_ions_photons
from .plotting import heatmap_with_zones


def events_selection(runs,thresholds,num_pulses=None):
    'Reads one or multiple runs from h5 files'
    'Makes a pulse selection based on the number of events per pulse between the defined thresholds'
    'If multiple runs are passed, will merge the runs, once hit selected'
    'Thresholds can be between one and three tuples (lower threshold, upper threshold)'
    'Downsamples by num_pulses'
    
    lower_threshold1, upper_threshold1 = thresholds[0]
    selected_dfevents1 = list()
    selected_dfpulses1 = list()
    selected_etofs1 = list()
    selected_pnccds1 = list()
    
    if len(thresholds) > 1:
        lower_threshold2, upper_threshold2 = thresholds[1]
        selected_dfevents2 = list()
        selected_dfpulses2 = list()
        selected_etofs2 = list()
        selected_pnccds2 = list()
    
    if len(thresholds) > 2:
        lower_threshold3, upper_threshold3 = thresholds[2]  
        selected_dfevents3 = list()
        selected_dfpulses3 = list()
        selected_etofs3 = list()
        selected_pnccds3 = list()
    
    dataframes = dict()
    
    if type(num_pulses) == int:
        num_pulses_run = int(num_pulses/len(runs))
    
    for run in runs:
        
        dfevent, dfpulse, etof, pnccd = read(run)
        
        selections = list()
        
        plt.figure()
        plt.scatter(dfpulse.pulseId,dfpulse.nevents_pulse,c='black',label='All pulses')
        
        if type(num_pulses) == int:
            dfpulse = dfpulse.sample(n=num_pulses_run)
            
        selected_dfpulse1 = dfpulse[lower_threshold1 < dfpulse.nevents_pulse][dfpulse.nevents_pulse < upper_threshold1]
        selected_dfevent1 = dfevent[dfevent.pulseId.isin(selected_dfpulse1.pulseId)]
        selected_etof1 = etof.sel(pulseId=etof.coords['pulseId'].isin(selected_dfpulse1.pulseId))
        selected_pnccd1 = pnccd.sel(trainId=pnccd.coords['trainId'].isin(selected_dfpulse1.trainId))
        selections.append((selected_dfevent1, selected_dfpulse1, selected_etof1, selected_pnccd1))
        plt.scatter(selected_dfpulse1.pulseId,selected_dfpulse1.nevents_pulse,c='r',label=f'Between {lower_threshold1} and {upper_threshold1}')
        
        if len(thresholds) > 1:
            
            selected_dfpulse2 = dfpulse[lower_threshold2 < dfpulse.nevents_pulse][dfpulse.nevents_pulse < upper_threshold2]
            selected_dfevent2 = dfevent[dfevent.pulseId.isin(selected_dfpulse2.pulseId)]
            selected_etof2 = etof.sel(pulseId=etof.coords['pulseId'].isin(selected_dfpulse2.pulseId))
            selected_pnccd2 = pnccd.sel(trainId=pnccd.coords['trainId'].isin(selected_dfpulse2.trainId))
            selections.append((selected_dfevent2, selected_dfpulse2, selected_etof2, selected_pnccd2))
            plt.scatter(selected_dfpulse2.pulseId,selected_dfpulse2.nevents_pulse,c='blue',label=f'Between {lower_threshold2} and {upper_threshold2}')
            
        if len(thresholds) > 2:
            
            selected_dfpulse3 = dfpulse[lower_threshold3 < dfpulse.nevents_pulse][dfpulse.nevents_pulse < upper_threshold3]
            selected_dfevent3 = dfevent[dfevent.pulseId.isin(selected_dfpulse3.pulseId)]
            selected_etof3 = etof.sel(pulseId=etof.coords['pulseId'].isin(selected_dfpulse3.pulseId))
            selected_pnccd3 = pnccd.sel(trainId=pnccd.coords['trainId'].isin(selected_dfpulse3.trainId))
            selections.append((selected_dfevent3, selected_dfpulse3, selected_etof3, selected_pnccd3))
            plt.scatter(selected_dfpulse3.pulseId,selected_dfpulse3.nevents_pulse,c='g',label=f'Between {lower_threshold3} and {upper_threshold3}')  
        
        dataframes[run] = selections
          
        plt.xlabel('Pulse ID')
        plt.ylabel('Number of events per pulse')
        plt.legend()
        plt.title(f'Events per pulse with respect to pulse ID for run {run}')
        plt.show()

        
    for key, values in dataframes.items():
        
        selected_dfevents1.append(values[0][0])
        selected_dfpulses1.append(values[0][1])
        selected_etofs1.append(values[0][2])
        selected_pnccds1.append(values[0][3])
        
        if len(thresholds) > 1:
            selected_dfevents2.append(values[1][0])
            selected_dfpulses2.append(values[1][1])
            selected_etofs2.append(values[1][2])
            selected_pnccds2.append(values[1][3])
            
        if len(thresholds) > 2: 
            selected_dfevents3.append(values[2][0])
            selected_dfpulses3.append(values[2][1])
            selected_etofs3.append(values[2][2])
            selected_pnccds3.append(values[2][3])
        
        
    merged_selection = list()
    
    merged_dfevent1 = pd.concat(selected_dfevents1)
    merged_dfevent1.reset_index(drop=True, inplace=True)
    
    merged_dfpulse1 = pd.concat(selected_dfpulses1)
    merged_dfpulse1.reset_index(drop=True, inplace=True)
    
    merged_etof1 = xr.concat(selected_etofs1, dim='pulseId')
    merged_pnccd1 = xr.concat(selected_pnccds1, dim='pulseId')
    
    merged_selection.append((merged_dfevent1, merged_dfpulse1, merged_etof1, merged_pnccd1))
    
    print(f"Number of pulses selected across {len(runs)} run(s) between {lower_threshold1} and {upper_threshold1} events: {len(merged_dfpulse1)}")
       
        
    if len(thresholds) > 1:
        merged_dfevent2 = pd.concat(selected_dfevents2)
        merged_dfevent2.reset_index(drop=True, inplace=True)
    
        merged_dfpulse2 = pd.concat(selected_dfpulses2)
        merged_dfpulse2.reset_index(drop=True, inplace=True)
    
        merged_etof2 = xr.concat(selected_etofs2, dim='pulseId')
        merged_pnccd2 = xr.concat(selected_pnccds2, dim='pulseId')
        
        merged_selection.append((merged_dfevent2, merged_dfpulse2, merged_etof2, merged_pnccd2))
        
        print(f"Number of pulses selected across {len(runs)} run(s) between {lower_threshold2} and {upper_threshold2} events: {len(merged_dfpulse2)}")
    
    
    if len(thresholds) > 2:
        merged_dfevent3 = pd.concat(selected_dfevents3)
        merged_dfevent3.reset_index(drop=True, inplace=True)
    
        merged_dfpulse3 = pd.concat(selected_dfpulses3)
        merged_dfpulse3.reset_index(drop=True, inplace=True)
    
        merged_etof3 = xr.concat(selected_etofs3, dim='pulseId')
        merged_pnccd3 = xr.concat(selected_pnccds3, dim='pulseId')
        
        merged_selection.append((merged_dfevent3, merged_dfpulse3, merged_etof3, merged_pnccd3))
        
        print(f"Number of pulses selected across {len(runs)} run(s) between {lower_threshold3} and {upper_threshold3} events: {len(merged_dfpulse3)}")
        
    
    return merged_selection


def events_selection_ions_electrons(runs,thresholds,num_pulses=None,tof_limit=None):
    'Reads one or multiple runs from h5 files'
    'Makes a pulse selection based on the number of events per pulse between the defined thresholds'
    'If multiple runs are passed, will merge the runs, once hit selected'
    'Thresholds can be between one and three tuples (lower threshold, upper threshold)'
    'Downsamples by num_pulses'
    
    lower_threshold1, upper_threshold1 = thresholds[0]
    selected_dfevents1 = list()
    selected_dfpulses1 = list()
    selected_etofs1 = list()
    
    if len(thresholds) > 1:
        lower_threshold2, upper_threshold2 = thresholds[1]
        selected_dfevents2 = list()
        selected_dfpulses2 = list()
        selected_etofs2 = list()
    
    if len(thresholds) > 2:
        lower_threshold3, upper_threshold3 = thresholds[2]  
        selected_dfevents3 = list()
        selected_dfpulses3 = list()
        selected_etofs3 = list()
    
    dataframes = dict()
    
    if type(num_pulses) == int:
        num_pulses_run = int(num_pulses/len(runs))
    
    for run in runs:
        
        print('Processing run number',run)
        
        dfevent, dfpulse, etof = read_ions_electrons(run,tof_limit)
        
        selections = list()
        
        if type(num_pulses) == int:
            dfpulse = dfpulse.sample(n=num_pulses_run)
            
        selected_dfpulse1 = dfpulse[lower_threshold1 < dfpulse.nevents_pulse][dfpulse.nevents_pulse < upper_threshold1]
        selected_dfevent1 = dfevent[dfevent.pulseId.isin(selected_dfpulse1.pulseId)]
        selected_etof1 = etof.sel(pulseId=etof.coords['pulseId'].isin(selected_dfpulse1.pulseId))
        selections.append((selected_dfevent1, selected_dfpulse1, selected_etof1))
        plt.scatter(selected_dfpulse1.pulseId,selected_dfpulse1.nevents_pulse,c='r',label=f'Between {lower_threshold1} and {upper_threshold1}')
        
        if len(thresholds) > 1:
            
            selected_dfpulse2 = dfpulse[lower_threshold2 < dfpulse.nevents_pulse][dfpulse.nevents_pulse < upper_threshold2]
            selected_dfevent2 = dfevent[dfevent.pulseId.isin(selected_dfpulse2.pulseId)]
            selected_etof2 = etof.sel(pulseId=etof.coords['pulseId'].isin(selected_dfpulse2.pulseId))
            selections.append((selected_dfevent2, selected_dfpulse2, selected_etof2))
            plt.scatter(selected_dfpulse2.pulseId,selected_dfpulse2.nevents_pulse,c='blue',label=f'Between {lower_threshold2} and {upper_threshold2}')
            
        if len(thresholds) > 2:
            
            selected_dfpulse3 = dfpulse[lower_threshold3 < dfpulse.nevents_pulse][dfpulse.nevents_pulse < upper_threshold3]
            selected_dfevent3 = dfevent[dfevent.pulseId.isin(selected_dfpulse3.pulseId)]
            selected_etof3 = etof.sel(pulseId=etof.coords['pulseId'].isin(selected_dfpulse3.pulseId))
            selections.append((selected_dfevent3, selected_dfpulse3, selected_etof3))
            plt.scatter(selected_dfpulse3.pulseId,selected_dfpulse3.nevents_pulse,c='g',label=f'Between {lower_threshold3} and {upper_threshold3}')  
        
        dataframes[run] = selections
        
    for key, values in dataframes.items():
        
        selected_dfevents1.append(values[0][0])
        selected_dfpulses1.append(values[0][1])
        selected_etofs1.append(values[0][2])
        
        if len(thresholds) > 1:
            selected_dfevents2.append(values[1][0])
            selected_dfpulses2.append(values[1][1])
            selected_etofs2.append(values[1][2])
            
        if len(thresholds) > 2: 
            selected_dfevents3.append(values[2][0])
            selected_dfpulses3.append(values[2][1])
            selected_etofs3.append(values[2][2])
        
    merged_selection = list()
    
    merged_dfevent1 = pd.concat(selected_dfevents1)
    merged_dfevent1.reset_index(drop=True, inplace=True)
    
    merged_dfpulse1 = pd.concat(selected_dfpulses1)
    merged_dfpulse1.reset_index(drop=True, inplace=True)
    
    merged_etof1 = xr.concat(selected_etofs1, dim='pulseId')
    
    merged_selection.append((merged_dfevent1, merged_dfpulse1, merged_etof1))
    
    print(f"Number of pulses selected across {len(runs)} run(s) between {lower_threshold1} and {upper_threshold1} events: {len(merged_dfpulse1)}")
           
    if len(thresholds) > 1:
        merged_dfevent2 = pd.concat(selected_dfevents2)
        merged_dfevent2.reset_index(drop=True, inplace=True)
    
        merged_dfpulse2 = pd.concat(selected_dfpulses2)
        merged_dfpulse2.reset_index(drop=True, inplace=True)
    
        merged_etof2 = xr.concat(selected_etofs2, dim='pulseId')
        
        merged_selection.append((merged_dfevent2, merged_dfpulse2, merged_etof2))
        
        print(f"Number of pulses selected across {len(runs)} run(s) between {lower_threshold2} and {upper_threshold2} events: {len(merged_dfpulse2)}") 
    
    if len(thresholds) > 2:
        merged_dfevent3 = pd.concat(selected_dfevents3)
        merged_dfevent3.reset_index(drop=True, inplace=True)
    
        merged_dfpulse3 = pd.concat(selected_dfpulses3)
        merged_dfpulse3.reset_index(drop=True, inplace=True)
    
        merged_etof3 = xr.concat(selected_etofs3, dim='pulseId')
        
        merged_selection.append((merged_dfevent3, merged_dfpulse3, merged_etof3))
        
        print(f"Number of pulses selected across {len(runs)} run(s) between {lower_threshold3} and {upper_threshold3} events: {len(merged_dfpulse3)}")
        
    return merged_selection


def events_selection_ions_photons(runs,thresholds,num_pulses=None):
    'Reads one or multiple runs from h5 files'
    'Makes a pulse selection based on the number of events per pulse between the defined thresholds'
    'If multiple runs are passed, will merge the runs, once hit selected'
    'Thresholds can be between one and three tuples (lower threshold, upper threshold)'
    'Downsamples by num_pulses'
    
    lower_threshold1, upper_threshold1 = thresholds[0]
    selected_dfevents1 = list()
    selected_dfpulses1 = list()
    selected_pnccds1 = list()
    
    if len(thresholds) > 1:
        lower_threshold2, upper_threshold2 = thresholds[1]
        selected_dfevents2 = list()
        selected_dfpulses2 = list()
        selected_pnccds2 = list()
    
    if len(thresholds) > 2:
        lower_threshold3, upper_threshold3 = thresholds[2]  
        selected_dfevents3 = list()
        selected_dfpulses3 = list()
        selected_pnccds3 = list()
    
    dataframes = dict()
    
    if type(num_pulses) == int:
        num_pulses_run = int(num_pulses/len(runs))
    
    for run in runs:
        
        print('Processing run number',run)
        
        dfevent, dfpulse, pnccd = read_ions_photons(run)
        
        selections = list()
        
        if type(num_pulses) == int:
            dfpulse = dfpulse.sample(n=num_pulses_run)
            
        selected_dfpulse1 = dfpulse[lower_threshold1 < dfpulse.nevents_pulse][dfpulse.nevents_pulse < upper_threshold1]
        selected_dfevent1 = dfevent[dfevent.pulseId.isin(selected_dfpulse1.pulseId)]
        selected_pnccd1 = pnccd.sel(trainId=pnccd.coords['trainId'].isin(selected_dfpulse1.trainId))
        selections.append((selected_dfevent1, selected_dfpulse1, selected_pnccd1))
        
        if len(thresholds) > 1:
            
            selected_dfpulse2 = dfpulse[lower_threshold2 < dfpulse.nevents_pulse][dfpulse.nevents_pulse < upper_threshold2]
            selected_dfevent2 = dfevent[dfevent.pulseId.isin(selected_dfpulse2.pulseId)]
            selected_pnccd2 = pnccd.sel(trainId=pnccd.coords['trainId'].isin(selected_dfpulse2.trainId))
            selections.append((selected_dfevent2, selected_dfpulse2, selected_pnccd2))
            
        if len(thresholds) > 2:
            
            selected_dfpulse3 = dfpulse[lower_threshold3 < dfpulse.nevents_pulse][dfpulse.nevents_pulse < upper_threshold3]
            selected_dfevent3 = dfevent[dfevent.pulseId.isin(selected_dfpulse3.pulseId)]
            selected_pnccd3 = pnccd.sel(trainId=pnccd.coords['trainId'].isin(selected_dfpulse3.trainId))
            selections.append((selected_dfevent3, selected_dfpulse3, selected_pnccd3))
            
        dataframes[run] = selections
        
    for key, values in dataframes.items():
        
        selected_dfevents1.append(values[0][0])
        selected_dfpulses1.append(values[0][1])
        selected_pnccds1.append(values[0][2])
        
        if len(thresholds) > 1:
            selected_dfevents2.append(values[1][0])
            selected_dfpulses2.append(values[1][1])
            selected_pnccds2.append(values[1][2])
            
        if len(thresholds) > 2: 
            selected_dfevents3.append(values[2][0])
            selected_dfpulses3.append(values[2][1])
            selected_pnccds3.append(values[2][2])
        
        
    merged_selection = list()
    
    merged_dfevent1 = pd.concat(selected_dfevents1)
    merged_dfevent1.reset_index(drop=True, inplace=True)
    
    merged_dfpulse1 = pd.concat(selected_dfpulses1)
    merged_dfpulse1.reset_index(drop=True, inplace=True)
    
    merged_pnccd1 = xr.concat(selected_pnccds1, dim='trainId')
    
    merged_selection.append((merged_dfevent1, merged_dfpulse1, merged_pnccd1))
    
    print(f"Number of pulses selected across {len(runs)} run(s) between {lower_threshold1} and {upper_threshold1} events: {len(merged_dfpulse1)}")
       
        
    if len(thresholds) > 1:
        merged_dfevent2 = pd.concat(selected_dfevents2)
        merged_dfevent2.reset_index(drop=True, inplace=True)
    
        merged_dfpulse2 = pd.concat(selected_dfpulses2)
        merged_dfpulse2.reset_index(drop=True, inplace=True)
    
        merged_pnccd2 = xr.concat(selected_pnccds2, dim='trainId')
        
        merged_selection.append((merged_dfevent2, merged_dfpulse2, merged_pnccd2))
        
        print(f"Number of pulses selected across {len(runs)} run(s) between {lower_threshold2} and {upper_threshold2} events: {len(merged_dfpulse2)}")
    
    
    if len(thresholds) > 2:
        merged_dfevent3 = pd.concat(selected_dfevents3)
        merged_dfevent3.reset_index(drop=True, inplace=True)
    
        merged_dfpulse3 = pd.concat(selected_dfpulses3)
        merged_dfpulse3.reset_index(drop=True, inplace=True)
    
        merged_pnccd3 = xr.concat(selected_pnccds3, dim='trainId')
        
        merged_selection.append((merged_dfevent3, merged_dfpulse3, merged_pnccd3))
        
        print(f"Number of pulses selected across {len(runs)} run(s) between {lower_threshold3} and {upper_threshold3} events: {len(merged_dfpulse3)}")

    
    return merged_selection


def ion_selection(runs,thresholds,num_pulses=None):
    'Only handles ion data'
    'Reads one or multiple runs from h5 files'
    'Makes a pulse selection based on the number of events per pulse between the defined thresholds'
    'If multiple runs are passed, will merge the runs, once hit selected'
    'Thresholds can be between one and three tuples (lower threshold, upper threshold)'
    'Downsamples by num_pulses'
    
    lower_threshold1, upper_threshold1 = thresholds[0]
    selected_dfevents1 = list()
    selected_dfpulses1 = list()
    
    if len(thresholds) > 1:
        lower_threshold2, upper_threshold2 = thresholds[1]
        selected_dfevents2 = list()
        selected_dfpulses2 = list()
    
    if len(thresholds) > 2:
        lower_threshold3, upper_threshold3 = thresholds[2]  
        selected_dfevents3 = list()
        selected_dfpulses3 = list()
    
    dataframes = dict()
    
    if type(num_pulses) == int:
        num_pulses_run = int(num_pulses/len(runs))
    
    for run in runs:
        
        print('Handling run', run)
        dfevent, dfpulse = read_ion(run)
        
        selections = list()
        # plt.figure()
        # plt.scatter(dfpulse.pulseId,dfpulse.nevents_pulse,c='black',label='All pulses')
        
        if type(num_pulses) == int:
            dfpulse = dfpulse.sample(n=num_pulses_run)
            
        selected_dfpulse1 = dfpulse[lower_threshold1 < dfpulse.nevents_pulse][dfpulse.nevents_pulse < upper_threshold1]
        selected_dfevent1 = dfevent[dfevent.pulseId.isin(selected_dfpulse1.pulseId)]
        selections.append((selected_dfevent1, selected_dfpulse1))
        
        if len(thresholds) > 1:
            
            selected_dfpulse2 = dfpulse[lower_threshold2 < dfpulse.nevents_pulse][dfpulse.nevents_pulse < upper_threshold2]
            selected_dfevent2 = dfevent[dfevent.pulseId.isin(selected_dfpulse2.pulseId)]
            selections.append((selected_dfevent2, selected_dfpulse2))
            
        if len(thresholds) > 2:
            
            selected_dfpulse3 = dfpulse[lower_threshold3 < dfpulse.nevents_pulse][dfpulse.nevents_pulse < upper_threshold3]
            selected_dfevent3 = dfevent[dfevent.pulseId.isin(selected_dfpulse3.pulseId)]
            selections.append((selected_dfevent3, selected_dfpulse3))
        
        dataframes[run] = selections
          
        # plt.xlabel('Pulse ID')
        # plt.ylabel('Number of events per pulse')
        # plt.legend()
        # plt.title(f'Events per pulse with respect to pulse ID for run {run}')
        # plt.show()

        
    for key, values in dataframes.items():
        
        selected_dfevents1.append(values[0][0])
        selected_dfpulses1.append(values[0][1])
        
        if len(thresholds) > 1:
            selected_dfevents2.append(values[1][0])
            selected_dfpulses2.append(values[1][1])
            
        if len(thresholds) > 2: 
            selected_dfevents3.append(values[2][0])
            selected_dfpulses3.append(values[2][1])
        
        
    merged_selection = list()
    
    merged_dfevent1 = pd.concat(selected_dfevents1)
    merged_dfevent1.reset_index(drop=True, inplace=True)
    
    merged_dfpulse1 = pd.concat(selected_dfpulses1)
    merged_dfpulse1.reset_index(drop=True, inplace=True)
    
    merged_selection.append((merged_dfevent1, merged_dfpulse1))
    
    print(f"Number of pulses selected across {len(runs)} run(s) between {lower_threshold1} and {upper_threshold1} events: {len(merged_dfpulse1)}")
       
        
    if len(thresholds) > 1:
        merged_dfevent2 = pd.concat(selected_dfevents2)
        merged_dfevent2.reset_index(drop=True, inplace=True)
    
        merged_dfpulse2 = pd.concat(selected_dfpulses2)
        merged_dfpulse2.reset_index(drop=True, inplace=True)
        
        merged_selection.append((merged_dfevent2, merged_dfpulse2))
        
        print(f"Number of pulses selected across {len(runs)} run(s) between {lower_threshold2} and {upper_threshold2} events: {len(merged_dfpulse2)}")
    
    
    if len(thresholds) > 2:
        merged_dfevent3 = pd.concat(selected_dfevents3)
        merged_dfevent3.reset_index(drop=True, inplace=True)
    
        merged_dfpulse3 = pd.concat(selected_dfpulses3)
        merged_dfpulse3.reset_index(drop=True, inplace=True)
        
        merged_selection.append((merged_dfevent3, merged_dfpulse3))
        
        print(f"Number of pulses selected across {len(runs)} run(s) between {lower_threshold3} and {upper_threshold3} events: {len(merged_dfpulse3)}")
        
    
    return merged_selection


def spatial_selection(dfevent,dfpulse,etof,zones):
    'Square selection from the heatmap using spatial coordinates'
    'Zone is a tuple representing a zone that is not tilted (xstart, ystart, width, height)'
    'Returns spatially selected dfevent,dfpulse,etof'
    
    heatmap_with_zones(dfevent,zones)
    
    selected_dfevents = []
    
    for zone in zones:
        
        xstart,ystart,width,height,angle = zone
    
        spatial_selected_dfevent = dfevent[dfevent.x > xstart][dfevent.x < xstart+width][dfevent.y > ystart][dfevent.y < ystart+height]
        selected_dfevents.append(spatial_selected_dfevent)
        
    merged_dfevent = pd.concat(selected_dfevents)
    merged_dfevent.reset_index(drop=True, inplace=True)
    
    spatial_selected_dfpulse = dfpulse[dfpulse.pulseId.isin(merged_dfevent.pulseId)]
    spatial_selected_etof = etof.sel(pulseId=etof.coords['pulseId'].isin(merged_dfevent.pulseId))
    
    return merged_dfevent,spatial_selected_dfpulse,spatial_selected_etof


def spatial_ion_selection(dfevent,dfpulse,zones):
    'Square ion selection from the heatmap using spatial coordinates'
    'Zone is a tuple representing a zone that is not tilted (xstart, ystart, width, height)'
    'Returns spatially selected dfevent,dfpulse'
    
    heatmap_with_zones(dfevent,zones)
    
    selected_dfevents = []
    
    for zone in zones:
        
        xstart,ystart,width,height,angle = zone
    
        spatial_selected_dfevent = dfevent[dfevent.x > xstart][dfevent.x < xstart+width][dfevent.y > ystart][dfevent.y < ystart+height]
        selected_dfevents.append(spatial_selected_dfevent)
        
    merged_dfevent = pd.concat(selected_dfevents)
    merged_dfevent.reset_index(drop=True, inplace=True)
    
    spatial_selected_dfpulse = dfpulse[dfpulse.pulseId.isin(merged_dfevent.pulseId)]
    
    return merged_dfevent,spatial_selected_dfpulse


def spatial_ion_selection_no_etof(dfevent,dfpulse,zones):
    'Square selection from the heatmap using spatial coordinates'
    'Zone is a tuple representing a zone that is not tilted (xstart, ystart, width, height)'
    'Returns spatially selected dfevent,dfpulse,etof'
    
    heatmap_with_zones(dfevent,zones)
    
    selected_dfevents = []
    
    for zone in zones:
        
        xstart,ystart,width,height,angle = zone
    
        spatial_selected_dfevent = dfevent[dfevent.x > xstart][dfevent.x < xstart+width][dfevent.y > ystart][dfevent.y < ystart+height]
        selected_dfevents.append(spatial_selected_dfevent)
        
    merged_dfevent = pd.concat(selected_dfevents)
    merged_dfevent.reset_index(drop=True, inplace=True)
    
    spatial_selected_dfpulse = dfpulse[dfpulse.pulseId.isin(merged_dfevent.pulseId)]
    
    return merged_dfevent,spatial_selected_dfpulse


def mq_selection(calibrated_dfevent,dfpulse,etof,lower_mq,upper_mq):
    'Selects based on m/q values. Need to input calibrated_dfevent! Returns m/q selected dfevent,dfpulse,etof.'
    
    mqselected_dfevent = calibrated_dfevent[lower_mq < calibrated_dfevent.mq][calibrated_dfevent.mq < upper_mq]
    mqselected_dfpulse = dfpulse[dfpulse.pulseId.isin(mqselected_dfevent.pulseId)]
    mqselected_etof = etof.sel(pulseId=etof.coords['pulseId'].isin(mqselected_dfevent.pulseId))
    
    return mqselected_dfevent,mqselected_dfpulse,mqselected_etof


def find_rectangle_corners(zone):
    
    x, y, width, height, angle_degrees = zone
    
    angle_radians = np.deg2rad(angle_degrees)
    c, s = np.cos(angle_radians), np.sin(angle_radians)
    
    # Calculate the coordinates of the other three corners
    corners = np.array([
        [x, y],
        [x + width * c, y + width * s],
        [x + width * c - height * s, y + width * s + height * c],
        [x - height * s, y + height * c]
    ])
    
    return corners


def is_inside(x, y, corners, zone):
    """
    Correct test: project point P onto the rectangle's local axes (u, v),
    then check 0 <= <P - O, u> <= width and 0 <= <P - O, v> <= height.
    """
    ox, oy, width, height, angle_degrees = zone
    angle_radians = np.deg2rad(angle_degrees)
    c, s = np.cos(angle_radians), np.sin(angle_radians)

    # local axes
    # u axis points along 'width' (c, s), v axis along 'height' (-s, c)
    dx, dy = x - ox, y - oy
    u_coord =  dx * c + dy * s         # projection onto u
    v_coord = -dx * s + dy * c         # projection onto v

    return (0 <= u_coord <= width) and (0 <= v_coord <= height)


def find_integer_coordinates(corners, zone):

    # bounding box to loop over
    min_x, min_y = np.floor(np.min(corners, axis=0))
    max_x, max_y = np.ceil(np.max(corners, axis=0))

    integer_coordinates = []
    for x in range(int(min_x), int(max_x) + 1):
        for y in range(int(min_y), int(max_y) + 1):
            if is_inside(x, y, corners, zone):
                integer_coordinates.append((x, y))
    
    return integer_coordinates


def find_outside_integer_coordinates(corners, zone, xlim=(0, 255), ylim=(0, 255)):
    """
    Outside = all (x,y) in domain minus inside-of-rect.
    """
    inside = set(find_integer_coordinates(corners, zone))
    integer_coordinates = []
    
    for xi in range(xlim[0], xlim[1] + 1):
        for yi in range(ylim[0], ylim[1] + 1):
            if (xi, yi) not in inside:
                integer_coordinates.append((xi, yi))
                
    return integer_coordinates


def _select_by_pairs(dfevent, integer_coordinates):
    """
    Using a merge is clean and vectorized.
    """
    coords_df = pd.DataFrame(integer_coordinates, columns=['x', 'y']).drop_duplicates()
    selected = dfevent.merge(coords_df, on=['x', 'y'], how='inner')
    return selected


def tilted_spatial_ion_selection(dfevent, dfpulse, etof, zones, xlim=(0,255), ylim=(0,255)):
    'Selection from the heatmap using spatial coordinates'
    'Zones is a list of tuple representing zones - tilted or not - (xstart, ystart, width, height, angle in degrees)'
    'Returns spatially selected dfevent,dfpulse,etof'
    
    heatmap_with_zones(dfevent, zones)

    integer_coordinates = []
    for zone in zones:
        corners = find_rectangle_corners(zone)
        integer_coordinates.extend(find_integer_coordinates(corners, zone))

    # Deduplicate and select by (x,y) PAIRS
    integer_coordinates = list(set(integer_coordinates))
    spatial_selected_dfevent = _select_by_pairs(dfevent, integer_coordinates)

    spatial_selected_dfpulse = dfpulse[dfpulse.pulseId.isin(spatial_selected_dfevent.pulseId.unique())]
    spatial_selected_etof = etof.sel(pulseId=etof.coords['pulseId'].isin(spatial_selected_dfevent.pulseId))

    return spatial_selected_dfevent, spatial_selected_dfpulse, spatial_selected_etof


def outside_tilted_spatial_ion_selection(dfevent, dfpulse, etof, zones, xlim=(0,255), ylim=(0,255)):
    """
    Select events OUTSIDE the union of the zones.
    We build the union of 'inside' pairs and then take the complement on the domain.
    """
    heatmap_with_zones(dfevent, zones)

    inside_pairs = set()
    for zone in zones:
        corners = find_rectangle_corners(zone)
        inside_pairs.update(find_integer_coordinates(corners, zone))

    # Domain from limits; build complement
    outside_pairs = []
    for xi in range(xlim[0], xlim[1] + 1):
        for yi in range(ylim[0], ylim[1] + 1):
            if (xi, yi) not in inside_pairs:
                outside_pairs.append((xi, yi))

    spatial_selected_dfevent = _select_by_pairs(dfevent, outside_pairs)
    spatial_selected_dfpulse = dfpulse[dfpulse.pulseId.isin(spatial_selected_dfevent.pulseId.unique())]
    spatial_selected_etof = etof.sel(pulseId=etof.coords['pulseId'].isin(spatial_selected_dfevent.pulseId))

    return spatial_selected_dfevent, spatial_selected_dfpulse, spatial_selected_etof


def _x_on_line_at_y(y_vals, A, B):
    """
    Given two points A=(xa,ya), B=(xb,yb) defining a (non-horizontal) line,
    return x_on_line for each y in y_vals assuming the infinite line through A and B.
    """
    xa, ya = A
    xb, yb = B
    dy = yb - ya
    if dy == 0:
        raise ValueError("The line is horizontal (yb == ya). 'Left = lower x' is undefined for a horizontal line.")
    t = (y_vals - ya) / dy
    return xa + t * (xb - xa)


def select_right_of_line(dfevent, dfpulse, etof, A, B, inclusive=True):
    """
    Keep events to the RIGHT of the line (higher x than the line at the same y).
    This is your requested 'outside' (drop everything to the left).

    A, B: tuples/lists (x, y) defining the line (non-horizontal).
    inclusive=True keeps points exactly on the line on the right side.
    """
    xs = dfevent['x'].to_numpy(dtype=float)
    ys = dfevent['y'].to_numpy(dtype=float)

    x_line = _x_on_line_at_y(ys, A, B)
    if inclusive:
        right_mask = xs >= x_line
    else:
        right_mask = xs > x_line

    selected_dfevent = dfevent.loc[right_mask]
    sel_pulse_ids = selected_dfevent.pulseId.unique()

    selected_dfpulse = dfpulse[dfpulse.pulseId.isin(sel_pulse_ids)]
    selected_etof = etof.sel(pulseId=etof.coords['pulseId'].isin(sel_pulse_ids))

    return selected_dfevent, selected_dfpulse, selected_etof


def select_left_of_line(dfevent, dfpulse, etof, A, B, inclusive=False):
    """
    (Optional helper) Keep events to the LEFT of the line (lower x than the line at the same y).
    Useful if you ever want the 'inside' instead.
    """
    xs = dfevent['x'].to_numpy(dtype=float)
    ys = dfevent['y'].to_numpy(dtype=float)

    x_line = _x_on_line_at_y(ys, A, B)
    if inclusive:
        left_mask = xs <= x_line
    else:
        left_mask = xs < x_line

    selected_dfevent = dfevent.loc[left_mask]
    sel_pulse_ids = selected_dfevent.pulseId.unique()

    selected_dfpulse = dfpulse[dfpulse.pulseId.isin(sel_pulse_ids)]
    selected_etof = etof.sel(pulseId=etof.coords['pulseId'].isin(sel_pulse_ids))

    return selected_dfevent, selected_dfpulse, selected_etof


def fish_plot_ion_selection(dfevent,zone):
    'Selection from the heatmap using spatial coordinates'
    'Zone is a tuple representing a zone - tilted or not - (xstart, ystart, width, height, angle in degrees)'
    'Returns spatially selected dfevent'
    
    heatmap_with_zones(dfevent,[zone])
    
    corners = find_rectangle_corners(zone)
    integer_coordinates = find_integer_coordinates(corners, zone)
    
    x_coords, y_coords = zip(*integer_coordinates)
    
    spatial_selected_dfevent = dfevent[dfevent.x.isin(x_coords)][dfevent.y.isin(y_coords)]
    
    return spatial_selected_dfevent
