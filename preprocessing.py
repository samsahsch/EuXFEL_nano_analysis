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





### Changing Variables


### Function by EuXFEL staff to compute number of pulses per train, and some number 16?

def get_stacking(run):
    bunch_table = run["SQS_RR_UTC/TSYS/TIMESERVER:outputBunchPattern",'data.bunchPatternTable'].ndarray()
    
    DESTINATION_MASK = 0xf << 18
    DESTINATION_T4D = 4 << 18   # SASE1/3 dump
    PHOTON_LINE_DEFLECTION = 1 << 27  # Soft kick (e.g. SA3)

    matched = (bunch_table & DESTINATION_MASK) == DESTINATION_T4D 
    matched &= (bunch_table & PHOTON_LINE_DEFLECTION) != 0

    SASE3_pattern = np.ones_like(matched[0]) &  matched[1]
    test = (  SASE3_pattern ) *1e8
    return np.sum( SASE3_pattern) , np.mean(np.diff(np.nonzero(SASE3_pattern)))



PULSES_PER_TRAIN, EVERYNTH = get_stacking(run)
N_CHANNELS_PULSE = int(EVERYNTH/2)*1760 # bunch interval? times 1760? gives number of channels per pulse
filename = 'datarun' + str(RUNID) + '.h5'


# Data where all trains are present
gasattenuator_transmission = run['SA3_XTD10_VAC/MDL/GATT_TRANSMISSION_MONITOR', 'Estimated_Tr.value'].xarray()


first_trainid = int(gasattenuator_transmission.trainId[0])
trains = gasattenuator_transmission.shape[0]


if os.path.exists(filename):
    os.remove(filename)
    print(f"The previous file '{filename}' has been deleted.")
else:
    print(f"The file '{filename}' will be created.")





### Dataframe per event



print('Loading per event data...')


### Retrieve x,y,toa,tot

x = run['SQS_AQS_CAM/CAM/TIMEPIX3:daqEventOutput', 'data.x'].ndarray()
y = run['SQS_AQS_CAM/CAM/TIMEPIX3:daqEventOutput', 'data.y'].ndarray()
toa = run['SQS_AQS_CAM/CAM/TIMEPIX3:daqEventOutput', 'data.toa'].ndarray()
tot = run['SQS_AQS_CAM/CAM/TIMEPIX3:daqEventOutput', 'data.tot'].ndarray()
events_trainid = run['SQS_AQS_CAM/CAM/TIMEPIX3:daqEventOutput', 'data.trainId'].ndarray()


print('Producing dataframe per event...')



# Find missing trains in events arrays
#DOES NOT HANDLE THE CASE WHERE TWO OR MORE CONSECUTIVE TRAINS ARE MISSING AT THE BEGINNING OR END

# Indices array of missing trains
missing_trains_events = np.array([], dtype=int)

# Compute differences in the train IDs to catch missing trains, find their indices
events_trainid_diff = np.diff(events_trainid)-1
nonzero_indices = np.nonzero(events_trainid_diff)[0]

# Handle the cases where two or more consecutive trains are missing
for idx in nonzero_indices:
    additional_indices = np.arange(idx,idx+events_trainid_diff[idx], dtype=int)
    missing_trains_events = np.append(missing_trains_events,additional_indices)

# Handle the cases where the first or/and last trains are missing
if events_trainid[0] != gasattenuator_transmission.trainId[0]:
    missing_trains_events = np.append(missing_trains_events, 0)
if events_trainid[-1] != gasattenuator_transmission.trainId[-1]:
    missing_trains_events = np.append(missing_trains_events, events_trainid.shape[0])



# Put the x,y,toa,tot matrices into a common list of matrices
events_arrays = [x,y,toa,tot]

# Bins to split the data by pulse
bins = np.linspace(0,(PULSES_PER_TRAIN-1)*TIME_BETWEEN_PULSES,PULSES_PER_TRAIN)

# Filler for empty or missing trains
false_record = np.append(np.array([1]), np.zeros(91, dtype='int'))

# Function to remove unncessary events data, resize the trains and sort by time based on toa
def remove_resize_sort(row):
    
    xrow, yrow, toarow, totrow = row
    
    # Find the index where the first one appears in toa row
    first_one_index = np.argmax(toarow == 1)
    
    # Slice all rows to remove ones
    resized_xrow = xrow[:first_one_index]
    resized_yrow = yrow[:first_one_index]
    resized_toarow = toarow[:first_one_index]-T0 # substract T0
    resized_totrow = totrow[:first_one_index]
    
    # Get sorting assignments from resized toa row
    sorting = np.argsort(resized_toarow)
    
    # Sort all rows based on the sorting assignments
    sorted_xrow = np.take_along_axis(resized_xrow, sorting, axis=0)
    sorted_yrow = np.take_along_axis(resized_yrow, sorting, axis=0)
    sorted_toarow = np.take_along_axis(resized_toarow, sorting, axis=0)
    sorted_totrow = np.take_along_axis(resized_totrow, sorting, axis=0)
    
    if sorted_toarow[-1] < 0: # Handle case where train only contains hits before T0
        final_xrow, final_yrow, final_toarow, final_totrow = (np.array([0]),)*4
        
    else:
        # Find the index of the first event after T0
        first_event_index = np.argmax(sorted_toarow > 0)
    
        # Slice all rows to remove events before T0
        final_xrow = sorted_xrow[first_event_index:]
        final_yrow = sorted_yrow[first_event_index:]
        final_toarow = sorted_toarow[first_event_index:]
        final_totrow = sorted_totrow[first_event_index:]
    
    # Find the number of events in each pulse based on the sorted toa row
    if len(final_toarow) == 1: # Handle case where train is essentially empty
        nevents = false_record
        
        final_tofrow = np.array([0])
        
    else:
        nevents = np.diff(np.append(np.searchsorted(sorted_toarow,bins),len(sorted_toarow)))
        
        start_indices = nevents[:-1]
        end_indices = nevents[1:]
        final_tofrow = np.concatenate([final_toarow[start:end] - bins[i:i+1] for i, (start, end) in enumerate(zip(start_indices, end_indices))])
        
        
    # Compute time of flight from time of arrival and number of events
    
    # Values to subtract of toa
    subtract_values = TIME_BETWEEN_PULSES * np.arange(PULSES_PER_TRAIN)
    
    # Create an array of the size of toa using nevents
    subtract_array = np.repeat(subtract_values, nevents)
    
    # Produce tof array
    final_tofrow = final_toarow - subtract_array
    
    return final_xrow, final_yrow, final_toarow, final_tofrow, final_totrow, nevents

# Apply the function to each row of each matrix in events_arrays
resized_sorted_events_arrays = np.array([remove_resize_sort(row) for row in zip(*events_arrays)], dtype=object)

# Insert placeholders where trains are missing
zero = np.array([0])
placeholder = np.array([zero,zero,zero,zero,zero,false_record], dtype=object)
final_arrays = np.insert(resized_sorted_events_arrays, missing_trains_events, placeholder, axis=0).T

# Flatten x,y,toa,tot data into arrays
final_xyt_events = [np.concatenate(row) for row in final_arrays[:5]]
finalx, finaly, finaltoa, finaltof, finaltot = final_xyt_events

# Compute number of events per pulse, number of events per train, and getting them to the same dimension
nevents = final_arrays[5]
nevents_pulse = np.concatenate(nevents).astype('int32')
nevents_train = np.sum(np.vstack(nevents),axis=1).astype('int32')
nevents_train_repeated = np.repeat(nevents_train,PULSES_PER_TRAIN)



### Compute train ID & pulse ID for events

trainid_part_events = np.char.add(str(RUNID), np.char.zfill(np.arange(1,trains+1).astype(str),5)).astype(int)
trainid_events = np.repeat(trainid_part_events,nevents_train)

pulseid_part_events = np.char.add(np.repeat(trainid_part_events,PULSES_PER_TRAIN).astype(str), np.char.zfill(np.tile(np.arange(1,PULSES_PER_TRAIN+1),trains).astype(str),3)).astype(int)
pulseid_events = np.repeat(pulseid_part_events,nevents_pulse)



print('Saving dataframe per event...')



### Build dictionary for event split data

dataevent = dict()
dataevent['trainId'] = trainid_events.astype('int32')
dataevent['pulseId'] = pulseid_events
dataevent['x'] = finalx.astype('int16')
dataevent['y'] = finaly.astype('int16')
dataevent['toa'] = finaltoa
dataevent['tof'] = finaltof
dataevent['tot'] = finaltot.astype('int16')



### Create and save dataframe per event

dfevent = pd.DataFrame(data=dataevent)
dfevent.to_hdf(filename, key='dfevent', mode='w') 






### Dataframe per pulse



print('Loading per pulse data...')


# Retrieve data

gasattenuator_pressure = run['SA3_XTD10_VAC/MDL/GATT_P_CELL', 'value.value'].xarray()

pulseenergy_sase3 = run['SA3_XTD10_XGM/XGM/DOOCS:output', 'data.intensitySa3TD'].xarray()
pulseenergy_sqs = run['SQS_DIAG1_XGMD/XGM/DOOCS:output', 'data.intensitySa3TD'].xarray()

pnccd_gain = run['SQS_NQS_PNCCD1MP/MDL/DAQ_GAIN', 'pNCCDGain.value'].xarray()
pnccd_voltage = run['SQS_NQS_PNCCD1MP/MDL/DAQ_MPOD', 'u0voltage.value'].xarray()

#voltage_channel = run['SQS_RACK_MPOD-1/MDL/MPOD_MAPPER', ''].xarray() IS INCLUDED IN RUN LOG

# Remove unnecessary data

pulseenergy_sase3 = pulseenergy_sase3[:,:PULSES_PER_TRAIN]
pulseenergy_sqs = pulseenergy_sqs[:,:PULSES_PER_TRAIN]


print('Producing dataframe per pulse...')



# Insert zeros when missing a train

zero_array = np.zeros(92)

shift_pulseenergy_sase3 = 0
shift_pulseenergy_sqs = 0

for trainid in range(trains):
        
    if trainid - shift_pulseenergy_sase3 > pulseenergy_sase3.shape[0] or trainid != int(pulseenergy_sase3.trainId[trainid - shift_pulseenergy_sase3]) - first_trainid:
        
        pulseenergy_sase3 = np.insert(pulseenergy_sase3, trainid, zero_array, axis=0)
        shift_pulseenergy_sase3 += 1
        
    if trainid - shift_pulseenergy_sqs > pulseenergy_sqs.shape[0] or trainid != int(pulseenergy_sqs.trainId[trainid - shift_pulseenergy_sqs]) - first_trainid:
        
        pulseenergy_sqs = np.insert(pulseenergy_sqs, trainid, zero_array, axis=0)
        shift_pulseenergy_sqs += 1



# Compute train ID & pulse ID, convert into per pulse data

trainid_part = np.char.zfill(np.repeat(np.arange(1,trains+1),PULSES_PER_TRAIN).astype(str),5)
trainid = np.char.add(str(RUNID), trainid_part).astype(int)

pulseid_part = np.char.zfill(np.tile(np.arange(1,PULSES_PER_TRAIN+1),trains).astype(str),3)
pulseid = np.char.add(trainid.astype(str), pulseid_part).astype(int)

gasattenuator_transmission = np.repeat(gasattenuator_transmission,PULSES_PER_TRAIN)
gasattenuator_pressure = np.repeat(gasattenuator_pressure,PULSES_PER_TRAIN)

pulseenergy_sase3 = np.array([pulseenergy_sase3]).reshape(-1)
pulseenergy_sqs = np.array([pulseenergy_sqs]).reshape(-1)

pnccd_gain = np.repeat(pnccd_gain,PULSES_PER_TRAIN)
pnccd_voltage = np.repeat(pnccd_voltage,PULSES_PER_TRAIN)



print('Saving dataframe per pulse...')



# Build dictionary for pulse split data

datapulse = dict()
datapulse['trainId'] = trainid.astype('int32')
datapulse['pulseId'] = pulseid
datapulse['gasattenuator_transmission'] = gasattenuator_transmission.astype('float32')
datapulse['gasattenuator_pressure'] = gasattenuator_pressure.astype('float32')
datapulse['pulseenergy_sase3'] = pulseenergy_sase3.astype('float32')
datapulse['pulseenergy_sqs'] = pulseenergy_sqs.astype('float32')
datapulse['pnccd_gain'] = pnccd_gain.astype('float16')
datapulse['pnccd_voltage'] = pnccd_voltage.astype('float32')
datapulse['nevents_pulse'] = nevents_pulse
datapulse['nevents_train'] = nevents_train_repeated



# Create save and read dataframe per pulse

dfpulse = pd.DataFrame(data=datapulse)
dfpulse.to_hdf(filename, key='dfpulse', mode='a')  





### Etof



print('Loading etof data...')


digitizer = run['SQS_DIGITIZER_UTC1/ADC/1:network', 'digitizers.channel_1_A.raw.samples'].ndarray()
digitizer_trainid = run['SQS_DIGITIZER_UTC1/ADC/1:network', 'digitizers.trainId'].ndarray()


print('Producing etof xarray...')


def removePeriodicNoiseRunData(traces, n, dead_time ):
    clean_trace = traces.astype(dtype=np.float64)
    for i in range(n):
        clean_trace[:, i::n] -= clean_trace[:,dead_time:][:,i::n].mean()
    return clean_trace

def slicePeriodicRunData(signal,offset,periodicity,N):
    return signal[:,offset:offset+N*periodicity].reshape([signal.shape[0]*N,periodicity])


# Clean and slice digitizer data

# clean each train one by one
clean_digitizer = removePeriodicNoiseRunData(digitizer, DIGITIZER_NOISE_PERIODICITY, DIGITIZER_END)

# slice clean digitizer data into trains and pulses 
sliced_etof_data = slicePeriodicRunData(clean_digitizer, DIGITIZER_OFFSET, N_CHANNELS_PULSE, PULSES_PER_TRAIN)


digitizer_pulseid_part = np.char.zfill(np.tile(np.arange(1,PULSES_PER_TRAIN+1),digitizer_trainid.shape[0]).astype(str),3)
digitizer_trainid = np.repeat( np.char.add(str(RUNID), np.chararray.zfill((digitizer_trainid-first_trainid+1).astype(str),5)).astype(int), PULSES_PER_TRAIN)
digitizer_pulseid = np.char.add(digitizer_trainid.astype(str), digitizer_pulseid_part).astype(int)


etof = xr.DataArray(sliced_etof_data)
etof = etof.assign_coords(dim_0=digitizer_pulseid).rename({'dim_0':'pulseId','dim_1':'data'})


print('Saving etof xarray...')





### PNCCD



print('Loading pnccd data...')

# Import pnccd raw images
pnccd_raw = run['SQS_NQS_PNCCD1MP/CAL/PNCCD_FMT-0:output', 'data.image'].xarray()



print('Producing pnccd xarray...')

# Rescaling train IDs so they fit the chosen identifiers
pnccd_corrected_ids = pnccd_raw.assign_coords(trainId=np.char.add(str(RUNID), np.chararray.zfill((pnccd_raw.trainId-first_trainid+1).astype(str),5)).astype(int))

# Rename array to a format that works with hdf5
pnccd = pnccd_corrected_ids.rename('pnccd')



print('Saving pnccd xarray...')

pnccd.to_netcdf(filename, mode='a', group='pnccd', engine='h5netcdf')





### Reset



end_time = time.time()
print(f"Preprocessing took {end_time - start_time:.2f} seconds")


%reset -f


import pandas as pd
import xarray as xr

def read(runid):
    'Read the preprocessed data of run with ID runid saved in the h5 file with a corresponding name'
    'Outputs dataframes per event, per pulse, and xarrays etof, pnccd in that order'
    
    filename = 'datarun' + str(RUNID) + '.h5'
    
    dfevent = pd.read_hdf(filename, 'dfevent')
    dfpulse = pd.read_hdf(filename, 'dfpulse')
    
    etof = xr.open_dataarray(filename, group="etof")
    pnccd = xr.open_dataarray(filename, group="pnccd")
    
    return dfevent, dfpulse, etof, pnccd


print('')