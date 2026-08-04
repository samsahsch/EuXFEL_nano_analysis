"""Binning of spectra by per-pulse ion count, and the waterfall / heatmap views."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .constants import TIME_BETWEEN_PULSES, CHANNELS_PER_PULSE


def nevents_binning(dfevent,dfpulse,etof,nbins_events,nbins_mq):
    'Binning dfevent dataframe into a number of bins nbins using number of events per pulse dfpulse.nevents_pulse'
    'Outputs nbins sized list of tuples (filtered_dfevent, filtered_dfpulse) and nbins sized array of histograms'
    'hists is divided by number of pulses in a certain number of events bin'
    
    nevents_min = min(dfpulse.nevents_pulse)
    nevents_max = max(dfpulse.nevents_pulse)
    
    bins = np.linspace(nevents_min,nevents_max,nbins_events+1).astype(int)
    
    filtered_dfevents = []
    filtered_dfpulses = []
    filtered_etofs = []
    hists = []
    
    for i in range(len(bins) - 1):

        start_edge = bins[i]
        end_edge = bins[i + 1]

        filtered_dfpulse = dfpulse[(dfpulse.nevents_pulse >= start_edge) & (dfpulse.nevents_pulse < end_edge)]
        filtered_dfevent = dfevent[dfevent.pulseId.isin(filtered_dfpulse.pulseId)]
        filtered_etof = etof.sel(pulseId=etof.coords['pulseId'].isin(filtered_dfpulse.pulseId))

        filtered_dfevents.append(filtered_dfevent)
        filtered_dfpulses.append(filtered_dfpulse)
        filtered_etofs.append(filtered_etof)
        
        hist, bin_edges = np.histogram(filtered_dfevent.mq, bins=np.linspace(0,200,nbins_mq+1),range=(0,200))
        hists.append(hist/len(filtered_dfpulse))
        
    return filtered_dfevents, filtered_dfpulses, filtered_etofs, np.array(hists), bins


def nevents_binning_tof(dfevent,dfpulse,etof,nbins_events,nbins_tof):
    'Bins into time of flight bins'
    'Binning dfevent dataframe into a number of bins nbins using number of events per pulse dfpulse.nevents_pulse'
    'Outputs nbins sized list of tuples (filtered_dfevent, filtered_dfpulse) and nbins sized array of histograms'
    'hists is divided by number of pulses in a certain number of events bin'
    
    nevents_min = min(dfpulse.nevents_pulse)
    nevents_max = max(dfpulse.nevents_pulse)
    
    bins = np.linspace(nevents_min,nevents_max,nbins_events+1).astype(int)
    
    filtered_dfevents = []
    filtered_dfpulses = []
    filtered_etofs = []
    hists = []
    
    for i in range(len(bins) - 1):

        start_edge = bins[i]
        end_edge = bins[i + 1]

        filtered_dfpulse = dfpulse[(dfpulse.nevents_pulse >= start_edge) & (dfpulse.nevents_pulse < end_edge)]
        filtered_dfevent = dfevent[dfevent.pulseId.isin(filtered_dfpulse.pulseId)]
        filtered_etof = etof.sel(pulseId=etof.coords['pulseId'].isin(filtered_dfpulse.pulseId))

        filtered_dfevents.append(filtered_dfevent)
        filtered_dfpulses.append(filtered_dfpulse)
        filtered_etofs.append(filtered_etof)
        
        hist, bin_edges = np.histogram(filtered_dfevent.tof, bins=np.linspace(0,TIME_BETWEEN_PULSES,nbins_tof+1),range=(0,TIME_BETWEEN_PULSES))
        hists.append(hist/len(filtered_dfpulse))
        
    return filtered_dfevents, filtered_dfpulses, filtered_etofs, np.array(hists), bins


def nions_binning(dfevent,dfpulse,nbins_events,nbins_mq):
    'Only handles ions'
    'Binning dfevent dataframe into a number of bins nbins using number of events per pulse dfpulse.nevents_pulse'
    'Outputs nbins sized list of tuples (filtered_dfevent, filtered_dfpulse) and nbins sized array of histograms'
    'hists is divided by number of pulses in a certain number of events bin'
    
    nevents_min = min(dfpulse.nevents_pulse)
    nevents_max = max(dfpulse.nevents_pulse)
    
    bins = np.linspace(nevents_min,nevents_max,nbins_events+1).astype(int)
    
    filtered_dfevents = []
    filtered_dfpulses = []
    hists = []
    
    for i in range(len(bins) - 1):

        start_edge = bins[i]
        end_edge = bins[i + 1]

        filtered_dfpulse = dfpulse[(dfpulse.nevents_pulse >= start_edge) & (dfpulse.nevents_pulse < end_edge)]
        filtered_dfevent = dfevent[dfevent.pulseId.isin(filtered_dfpulse.pulseId)]

        filtered_dfevents.append(filtered_dfevent)
        filtered_dfpulses.append(filtered_dfpulse)
        
        hist, bin_edges = np.histogram(filtered_dfevent.mq, bins=np.linspace(0,200,nbins_mq+1),range=(0,200))
        hists.append(hist/len(filtered_dfpulse))
        
    return filtered_dfevents, filtered_dfpulses, np.array(hists), bins


def nions_binning_tof(dfevent,dfpulse,nbins_events,nbins_tof):
    'Bins into time of flight bins'
    'Only handles ions'
    'Binning dfevent dataframe into a number of bins nbins using number of events per pulse dfpulse.nevents_pulse'
    'Outputs nbins sized list of tuples (filtered_dfevent, filtered_dfpulse) and nbins sized array of histograms'
    'hists is divided by number of pulses in a certain number of events bin'
    
    nevents_min = min(dfpulse.nevents_pulse)
    nevents_max = max(dfpulse.nevents_pulse)
    
    bins = np.linspace(nevents_min,nevents_max,nbins_events+1).astype(int)
    
    filtered_dfevents = []
    filtered_dfpulses = []
    hists = []
    
    for i in range(len(bins) - 1):

        start_edge = bins[i]
        end_edge = bins[i + 1]

        filtered_dfpulse = dfpulse[(dfpulse.nevents_pulse >= start_edge) & (dfpulse.nevents_pulse < end_edge)]
        filtered_dfevent = dfevent[dfevent.pulseId.isin(filtered_dfpulse.pulseId)]

        filtered_dfevents.append(filtered_dfevent)
        filtered_dfpulses.append(filtered_dfpulse)
        
        hist, bin_edges = np.histogram(filtered_dfevent.tof, bins=np.linspace(0,TIME_BETWEEN_PULSES,nbins_tof+1),range=(0,TIME_BETWEEN_PULSES))
        hists.append(hist/len(filtered_dfpulse))
        
    return filtered_dfevents, filtered_dfpulses, np.array(hists), bins


def nevents_binning_plot(dfevent,dfpulse,nbins_events,nbins_mq,xlimits=(0,200)):
    'Bins by m/q'
    'Binning dfevent dataframe into a number of bins nbins using number of events per pulse dfpulse.nevents_pulse'
    'Outputs nbins sized list of tuples (filtered_dfevent, filtered_dfpulse)'
    
    x_lower, x_upper = xlimits
    
    nevents_min = min(dfpulse.nevents_pulse)
    nevents_max = max(dfpulse.nevents_pulse)
    bins = np.linspace(nevents_min,nevents_max,nbins_events+1).astype(int)
    
    filtered_dfs = []
    plt.figure(figsize=(20,8))
    
    for i in range(len(bins) - 1):

        start_edge = bins[i]
        end_edge = bins[i + 1]

        filtered_dfpulse = dfpulse[(dfpulse.nevents_pulse >= start_edge) & (dfpulse.nevents_pulse < end_edge)]
        filtered_dfevent = dfevent[dfevent.pulseId.isin(filtered_dfpulse.pulseId)]

        filtered_dfs.append((filtered_dfevent,filtered_dfpulse))
        
        hist, bin_edges = np.histogram(filtered_dfevent.mq, bins=np.linspace(0,200,nbins_mq),range=(0,200))
        
        plt.plot(bin_edges[:-1], hist/max(hist), label=f'{start_edge}-{end_edge}')
        
    plt.xlabel('m/q')
    plt.ylabel('Relatively normalized number of hits per bin')
    plt.title('Normalized ions time of flight')
    plt.legend()
    plt.xlim(x_lower, x_upper)
    plt.show()   
        
    return filtered_dfs


def nevents_binning_plot_tof(dfevent,dfpulse,nbins_events,nbins_tof,xlimits=(0,TIME_BETWEEN_PULSES)):
    'Bins by time of flight'
    'Binning dfevent dataframe into a number of bins nbins using number of events per pulse dfpulse.nevents_pulse'
    'Outputs nbins sized list of tuples (filtered_dfevent, filtered_dfpulse)'
    
    x_lower, x_upper = xlimits
    
    nevents_min = min(dfpulse.nevents_pulse)
    nevents_max = max(dfpulse.nevents_pulse)
    bins = np.linspace(nevents_min,nevents_max,nbins_events+1).astype(int)
    
    filtered_dfs = []
    plt.figure(figsize=(20,8))
    
    for i in range(len(bins) - 1):

        start_edge = bins[i]
        end_edge = bins[i + 1]

        filtered_dfpulse = dfpulse[(dfpulse.nevents_pulse >= start_edge) & (dfpulse.nevents_pulse < end_edge)]
        filtered_dfevent = dfevent[dfevent.pulseId.isin(filtered_dfpulse.pulseId)]

        filtered_dfs.append((filtered_dfevent,filtered_dfpulse))
        
        hist, bin_edges = np.histogram(filtered_dfevent.tof, bins=np.linspace(0,TIME_BETWEEN_PULSES,nbins_tof),range=(0,TIME_BETWEEN_PULSES))
        
        plt.plot(bin_edges[:-1], hist/max(hist), label=f'{start_edge}-{end_edge}')
        
    plt.xlabel('Time of flight (s)')
    plt.ylabel('Relatively normalized number of hits per bin')
    plt.title('Normalized ions time of flight')
    plt.legend()
    plt.xlim(x_lower, x_upper)
    plt.show()   
        
    return filtered_dfs


def nevents_binning_cov(dfevent,dfpulse,etof,nbins_events,nbins_ion_tof,nbins_e_tof,max_ion_limit=TIME_BETWEEN_PULSES,max_e_limit=TIME_BETWEEN_PULSES):
    'Binning dfevent and etof by numbers of events, and number of bins along time of flight'
    'Can select maximal time limit for electrons and ions'
    
    channel_time = TIME_BETWEEN_PULSES/CHANNELS_PER_PULSE
    
    nevents_min = min(dfpulse.nevents_pulse)
    nevents_max = max(dfpulse.nevents_pulse)
    
    bins = np.linspace(nevents_min,nevents_max,nbins_events+1).astype(int)
    
    max_int_ion_limit = int(max_ion_limit/channel_time)
    new_int_ion_limit = max_int_ion_limit - max_int_ion_limit % nbins_ion_tof
    new_ion_limit = new_int_ion_limit*channel_time
    
    max_int_e_limit = int(max_e_limit/channel_time)
    new_int_e_limit = max_int_e_limit - max_int_e_limit % nbins_e_tof
    e_group_size = int(new_int_e_limit/nbins_e_tof)
    
    shortened_dfevent = dfevent[dfevent.tof < new_ion_limit]
    shortened_etof = etof[:,:new_int_e_limit]
    
    hists = []
    hists_etof = []
    
    for i in range(len(bins) - 1):

        start_edge = bins[i]
        end_edge = bins[i + 1]

        filtered_dfpulse = dfpulse[(dfpulse.nevents_pulse >= start_edge) & (dfpulse.nevents_pulse < end_edge)]
        filtered_dfevent = shortened_dfevent[shortened_dfevent.pulseId.isin(filtered_dfpulse.pulseId)]
        filtered_etof = shortened_etof.sel(pulseId=shortened_etof.coords['pulseId'].isin(filtered_dfpulse.pulseId))
        
        hist, bin_edges = np.histogram(filtered_dfevent.tof, bins=nbins_ion_tof)
        hists.append(hist)
        
        numpy_etof = filtered_etof.to_numpy()
        reshaped_etof = numpy_etof.reshape((numpy_etof.shape[0], -1, e_group_size))
        summed_etof = np.sum(reshaped_etof, axis=-1)
        avg_etof = -np.mean(summed_etof, axis=0)
        hists_etof.append(avg_etof)
        
    hists = np.array(hists)
    hists_etof = np.array(hists_etof)
        
    return hists, hists_etof, bins


def stacked_ion_tof_max(hists,nbins_tof,bins,xlimits=(0,TIME_BETWEEN_PULSES),backgrd_dfevent=None):
    "Plots ion tof for each number of events defined on top of each other"
    "Can determine number of bins along time of flight and limits along x"
    
    precision = TIME_BETWEEN_PULSES/nbins_tof
    x_lower, x_upper = xlimits
    hist_lower, hist_upper = int(x_lower/precision), int(x_upper/precision)
    
    x_edges = np.linspace(x_lower, x_upper, hist_upper-hist_lower)
    plt.figure(figsize=(20, 8))
    
    for i in range(len(hists)):
        
        shortened_hist = hists[i][hist_lower:hist_upper]
        plt.plot(x_edges, shortened_hist/max(shortened_hist), label=f'{bins[i]}-{bins[i+1]}')
        
    if isinstance(backgrd_dfevent, pd.DataFrame):
        hist, bin_edges = np.histogram(backgrd_dfevent.tof, bins=np.linspace(0,TIME_BETWEEN_PULSES,nbins_tof+1),range=(0,TIME_BETWEEN_PULSES))
        shortened_hist = hist[hist_lower:hist_upper]
        plt.plot(x_edges, shortened_hist/max(shortened_hist), label='Background')

    plt.xlabel('Time of flight (s)')
    plt.ylabel('Individually normalized signal')
    plt.title('Normalized ions time of flight')
    plt.legend()
    plt.xlim(x_lower, x_upper)
    plt.show()


def stacked_ion_tof_sum(hists,nbins_tof,bins,xlimits=(0,TIME_BETWEEN_PULSES),backgrd_dfevent=None):
    "Plots ion tof for each number of events defined on top of each other"
    "Can determine number of bins along time of flight and limits along x"
    
    precision = TIME_BETWEEN_PULSES/nbins_tof
    x_lower, x_upper = xlimits
    hist_lower, hist_upper = int(x_lower/precision), int(x_upper/precision)
    
    x_edges = np.linspace(x_lower, x_upper, hist_upper-hist_lower)
    plt.figure(figsize=(20, 8))
    
    for i in range(len(hists)):
        
        shortened_hist = hists[i][hist_lower:hist_upper]
        plt.plot(x_edges, shortened_hist/sum(shortened_hist), label=f'{bins[i]}-{bins[i+1]}')
        
    if isinstance(backgrd_dfevent, pd.DataFrame):
        hist, bin_edges = np.histogram(backgrd_dfevent.tof, bins=np.linspace(0,TIME_BETWEEN_PULSES,nbins_tof+1),range=(0,TIME_BETWEEN_PULSES))
        shortened_hist = hist[hist_lower:hist_upper]
        plt.plot(x_edges, shortened_hist/sum(shortened_hist), label='Background')

    plt.xlabel('Time of flight (s)')
    plt.ylabel('Individually normalized signal')
    plt.title('Normalized ions time of flight')
    plt.legend()
    plt.xlim(x_lower, x_upper)
    plt.show()


def waterfall_rel(hists,nbins_mq,xlimits=(0,200)):
    "Waterfall plot of relative normalization with respect to m/q using hists, which is a list of histograms"

    x_lower, x_upper = xlimits
    nbins_events = len(hists)
    bin_edges = np.linspace(0, 200, nbins_mq+1)

    plt.figure(figsize=(20, 8))
    colormap = plt.cm.inferno
    color_index = np.linspace(0.2, 0.8, nbins_events)

    for i in range(nbins_events):
        
        hist_norm = hists[i] / max(hists[i]) + i

        line_color = colormap(color_index[i])
        plt.plot(bin_edges[:-1], hist_norm, color=line_color)

    plt.title('Relative waterfall plot')
    plt.xlabel('m/q')
    plt.ylabel('Relative normalized counts')
    plt.xlim(x_lower, x_upper)
    plt.show()


def nevents_heatmap_rel(hists,nbins_mq,bins,xlimits=(0,200)):
    "Heatmap of relative normalized counts with number of events slices on the y axis, with respect to m/q on the x axis using hists, which is a list of histograms"
    
    precision = 200/nbins_mq
    x_lower, x_upper = xlimits
    hist_lower, hist_upper = int(x_lower/precision), int(x_upper/precision)
    nbins_events = len(hists)
    hists_norm = []

    for i in range(nbins_events):

        shortened_hist = hists[i][hist_lower:hist_upper]
        hist_norm = shortened_hist / max(shortened_hist)
        hists_norm.append(hist_norm)

    x_edges = np.linspace(x_lower, x_upper, hist_upper-hist_lower)
    y_edges = bins[:-1]
    X, Y = np.meshgrid(x_edges, y_edges)

    plt.figure(figsize=(20, 8))
    c = plt.pcolormesh(X, Y, hists_norm, shading='auto')
    plt.colorbar(c, label='Relative normalized counts', extend='max')
    plt.xlabel('m/q')
    plt.ylabel('Number of events slice')
    plt.title('Relative heatmap for number of events slices with respect to m/q')
    plt.xlim(x_lower, x_upper)
    plt.show()


def nevents_heatmap_rel_tof(hists,nbins_tof,bins,xlimits=(0,TIME_BETWEEN_PULSES)):
    "Heatmap of relative normalized counts with number of events slices on the y axis, with respect to tof on the x axis using hists, which is a list of histograms"
    
    precision = TIME_BETWEEN_PULSES/nbins_tof
    x_lower, x_upper = xlimits
    hist_lower, hist_upper = int(x_lower/precision), int(x_upper/precision)
    nbins_events = len(hists)
    hists_norm = []

    for i in range(nbins_events):

        shortened_hist = hists[i][hist_lower:hist_upper]
        hist_norm = shortened_hist / max(shortened_hist)
        hists_norm.append(hist_norm)

    x_edges = np.linspace(x_lower, x_upper, hist_upper-hist_lower)
    X, Y = np.meshgrid(x_edges, bins[:-1])

    plt.figure(figsize=(20, 8))
    c = plt.pcolormesh(X, Y, hists_norm, shading='auto')
    plt.colorbar(c, label='Relative normalized counts', extend='max')
    plt.xlabel('Time of flight (s)')
    plt.ylabel('Number of events slice')
    plt.title('Relative heatmap for number of events slices with respect to tof')
    plt.xlim(x_lower, x_upper)
    # plt.yticks(bins[:-1])
    plt.show()


def waterfall_abs(hists,nbins_mq,xlimits=(0,200)):
    "Waterfall plot of absolute normalization with respect to m/q using hists, which is a list of histograms"

    x_lower, x_upper = xlimits
    nbins_events = len(hists)
    bin_edges = np.linspace(0, 200, nbins_mq+1)
    hists_norm = hists/np.max(hists)

    plt.figure(figsize=(20, 8))
    colormap = plt.cm.inferno
    color_index = np.linspace(0.2, 0.8, nbins_events)

    for i in range(nbins_events):

        line_color = colormap(color_index[i])
        plt.plot(bin_edges[:-1], hists_norm[i] + i, color=line_color)

    plt.title('Absolute waterfall plot')
    plt.xlabel('m/q')
    plt.ylabel('Relative normalized counts')
    plt.xlim(x_lower, x_upper)
    plt.show()


def nevents_heatmap_abs(hists,nbins_mq,bins,xlimits=(0,200)):
    "Heatmap of absolute normalized counts with number of events slices on the y axis, with respect to m/q on the x axis using hists, which is a list of histograms"
    
    precision = 200/nbins_mq
    x_lower, x_upper = xlimits
    hist_lower, hist_upper = int(x_lower/precision), int(x_upper/precision)
    hists_shortened = hists[:,hist_lower:hist_upper]
    nbins_events = len(hists_shortened)
    hists_norm = hists_shortened/np.max(hists_shortened)

    x_edges = np.linspace(0, 200, nbins_mq)
    y_edges = bins[:-1]
    X, Y = np.meshgrid(x_edges, y_edges)
    
    plt.figure(figsize=(20, 8))
    c = plt.pcolormesh(X, Y, hists_norm, shading='auto')
    plt.colorbar(c, label='Relative normalized counts', extend='max')
    plt.xlabel('m/q')
    plt.ylabel('Number of events slice')
    plt.title('Absolute heatmap for number of events slices with respect to m/q')
    plt.xlim(x_lower, x_upper)
    plt.show()


def nevents_heatmap_abs_tof(hists,nbins_tof,bins,xlimits=(0,TIME_BETWEEN_PULSES)):
    "Heatmap of absolute normalized counts with number of events slices on the y axis, with respect to tof on the x axis using hists, which is a list of histograms"
    
    precision = TIME_BETWEEN_PULSES/nbins_tof
    x_lower, x_upper = xlimits
    hist_lower, hist_upper = int(x_lower/precision), int(x_upper/precision)
    hists_shortened = hists[:,hist_lower:hist_upper]
    nbins_events = len(hists_shortened)
    hists_norm = hists_shortened/np.max(hists_shortened)

    x_edges = np.linspace(x_lower, x_upper, hist_upper-hist_lower)
    X, Y = np.meshgrid(x_edges, bins[:-1])

    plt.figure(figsize=(20, 8))
    c = plt.pcolormesh(X, Y, hists_norm, shading='auto')
    plt.colorbar(c, label='Relative normalized counts', extend='max')
    plt.xlabel('Time of flight (s)')
    plt.ylabel('Number of events slice')
    plt.title('Absolute heatmap for number of events slices with respect to tof')
    plt.xlim(x_lower, x_upper)
    # plt.yticks(bins[:-1])
    plt.show()


def waterfall_etof(filtered_etofs,xlimits=(0,200)):
    'Waterfall plot of etof data using list of etofs filtered_etofs'

    x_lower, x_upper = xlimits
    nbins = len(filtered_etofs)
    channel_time = TIME_BETWEEN_PULSES/CHANNELS_PER_PULSE
    xaxis = np.arange(14080)*channel_time

    plt.figure(figsize=(20, 8))
    colormap = plt.cm.inferno
    color_index = np.linspace(0.2, 0.8, nbins)

    for i in range(nbins):
        
        summed_etof = -np.sum(filtered_etofs[i],axis=0)
        line_color = colormap(color_index[i])
        plt.plot(xaxis, summed_etof/np.max(summed_etof) + i, color=line_color)

    plt.title('Relative electron waterfall plot')
    plt.xlabel('Time of flight (s)')
    plt.ylabel('Normalized signal')
    plt.show()
