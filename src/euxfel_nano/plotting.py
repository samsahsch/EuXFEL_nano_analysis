"""Diagnostic and exploratory plots for the ion and electron detectors."""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from .constants import TIME_BETWEEN_PULSES, CHANNELS_PER_PULSE


def heatmap(dfevent):
    'Creates heatmap of the ions hits, based on a dfevent dataframe'
    
    counts_df = dfevent.groupby(['x', 'y']).size().reset_index(name='count')
    heatmap_data = counts_df.pivot(index='y', columns='x', values='count')
    
    plt.figure()
    ax = sns.heatmap(heatmap_data, cmap='viridis',cbar_kws={'label': 'Number of events'})
    plt.title('Ion heatmap')
    plt.show()


def ion_tof(dfevent,xlimits=(0,TIME_BETWEEN_PULSES),nbins=1000):
    'Plots ion time of flight data using dfevent dataframe'
    
    bounded_dfevent = dfevent[dfevent.tof > xlimits[0]][dfevent.tof < xlimits[1]]
    hist, bin_edges = np.histogram(bounded_dfevent.tof, bins=nbins)
    
    plt.figure()
    plt.plot(bin_edges[:-1], hist)
    plt.xlabel('Time of flight (s)')
    plt.ylabel('Number of hits per bin')
    plt.title('Ions time of flight')
    # plt.xlim(xlimits[0],xlimits[1])
    plt.show()


def e_tof(etof):
    'Plots electron time of flight data using etof xarray data'
    
    channel_time = TIME_BETWEEN_PULSES/CHANNELS_PER_PULSE
    
    xaxis = np.arange(14080)*channel_time
    avg_selected_etof = -np.mean(etof, axis=0)
    
    plt.figure()
    plt.plot(xaxis,avg_selected_etof/max(avg_selected_etof))
    plt.xlabel('Time of flight (s)')
    plt.ylabel('Normalized signal')
    plt.title('Electrons time of flight')
    plt.show()


def events_selection_plots(runs,thresholds,downsampling=None):
    'Runs functions events_selection, heatmap, e_tof, ion_tof'
    'Downsamples by downsampling integer value if one is given'
    
    selections = events_selection(runs,thresholds,downsampling)
    
    print(f'\n Plots for selection between {thresholds[0][0]} and {thresholds[0][1]} events:')
    selected_dfevent1, selected_dfpulse1, selected_etof1, selected_pnccd1 = selections[0]
    heatmap(selected_dfevent1)
    ion_tof(selected_dfevent1)
    e_tof(selected_etof1)
    
    if len(selections) > 1:
        print(f'Plots for selection between {thresholds[1][0]} and {thresholds[1][1]} events:')
        selected_dfevent2, selected_dfpulse2, selected_etof2, selected_pnccd2 = selections[1]
        heatmap(selected_dfevent2)
        ion_tof(selected_dfevent2)
        e_tof(selected_etof2)
    
    elif len(selections) > 2:
        print(f'Plots for selection between {thresholds[2][0]} and {thresholds[2][1]} events:')
        selected_dfevent3, selected_dfpulse3, selected_etof3, selected_pnccd3 = selections[2]
        heatmap(selected_dfevent3)
        ion_tof(selected_dfevent3)
        e_tof(selected_etof3)
    
    return selections


def pulse_filtered_ion_tof(dfevent,pulse_number=None,xlim=TIME_BETWEEN_PULSES):
    'Plots ion time of flight data using dfevent dataframe'
    
    if pulse_number != None:
        pulse_filtered_dfevent = dfevent[dfevent.pulseId.astype(str).str.endswith(pulse_number)]
    else:
        pulse_filtered_dfevent = dfevent
    
    bounded_dfevent = pulse_filtered_dfevent[pulse_filtered_dfevent.tof < TIME_BETWEEN_PULSES]
    hist, bin_edges = np.histogram(bounded_dfevent.tof, bins=1000)
    
    plt.figure()
    plt.plot(bin_edges[:-1], hist/max(hist))
    plt.xlabel('Time of flight (s)')
    plt.ylabel('Normalized signal')
    plt.title('Ions time of flight')
    plt.xlim(0,xlim)
    plt.yscale('log')
    plt.show()
    
    plt.figure()
    plt.plot(bin_edges[:-1], hist/max(hist))
    plt.xlabel('Time of flight (s)')
    plt.ylabel('Normalized signal')
    plt.title('Ions time of flight')
    plt.xlim(0,TIME_BETWEEN_PULSES)
    plt.yscale('log')
    plt.show()


def heatmap_with_zones(dfevent,zones):
    'Creates heatmap of the ions hits, based on a dfevent dataframe'
    'Draws a rectangle around zones defined by a list of tuples where each tuple represents a tilted zone (xstart, ystart, width, height, angle in degrees)'
    
    counts_df = dfevent.groupby(['x', 'y']).size().reset_index(name='count')
    heatmap_data = counts_df.pivot(index='y', columns='x', values='count')
    
    plt.figure()
    ax = sns.heatmap(heatmap_data, cmap='viridis', cbar_kws={'label': 'Number of events'})
    
    xlim = int(ax.get_xticklabels()[0].get_text())
    ylim = int(ax.get_yticklabels()[0].get_text())

    for zone in zones:
        x, y, width, height, angle = zone
        x_adjusted = x - xlim
        y_adjusted = y - ylim
        
        rect = plt.Rectangle((x_adjusted, y_adjusted), width, height, fill=False, edgecolor='red', lw=1, angle=angle)
        ax.add_patch(rect)
    
    plt.title('Ion heatmap')
    plt.show()


def big_ion_tof(dfevent):
    'Plots widget of big ion time of flight data using dfevent dataframe'
    
    bounded_dfevent = dfevent[dfevent.tof < TIME_BETWEEN_PULSES]
    hist, bin_edges = np.histogram(bounded_dfevent.tof, bins=1000)
    
    plt.figure(figsize=(18, 8))
    plt.plot(bin_edges[:-1], hist, c='g')
    plt.xlabel('Time of flight (s)')
    plt.ylabel('Number of hits per bin')
    plt.title('Ions time of flight')
    plt.show()


def autoscale_y(ax,margin=0.1):
    """This function rescales the y-axis based on the data that is visible given the current xlim of the axis.
    ax -- a matplotlib axes object
    margin -- the fraction of the total height of the y-data to pad the upper and lower ylims"""

    import numpy as np

    def get_bottom_top(line):
        xd = line.get_xdata()
        yd = line.get_ydata()
        lo,hi = ax.get_xlim()
        y_displayed = yd[((xd>lo) & (xd<hi))]
        h = np.max(y_displayed) - np.min(y_displayed)
        bot = np.min(y_displayed)-margin*h
        top = np.max(y_displayed)+margin*h
        return bot,top

    lines = ax.get_lines()
    bot,top = np.inf, -np.inf

    for line in lines:
        new_bot, new_top = get_bottom_top(line)
        if new_bot < bot: bot = new_bot
        if new_top > top: top = new_top

    ax.set_ylim(bot,top)


def zoomed_ion_tof(dfevent,anchor):
    'Plots zoom around anchor point of ion time of flight data using dfevent dataframe'
    
    bounded_dfevent = dfevent[dfevent.tof < TIME_BETWEEN_PULSES]
    hist, bin_edges = np.histogram(bounded_dfevent.tof, bins=1000)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(bin_edges[:-1], hist, c='g')
    plt.xlabel('Time of flight (s)')
    plt.ylabel('Number of hits per bin')
    plt.title('Ions time of flight')
    plt.xlim(anchor-1e-7,anchor+1e-7)
    autoscale_y(ax)
    ax.axvline(x=anchor, color='black', linestyle='--')
    plt.show()


def fish_plot_x(dfevent,zone,tof_bins=1000,mq_bins=500):
    'Produces fish plots along x with respect to time of flight and m/q from dfevent dataframe and a zone defined as (startx, starty, width, height, angle in degrees)'
    
    fish_dfevent = fish_plot_ion_selection(dfevent,zone)
    
    tof_bin_edges = np.linspace(0,TIME_BETWEEN_PULSES,tof_bins+1)
    fish_dfevent['tof_binned'] = pd.cut(fish_dfevent['tof'], bins=tof_bin_edges, labels=tof_bin_edges[:-1].astype('str'))
    tof_pivot_table = fish_dfevent.pivot_table(index='x', columns='tof_binned', aggfunc='size', fill_value=0)
    max_tof_value = tof_pivot_table.values.max()
    normalized_tof = tof_pivot_table.values / max_tof_value
    
    mq_bin_edges = np.linspace(0,200,mq_bins+1)
    fish_dfevent['mq_binned'] = pd.cut(fish_dfevent['mq'], bins=mq_bin_edges, labels=mq_bin_edges[:-1].astype('str'))
    mq_pivot_table = fish_dfevent.pivot_table(index='x', columns='mq_binned', aggfunc='size', fill_value=0)
    max_mq_value = mq_pivot_table.values.max()
    normalized_mq = mq_pivot_table.values / max_mq_value
    
    
    fig, axes = plt.subplots(2, 1, figsize=(30, 12))
    
    cax_tof = axes[0].imshow(normalized_tof, cmap='viridis', aspect='auto', norm=LogNorm(), extent=[tof_bin_edges.min(), tof_bin_edges.max(), 0, 1])
    axes[0].set_xlabel('Time of flight (s)')
    axes[0].set_xlim(0,5e-7)
    axes[0].set_ylabel('x')
    axes[0].set_yticklabels(np.linspace(256,0,6,dtype=int))
    axes[0].set_title('Fish plot along x with respect to time of flight')
    cbar = plt.colorbar(cax_tof, ax=axes[0], label='Normalized number of events', norm=LogNorm())
    
    cax_mq = axes[1].imshow(normalized_mq, cmap='viridis', aspect='auto', norm=LogNorm(), extent=[mq_bin_edges.min(), mq_bin_edges.max(), 0, 1])
    axes[1].set_xlabel('m/q')
    axes[1].set_ylabel('x')
    axes[1].set_yticklabels(np.linspace(256,0,6,dtype=int))
    axes[1].set_title('Fish plot along x with respect to m/q')
    cbar = plt.colorbar(cax_mq, ax=axes[1], label='Normalized number of events', norm=LogNorm())
    
    plt.show()


def fish_plot_y(dfevent,zone,tof_bins=1000,mq_bins=500):
    'Produces a fish plot along y from dfevent dataframe and a zone defined as (startx, starty, width, height, angle in degrees)'
    
    fish_dfevent = fish_plot_ion_selection(dfevent,zone)
    
    tof_bin_edges = np.linspace(0,TIME_BETWEEN_PULSES,tof_bins+1)
    fish_dfevent['tof_binned'] = pd.cut(fish_dfevent['tof'], bins=tof_bin_edges, labels=tof_bin_edges[:-1].astype('str'))
    tof_pivot_table = fish_dfevent.pivot_table(index='y', columns='tof_binned', aggfunc='size', fill_value=0)
    max_tof_value = tof_pivot_table.values.max()
    normalized_tof = tof_pivot_table.values / max_tof_value
    
    mq_bin_edges = np.linspace(0,200,mq_bins+1)
    fish_dfevent['mq_binned'] = pd.cut(fish_dfevent['mq'], bins=mq_bin_edges, labels=mq_bin_edges[:-1].astype('str'))
    mq_pivot_table = fish_dfevent.pivot_table(index='y', columns='mq_binned', aggfunc='size', fill_value=0)
    max_mq_value = mq_pivot_table.values.max()
    normalized_mq = mq_pivot_table.values / max_mq_value
    
    fig, axes = plt.subplots(2, 1, figsize=(30, 12))
    
    cax_tof = axes[0].imshow(normalized_tof, cmap='viridis', aspect='auto', norm=LogNorm(), extent=[tof_bin_edges.min(), tof_bin_edges.max(), 0, 1])
    axes[0].set_xlabel('Time of flight (s)')
    axes[0].set_ylabel('y')
    axes[0].set_yticklabels(np.linspace(256,0,6,dtype=int))
    axes[0].set_title('Fish plot along y with respect to time of flight')
    cbar = plt.colorbar(cax_tof, ax=axes[0], label='Normalized number of events', norm=LogNorm())
    
    cax_mq = axes[1].imshow(normalized_mq, cmap='viridis', aspect='auto', norm=LogNorm(), extent=[mq_bin_edges.min(), mq_bin_edges.max(), 0, 1])
    axes[1].set_xlabel('m/q')
    axes[1].set_ylabel('y')
    axes[1].set_yticklabels(np.linspace(256,0,6,dtype=int))
    axes[1].set_title('Fish plot along y with respect to m/q')
    cbar = plt.colorbar(cax_mq, ax=axes[1], label='Normalized number of events', norm=LogNorm())

    plt.show()


def big_mq_plot(dfevent,nbins_mq=1500,xlimits=(0,200)):
    'Plots big intensity vs m/q plot using dfevent'
    
    x_lower, x_upper = xlimits
    
    plt.figure(figsize=(18,8))
        
    hist, bin_edges = np.histogram(dfevent.mq, bins=np.linspace(0,200,nbins_mq+1))
    plt.plot(bin_edges[:-1], hist/max(hist))
    
    plt.xlabel('m/q')
    plt.ylabel('Relatively normalized number of hits per bin')
    plt.title('Normalized ions time of flight')
    plt.xlim(x_lower, x_upper)
    plt.show()
