"""Covariance and partial-covariance maps between ion and electron channels."""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from .constants import TIME_BETWEEN_PULSES, channel_time


def mq_np_covariance(dfevent,mq_bins=200,log=True,vmin=None,vmax=None):
    'Produces a positive and a negative covariance map of m/q vs m/q employing the numpy cov function'
    'Uses dfevent as input, can select number of mq bins, can produce plot as log, standard, or between defined ranges'
    
    mq_bin_edges = np.linspace(0,200,mq_bins+1)

    dfevent['mq_bin'] = pd.cut(dfevent['mq'], bins=mq_bin_edges)

    result_matrix = pd.crosstab(dfevent['pulseId'], dfevent['mq_bin'])
    result_numpy_matrix = result_matrix.values
    
    cov_matrix = np.cov(result_numpy_matrix, rowvar=False)
    

    plt.figure(figsize=(10, 8))
    if log == True:
        ax = sns.heatmap(cov_matrix, cmap='viridis', fmt='.2f', norm=LogNorm())
    elif vmax == None:
        ax = sns.heatmap(cov_matrix, cmap='viridis', fmt='.2f')
    else:
        ax = sns.heatmap(cov_matrix, cmap='viridis', fmt='.2f', vmin=0, vmax=vmax)

    tick_positions = np.linspace(0, mq_bins, 11)
    tick_labels = np.linspace(0, 200, 11).astype(int)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    ax.invert_yaxis()

    plt.title('Positive Numpy Covariance Heatmap  m/q vs m/q')
    plt.xlabel('m/q')
    plt.ylabel('m/q')
    plt.show()
    

    plt.figure(figsize=(10, 8))
    if log == True:
        ax = sns.heatmap(-cov_matrix, cmap='viridis', fmt='.2f', norm=LogNorm())
    elif vmin == None:
        ax = sns.heatmap(-cov_matrix, cmap='viridis', fmt='.2f')
    else:
        ax = sns.heatmap(-cov_matrix, cmap='viridis', fmt='.2f', vmin=0, vmax=-vmin)

    tick_positions = np.linspace(0, mq_bins, 11)
    tick_labels = np.linspace(0, 200, 11).astype(int)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    ax.invert_yaxis()

    plt.title('Negative Numpy Covariance Heatmap  m/q vs m/q')
    plt.xlabel('m/q')
    plt.ylabel('m/q')
    plt.show()


def mq_covariance_1d(dfevent,mq_bin_range,mq_bins=200):
    'Produces 1d covariance of an m/q bin range vs m/q employing the numpy cov function'
    'Uses dfevent as input, can select number of mq bins'
    
    mq_bin_edges = np.linspace(0,200,mq_bins+1)

    dfevent['mq_bin'] = pd.cut(dfevent['mq'], bins=mq_bin_edges)

    result_matrix = pd.crosstab(dfevent['pulseId'], dfevent['mq_bin'])
    result_numpy_matrix = result_matrix.values

    array_hyd = result_matrix.values[:,mq_bin_range[0]:mq_bin_range[1]].sum(axis=1)
    shape = result_numpy_matrix.shape[1]

    # Calculate the covariance between each row of result_numpy_matrix and array_hyd
    covariances = np.array([np.cov(np.column_stack((result_numpy_matrix[:, j], array_hyd)), rowvar=False)[0, 1] for j in range(shape)])

    xaxis = np.linspace(0,200,shape)
    fig, ax = plt.subplots()
    ax.plot(xaxis,covariances)
    plt.xlabel('m/q')
    plt.ylabel('Covariance with m/q')
    plt.title(f'Covariance Plot between m/q bin range {mq_bin_range} and m/q')
    ax.set_yscale('symlog', linthresh=10)
    plt.show()


def fix_missing_row(dfevent,dfpulse,mq_bins=200):
    'Fixes the missing row in dfevent dataframe when computing the cross-tabulation of pulseId and mq_bin'
    
    mq_bin_edges = np.linspace(0,200,mq_bins+1)

    dfevent['mq_bin'] = pd.cut(dfevent['mq'], bins=mq_bin_edges)

    result_matrix = pd.crosstab(dfevent['pulseId'], dfevent['mq_bin'])
    result_numpy_matrix = result_matrix.values
    
    resultlist = result_matrix.index.to_list()
    resultlist.append(0)
    selectedlist = dfpulse[dfpulse.pulseId.isin(dfevent.pulseId)].pulseId.to_list()
    truefalse = np.equal(resultlist,selectedlist)
    first_instance = np.argmax(~truefalse)
    missing_pulse = int(dfpulse.iloc[first_instance].pulseId)

    new_dfevent = dfevent[dfevent.pulseId != missing_pulse]
    
    return new_dfevent


def calc_corrs(array1, array2, pcovparams, alpha=1):
    print('calculating covariance')

    assert len(pcovparams)==len(array1)==len(array2)
    numshots=len(array1)
    
    # heavy stuff
    syx=np.einsum('ij,ik->jk', array1, array2)
    print('calculated syx')
    syi=np.einsum('ij,i->j', array1, pcovparams)
    print('calculated syi')
    six=np.einsum('ij,i->j', array2, pcovparams)
    print('calculated six')

    # lighter stuff
    sy=array1.sum(axis=0)
    sx=array2.sum(axis=0)
    si=pcovparams.sum(axis=0)
    
    syy=(array1**2).sum(axis=0)
    sxx=(array2**2).sum(axis=0)
    sii=(pcovparams**2).sum()

    sysx=np.outer(sy, sx)
    sisx=si*sx
    sysi=sy*si
    
    # calculate covariances
    covyx=(syx-sysx/numshots)/(numshots-1)
    covyi=(syi-sysi/numshots)/(numshots-1)
    covix=(six-sisx/numshots)/(numshots-1)

    covyy=(syy-sy**2/numshots)/(numshots-1)
    covxx=(sxx-sx**2/numshots)/(numshots-1)
    covii=(sii-si**2/numshots)/(numshots-1) # renamed from varii

    # calculate partial covariances
    pcovyx=(numshots-1)/(numshots-2) * (covyx - alpha * np.outer(covyi, covix)/covii)
    pcovyy=(numshots-1)/(numshots-2) * (covyy - (covyi**2)/covii)
    pcovxx=(numshots-1)/(numshots-2) * (covxx - (covix**2)/covii)
    
    # calculate correlation
    corryx = covyx / np.sqrt(np.outer(covyy, covxx))
    # calculate partial correlation
    pcorryx = pcovyx / np.sqrt(np.outer(pcovyy, pcovxx))
    
    return covyx, pcovyx, corryx, pcorryx


def mq_covariance(dfevent,dfpulse,mq_bins=200,log=True,vmin=None,vmax=None):
    'Produces covariance maps of m/q vs m/q employing the calc_corrs function'
    'Uses dfevent and dfpulse as inputs, can select number of mq bins, can produce plot as log, standard, or between defined ranges'
    
    mq_bin_edges = np.linspace(0,200,mq_bins+1)

    dfevent['mq_bin'] = pd.cut(dfevent['mq'], bins=mq_bin_edges)

    result_matrix = pd.crosstab(dfevent['pulseId'], dfevent['mq_bin'])
    result_numpy_matrix = result_matrix.values
    
    nevents_pulse = dfpulse[dfpulse.pulseId.isin(dfevent.pulseId)].nevents_pulse
    
    covyx, pcovyx, corryx, pcorryx = calc_corrs(result_numpy_matrix, result_numpy_matrix, nevents_pulse)
    
    
    tick_positions = np.linspace(0, mq_bins, 11)
    tick_labels = np.linspace(0, 200, 11).astype(int)
    
    plt.figure(figsize=(10, 8))
    if log == True:
        ax = sns.heatmap(covyx, cmap='viridis', fmt='.2f', norm=LogNorm())

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)
        ax.invert_yaxis()

        plt.title('Positive Covariance Heatmap m/q vs m/q')
        plt.xlabel('m/q')
        plt.ylabel('m/q')
        plt.show()

        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(-covyx, cmap='viridis', fmt='.2f', norm=LogNorm())

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)
        ax.invert_yaxis()

        plt.title('Negative Covariance Heatmap m/q vs m/q')
        plt.xlabel('m/q')
        plt.ylabel('m/q')
        plt.show()
    
    elif vmax == None:
        ax = sns.heatmap(covyx, cmap='seismic', fmt='.2f')
        
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)
        ax.invert_yaxis()

        plt.title('Covariance Heatmap m/q vs m/q')
        plt.xlabel('m/q')
        plt.ylabel('m/q')
        plt.show()
        
    else:
        ax = sns.heatmap(covyx, cmap='seismic', fmt='.2f', vmin=vmin, vmax=vmax)
        
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)
        ax.invert_yaxis()

        plt.title('Covariance Heatmap m/q vs m/q')
        plt.xlabel('m/q')
        plt.ylabel('m/q')
        plt.show()


def mq_partial_covariance(dfevent,dfpulse,mq_bins=200,alpha=1,log=True,vmin=None,vmax=None):
    'Produces partial covariance maps of m/q vs m/q employing the calc_corrs function'
    'Uses dfevent and dfpulse as inputs, can select number of mq bins and factor alpha, can produce plot as log, standard, or between defined ranges'
    
    mq_bin_edges = np.linspace(0,200,mq_bins+1)

    dfevent['mq_bin'] = pd.cut(dfevent['mq'], bins=mq_bin_edges)

    result_matrix = pd.crosstab(dfevent['pulseId'], dfevent['mq_bin'])
    result_numpy_matrix = result_matrix.values
    
    nevents_pulse = dfpulse[dfpulse.pulseId.isin(dfevent.pulseId)].nevents_pulse
    
    covyx, pcovyx, corryx, pcorryx = calc_corrs(result_numpy_matrix, -result_numpy_matrix, nevents_pulse, alpha)
    
    
    tick_positions = np.linspace(0, mq_bins, 11)
    tick_labels = np.linspace(0, 200, 11).astype(int)
    
    plt.figure(figsize=(10, 8))
    if log == True:
        ax = sns.heatmap(pcovyx, cmap='viridis', fmt='.2f', norm=LogNorm())

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)
        ax.invert_yaxis()

        plt.title('Positive Partial Covariance Heatmap m/q vs m/q')
        plt.xlabel('m/q')
        plt.ylabel('m/q')
        plt.show()

        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(-pcovyx, cmap='viridis', fmt='.2f', norm=LogNorm())

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)
        ax.invert_yaxis()

        plt.title('Negative Partial Covariance Heatmap m/q vs m/q')
        plt.xlabel('m/q')
        plt.ylabel('m/q')
        plt.show()
    
    elif vmax == None:
        ax = sns.heatmap(pcovyx, cmap='seismic', fmt='.2f')
        
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)
        ax.invert_yaxis()

        plt.title('Partial Covariance Heatmap m/q vs m/q')
        plt.xlabel('m/q')
        plt.ylabel('m/q')
        plt.show()
        
    else:
        ax = sns.heatmap(pcovyx, cmap='seismic', fmt='.2f', vmin=vmin, vmax=vmax)
        
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)
        ax.invert_yaxis()

        plt.title('Partial Covariance Heatmap m/q vs m/q')
        plt.xlabel('m/q')
        plt.ylabel('m/q')
        plt.show()


def ion_ion_covariance(dfevent, dfpulse, nbins_ion_tof, max_ion_limit=TIME_BETWEEN_PULSES, alpha=1, log=True, vmin=None, vmax=None):
    'Produces numpy covariance maps of ion tof vs ion tof employing the calc_corrs function'
    
    # Create bins for ion TOF
    ion_tof_bin_edges = np.linspace(0, max_ion_limit, nbins_ion_tof+1)
    dfevent['tof_bin'] = pd.cut(dfevent['tof'], bins=ion_tof_bin_edges)

    # Create ion matrix
    ion_matrix = pd.crosstab(dfevent['pulseId'], dfevent['tof_bin'])
    
    # Get number of events per pulse
    nevents_pulse = dfpulse[dfpulse.pulseId.isin(ion_matrix.index.to_numpy())].nevents_pulse

    # Calculate correlations
    covyx, pcovyx, corryx, pcorryx = calc_corrs(ion_matrix.values, ion_matrix.values, nevents_pulse, alpha)

    # Setup tick positions and labels
    ion_tick_positions = np.linspace(0, nbins_ion_tof, 5).astype(int)
    formatted_ion_tick_labels = np.linspace(0, max_ion_limit, 5)
    
    # Plotting functions
    def plot_heatmap(data, title, is_negative=False):
        plt.figure(figsize=(10, 8))
        if log:
            data_to_plot = -data if is_negative else data
            ax = sns.heatmap(data_to_plot, cmap='viridis', fmt='.2f', norm=LogNorm())
        elif vmax is None:
            ax = sns.heatmap(data, cmap='seismic', fmt='.2f')
        else:
            ax = sns.heatmap(data, cmap='seismic', fmt='.2f', vmin=vmin, vmax=vmax)
            
        ax.set_yticks(ion_tick_positions)
        ax.set_yticklabels(formatted_ion_tick_labels)
        ax.set_xticks(ion_tick_positions)
        ax.set_xticklabels(formatted_ion_tick_labels)
        ax.invert_yaxis()
        plt.title(title)
        plt.ylabel('Ion tof')
        plt.xlabel('Ion tof')
        plt.show()
    
    # Plot covariance maps
    if log:
        plot_heatmap(covyx, 'Positive Covariance Heatmap ion tof vs ion tof')
        plot_heatmap(covyx, 'Negative Covariance Heatmap ion tof vs ion tof', is_negative=True)
    else:
        plot_heatmap(covyx, 'Covariance Heatmap ion tof vs ion tof')
    
    # Plot partial covariance maps
    if log:
        plot_heatmap(pcovyx, 'Positive Partial Covariance Heatmap ion tof vs ion tof')
        plot_heatmap(pcovyx, 'Negative Partial Covariance Heatmap ion tof vs ion tof', is_negative=True)
    else:
        plot_heatmap(pcovyx, 'Partial Covariance Heatmap ion tof vs ion tof')


def etof_ion_covariance(dfevent,dfpulse,etof,nbins_ion_tof,nbins_e_tof,max_ion_limit=TIME_BETWEEN_PULSES,max_e_limit=TIME_BETWEEN_PULSES,alpha=1,log=True,vmin=None,vmax=None):
    'Produces numpy covariance maps of etof vs ion employing the calc_corrs function'
    'Uses dfevent and etof as inputs, can select number of bins along ion tof and electron tof, can produce plot as log, standard, or between defined ranges, and set alpha'
    'Can select maximal time limit for electrons and ions'
    
    ion_tof_bin_edges = np.linspace(0,max_ion_limit,nbins_ion_tof+1)
    dfevent['tof_bin'] = pd.cut(dfevent['tof'], bins=ion_tof_bin_edges)

    ion_matrix = pd.crosstab(dfevent['pulseId'], dfevent['tof_bin'])


    coords_etof = etof.assign_coords(data=np.arange(0,TIME_BETWEEN_PULSES,channel_time))

    e_tof_bin_edges = np.linspace(0,max_e_limit,nbins_e_tof+1)
    binned_etof = coords_etof.groupby_bins("data", e_tof_bin_edges).sum()
    e_matrix = binned_etof.to_pandas()
    
    ion_list = ion_matrix.index.to_list()
    # ion_list.append(0)
    e_list = e_matrix.index.to_list()
    truefalse = np.equal(ion_list,e_list)
    first_instance = np.argmax(~truefalse)

    new_e_matrix = e_matrix#.drop(e_list[first_instance])


    nevents_pulse = dfpulse[dfpulse.pulseId.isin(ion_matrix.index.to_numpy())].nevents_pulse

    covyx, pcovyx, corryx, pcorryx = calc_corrs(ion_matrix.values, -new_e_matrix.values, nevents_pulse, alpha)


    ion_tick_positions = np.linspace(0, nbins_ion_tof, 5).astype(int)
    formatted_ion_tick_labels = np.linspace(0, max_ion_limit, 5)

    e_tick_positions = np.linspace(0, nbins_e_tof, 5).astype(int)
    formatted_e_tick_labels = np.linspace(0, max_e_limit, 5)
    
    
    plt.figure(figsize=(10, 8))
    if log == True:
        ax = sns.heatmap(covyx, cmap='viridis', fmt='.2f', norm=LogNorm())
        ax.set_yticks(ion_tick_positions)
        ax.set_yticklabels(formatted_ion_tick_labels)
        ax.set_xticks(e_tick_positions)
        ax.set_xticklabels(formatted_e_tick_labels)
        ax.invert_yaxis()
        plt.title('Positive Covariance Heatmap etof vs ion tof')
        plt.ylabel('Ion tof')
        plt.xlabel('Electron tof')
        plt.show()

        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(-covyx, cmap='viridis', fmt='.2f', norm=LogNorm())
        ax.set_yticks(ion_tick_positions)
        ax.set_yticklabels(formatted_ion_tick_labels)
        ax.set_xticks(e_tick_positions)
        ax.set_xticklabels(formatted_e_tick_labels)
        ax.invert_yaxis()
        plt.title('Negative Covariance Heatmap etof vs ion tof')
        plt.ylabel('Ion tof')
        plt.xlabel('Electron tof')
        plt.show()
    
    elif vmax == None:
        ax = sns.heatmap(covyx, cmap='seismic', fmt='.2f')
        ax.set_yticks(ion_tick_positions)
        ax.set_yticklabels(formatted_ion_tick_labels)
        ax.set_xticks(e_tick_positions)
        ax.set_xticklabels(formatted_e_tick_labels)
        ax.invert_yaxis()
        plt.title('Covariance Heatmap etof vs ion tof')
        plt.ylabel('Ion tof')
        plt.xlabel('Electron tof')
        plt.show()
        
    else:
        ax = sns.heatmap(covyx, cmap='seismic', fmt='.2f', vmin=vmin, vmax=vmax)
        ax.set_yticks(ion_tick_positions)
        ax.set_yticklabels(formatted_ion_tick_labels)
        ax.set_xticks(e_tick_positions)
        ax.set_xticklabels(formatted_e_tick_labels)
        ax.invert_yaxis()
        plt.title('Covariance Heatmap etof vs ion tof')
        plt.ylabel('Ion tof')
        plt.xlabel('Electron tof')
        plt.show()
    
    
    plt.figure(figsize=(10, 8))
    if log == True:
        ax = sns.heatmap(pcovyx, cmap='viridis', fmt='.2f', norm=LogNorm())
        ax.set_yticks(ion_tick_positions)
        ax.set_yticklabels(formatted_ion_tick_labels)
        ax.set_xticks(e_tick_positions)
        ax.set_xticklabels(formatted_e_tick_labels)
        ax.invert_yaxis()
        plt.title('Positive Partial Covariance Heatmap etof vs ion tof')
        plt.ylabel('Ion tof')
        plt.xlabel('Electron tof')
        plt.show()

        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(-pcovyx, cmap='viridis', fmt='.2f', norm=LogNorm())
        ax.set_yticks(ion_tick_positions)
        ax.set_yticklabels(formatted_ion_tick_labels)
        ax.set_xticks(e_tick_positions)
        ax.set_xticklabels(formatted_e_tick_labels)
        ax.invert_yaxis()
        plt.title('Negative Partial Covariance Heatmap etof vs ion tof')
        plt.ylabel('Ion tof')
        plt.xlabel('Electron tof')
        plt.show()
    
    elif vmax == None:
        ax = sns.heatmap(pcovyx, cmap='seismic', fmt='.2f')
        ax.set_yticks(ion_tick_positions)
        ax.set_yticklabels(formatted_ion_tick_labels)
        ax.set_xticks(e_tick_positions)
        ax.set_xticklabels(formatted_e_tick_labels)
        ax.invert_yaxis()
        plt.title('Partial Covariance Heatmap etof vs ion tof')
        plt.ylabel('Ion tof')
        plt.xlabel('Electron tof')
        plt.show()
        
    else:
        ax = sns.heatmap(pcovyx, cmap='seismic', fmt='.2f', vmin=vmin, vmax=vmax)
        ax.set_yticks(ion_tick_positions)
        ax.set_yticklabels(formatted_ion_tick_labels)
        ax.set_xticks(e_tick_positions)
        ax.set_xticklabels(formatted_e_tick_labels)
        ax.invert_yaxis()
        plt.title('Partial Covariance Heatmap etof vs ion tof')
        plt.ylabel('Ion tof')
        plt.xlabel('Electron tof')
        plt.show()
