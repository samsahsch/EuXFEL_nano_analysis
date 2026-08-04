"""pnCCD dark correction, display and integration (coherent diffractive imaging)."""

import numpy as np
import matplotlib.pyplot as plt


def getBadPixelMask(avg_dark, dark_thr, std_dark, std_thr):
    mask = np.ones_like(avg_dark)

    mask[avg_dark>dark_thr]=0
    mask[std_dark>std_thr]=0
    mask[std_dark==0]=0

    return mask


def correct_pnccd(pnccd, dark_pnccd):
    'Correct pnccd using dark run pnccd input'
    
    avg_dark= dark_pnccd.mean(axis=0)
    std_dark= dark_pnccd.std(axis=0)
    
    mask = getBadPixelMask(avg_dark, 60000, std_dark, 600)
    
    corr_pnccd = (pnccd - avg_dark)*mask
    
    pretty_pnccd = corr_pnccd.isel(dim_0=slice(412, 512), dim_1=slice(412, 612))
    
    return pretty_pnccd


def pnccd_image(corr_pnccd, trainId):
    'Show pnccd image for given train id'
    
    image_data = corr_pnccd.sel(trainId=trainId).squeeze()
    
    plt.figure(figsize=(10,10))
    plt.imshow(image_data, vmin=10)
    plt.title(f'PNCCD image of train {trainId}')
    plt.show()


def integrate_pnccd(corr_pnccd, trainId, prnt=True):
    'Integrate upper half of pnncd for given train id'
    
    image_data = corr_pnccd.sel(trainId=trainId).squeeze()
    number = np.round(image_data.where(image_data > 10).sum().item())
    
    if prnt:
        print(f'Integrating yields {number}')
    else:
        return number
