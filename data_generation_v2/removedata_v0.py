import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import afterglowpy as grb
import matplotlib.cm as cm
import fakedata_v6 as fd6

# A function that masks an entire frequency band band
def removeband(data,band):
    time, freq, flux, UB_err, LB_err = data
    time = np.array(time)
    freq = np.array(freq)
    flux = np.array(flux)
    UB_err = np.array(UB_err)
    LB_err = np.array(LB_err)
    if band == 'x-rays':
        mask = freq < 1e16
    
    elif band == 'radio':
        mask = freq >= 1e11
        
    else:
        mask = (freq < 1e11) | (freq>=1e16)
    
    data_out = time[mask],freq[mask],flux[mask],UB_err[mask],LB_err[mask]
    
    return data_out

# A function that thins the number of measurements per band
def thinning(data, num):
    time, freq, flux, UB_err, LB_err = data
    
    time = np.array(time)
    freq = np.array(freq)
    flux = np.array(flux)
    UB_err = np.array(UB_err)
    LB_err = np.array(LB_err)
    # Define frequency bands
    bands = {
        'x-rays': freq >= 1e16,
        'optical': (freq < 1e16) & (freq >= 1e11),
        'radio': freq < 1e11
    }
    
    thinned_data = []

    for band, mask in bands.items():
        # Filter data for the specific band
        band_time = time[mask]
        band_freq = freq[mask]
        band_flux = flux[mask]
        band_UB_err = UB_err[mask]
        band_LB_err = LB_err[mask]
        
        # Identify unique frequencies in this band
        unique_freqs = np.unique(band_freq)
        
        # Sort frequencies by number of occurrences
        freq_counts = {uf: np.sum(band_freq == uf) for uf in unique_freqs}
        sorted_freqs = sorted(freq_counts, key=freq_counts.get)
        
        # Select the least common `num` frequencies
        selected_freqs = sorted_freqs[:num]
        
        # Create mask for the selected frequencies
        thinning_mask = np.isin(band_freq, selected_freqs)
        
        # Collect thinned data for the band
        thinned_band_data = (
            band_time[thinning_mask],
            band_freq[thinning_mask],
            band_flux[thinning_mask],
            band_UB_err[thinning_mask],
            band_LB_err[thinning_mask]
        )
        thinned_data.append(thinned_band_data)
    
    # Combine thinned data from all bands
    combined_data = tuple(np.concatenate([band[i] for band in thinned_data]) for i in range(5))
    return combined_data
    
# A function that splits the data into post jetbreak and pre jetbreak datasets
def jetbreak(data,jb_time):
    time, freq, flux, UB_err, LB_err = data
    time = np.array(time)
    freq = np.array(freq)
    flux = np.array(flux)
    UB_err = np.array(UB_err)
    LB_err = np.array(LB_err)
    
    prejb = time < jb_time
    postjb = time >= jb_time
    prejb_data = time[prejb],freq[prejb],flux[prejb],UB_err[prejb],LB_err[prejb]
    postjb_data = time[postjb],freq[postjb],flux[postjb],UB_err[postjb],LB_err[postjb]
    return prejb_data, postjb_data
    
