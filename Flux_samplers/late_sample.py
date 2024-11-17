import numpy as np
import afterglowpy as grb
import emcee
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import curve_fit
import corner
import os
import sys
import multiprocessing
from multiprocessing import Pool
import time
from functools import partial
import concurrent.futures
import sampler as splr

def autocorr(x):
    """Compute the autocorrelation of a 1D numpy array."""
    result = np.correlate(x, x, mode='full')  # Cross-correlate with itself
    mid = result.size // 2  # Get the middle index (center of correlation)
    return result[mid:] / result[mid]  # Normalize by the zero lag correlation


z = 2
d_L = 4.88e28

# core angle, log10 density, electron distribution, log10 thermal fraction, log10 magnetic fraction, log10 isotropic energy, observation angle
initial = np.array([0.1,0.0,2.33,-1.0,-3.0,51.0])

#unpack data
time, freq, flux, flux_err = np.genfromtxt('../Data_Generation/data/data_second_half.csv',delimiter=',',skip_header=1,unpack=True)


#if __name__ == "__main__":
    #splr.run_parallel_optimization([time,freq],flux,initial,flux_err,processes=2)
#splr.run_optimization([time,freq],flux,initial,flux_err,z=z,d_L=d_L)

if __name__ == "__main__":
    splr.run_sampling([time,freq],flux,initial,flux_err,d_L=d_L,z=z,steps=30000,processes=40,genfile=1,parallel_optimization=0,nwalkers=32,filename='../../../Large_data/late_samples.h5')


    file_path = '../../../Large_data/late_samples.h5'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The HDF5 file '{file_path}' does not exist.")
    backend = emcee.backends.HDFBackend(file_path)
    if backend.iteration == 0:
        raise ValueError("The HDF5 file is empty. No data was stored.")
    reader = emcee.backends.HDFBackend(file_path)


    try:
        tau = reader.get_autocorr_time()
        burnin = int(2 * np.max(tau))
        thin = int(0.5 * np.min(tau))
    except emcee.autocorr.AutocorrError:
        print("Warning: Autocorrelation time estimation failed. Proceeding with current chain length.")
        burnin = len(reader.get_chain()) // 3  # Use some default burn-in, e.g., one-third of the chain length
        thin = 1  # No thinning
    labels = ["thetaCore", "n0", "p", "epsilon_e", "epsilon_B", "E0"]
    # Plot the sampling results
    samples = reader.get_chain()
    fig, axes = plt.subplots(len(labels), figsize=(10, 6), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")
    fig.savefig('./graph/late_steps.png')
    plt.close(fig)
    flat_samples = reader.get_chain(discard=burnin,thin=thin,flat=True)
    fig2 = corner.corner(flat_samples, labels=labels)
    fig2.savefig('./graph/late_contour.png')
    plt.close(fig2)
    max_samples = 1000
    
    # Use a subset of the samples to speed up the computation
    subset_samples = flat_samples[:max_samples, 0]  # First parameter (example)

    ac = autocorr(subset_samples)

    # Plot the autocorrelation for the first parameter
    plt.plot(ac[:100])  # Plot the first 100 lags
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title('Autocorrelation for First Parameter')
    plt.savefig('./graph/late_ac.png')
    plt.close()

