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

z = 1.619
d_L = 39.755*9.461e26

# core angle, log10 density, electron distribution, log10 thermal fraction, log10 magnetic fraction, log10 isotropic energy, observation angle
initial = np.array([0.1,1.0,2.3,-1.0,-3.0,53.0])

#unpack data
time, freq, flux, flux_err = np.genfromtxt('./data/990510.csv',delimiter=',',skip_header=1,unpack=True)

#format the data
# Get unique times and frequencies, and map them to indices
t, time_indices = np.unique(time, return_inverse=True)
nu, freq_indices = np.unique(freq, return_inverse=True)

# Initialize an empty array with NaN values
F = np.full((len(t), len(nu)), np.nan)
f_err = np.full((len(t),len(nu)), np.nan)

# Populate the arrays using indices
F[time_indices, freq_indices] = np.log(flux)
f_err[time_indices, freq_indices] = flux_err/flux
t_err = np.zeros((len(t), len(nu)))
nu_err = np.zeros((len(t), len(nu)))

err = [f_err,t_err,nu_err]
truth = [0.1,0.0,2.3,-2,-4,53]
"""
for freq_idx, nu_value in enumerate(nu):
    plt.errorbar(np.log10(t),F[:,freq_idx],yerr=f_err[:,freq_idx],fmt='.', label=f'{nu_value:.2e} Hz')
plt.legend()
plt.show()
"""



#if __name__ == "__main__":
    #splr.run_parallel_optimization([time,freq],flux,initial,flux_err,processes=2)
#splr.run_optimization([time,freq],flux,initial,flux_err)

if __name__ == "__main__":
    splr.run_sampling([time,freq],flux,initial,flux_err,d_L=d_L,z=z,steps=10,processes=2,genfile=1,parallel_optimization=2,nwalkers=32,filename='../../../Large_data/Real_Data6.h5')


    file_path = '../../../Large_data/Real_Data6.h5'
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
    fig.savefig('./graph/990510_steps6.png')
    plt.close(fig)
    flat_samples = reader.get_chain(discard=burnin,thin=thin,flat=True)
    fig2 = corner.corner(flat_samples, labels=labels, truths=truth)
    fig2.savefig('./graph/990510_contour6.png')
    plt.close(fig2)

