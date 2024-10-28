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


file_path = './data/combo.h5'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The HDF5 file '{file_path}' does not exist.")
backend = emcee.backends.HDFBackend(file_path)
if backend.iteration == 0:
    raise ValueError("The HDF5 file is empty. No data was stored.")
sampler = emcee.backends.HDFBackend(file_path)

ndim = 6
labels = ["thetaCore", "n0", "p", "epsilon_e", "epsilon_B", "E0"]

try:
    tau = sampler.get_autocorr_time()
    burnin = int(2 * np.max(tau))
    thin = int(0.5 * np.min(tau))
except emcee.autocorr.AutocorrError:
    print("Warning: Autocorrelation time estimation failed. Proceeding with current chain length.")
    burnin = len(sampler.get_chain()) // 3  # Use some default burn-in, e.g., one-third of the chain length
    thin = 1  # No thinning

flat_samples = sampler.get_chain(discard=burnin,thin=thin,flat=True)
mcmc = np.zeros([6,3])
q = np.zeros([6,2])
for i in range(ndim):
    mcmc[i] = np.percentile(flat_samples[:, i], [16, 50, 84])
    q[i] = np.diff(mcmc[i])
print(labels[0],":",mcmc[0][1],"+",q[0][0],"-",q[0][1])
print(labels[1],":",10**mcmc[1][1],"+",(10**mcmc[1][1])*q[1][0],"-",(10**mcmc[1][1])*q[1][1])
print(labels[2],":",mcmc[2][1],"+",q[2][0],"-",q[2][1])
print(labels[3],":",10**mcmc[3][1],"+",(10**mcmc[3][1])*q[3][0],"-",(10**mcmc[3][1])*q[3][1])
print(labels[4],":",10**mcmc[4][1],"+",(10**mcmc[4][1])*q[4][0],"-",(10**mcmc[4][1])*q[4][1])
print(labels[5],":",10**mcmc[5][1],"+",(10**mcmc[5][1])*q[5][0],"-",(10**mcmc[5][1])*q[5][1])


nu = np.geomspace(1e0, 1e20, num=7)
t = np.geomspace(1e-1 * grb.day2sec, 1e3 * grb.day2sec, num=17)
t_best = np.geomspace(1e-1 * grb.day2sec, 1e3 * grb.day2sec, num=100)
thetaObs = 0.0
xi_N = 1.0
d_L = 1.0e26
z = 0.01


# Prepare the figure
fig, ax = plt.subplots(1, 1)
flat_samples = sampler.get_chain(flat=True)
inds = np.random.randint(len(flat_samples), size=100)

# Prepare the input for Flux calculation
x = [t, nu]
x_best = [t_best,nu]
F = np.array(splr.Flux(x,0.1,0,2.3,-2,-4,53,0.0,1,1e26,0.01))
best = np.array(splr.Flux(x_best,mcmc[0][1],mcmc[1][1],mcmc[2][1],mcmc[3][1],mcmc[4][1],mcmc[5][1],0.0,1.0,1e26,0.01))
noise_level = abs(0.05*F)
F_noise = F + np.random.normal(0, noise_level, size=F.shape)
yerr = abs(0.05*F)

# Loop through the selected samples
for ind in inds:
    sample = flat_samples[ind]
    thetaCore, n0, p, epsilon_e, epsilon_B, E0 = sample[:6]
    # Calculate the flux for the current sample across all time and frequency combinations
    y0 = splr.Flux(x, thetaCore, n0, p, epsilon_e, epsilon_B, E0, thetaObs, xi_N, d_L, z)
    y0 = np.array(y0)
    # Plot the flux values for each frequency, one line per frequency
    for freq_idx, nu_value in enumerate(nu):
        ax.plot(t, y0[:, freq_idx], alpha=0.1)

# Plot the observational data
for freq_idx, nu_value in enumerate(nu):
    ax.errorbar(t, F_noise[:, freq_idx],yerr=yerr[:, freq_idx], fmt='.', label=f'{nu_value:.2e} Hz')
    ax.plot(t_best,best[:, freq_idx],'-')
ax.set(xscale='log', xlabel=r'$t$ (s)', ylabel=r'$\log_{10}(F_\nu)$ (mJy)')
ax.legend()

# Save the figure
fig.savefig('./graph/fitrange_all_frequencies.png')
plt.close(fig)