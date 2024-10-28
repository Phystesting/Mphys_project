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

nu = np.geomspace(1e0, 1e20, num=7)
t = np.geomspace(1e-1 * grb.day2sec, 1e3 * grb.day2sec, num=17)

x = [t,nu]
F = np.array(splr.Flux(x,0.1,0,2.3,-2,-4,53,0.0,1,1e26,0.01))
noise_level = abs(0.05*F)
F_noise = F + np.random.normal(0, noise_level, size=F.shape)
yerr = abs(0.05*F)

file_path = './data/combo.h5'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The HDF5 file '{file_path}' does not exist.")
backend = emcee.backends.HDFBackend(file_path)
if backend.iteration == 0:
    raise ValueError("The HDF5 file is empty. No data was stored.")
sampler = emcee.backends.HDFBackend(file_path)

try:
    tau = sampler.get_autocorr_time()
    burnin = int(2 * np.max(tau))
    thin = int(0.5 * np.min(tau))
except emcee.autocorr.AutocorrError:
    print("Warning: Autocorrelation time estimation failed. Proceeding with current chain length.")
    burnin = len(sampler.get_chain()) // 3  # Use some default burn-in, e.g., one-third of the chain length
    thin = 1  # No thinning

ndim = 6

flat_samples = sampler.get_chain(discard=burnin,thin=thin,flat=True)
mcmc = np.zeros([6,3])
q = np.zeros([6,2])
for i in range(ndim):
    mcmc[i] = np.percentile(flat_samples[:, i], [16, 50, 84])
    q[i] = np.diff(mcmc[i])
    
best = np.array(splr.Flux(x,mcmc[0][1],mcmc[1][1],mcmc[2][1],mcmc[3][1],mcmc[4][1],mcmc[5][1],0.0,1.0,1e26,0.01))

deviation = abs(F_noise - best) / yerr

T, Nu = np.meshgrid(np.log10(t), np.log10(nu), indexing='ij')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.grid()
ax2.grid()

# Plot the original flux with noise on the left subplot
c1 = ax1.pcolormesh(T, Nu, F_noise, shading='gouraud', cmap='viridis')
cbar1 = plt.colorbar(c1, ax=ax1)
cbar1.set_label('Flux', labelpad=10)
ax1.set_title('Spectral Time Flux')
ax1.set_xlabel('log(Time (t))', labelpad=10)
ax1.set_ylabel('log(Frequency (ν))', labelpad=10)

# Plot the deviation on the right subplot
c2 = ax2.pcolormesh(T, Nu, deviation, shading='gouraud', cmap='coolwarm')
cbar2 = plt.colorbar(c2, ax=ax2)
cbar2.set_label('Deviation', labelpad=10)
ax2.set_title('Deviation (F_noise - Best) / yerr')
ax2.set_xlabel('log(Time (t))', labelpad=10)
ax2.set_ylabel('log(Frequency (ν))', labelpad=10)

# Adjust layout to fit all elements and save the figure
plt.tight_layout()
fig.savefig('./graph/6param_colourmesh_with_deviation.png', dpi=300)
plt.close(fig)