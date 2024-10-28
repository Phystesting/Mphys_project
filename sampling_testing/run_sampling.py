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
    
initial = np.array([0.2,-2,2.5,-3,-3,55])

nu = np.geomspace(1e0, 1e20, num=7)
t = np.geomspace(1e-1 * grb.day2sec, 1e3 * grb.day2sec, num=17)

x = [t,nu]
print(nu[6])
F = np.array(splr.Flux(x,0.1,0,2.3,-2,-4,53,0.0,1,1e26,0.01))
time_error_1d = 0.05*t
spec_error_1d = 0.05*nu
err_time = np.tile(time_error_1d[:, np.newaxis], (1, len(nu)))
err_spec = np.tile(spec_error_1d, (len(t), 1))

#F2 = np.array(Flux(x,0.1,0,2.3,-2,-4,53,0.0,1,1e26,0.01))
noise_level = abs(0.05*F)
F_noise = F + np.random.normal(0, noise_level, size=F.shape)
err_flux = abs(0.05*F)
err = [err_flux,err_time,err_spec]
truth = [0.1,0.0,2.3,-2,-4,53]

fig, ax = plt.subplots(1, 1)
ax.set(xscale='log', xlabel=r'$t$ (s)', ylabel=r'$\log_{10}(F_\nu)$ (mJy)')
#for freq_idx, nu_value in enumerate(nu):
ax.errorbar(t, F_noise[:, 1],xerr=err_time[:, 1],yerr=err_flux[:, 1], fmt='.', label=f'{10:.2e} Hz')
plt.show()
#if __name__ == "__main__":
    #run_parallel_optimization(x,F_noise,initial,yerr,processes=6)
#run_optimization(x,F_noise,initial,yerr)
"""
if __name__ == "__main__":
    splr.run_sampling(x,F,initial,err,steps=30000,processes=6,genfile=1,filename='./data/err_3D.h5')


    file_path = './data/err_3D.h5'
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
    fig, axes = plt.subplots(len(labels), figsize=(10, 7), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")
    fig.savefig('./graph/err_3D_steps.png')
    plt.close(fig)
    flat_samples = reader.get_chain(discard=burnin,thin=thin,flat=True)#
    fig2 = corner.corner(flat_samples, labels=labels, truths=truth)
    fig2.savefig('./graph/err_3D_contour.png')
    plt.close(fig2)
    """