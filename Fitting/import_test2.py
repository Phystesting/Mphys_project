import numpy as np
import afterglowpy as grb
import emcee
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import curve_fit
import corner
import os
from multiprocessing import Pool
import time
import h5py
import tfreq_sampler as ts
import Jet_sampler as js
import Multimode as M


initial = np.array([0.1,0.1,0.0,2.3,-1,-3,53])
truth = [0.1,0.1,0.0,2.3,-1,-3,53]


# Load the data
x, ydata = np.genfromtxt('./data/test_curve.txt', delimiter=',', skip_header=11, unpack=True)
y = np.log(ydata)
x2, ydata2 = np.genfromtxt('./data/test_spec.txt', delimiter=',', skip_header=2, unpack=True)
y2 = np.log(ydata2)

# Add random noise to the data
noise_level = 0.1  # Standard deviation of the noise
y_noisy = y + np.random.normal(0, noise_level, size=y.shape)
y2_noisy = y2 + np.random.normal(0, noise_level, size=y2.shape)

# Set up the error profile: 5% of the signal plus the noise level as the error
yerr = abs(y_noisy * 0.05) + noise_level
yerr2 = abs(y2_noisy * 0.05) + noise_level
if __name__ == '__main__':
    fig, ax = plt.subplots(1, 1)
    ax.errorbar(x2, y2_noisy,yerr2)

    ax.set_xscale('log')
    ax.set_xlabel(r'$\nu$ (Hz)')
    ax.set_ylabel(r'$F_\nu$[1 day] (mJy)')

    fig.tight_layout()
    print("Saving figure spec.png")
    fig.savefig("time.png")
    plt.close(fig)
       
       
    """
    sampler = js.run_sampling(x,y_noisy,initial,yerr=yerr,processes=10,filename='./data/time_v1.h5',genfile=1,steps=10000,datatype=0,fixed=1e18,d_L=1e26,z=0.01)
    sampler2 = js.run_sampling(x2,y2_noisy,initial,yerr=yerr2,processes=10,filename='./data/spec_v1.h5',genfile=1,steps=10000,datatype=1,fixed=86400,d_L=1e26,z=0.01)
    #js.optimize(x2,y2,initial,datatype=1,fixed=86400,d_L=1e28,z=0.55)
    
    #t_sample,spec_sample = js.run_combined_sampling(x,y_noisy,x2,y2_noisy,initial,time_yerr=yerr,spec_yerr=yerr2)
    file_path = './data/time_v1.h5'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The HDF5 file '{file_path}' does not exist.")
    backend = emcee.backends.HDFBackend(file_path)
    if backend.iteration == 0:
        raise ValueError("The HDF5 file is empty. No data was stored.")
    reader = emcee.backends.HDFBackend(file_path)
    #chain = reader.get_chain()[0][0]
    #chain_shape = chain.shape
    #print(chain)
    js.plot_results(reader,truth)

    file_path = './data/spec_v1.h5'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The HDF5 file '{file_path}' does not exist.")
    backend = emcee.backends.HDFBackend(file_path)
    if backend.iteration == 0:
        raise ValueError("The HDF5 file is empty. No data was stored.")
    reader = emcee.backends.HDFBackend(file_path)
    #chain = reader.get_chain()[0][0]
    #chain_shape = chain.shape
    #print(chain)
    js.plot_results(reader,truth)

    
    file1 = "./data/time_v1.h5"
    file2 = "./data/spec_v1.h5"

    # Lists to store the chains from each file
    chains = []

    # Read the first HDF5 file
    with h5py.File(file1, "r") as f1:
        backend1 = emcee.backends.HDFBackend(file1)
        chain1 = backend1.get_chain()  # Shape: (nsteps, nwalkers, ndim)
        chains.append(chain1)

    # Read the second HDF5 file
    with h5py.File(file2, "r") as f2:
        backend2 = emcee.backends.HDFBackend(file2)
        chain2 = backend2.get_chain()  # Shape: (nsteps, nwalkers, ndim)
        chains.append(chain2)

    # Concatenate the chains along the walker axis (axis=1)
    # This will create a new array with the shape: (nsteps, total_nwalkers, ndim)
    combined_chain = np.concatenate(chains, axis=1)
    
    print(f"Combined chain shape: {combined_chain.shape}")
    # Number of steps to discard (e.g., burn-in period)
    discard = 1000

    # Thinning factor
    thin = 10

    # Assuming 'chain' has the shape (nsteps, nwalkers, ndim)
    # Step 1: Discard the first 'discard' steps
    combined_discarded = combined_chain[discard:]

    # Step 2: Thin the chain by taking every 'thin'-th step
    combined_thinned = combined_discarded[::thin]
    
    # Flatten the chain: (nsteps * nwalkers, ndim)
    flat_combined = combined_discarded.reshape((-1, combined_thinned.shape[2]))
    
    labels = ["thetaObs", "thetaCore", "n0", "p", "epsilon_e", "epsilon_B", "E0"]
	# Plot the sampling results
    #samples = sampler.get_chain()
    fig, axes = plt.subplots(len(labels), figsize=(10, 7), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(combined_chain[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(combined_chain))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")
    fig.savefig('./data/steps.png')
    plt.close(fig)
    #flat_samples = sampler.get_chain(discard=burnin,thin=thin,flat=True)#
    fig2 = corner.corner(flat_combined, labels=labels, truths=truth)
    fig2.savefig('./data/corner.png')
    plt.close(fig2)
    """
    
    #js.fit_spec(x2,y2_noisy,initial,yerr2,reader,86400,1,1e26,0.01)
    #js.optimize(x,y_noisy,initial,yerr)
    #sampler_pt, pt_samples = M.run_manual_parallel_tempering(x, y_noisy, initial, steps=1000, ntemps=10)
    #param_labels = ["thetaObs", "thetaCore", "n0", "p", "epsilon_e", "epsilon_B", "E0"]
    #M.plot_parallel_tempering_results(sampler_pt, pt_samples, labels=param_labels)
    