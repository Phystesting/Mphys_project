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
from functools import partial


def ag_time(x, thetaObs, thetaCore, n0, p, epsilon_e, epsilon_B, E0, fixed,xi_N,d_L,z):
    # Function to calculate the model
    Z = {
        'jetType': grb.jet.TopHat,
        'specType': grb.jet.SimpleSpec,
        'thetaObs': thetaObs,
        'E0': 10**E0,
        'thetaCore': thetaCore,
        'n0': 10**n0,
        'p': p,
        'epsilon_e': 10**epsilon_e,
        'epsilon_B': 10**epsilon_B,
        'xi_N': xi_N,
        'd_L': d_L,
        'z': z
	}
    nu = np.empty(x.shape)
    nu[:] = fixed
    return np.log(grb.fluxDensity(x, nu, **Z))
	
def ag_freq(x, thetaObs, thetaCore, n0, p, epsilon_e, epsilon_B, E0,fixed,xi_N,d_L,z):
	# Function to calculate the model
	Z = {
        'jetType': grb.jet.TopHat,
        'specType': grb.jet.SimpleSpec,
        'thetaObs': thetaObs,
        'E0': 10**E0,
        'thetaCore': thetaCore,
        'n0': 10**n0,
        'p': p,
        'epsilon_e': 10**epsilon_e,
        'epsilon_B': 10**epsilon_B,
        'xi_N': xi_N,
        'd_L': d_L,
        'z': z
    }
	return np.log(grb.fluxDensity(fixed, x, **Z))


def log_likelihood(theta, x, y, yerr, datatype, fixed,xi_N,d_L,z):
    thetaObs, thetaCore, n0, p, epsilon_e, epsilon_B, E0 = theta

    # Choose the appropriate model function based on the datatype
    if datatype == 0:
        model = ag_time(x, thetaObs, thetaCore, n0, p, epsilon_e, epsilon_B, E0, fixed,xi_N,d_L,z)
    elif datatype == 1:
        model = ag_freq(x, thetaObs, thetaCore, n0, p, epsilon_e, epsilon_B, E0, fixed,xi_N,d_L,z)
    else:
        raise ValueError("Invalid datatype. Must be 0 for ag_time or 1 for ag_freq.")
    
    # Check if the model returns finite values
    if not np.all(np.isfinite(model)):
        print(thetaObs, thetaCore, n0, p, epsilon_e, epsilon_B, E0)
        raise ValueError("Model returned non-finite values.")
    
    # Calculate sigma squared (variance)
    sigma2 = np.sqrt(model**2 + yerr**2)
    #print(thetaObs, thetaCore, n0, p, epsilon_e, epsilon_B, E0)
    #print(-0.5 * np.sum(((y - model) ** 2 / sigma2)))
    # Return the log-likelihood
    return -np.sum(((y - model) ** 2 / sigma2)) # +0.01*np.log(sigma2) (optional)

def log_probability(theta,x,y,yerr,datatype,fixed,xi_N,d_L,z):
    # Function for log-probability calculation
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr, datatype,fixed,xi_N,d_L,z)

def log_prior(theta):
    # Function for log-prior calculation
    thetaObs, thetaCore, n0, p, epsilon_e, epsilon_B, E0 = theta
    if (45 < E0 < 57
        and 0.0 < thetaObs < np.pi * 0.5
        and 0.01 < thetaCore < 0.8
        and -10.0 < n0 < 10.0
        and 2.1 < p < 5.0
        and -5.0 < epsilon_e < 0.0
        and -5.0 < epsilon_B < 0.0):
        return 0.0
    return -np.inf

def optimize(x, y, initial, yerr=None, datatype=0,fixed=1e18,xi_N=1.0,d_L=1.0e26,z=0.01):
    if yerr.any() == None:
        yerr = np.zeros(len(x))
    #calculate least squares fit parameters
    if datatype == 0:
        fit = ag_time
    elif datatype ==1:
        fit = ag_freq
    fit_fixed = partial(fit,fixed=fixed,xi_N=xi_N,d_L=d_L,z=z)
    bounds = [(0,0.01,-10,2.1,-5,-5,45),(0.8,np.pi*0.5,10,5,0,0,57)]
    soln = curve_fit(fit_fixed,x,y,sigma=yerr, p0=initial, bounds=bounds)
    thetaObs_ml, thetaCore_ml, n0_ml, p_ml, epsilon_e_ml, epsilon_B_ml, E0_ml = soln[0]

    print("Log likelihood estimates:")
    print("E0 = {0:.3e}".format(10**E0_ml))
    print("thetaObs = {0:.3f}".format(thetaObs_ml))
    print("thetaCore = {0:.3f}".format(thetaCore_ml))
    print("n0 = {0:.10f}".format(10**n0_ml))
    print("p = {0:.3f}".format(p_ml))
    print("epsilon_e = {0:.5f}".format(10**epsilon_e_ml))
    print("epsilon_B = {0:.5f}".format(10**epsilon_B_ml))
    return soln[0]

   


def run_optimization(x, y, initial, yerr=None, datatype=0,fixed=1e18,xi_N=1.0,d_L=1.0e26,z=0.01):
    if yerr.any() == None:
        yerr = np.zeros(len(x))
    # Perform the parameter optimization
    bounds = [(0, 0.8), (0.01, np.pi*0.5), (-10.0, 10.0), 
        (2.1, 5.0), (-5.0, 0.0), (-5.0, 0.0), (45.0, 57.0)]
    # Use the wrapper function to fix datatype for the residual calculation
    likelihood_fixed = partial(log_likelihood, datatype=datatype,fixed=fixed,xi_N=xi_N,d_L=d_L,z=z)
    nll = lambda *args: -likelihood_fixed(*args)
    soln = minimize(nll, initial, args=(x, y, yerr), bounds=bounds, method='SLSQP')
    thetaObs_ml, thetaCore_ml, n0_ml, p_ml, epsilon_e_ml, epsilon_B_ml, E0_ml = soln.x
    print("Log likelihood estimates:")
    print("E0 = {0:.3e}".format(10**E0_ml))
    print("thetaObs = {0:.3f}".format(thetaObs_ml))
    print("thetaCore = {0:.3f}".format(thetaCore_ml))
    print("n0 = {0:.10f}".format(10**n0_ml))
    print("p = {0:.3f}".format(p_ml))
    print("epsilon_e = {0:.5f}".format(10**epsilon_e_ml))
    print("epsilon_B = {0:.5f}".format(10**epsilon_B_ml))
    return soln

def run_sampling(x, y, initial, genfile=0, yerr=None, datatype=0,fixed=1.0e18,xi_N=1.0,d_L=1.0e26,z=0.01, steps=100, nwalkers=32, processes=4, filename="./data/test_sample.h5"):
    if yerr.any() == None:
        yerr = np.zeros(len(x))
    # Run the minimize to find starting point
    soln = optimize(x, y, initial,yerr, datatype,fixed,xi_N,d_L,z)
    pos = soln + 1e-4 * np.random.randn(nwalkers, len(soln))
    ndim = len(soln)

    # Use the wrapper function to fix datatype for the MCMC sampling
    log_prob_fixed = partial(log_probability,datatype=datatype,fixed=fixed,xi_N=xi_N,d_L=d_L,z=z)
    
    if genfile == 1:
        # Set up the backend
        # Don't forget to clear it in case the file already exists
        backend = emcee.backends.HDFBackend(filename)
        backend.reset(nwalkers, ndim)
        # Run the MCMC sampling
        with Pool(processes=processes) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_fixed, args=(x, y, yerr), pool=pool, backend=backend)
            sampler.run_mcmc(pos, steps, progress=True)
    elif genfile == 0:
        # Run the MCMC sampling
        with Pool(processes=processes) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_fixed, args=(x, y, yerr), pool=pool, backend=backend)
            sampler.run_mcmc(pos, steps, progress=True)
    else:
        raise ValueError("Invalid genfile. Must be 0 for to run without generating a file or 1 to do so.")
    
    return sampler

def run_manual_parallel_tempering(x, y, initial, yerr=None, datatype=0, fixed=1.0e18,xi_N=1.0,d_L=1.0e26,z=0.01, steps=100, ntemps=10, nwalkers=32, swap_interval=10):
    if yerr is None:
        yerr = np.zeros(len(x))
    
    ndim = len(initial)
    
    # Define a log-probability function wrapper for each temperature
    def log_prob_temp(theta, temp, *args):
        return log_probability(theta, *args) / temp
    
    # Create the samplers, one for each temperature
    temperatures = np.logspace(0, np.log10(10), ntemps)  # Linear spacing in log space
    samplers = [emcee.EnsembleSampler(nwalkers, ndim, partial(log_prob_temp, temp=temp), args=(x, y, yerr, datatype, fixed, xi_N, d_L, z)) for temp in temperatures]
    
    soln = optimize(x, y, initial,yerr, datatype,fixed,xi_N,d_L,z)
    
    # Initialize walker positions for each temperature
    pos = [soln for _ in range(ntemps)]
    
    # Storage for the samples
    all_samples = [[] for _ in range(ntemps)]
    
    # Run the MCMC with periodic swapping
    for step in range(steps):
        # Perform a sampling step for each sampler
        for i, sampler in enumerate(samplers):
            pos[i], _, _ = sampler.run_mcmc(pos[i], 1, progress=False)
            all_samples[i].append(pos[i])
        
        # Swap positions between chains every 'swap_interval' steps
        if (step + 1) % swap_interval == 0:
            for i in range(ntemps - 1):
                # Choose walkers to swap
                for j in range(nwalkers):
                    # Metropolis-Hastings acceptance criteria for swapping
                    beta = 1.0 / temperatures[i + 1] - 1.0 / temperatures[i]
                    delta_log_prob = (samplers[i + 1].get_log_prob()[j] - samplers[i].get_log_prob()[j])
                    if np.log(np.random.rand()) < beta * delta_log_prob:
                        # Swap the walkers
                        pos[i][j], pos[i + 1][j] = pos[i + 1][j], pos[i][j]
    
    # Flatten the samples and return results for the lowest temperature
    samples_lowest_temp = np.vstack(all_samples[0])
    
    return samplers, samples_lowest_temp

def plot_results(sampler,truth,filename1='./graph/probable_parameters.png',filename2='./graph/parameter_steps.png'):
    try:
        tau = sampler.get_autocorr_time()
        burnin = int(2 * np.max(tau))
        thin = int(0.5 * np.min(tau))
    except emcee.autocorr.AutocorrError:
        print("Warning: Autocorrelation time estimation failed. Proceeding with current chain length.")
        tau = sampler.get_autocorr_time()
        burnin = len(sampler.get_chain()) // 3  # Use some default burn-in, e.g., one-third of the chain length
        thin = 1  # No thinning
    labels = ["thetaObs", "thetaCore", "n0", "p", "epsilon_e", "epsilon_B", "E0"]
	# Plot the sampling results
    samples = sampler.get_chain()
    fig, axes = plt.subplots(len(labels), figsize=(10, 7), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")
    fig.savefig(filename2)
    plt.close(fig)
    flat_samples = sampler.get_chain(discard=burnin,thin=thin,flat=True)#
    fig2 = corner.corner(flat_samples, labels=labels, truths=truth)
    fig2.savefig(filename1)
    plt.close(fig2)




def fit_time(x,y,initial,yerr,sampler,fixed,xi_N,d_L,z):
    datatype = 0
    x0 = np.geomspace(x[0], x[-1], num=100)
    soln = optimize(x, y, initial,yerr, datatype,fixed,xi_N,d_L,z)
    thetaObs_ml, thetaCore_ml, n0_ml, p_ml, epsilon_e_ml, epsilon_B_ml, E0_ml = soln
    y_fit = ag_freq(x0,thetaObs_ml, thetaCore_ml, n0_ml, p_ml, epsilon_e_ml, epsilon_B_ml, E0_ml,fixed ,xi_N,d_L,z)
    fig, ax = plt.subplots(1, 1)
    flat_samples = sampler.get_chain(flat=True)
    inds = np.random.randint(len(flat_samples), size=100)
    for ind in inds:
        sample = flat_samples[ind]
        thetaObs,thetaCore,n0,p,epsilon_e,epsilon_B,E0 = sample[:7]
        y0 = ag_time(x0, thetaObs, thetaCore, n0, p, epsilon_e, epsilon_B, E0,fixed ,xi_N,d_L,z)
        ax.plot(x0, y0, "C1", alpha=0.1,zorder=1)
    ax.errorbar(x, y,yerr,capsize=2,fmt='.',color='black',zorder=2)
    ax.plot(x0, y_fit,'-',color='blue',zorder=3)
    ax.set(xscale='log', xlabel=r'$t$ (s)', ylabel=r'$ln(F_\nu)$[$10^{18}$ Hz] (ln(mJy))')
    fig.savefig('./graph/fitrange2.png')
    plt.close(fig)

def fit_spec(x,y,initial,yerr,sampler,fixed,xi_N,d_L,z):
    datatype = 1
    x0 = np.geomspace(x[0], x[-1], num=100)
    soln = optimize(x, y, initial,yerr, datatype,fixed,xi_N,d_L,z)
    thetaObs_ml, thetaCore_ml, n0_ml, p_ml, epsilon_e_ml, epsilon_B_ml, E0_ml = soln
    y_fit = ag_freq(x0,thetaObs_ml, thetaCore_ml, n0_ml, p_ml, epsilon_e_ml, epsilon_B_ml, E0_ml,fixed ,xi_N,d_L,z)
    fig, ax = plt.subplots(1, 1)
    flat_samples = sampler.get_chain(flat=True)
    inds = np.random.randint(len(flat_samples), size=100)
    for ind in inds:
        sample = flat_samples[ind]
        thetaObs,thetaCore,n0,p,epsilon_e,epsilon_B,E0 = sample[:7]
        y0 = ag_freq(x0, thetaObs, thetaCore, n0, p, epsilon_e, epsilon_B, E0,fixed ,xi_N,d_L,z)
        ax.plot(x0, y0, "C1", alpha=0.1,zorder=1)
    ax.errorbar(x, y,yerr,capsize=2,fmt='.',color='black',zorder=2)
    ax.plot(x0, y_fit,'-',color='blue',zorder=3)
    
    ax.set(xscale='log', xlabel=r'$frequency$ (Hz)', ylabel=r'$ln(F_\nu)$[$10^{18}$ Hz] (ln(mJy))')
    fig.savefig('./graph/fitrange2.png')
    plt.close(fig)
    
def plot_parallel_tempering_results(sampler, samples, labels=None, filename_corner='./graph/parallel_tempering_corner.png', filename_trace='./graph/parallel_tempering_trace.png'):
    # Corner plot for the posterior distributions
    fig_corner = corner.corner(samples, labels=labels, show_titles=True)
    fig_corner.suptitle('Corner Plot of Parallel Tempering MCMC')
    fig_corner.savefig(filename_corner)
    plt.close(fig_corner)

    # Trace plots to visualize the sampling history for each parameter
    ndim = samples.shape[1]
    fig_trace, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    
    for i in range(ndim):
        # Extract the samples for each walker and parameter for the lowest temperature chain
        walker_samples = sampler.chain[0, :, :, i]
        for walker in walker_samples:
            axes[i].plot(walker, alpha=0.3, color='k')  # Plot each walker trace
        axes[i].set_ylabel(labels[i] if labels else f"Param {i+1}")
        axes[i].yaxis.set_label_coords(-0.1, 0.5)
    
    axes[-1].set_xlabel("Step Number")
    fig_trace.suptitle('Trace Plot of Parallel Tempering MCMC')
    fig_trace.tight_layout()
    fig_trace.savefig(filename_trace)
    plt.close(fig_trace)