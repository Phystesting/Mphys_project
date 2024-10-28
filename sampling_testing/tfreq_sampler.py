import numpy as np
import afterglowpy as grb
import emcee
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import corner
import os
from multiprocessing import Pool
import time
from functools import partial

def ag_time(t, thetaObs, thetaCore, thetaWing, n0, p, epsilon_e, epsilon_B, E0, nu_in,xi_N,d_L,z):
    # Function to calculate the model
    Z = {
        'jetType': grb.jet.Gaussian,
        'specType': grb.jet.SimpleSpec,
        'thetaObs': thetaObs,
        'E0': 10**E0,
        'thetaCore': thetaCore,
        'thetaWing': thetaCore + (11*thetaCore * thetaWing),
        'n0': 10**n0,
        'p': p,
        'epsilon_e': 10**epsilon_e,
        'epsilon_B': 10**epsilon_B,
        'xi_N': xi_N,
        'd_L': d_L,
        'z': z
	}
    nu = np.empty(t.shape)
    nu[:] = nu_in
    return np.log(grb.fluxDensity(t, nu, **Z))
	
def ag_freq(nu, thetaObs, thetaCore, thetaWing, n0, p, epsilon_e, epsilon_B, E0,t,xi_N,d_L,z):
	# Function to calculate the model
	Z = {
        'jetType': grb.jet.Gaussian,
        'specType': grb.jet.SimpleSpec,
        'thetaObs': thetaObs,
        'E0': 10**E0,
        'thetaCore': thetaCore,
        'thetaWing': thetaCore + (11*thetaCore * thetaWing),
        'n0': 10**n0,
        'p': p,
        'epsilon_e': 10**epsilon_e,
        'epsilon_B': 10**epsilon_B,
        'xi_N': xi_N,
        'd_L': d_L,
        'z': z
    }
	return np.log(grb.fluxDensity(t, nu, **Z))


def log_likelihood(theta, x, y, yerr, datatype, fixed,xi_N,d_L,z):
    thetaObs, thetaCore, thetaWing, n0, p, epsilon_e, epsilon_B, E0 = theta

    # Choose the appropriate model function based on the datatype
    if datatype == 0:
        model = ag_time(x, thetaObs, thetaCore, thetaWing, n0, p, epsilon_e, epsilon_B, E0, fixed,xi_N,d_L,z)
    elif datatype == 1:
        model = ag_freq(x, thetaObs, thetaCore, thetaWing, n0, p, epsilon_e, epsilon_B, E0, fixed,xi_N,d_L,z)
    else:
        raise ValueError("Invalid datatype. Must be 0 for ag_time or 1 for ag_freq.")
    
    # Check if the model returns finite values
    if not np.all(np.isfinite(model)):
        print(thetaObs, thetaCore, thetaWing, n0, p, epsilon_e, epsilon_B, E0)
        raise ValueError("Model returned non-finite values.")
    
    # Calculate sigma squared (variance)
    sigma2 = model**2 + yerr**2
    print(-0.5 * np.sum(((y - model) ** 2 / sigma2)))
    # Return the log-likelihood
    return -0.5 * np.sum(((y - model) ** 2 / sigma2)) # +0.01*np.log(sigma2) (optional)

def log_probability(theta,x,y,yerr,datatype,fixed):
    # Function for log-probability calculation
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr, datatype,fixed)

def log_prior(theta):
    # Function for log-prior calculation
    thetaObs, thetaCore, thetaWing, n0, p, epsilon_e, epsilon_B, E0 = theta
    if (45 < E0 < 57
        and 0.0 < thetaObs < 0.8
        and 0.01 < thetaCore < np.pi*0.5
        and 0.0 < thetaWing < 1.0
        and -10.0 < n0 < 10.0
        and 2.1 < p < 5.0
        and -5.0 < epsilon_e < 0.0
        and -5.0 < epsilon_B < 0.0):
        return 0.0
    return -np.inf


def run_optimization(x, y, initial, yerr=None, datatype=0,fixed=1e18,xi_N=1.0,d_L=1.36e26,z=0.01):
    yerr = np.zeros(len(x))
    # Perform the parameter optimization
    bounds = [(0, 0.8), (0.01, np.pi*0.5), (0.0, 1.0), (-10.0, 10.0), 
        (2.1, 5.0), (-5.0, 0.0), (-5.0, 0.0), (45.0, 57.0)]
    # Use the wrapper function to fix datatype for the residual calculation
    likelihood_fixed = partial(log_likelihood, datatype=datatype,fixed=fixed,xi_N=xi_N,d_L=d_L,z=z)
    nll = lambda *args: -likelihood_fixed(*args)
    soln = minimize(nll, initial, args=(x, y, yerr), bounds=bounds, method='SLSQP')
    thetaObs_ml, thetaCore_ml, thetaWing_ml, n0_ml, p_ml, epsilon_e_ml, epsilon_B_ml, E0_ml = soln.x
    print("Log likelihood estimates:")
    print("E0 = {0:.3e}".format(10**E0_ml))
    print("thetaObs = {0:.3f}".format(thetaObs_ml))
    print("thetaCore = {0:.3f}".format(thetaCore_ml))
    print("thetaWing = {0:.3f}".format(thetaCore_ml + (11*thetaCore_ml) * thetaWing_ml))
    print("n0 = {0:.5f}".format(10**n0_ml))
    print("p = {0:.3f}".format(p_ml))
    print("epsilon_e = {0:.5f}".format(10**epsilon_e_ml))
    print("epsilon_B = {0:.5f}".format(10**epsilon_B_ml))
    return soln

def run_sampling(x, y, initial, yerr=None, datatype=0,fixed=1.0e18, steps=100, nwalkers=32, processes=4, filename="./data/test_sample.h5"):
    yerr = np.zeros(len(x))
    # Run the minimize to find starting point
    soln = run_optimization(x, y, initial, datatype,fixed)
    pos = soln.x + 1e-4 * np.random.randn(nwalkers, len(soln.x))
    ndim = len(soln.x)
    
    # Set up the backend
    # Don't forget to clear it in case the file already exists
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)
    
    # Use the wrapper function to fix datatype for the MCMC sampling
    log_prob_fixed = partial(log_probability,datatype=datatype,fixed=fixed)
    
    # Run the MCMC sampling
    with Pool(processes=processes) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_fixed, args=(x, y, yerr), pool=pool, backend=backend)
        sampler.run_mcmc(pos, steps, progress=True)
    
    return sampler



def plot_results(sampler,truth,filename1='./graph/probable_parameters.png',filename2='./graph/parameter_steps.png'):
    tau = sampler.get_autocorr_time()
    burnin = int(2 * np.max(tau))
    thin = int(0.5 * np.min(tau))
    labels = ["thetaObs", "thetaCore", "ThetaWing", "n0", "p", "epsilon_e", "epsilon_B", "E0"]
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

    flat_samples = sampler.get_chain(discard=burnin,thin=thin,flat=True)
    fig2 = corner.corner(flat_samples, labels=labels, truths=truth)
    fig2.savefig(filename1)
    plt.close(fig2)




def fit(x,y,sampler):
    fig, ax = plt.subplots(1, 1)
    flat_samples = sampler.get_chain(flat=True)
    inds = np.random.randint(len(flat_samples), size=100)
    for ind in inds:
        sample = flat_samples[ind]
        thetaObs,thetaCore,thetaWing,n0,p,epsilon_e,epsilon_B,E0 = sample[:8]
        y0 = ag_py(x,thetaObs,thetaCore,thetaWing,n0,p,epsilon_e,epsilon_B,E0)
        ax.plot(x, y0, "C1", alpha=0.1)
    ax.plot(x, y, '.')
    ax.set(xscale='log', xlabel=r'$t$ (s)', ylabel=r'$F_\nu$[$10^{18}$ Hz] (mJy)')
    fig.savefig('./graph/fitrange.png')
    plt.close(fig)
