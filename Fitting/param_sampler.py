import numpy as np
import afterglowpy as grb
import emcee
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import corner
import os
from multiprocessing import Pool
import time

def ag_py(t, thetaObs, thetaCore, thetaWing, n0, p, epsilon_e, epsilon_B, E0):
    # Function to calculate the model
    Z = {
        'jetType': grb.jet.Gaussian,
        'specType': grb.jet.SimpleSpec,
        'thetaObs': thetaObs,
        'E0': 10**E0,
        'thetaCore': thetaCore,
        'thetaWing': thetaCore + (0.4 - thetaCore) * thetaWing,
        'n0': 10**n0,
        'p': p,
        'epsilon_e': 10**epsilon_e,
        'epsilon_B': 10**epsilon_B,
        'xi_N': 1.0,
        'd_L': 1.36e26,
        'z': 0.01
    }
    nu = np.empty(t.shape)
    nu[:] = 1.0e18
    return np.log(grb.fluxDensity(t, nu, **Z))

def residual(theta, x, y):
    # Function to calculate the residual
    thetaObs, thetaCore, thetaWing, n0, p, epsilon_e, epsilon_B, E0 = theta
    model = ag_py(x, thetaObs, thetaCore, thetaWing, n0, p, epsilon_e, epsilon_B, E0)
    if not np.all(np.isfinite(model)):
        raise ValueError("ag_py returned non-finite values.")
    return sum(((y - model) ** 2) / abs(model))

def log_likelihood(theta, x, y):
    # Function for log-likelihood calculation
    return -0.5 * residual(theta, x, y)

def log_prior(theta):
    # Function for log-prior calculation
    thetaObs, thetaCore, thetaWing, n0, p, epsilon_e, epsilon_B, E0 = theta
    if (51 < E0 < 54
        and 0.0 < thetaObs < np.pi * 0.5
        and 0.01 < thetaCore < 0.4
        and 0.0 < thetaWing < 1.0
        and -5.0 < n0 < 3.0
        and 2.1 < p < 3.0
        and -5.0 < epsilon_e < 0.0
        and -5.0 < epsilon_B < 0.0):
        return 0.0
    return -np.inf

def log_probability(theta, x, y):
    # Function for log-probability calculation
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y)

def run_optimization(x, y,initial):
    # Perform the parameter optimization
    bounds = [(0, np.pi * 0.5), (0.01, 0.4), (0.0, 1.0), (-4.0, 3.0), 
                (2.1, 2.8), (-4.0, 0.0), (-4.0, 0.0), (51.0, 54.0)]

    soln = minimize(residual, initial, args=(x, y), bounds=bounds, method='SLSQP')
    thetaObs_ml, thetaCore_ml, thetaWing_ml, n0_ml, p_ml, epsilon_e_ml, epsilon_B_ml, E0_ml = soln.x
    print("Chi squared estimates:")
    print("E0 = {0:.3f}".format(10**E0_ml))
    print("thetaObs = {0:.3f}".format(thetaObs_ml))
    print("thetaCore = {0:.3f}".format(thetaCore_ml))
    print("thetaWing = {0:.3f}".format(thetaCore_ml + (0.4-thetaCore_ml) * thetaWing_ml))
    print("n0 = {0:.4f}".format(10**n0_ml))
    print("p = {0:.3f}".format(p_ml))
    print("epsilon_e = {0:.4f}".format(10**epsilon_e_ml))
    print("epsilon_B = {0:.4f}".format(10**epsilon_B_ml))
    return soln



def run_sampling(x, y, initial, steps=100, nwalkers=32, processes=4,filename="./data/test_sample.h5"): #don't call this without protection
    # Run the MCMC sampling
    soln = run_optimization(x,y,initial)
    pos = soln.x + 1e-4 * np.random.randn(nwalkers, len(soln.x))
    ndim = len(soln.x)
    # Set up the backend
    # Don't forget to clear it in case the file already exists
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)
    with Pool(processes=processes) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y), pool=pool,backend=backend)
        sampler.run_mcmc(pos, steps, progress=True)
    return sampler



def plot_results(sampler,truth,filename1='./graph/probable_parameters.png',filename2='./graph/parameter_steps.png'):
    """
    tau = sampler.get_autocorr_time()
    burnin = int(2 * np.max(tau))
    thin = int(0.5 * np.min(tau))
    """
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

    flat_samples = sampler.get_chain(flat=True)
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

