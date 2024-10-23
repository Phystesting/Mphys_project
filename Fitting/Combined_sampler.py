import numpy as np
import afterglowpy as grb
import emcee
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import curve_fit
import corner
import os
import multiprocessing
from multiprocessing import Pool
import time
from functools import partial
from scipy.stats import ks_2samp

#create a function to stack the lists
def appendList(l,element):
    l.append(element)
    return l

def Flux(x, thetaObs, thetaCore, n0, p, epsilon_e, epsilon_B, E0,xi_N,d_L,z):
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
    t = x[0]
    nu = x[1]
    Flux = [np.log(grb.fluxDensity(t[0], nu, **Z))]
    #print(Flux)
    for i in t[1:]:
        Flux = appendList(Flux,np.log(grb.fluxDensity(i, nu, **Z)))
        #print(Flux)
        #print('')
    return Flux

def log_likelihood(theta, x, y, yerr,xi_N,d_L,z):
    thetaObs, thetaCore, n0, p, epsilon_e, epsilon_B, E0 = theta


    model = Flux(x, thetaObs, thetaCore, n0, p, epsilon_e, epsilon_B, E0,xi_N,d_L,z)
    
    
    # Check if the model returns finite values
    if not np.all(np.isfinite(model)):
        print(thetaObs, thetaCore, n0, p, epsilon_e, epsilon_B, E0)
        raise ValueError("Model returned non-finite values.")
    model = np.array(model)
    y = np.array(y)
    yerr = np.array(yerr)

    sigma2 = yerr**2 + model**2
    #print(-0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2)))
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

def log_probability(theta,x,y,yerr,xi_N,d_L,z):
    # Function for log-probability calculation
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr,xi_N,d_L,z)

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


def run_optimization(x, y, initial, yerr,xi_N=1.0,d_L=1.0e26,z=0.01):
    # Perform the parameter optimization
    bounds = [(0, 0.8), (0.01, np.pi*0.5), (-10.0, 10.0), 
        (2.1, 5.0), (-5.0, 0.0), (-5.0, 0.0), (45.0, 57.0)]
    # Use the wrapper function to fix datatype for the residual calculation
    likelihood_fixed = partial(log_likelihood,xi_N=xi_N,d_L=d_L,z=z)
    nll = lambda *args: -likelihood_fixed(*args)
    soln = minimize(nll, initial, args=(x, y, yerr), bounds=bounds, method='Nelder-Mead')
    thetaObs_ml, thetaCore_ml, n0_ml, p_ml, epsilon_e_ml, epsilon_B_ml, E0_ml = soln.x
    print("Maximum likelihood estimates:")
    print("E0 = {0:.3e}".format(10**E0_ml))
    print("thetaObs = {0:.3f}".format(thetaObs_ml))
    print("thetaCore = {0:.3f}".format(thetaCore_ml))
    print("n0 = {0:.10f}".format(10**n0_ml))
    print("p = {0:.3f}".format(p_ml))
    print("epsilon_e = {0:.5f}".format(10**epsilon_e_ml))
    print("epsilon_B = {0:.5f}".format(10**epsilon_B_ml))
    return soln

def run_sampling(x, y, initial, yerr, genfile=0,xi_N=1.0,d_L=1.0e26,z=0.01, steps=100, nwalkers=32, processes=4, filename="./data/test_sample.h5"):
    # Run the minimize to find starting point
    print("Finding optimal starting parameters...")
    soln = run_optimization(x, y, initial,yerr,xi_N,d_L,z)
    pos = soln.x + 1e-4 * np.random.randn(nwalkers, len(soln.x))
    ndim = len(soln.x)
    
    # Use the wrapper function to fix datatype for the MCMC sampling
    log_prob_fixed = partial(log_probability,xi_N=xi_N,d_L=d_L,z=z)
    total_cores = multiprocessing.cpu_count()
    print("Utilizing {0:.1f}% of avaliable processes".format(100*processes/total_cores))
    print("Beginning sampling...")
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
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_fixed, args=(x, y, yerr), pool=pool)
            sampler.run_mcmc(pos, steps, progress=True)
    else:
        raise ValueError("Invalid genfile. Must be 0 for to run without generating a file or 1 to do so.")
    
    return sampler

initial = np.array([0.2,0.3,0.1,3.0,-2,-5,50])

nu = np.geomspace(1e8, 1e20, num=10)
t = np.geomspace(1e-1, 1e3, num=10)

x = [t,nu]
F = Flux(x,0.1,0.1,0,2.3,-2,-4,53,1,1e26,0.01)
yerr = np.zeros((len(t),len(nu)))
truth = [0.1,0.1,0.0,2.3,-1,-3,53]

if __name__ == "__main__":
    run_sampling(x,F,initial,yerr,processes=20,genfile=1,filename='./data/combo.h5')


    file_path = './data/combo.h5'
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
    labels = ["thetaObs", "thetaCore", "n0", "p", "epsilon_e", "epsilon_B", "E0"]
    # Plot the sampling results
    samples = reader.get_chain()
    fig, axes = plt.subplots(len(labels), figsize=(10, 7), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")
    fig.savefig('./graph/combo_steps.png')
    plt.close(fig)
    flat_samples = reader.get_chain(discard=burnin,thin=thin,flat=True)#
    fig2 = corner.corner(flat_samples, labels=labels, truths=truth)
    fig2.savefig('./graph/combo_param.png')
    plt.close(fig2)

