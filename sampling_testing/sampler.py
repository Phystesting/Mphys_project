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

#create a function to stack the lists
def appendList(l,element):
    l.append(element)
    return l

def Flux(x,thetaCore, n0, p, epsilon_e, epsilon_B, E0, thetaObs,xi_N,d_L,z):
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

def log_likelihood(theta, x, y, err, thetaObs, xi_N, d_L, z):
    # Unpack the parameters
    thetaCore, n0, p, epsilon_e, epsilon_B, E0 = theta

    # Split the err array into err_flux, err_time, and err_spec
    err_flux, err_time, err_spec = err

    # Calculate the model flux
    model = Flux(x, thetaCore, n0, p, epsilon_e, epsilon_B, E0, thetaObs, xi_N, d_L, z)
    
    # Check if the model returns finite values
    if not np.all(np.isfinite(model)):
        raise ValueError("Model returned non-finite values.")
    
    model = np.array(model)
    y = np.array(y)
    err_flux = np.array(err_flux)
    err_time = np.array(err_time)
    err_spec = np.array(err_spec)

    # Create a mask for non-NaN values in y
    mask = ~np.isnan(y)
    y = y[mask]
    err_flux = err_flux[mask]
    err_time = err_time[mask]
    err_spec = err_spec[mask]
    model = model[mask]
    
    # Calculate the combined error term (flux, time, and spectral errors)
    sigma2 = err_flux**2 + err_time**2 + err_spec**2 + model**2
    
    # Optional penalty for parameters outside certain bounds
    penalty_factor = 2 * np.where(abs(n0) > 4, (abs(n0) - 4) ** 2, 0)
    
    # Calculate the negative log-likelihood
    return -0.5 * np.sum(np.exp(abs(y - model)) * (y - model)**2 / sigma2) - penalty_factor

def log_probability(theta, x, y, err, thetaObs, xi_N, d_L, z):
    # Calculate the log-prior
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    # Combine the log-prior and log-likelihood
    return lp + log_likelihood(theta, x, y, err, thetaObs, xi_N, d_L, z)
    
def log_prior(theta):
    # Function for log-prior calculation
    thetaCore, n0, p, epsilon_e, epsilon_B, E0 = theta
    if (45 < E0 < 57
        and 0.01 < thetaCore < np.pi*0.5
        and -10.0 < n0 < 10.0
        and 2.1 < p < 5.0
        and -5.0 < epsilon_e < 0.0
        and -5.0 < epsilon_B < 0.0):
        return 0.0
    return -np.inf

def run_optimization(x, y, initial, err, thetaObs=0.0, xi_N=1.0, d_L=1.0e26, z=0.01):
    # Perform the parameter optimization with updated likelihood
    bounds = [(0.01, np.pi*0.5), (-10.0, 10.0), (2.1, 5.0), (-5.0, 0.0), (-5.0, 0.0), (45.0, 57.0)]
    likelihood_fixed = partial(log_likelihood, thetaObs=thetaObs, xi_N=xi_N, d_L=d_L, z=z)
    nll = lambda *args: -likelihood_fixed(*args)
    start = time.time()
    soln = minimize(nll, initial, args=(x, y, err), bounds=bounds, method='Nelder-mead')
    end = time.time()
    serial_time = end - start
    print(f"Most probable parameters identified in {serial_time:.1f} seconds")
    thetaCore_ml, n0_ml, p_ml, epsilon_e_ml, epsilon_B_ml, E0_ml = soln.x
    print(f"E0 = {10**E0_ml:.3e}")
    print(f"thetaCore = {thetaCore_ml:.3f}")
    print(f"n0 = {10**n0_ml:.10f}")
    print(f"p = {p_ml:.3f}")
    print(f"epsilon_e = {10**epsilon_e_ml:.5f}")
    print(f"epsilon_B = {10**epsilon_B_ml:.5f}")
    print(f"Residual (negative log-likelihood) = {soln.fun:.5f}")
    print("-" * 30)
    return soln

def run_parallel_optimization(x, y, initial, err,processes=4,thetaObs=0.0, xi_N=1.0, d_L=1.0e26, z=0.01):
     # Bounds for the parameters
    bounds = [(0.01, 0.5), (-2, 2), (2.1, 3), (-3.0, -1), (-5.0, -3), (45.0, 57.0)]

    # Generate random initial points for four parallel optimizations
    initial_points = [
        [np.random.uniform(low, high) for (low, high) in bounds] 
        for _ in range(processes)
    ]
    # Run optimizations in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=processes) as executor:
        futures = [
            executor.submit(run_optimization, x, y, initial, err)
            for initial in initial_points
        ]
        
        # Collect the results
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    # Select the solution with the lowest function value (lowest residual)
    best_solution = min(results, key=lambda sol: sol.fun)
    # Extract the best parameters
    thetaCore_ml, n0_ml, p_ml, epsilon_e_ml, epsilon_B_ml, E0_ml = best_solution.x

    print("Best parameters identified:")
    print(f"E0 = {10**E0_ml:.3e}")
    print(f"thetaCore = {thetaCore_ml:.3f}")
    print(f"n0 = {10**n0_ml:.10f}")
    print(f"p = {p_ml:.3f}")
    print(f"epsilon_e = {10**epsilon_e_ml:.5f}")
    print(f"epsilon_B = {10**epsilon_B_ml:.5f}")
    print(f"Residual (negative log-likelihood) = {best_solution.fun:.5f}")
    return best_solution

def run_sampling(x, y, initial, err, genfile=0,thetaObs=0.0,xi_N=1.0,d_L=1.0e26,z=0.01, steps=100, nwalkers=32, processes=4, filename="./data/test_sample.h5"):
    ndim = len(initial)
    
    # Use the wrapper function to fix datatype for the MCMC sampling
    log_prob_fixed = partial(log_probability,thetaObs=thetaObs,xi_N=xi_N,d_L=d_L,z=z)
    total_cores = multiprocessing.cpu_count()
    
    if genfile == 1:
        # Set up the backend
        backend = emcee.backends.HDFBackend(filename)
        try:
            # Load the last positions from the backend
            pos = backend.get_last_sample().coords
            print("Resuming from the last saved position.")
        except AttributeError:
            # If no previous sampling exists in the file, initialize the walkers
            print("No previous sampling found in the file. Starting fresh.")
            print("Finding optimal starting parameters...")
            soln = run_optimization(x, y, initial,err,thetaObs,xi_N,d_L,z)
            pos = soln.x + 1e-4 * np.random.randn(nwalkers, ndim)
            backend.reset(nwalkers, ndim)
    else:
        print("Finding optimal starting parameters...")
        soln = run_optimization(x, y, initial,err,thetaObs,xi_N,d_L,z)
        pos = soln.x + 1e-4 * np.random.randn(nwalkers, ndim)
    print("Utilizing {0:.1f}% of avaliable processes".format(100*processes/total_cores))
    print("Beginning sampling...")
    # Run the MCMC sampling with a multiprocessing pool
    with Pool(processes=processes) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_fixed, args=(x, y, err), pool=pool, backend=backend if genfile == 1 else None)
        sampler.run_mcmc(pos, steps, progress=True)
    return sampler