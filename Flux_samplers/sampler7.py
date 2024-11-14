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

def Flux(x,thetaCore, log_n0, p, log_epsilon_e, log_epsilon_B, log_E0, thetaObs,xi_N,d_L,z):
    # Function to calculate the model
    Z = {
        'jetType': grb.jet.TopHat,
        'specType': grb.jet.SimpleSpec,
        'thetaObs': thetaObs,
        'E0': 10**log_E0,
        'thetaCore': thetaCore,
        'n0': 10**log_n0,
        'p': p,
        'epsilon_e': 10**log_epsilon_e,
        'epsilon_B': 10**log_epsilon_B,
        'xi_N': xi_N,
        'd_L': d_L,
        'z': z
	}
    t = x[0]
    nu = x[1]
    
    Flux = grb.fluxDensity(t, nu, **Z)
    return Flux

def log_likelihood(theta, x, y, err_flux, xi_N, d_L, z):
    # Unpack the parameters
    thetaCore, log_n0, p, log_epsilon_e, log_epsilon_B, log_E0, thetaObs = theta

    # Calculate the model flux
    model = Flux(x, thetaCore, log_n0, p, log_epsilon_e, log_epsilon_B, log_E0, thetaObs, xi_N, d_L, z)
    log_model = np.log(model)
    
    # Check if the model returns finite values
    if not np.all(np.isfinite(model)):
        raise ValueError("Model returned non-finite values.")
    
    log_y = np.log(y)
    
    #convert errors to log space for fitting
    Ub_err = abs(np.log(y + err_flux) - log_y)
    Lb_err = abs(np.log(y - err_flux) - log_y)
    
    #select errors for this iteration
    log_err = np.zeros(len(err_flux))
    
    #if model is above data use ub if it's below use lb
    for index, model_value in enumerate(model):
        if model_value > log_y[index]:
            log_err[index] = Ub_err[index]
        else:
            log_err[index] = Lb_err[index]
    
    # Calculate the combined error term (flux, time, and spectral errors)
    sigma2 = log_err**2
    
    # Optional penalty for parameters outside certain bounds
    #penalty_factor = 2 * np.where(abs(n0) > 4, (abs(n0) - 4) ** 2, 0) + np.where(0.8,100*(-1 + 1/np.cos(2*thetaObs)), 0)
    # Calculate the negative log-likelihood
    #print(-0.5 * np.sum((log_y - log_model)**2 / sigma2 ))
    return -0.5 * np.sum((log_y - log_model)**2 / sigma2)

def log_prior(theta):
    # Function for log-prior calculation
    thetaCore, log_n0, p, log_epsilon_e, log_epsilon_B, log_E0, thetaObs = theta
    if (45 < log_E0 < 57
        and 0.01 < thetaCore < np.pi*0.5
        and -10.0 < log_n0 < 10.0
        and 2.1 < p < 5.0
        and -5.0 < log_epsilon_e < 0.0
        and -5.0 < log_epsilon_B < 0.0
        and 0.0 < thetaObs < 0.8):
        return 0
    return -np.inf
    
def log_probability(theta, x, y, err, xi_N, d_L, z):
    # Calculate the log-prior
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    # Combine the log-prior and log-likelihood
    return lp + log_likelihood(theta, x, y, err, xi_N, d_L, z)
    

def run_optimization(x, y, initial, err, xi_N=1.0, d_L=1.0e26, z=0.01):
    # Perform the parameter optimization with updated likelihood
    bounds = [(0.01, np.pi*0.5), (-10.0, 10.0), (2.1, 5.0), (-5.0, 0.0), (-5.0, 0.0), (45.0, 57.0),(0.0,0.8)]
    likelihood_fixed = partial(log_likelihood, xi_N=xi_N, d_L=d_L, z=z)
    nll = lambda *args: -likelihood_fixed(*args)
    start = time.time()
    soln = minimize(nll, initial, args=(x, y, err), bounds=bounds, method='Nelder-mead')
    end = time.time()
    serial_time = end - start
    print(f"Most probable parameters identified in {serial_time:.1f} seconds")
    thetaCore_ml, log_n0_ml, p_ml, log_epsilon_e_ml, log_epsilon_B_ml, log_E0_ml, thetaObs_ml = soln.x
    print(f"E0 = {10**log_E0_ml:.3e}")
    print(f"thetaCore = {thetaCore_ml:.3f}")
    print(f"n0 = {10**log_n0_ml:.10f}")
    print(f"p = {p_ml:.3f}")
    print(f"epsilon_e = {10**log_epsilon_e_ml:.5f}")
    print(f"epsilon_B = {10**log_epsilon_B_ml:.5f}")
    print(f"thetaObs = {thetaObs_ml:.3f}")
    print(f"Residual (negative log-likelihood) = {soln.fun:.5f}")
    print("-" * 30)
    return soln

def run_parallel_optimization(x, y, initial, err,processes=4, xi_N=1.0, d_L=1.0e26, z=0.01):
     # Bounds for the parameters
    bounds = [(0.01, 0.5), (-2, 2), (2.1, 3), (-3.0, -1), (-5.0, -3), (45.0, 57.0), (0.0, 0.8)]

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
    thetaCore_ml, log_n0_ml, p_ml, log_epsilon_e_ml, log_epsilon_B_ml, log_E0_ml,thetaObs_ml = best_solution.x

    print("Best parameters identified:")
    print(f"E0 = {10**log_E0_ml:.3e}")
    print(f"thetaCore = {thetaCore_ml:.3f}")
    print(f"n0 = {10**log_n0_ml:.10f}")
    print(f"p = {p_ml:.3f}")
    print(f"epsilon_e = {10**log_epsilon_e_ml:.5f}")
    print(f"epsilon_B = {10**log_epsilon_B_ml:.5f}")
    print(f"thetaObs = {thetaObs_ml:.3f}")
    print(f"Residual (negative log-likelihood) = {best_solution.fun:.5f}")
    return best_solution

def run_sampling(x, y, initial, err, genfile=0,parallel_optimization=0,xi_N=1.0,d_L=1.0e26,z=0.01, steps=100, nwalkers=32, processes=4, filename="./data/test_sample.h5"):
    ndim = len(initial)
    
    # Use the wrapper function to fix datatype for the MCMC sampling
    log_prob_fixed = partial(log_probability,xi_N=xi_N,d_L=d_L,z=z)
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
            if parallel_optimization == 0:
                soln = run_optimization(x, y, initial,err,xi_N,d_L,z)
            else:
                soln = run_parallel_optimization(x, y, initial,err,parallel_optimization,xi_N,d_L,z)
            pos = soln.x + 1e-4 * np.random.randn(nwalkers, ndim)
            backend.reset(nwalkers, ndim)
    else:
        print("Finding optimal starting parameters...")
        if parallel_optimization == 0:
            soln = run_optimization(x, y, initial,err,xi_N,d_L,z)
        else:
            soln = run_parallel_optimization(x, y, initial,err,parallel_optimization,xi_N,d_L,z)
        pos = soln.x + 1e-4 * np.random.randn(nwalkers, ndim)
    print("Utilizing {0:.1f}% of avaliable processes".format(100*processes/total_cores))
    print("Beginning sampling...")
    # Run the MCMC sampling with a multiprocessing pool
    with Pool(processes=processes) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_fixed, args=(x, y, err), pool=pool, backend=backend if genfile == 1 else None)
        sampler.run_mcmc(pos, steps, progress=True)
    return sampler
