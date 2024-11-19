import numpy as np
import afterglowpy as grb
import emcee
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from multiprocessing import Pool
import time
from functools import partial
import concurrent.futures


def Flux(x, thetaCore, log_n0, p, log_epsilon_e, log_epsilon_B, log_E0, thetaObs, xi_N, d_L, z):
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
        'z': z,
    }
    t = x[0]
    nu = x[1]
    Flux = grb.fluxDensity(t, nu, **Z)
    return Flux


def log_likelihood(theta, x, y, err_flux, param_names, fixed_params, xi_N, d_L, z):

    # Create a dictionary for the combined parameters
    params = {name: value for name, value in zip(param_names, theta)}
    params.update(fixed_params)

    # Unpack the parameters
    thetaCore = params["thetaCore"]
    log_n0 = params["log_n0"]
    p = params["p"]
    log_epsilon_e = params["log_epsilon_e"]
    log_epsilon_B = params["log_epsilon_B"]
    log_E0 = params["log_E0"]
    thetaObs = params["thetaObs"]

    # Calculate the model flux
    model = Flux(x, thetaCore, log_n0, p, log_epsilon_e, log_epsilon_B, log_E0, thetaObs, xi_N, d_L, z)
    log_model = np.log(model)

    if not np.all(np.isfinite(log_model)):
        raise ValueError("Model returned non-finite values.")

    log_y = np.log(y)

    # Convert errors to log space
    Lb_err, Ub_err = err_flux
    log_Ub_err = np.abs(np.log(y + Ub_err) - log_y)
    log_Lb_err = np.where(Lb_err == y, np.inf, np.abs(np.log(y - Lb_err) - log_y))
    #print(log_Lb_err)
    # Select errors for the current iteration
    log_err = np.where(model > log_y, log_Ub_err, log_Lb_err)

    mask = log_err != np.inf

    # Perform the calculation only when no log_err values are zero
    sigma2 = log_err[mask]**2
    log_likelihood_value = -0.5*sum(np.where(log_err == 0 ,np.exp((log_y[mask] - log_model[mask])),(log_y[mask] - log_model[mask])**2 / sigma2))
    print(log_likelihood_value)
    return log_likelihood_value



def log_prior(theta, param_names):
    priors = {
        "thetaCore": (0.01, np.pi * 0.5),
        "log_n0": (-10.0, 10.0),
        "p": (2.1, 5.0),
        "log_epsilon_e": (-5.0, 0.0),
        "log_epsilon_B": (-5.0, 0.0),
        "log_E0": (45.0, 57.0),
        "thetaObs": (0.0, 0.8),
    }

    for value, name in zip(theta, param_names):
        low, high = priors[name]
        if not (low < value < high):
            return -np.inf  # Outside bounds
    
    return 0.0  # Uniform prior within bounds


def log_probability(theta, x, y, err_flux, param_names, fixed_params, xi_N, d_L, z):
    lp = log_prior(theta, param_names)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, err_flux, param_names, fixed_params, xi_N, d_L, z)


def run_optimization(x, y, initial,fixed_params, err_flux, xi_N, d_L, z):
    # Split into fitting and fixed parameters

    param_names = list(initial.keys())
    initial_guesses = list(initial.values())

    bounds = {
        "thetaCore": (0.01, np.pi * 0.5),
        "log_n0": (-10.0, 10.0),
        "p": (2.1, 5.0),
        "log_epsilon_e": (-5.0, 0.0),
        "log_epsilon_B": (-5.0, 0.0),
        "log_E0": (45.0, 57.0),
        "thetaObs": (0.0, 0.8),
    }
    fit_bounds = [bounds[param] for param in param_names]

    likelihood = partial(
        log_likelihood, param_names=param_names, fixed_params=fixed_params, xi_N=xi_N, d_L=d_L, z=z
    )
    nll = lambda *args: -likelihood(*args)
    result = minimize(nll, initial_guesses, args=(x, y, err_flux), bounds=fit_bounds, method="L-BFGS-B")

    print("Optimization complete.")
    for name, value in zip(param_names, result.x):
        print(f"{name}: {value:.6f}")  # Adjust decimal places as needed
    print(f"Residual (negative log-likelihood) = {result.fun:.5f}")
    print("-" * 30)
    return result


def run_sampling(x, y, initial,fixed_params, err_flux, xi_N, d_L, z, nwalkers, steps, processes,filename):

    param_names = list(initial.keys())
    initial_guesses = list(initial.values())

    ndim = len(param_names)
    pos = initial_guesses + 1e-4 * np.random.randn(nwalkers, ndim)

    log_prob = partial(
        log_probability, param_names=param_names, fixed_params=fixed_params, xi_N=xi_N, d_L=d_L, z=z
    )
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
        soln = run_optimization(x, y, initial,fixed_params,err_flux,xi_N,d_L,z)
        pos = soln.x + 1e-4 * np.random.randn(nwalkers, ndim)
        backend.reset(nwalkers, ndim)

    with Pool(processes=processes) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=(x, y, err_flux), pool=pool,backend=backend)
        sampler.run_mcmc(pos, steps, progress=True)

    print("Sampling complete.")
    return sampler

