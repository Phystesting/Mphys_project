import numpy as np
import afterglowpy as grb
import emcee
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from multiprocessing import Pool
import time
from functools import partial
import concurrent.futures


def Flux(x, thetaCore, log_n0, p, log_epsilon_e, log_epsilon_B, log_E0, thetaObs, xi_N, d_L, z, jet_type):
    # Function to calculate the model
    thetaWing = 4 * thetaCore
    E0 = 10 ** log_E0
    n0 = 10 ** log_n0
    epsilon_e = 10 ** log_epsilon_e
    epsilon_B = 10 ** log_epsilon_B
    Z = {
        'jetType': jet_type,
        'specType': grb.jet.SimpleSpec,
        'thetaObs': thetaObs,
        'E0': E0,
        'thetaCore': thetaCore,
        'thetaWing': thetaWing,
        'n0': n0,
        'p': p,
        'epsilon_e': epsilon_e,
        'epsilon_B': epsilon_B,
        'xi_N': xi_N,
        'd_L': d_L,
        'z': z,
    }
    t = x[0]
    nu = x[1]

    try:
        # Attempt to calculate the flux for all time-frequency pairs at once
        Flux = grb.fluxDensity(t, nu, **Z)

        # If the result contains non-finite values, retry for individual pairs
        if isinstance(Flux, np.ndarray) and not np.all(np.isfinite(Flux)):
            raise ValueError("Non-finite values detected in bulk computation.")

        if not isinstance(Flux, np.ndarray) and not np.isfinite(Flux):
            raise ValueError("Non-finite value detected in single flux computation.")

    except Exception as e:
        print(f"Error in bulk fluxDensity computation: {e}")
        print(f"Retrying with individual time-frequency pairs...")
        print(thetaCore,thetaWing,thetaObs,E0,n0,epsilon_e,epsilon_B,p)
        # Initialize an array to store the results
        Flux = np.zeros_like(t, dtype=float)

        # Process each time-frequency pair individually
        for i, (t_i, nu_i) in enumerate(zip(t, nu)):
            try:
                Flux[i] = grb.fluxDensity(t_i, nu_i, **Z)[0]

                # Check if the result is finite
                if not np.isfinite(Flux[i]):
                    raise ValueError(f"Non-finite value at t={t_i}, nu={nu_i}")
            
            except Exception as single_error:
                print(f"Error in single fluxDensity computation for t={t_i}, nu={nu_i}: {single_error}")
                Flux[i] = 1e-300  # Assign a very small flux value for problematic pairs

    return Flux



def log_likelihood(theta, x, y, err_flux, param_names, fixed_params, xi_N, d_L, z, jet_type):
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
    try:
        model = Flux(x, thetaCore, log_n0, p, log_epsilon_e, log_epsilon_B, log_E0, thetaObs, xi_N, d_L, z, jet_type)
        #print(min(model))
        log_model = np.log(model)
        if not np.isfinite(log_model).all():
            raise ValueError("Model log-flux contains non-finite values.")
    except Exception as e:
        return -1e10  # Return a very large negative log-likelihood


    # Select error for this iteration
    Lb_err, Ub_err = err_flux
    log_y = np.log(y)
    Lb_err, Ub_err = err_flux
    
    # Convert errors to log space for fitting
    log_Ub_err = abs(np.log(y + Ub_err) - log_y)
    log_Lb_err = abs(np.log(y - Lb_err) - log_y)
    
    # Select error for this iteration
    log_err = np.where(log_model > log_y, log_Ub_err, log_Lb_err)

    # Apply weighting to optical data
    sigma2 = log_err**2
    residuals = ((log_y - log_model)**2) / sigma2

    #print(-0.5 * np.sum(np.abs(np.log(sigma2)) + residuals))
    #return -0.5 * np.sum(np.abs(np.log(sigma2)) + residuals)
    #print(-0.5 * np.sum(residuals))
    return -0.5 * np.sum(residuals)


def log_prior(theta, param_names):
    mp = 1.67e-24 # g
    c = 3e10 # cm/s
    priors = {
        "thetaCore": (0.01, np.pi * 0.5),
        "log_n0": (-10.0, 10.0),
        "p": (2.1, 5.0),
        "log_epsilon_e": (-5.0, 0.0),
        "log_epsilon_B": (-5.0, -1.0),
        "log_E0": (45.0, 57.0),
        "thetaObs": (0.0, np.pi * 0.5),
    }

    for value, name in zip(theta, param_names):
        low, high = priors[name]
        if not (low < value < high):
            return -np.inf  # Outside bounds
    param_dict = dict(zip(param_names, theta))
    E0 = 10**param_dict.get("log_E0")
    n0 = 10**param_dict.get("log_n0")
    thetaCore = param_dict.get("thetaCore")
    Gamma = (E0/(n0*mp*(c**5)))**(1./8.) #calculate lorentz factor at t = 1s
    if not (100 < Gamma < 10000):
        return -np.inf
    
    return 0


def log_probability(theta, x, y, err_flux, param_names, fixed_params, xi_N, d_L, z, jet_type):
    lp = log_prior(theta, param_names)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, err_flux, param_names, fixed_params, xi_N, d_L, z, jet_type)


def run_optimization(x, y, initial, fixed_params, err_flux, xi_N, d_L, z, jet_type):
    # Split into fitting and fixed parameters
    param_names = list(initial.keys())
    initial_guesses = list(initial.values())

    bounds = {
        "thetaCore": (0.01, np.pi * 0.5),
        "log_n0": (-10.0, 10.0),
        "p": (2.1, 5.0),
        "log_epsilon_e": (-5.0, 0.0),
        "log_epsilon_B": (-5.0, -2.0),
        "log_E0": (45.0, 57.0),
        "thetaObs": (0.0, np.pi * 0.5),
    }
    fit_bounds = [bounds[param] for param in param_names]

    likelihood = partial(
        log_likelihood, param_names=param_names, fixed_params=fixed_params, xi_N=xi_N, d_L=d_L, z=z, jet_type=jet_type
    )
    nll = lambda *args: -likelihood(*args)
    result = minimize(nll, initial_guesses, args=(x, y, err_flux), bounds=fit_bounds, method="L-BFGS-B")

    print("Optimization complete.")
    for name, value in zip(param_names, result.x):
        print(f"{name}: {value:.6f}")  # Adjust decimal places as needed
    print(f"Residual (negative log-likelihood) = {result.fun:.5f}")
    print("-" * 30)
    return result

    
def run_sampling(x, y, initial, fixed_params, err_flux, xi_N, d_L, z, jet_type, nwalkers, steps, processes, filename):
    param_names = list(initial.keys())
    initial_guesses = list(initial.values())
    #print(initial_guesses,param_names)
    ndim = len(param_names)
    pos = initial_guesses + 1e-4 * np.random.randn(nwalkers, ndim)

    log_prob = partial(
        log_probability, param_names=param_names, fixed_params=fixed_params, xi_N=xi_N, d_L=d_L, z=z, jet_type=jet_type
    )

    # Set up the backend
    backend = emcee.backends.HDFBackend(filename)
    
    # Attach parameter names as metadata
    with backend.open("a") as f:
        if "param_names" not in f.attrs:  # Avoid overwriting existing labels
            f.attrs["param_names"] = param_names

    try:
        # Load the last positions from the backend
        pos = backend.get_last_sample().coords
        print("Resuming from the last saved position.")
    except AttributeError:
        # If no previous sampling exists in the file, initialize the walkers
        print("No previous sampling found in the file. Starting fresh.")
        print("Finding optimal starting parameters...")
        soln = run_optimization(x, y, initial, fixed_params, err_flux, xi_N, d_L, z, jet_type)
        pos = soln.x + 1e-4 * np.random.randn(nwalkers, ndim)
        backend.reset(nwalkers, ndim)

    with Pool(processes=processes) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=(x, y, err_flux), pool=pool, backend=backend)
        sampler.run_mcmc(pos, steps, progress=True)

    print("Sampling complete.")
    return sampler

