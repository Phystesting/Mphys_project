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
    Z = {
        'jetType': jet_type,  # Use the input jet type
        'specType': grb.jet.SimpleSpec,
        'thetaObs': thetaObs,
        'E0': 10**log_E0,
        'thetaCore': thetaCore,
        'thetaWing': 4*thetaCore,
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

    try:
        Flux = grb.fluxDensity(t, nu, **Z)
        # Check if all elements of Flux are finite
        if isinstance(Flux, np.ndarray):
            if not np.all(np.isfinite(Flux)):
                raise ValueError("Flux computation returned non-finite values.")
        elif not np.isfinite(Flux):
            raise ValueError("Flux computation returned a non-finite value.")
    except Exception as e:
        print(f"Error in fluxDensity computation: {e}")
        return np.full_like(t, 1e-300)  # Return a very small flux value

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
        #print(model)
        log_model = np.log(model)
        if not np.isfinite(log_model).all():
            raise ValueError("Model log-flux contains non-finite values.")
    except Exception as e:
        #print(f"Error in log-likelihood flux calculation: {e}")
        return -1e10  # Return a very large negative log-likelihood

    log_y = np.log(y)
    Lb_err, Ub_err = err_flux
    
    # Convert errors to log space for fitting
    log_Ub_err = abs(np.log(y + Ub_err) - log_y)
    log_Lb_err = abs(np.log(y - Lb_err) - log_y)
    
    # Select error for this iteration
    log_err = np.where(log_model > log_y, log_Ub_err, log_Lb_err)
    
    # Calculate the combined error term
    sigma2 = log_err**2
    #print(-0.5 * np.sum((log_y - log_model)**2 / sigma2))
    return -0.5 * np.sum((log_y - log_model)**2 / sigma2)


def log_prior(theta, param_names):
    mp = 1.67e27 # kg
    c = 3e10 # cm/s
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
    param_dict = dict(zip(param_names, theta))
    E0 = 10**param_dict.get("log_E0")
    n0 = 10**param_dict.get("log_n0")
    Gamma = (E0/(n0*mp*(c**5)))**(1/8) #calculate lorentz factor at t = 1s
    
    return -(np.exp((Gamma-1000)/200)+np.exp(-(Gamma-100)/10))  # discourages Lorentz values above 2000 and below 100


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
        "log_epsilon_B": (-5.0, 0.0),
        "log_E0": (45.0, 57.0),
        "thetaObs": (0.0, 0.8),
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

    ndim = len(param_names)
    pos = initial_guesses + 1e-4 * np.random.randn(nwalkers, ndim)

    log_prob = partial(
        log_probability, param_names=param_names, fixed_params=fixed_params, xi_N=xi_N, d_L=d_L, z=z, jet_type=jet_type
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
        soln = run_optimization(x, y, initial, fixed_params, err_flux, xi_N, d_L, z, jet_type)
        pos = soln.x + 1e-4 * np.random.randn(nwalkers, ndim)
        backend.reset(nwalkers, ndim)

    with Pool(processes=processes) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=(x, y, err_flux), pool=pool, backend=backend)
        sampler.run_mcmc(pos, steps, progress=True)

    print("Sampling complete.")
    return sampler
