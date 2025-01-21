import numpy as np
import afterglowpy as grb
import emcee
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from multiprocessing import Pool
import time
from functools import partial
import concurrent.futures

def get_changes(lst):
    if not lst:
        return []
    changes = [lst[0]]
    indices = [0]  # Track starting indices of changes
    for i in range(1, len(lst)):
        if lst[i] != lst[i - 1]:
            changes.append(lst[i])
            indices.append(i)
    return changes, indices

def Flux(x, thetaCore, log_n0, p, log_epsilon_e, log_epsilon_B, log_E0, thetaObs, xi_N, d_L, z, jet_type):
    Z = {
        'jetType': jet_type,
        'specType': grb.jet.SimpleSpec,
        'thetaObs': thetaObs,
        'E0': 10**log_E0,
        'thetaCore': thetaCore,
        'thetaWing': 4 * thetaCore,
        'n0': 10**log_n0,
        'p': p,
        'epsilon_e': 10**log_epsilon_e,
        'epsilon_B': 10**log_epsilon_B,
        'xi_N': xi_N,
        'd_L': d_L,
        'z': z,
    }

    t = np.array(x[0])
    nu = np.array(x[1])

    # Identify unique frequency segments
    unique_freq, segment_indices = get_changes(nu.tolist())
    segment_indices.append(len(nu))  # Add end index for the last segment

    Flux = []

    for i in range(len(unique_freq)):
        # Create a mask for the current segment
        start_idx = segment_indices[i]
        end_idx = segment_indices[i + 1]
        mask = slice(start_idx, end_idx)

        # Compute flux for the segment
        flux_segment = grb.fluxDensity(t[mask], unique_freq[i], **Z)
        Flux.extend(flux_segment)

    # Convert to NumPy array to match shape if needed
    Flux = np.array(Flux)

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
        log_model = np.log(model)
        #print(log_E0,thetaObs,log_epsilon_e,thetaCore,log_n0,p,log_epsilon_B)
        if not np.isfinite(log_model).all():
            raise ValueError("Model log-flux contains non-finite values.")
    except Exception as e:
        print(f"Error in log-likelihood flux calculation: {e}")
        return -1e10  # Return a very large negative log-likelihood

    log_y = np.log(y)
    Lb_err, Ub_err = err_flux
    
    # Convert errors to log space for fitting
    log_Ub_err = abs(np.log(y + Ub_err) - log_y)
    log_Lb_err = abs(np.log(y - Lb_err) - log_y)
    #print(max(log_model)/np.log(10))
    # Select error for this iteration
    log_err = np.where(log_model > log_y, log_Ub_err, log_Lb_err)
    #print(min(log_model),min(log_err))
    # Calculate the combined error term
    sigma2 = log_err**2
    #print(-0.5 * np.sum(np.log(sigma2) + ((log_y - log_model)**2) / sigma2))
    return -0.5 * np.sum(np.log(sigma2) + ((log_y - log_model)**2) / sigma2)


def log_prior(theta, param_names):
    mp = 1.67e-24 # g
    c = 3e10 # cm/s
    priors = {
        "thetaCore": (0.01, np.pi * 0.5),
        "log_n0": (-10.0, 10.0),
        "p": (2.1, 5.0),
        "log_epsilon_e": (-5.0, 0.0),
        "log_epsilon_B": (-5.0, 0.0),
        "log_E0": (45.0, 57.0),
        "thetaObs": (0.0, np.pi * 0.5),
    }
    
    param_dict = dict(zip(param_names, theta))
    E0 = 10**param_dict.get("log_E0")
    n0 = 10**param_dict.get("log_n0")
    Gamma = (E0/(n0*mp*(c**5)))**(1./8.) #calculate lorentz factor at t = 1s
    
    for value, name in zip(theta, param_names):
        low, high = priors[name]
        if not (low < value < high):
            return -np.inf  # Outside bounds
    
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
        "log_epsilon_B": (-5.0, 0.0),
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

