import numpy as np
import afterglowpy as grb
import emcee
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from multiprocessing import Pool
from functools import partial
import warnings

# Adding error testing into the functions

def Flux(x, thetaCore, log_n0, p, log_epsilon_e, log_epsilon_B, log_E0, thetaObs, xi_N, d_L, z):
    """Calculate the model flux with error testing."""
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

    # Compute flux density
    Flux = grb.fluxDensity(t, nu, **Z)

    # Check for non-finite values
    if not np.all(np.isfinite(Flux)):
        raise ValueError("Non-finite values detected in Flux computation.")
    return Flux


def log_likelihood(theta, x, y, err_flux, param_names, fixed_params, xi_N, d_L, z):
    """Log-likelihood function with debugging and error testing."""
    params = {name: value for name, value in zip(param_names, theta)}
    params.update(fixed_params)

    # Unpack parameters
    thetaCore = params["thetaCore"]
    log_n0 = params["log_n0"]
    p = params["p"]
    log_epsilon_e = params["log_epsilon_e"]
    log_epsilon_B = params["log_epsilon_B"]
    log_E0 = params["log_E0"]
    thetaObs = params["thetaObs"]

    # Compute model flux
    try:
        model = Flux(x, thetaCore, log_n0, p, log_epsilon_e, log_epsilon_B, log_E0, thetaObs, xi_N, d_L, z)
    except Exception as e:
        raise RuntimeError(f"Error in Flux computation: {e}")

    # Convert to log-space
    log_y = np.log(y)
    log_model = np.log(model)

    if not np.all(np.isfinite(log_model)):
        raise ValueError("Non-finite log(model) values detected.")

    # Error conversion to log space
    Lb_err, Ub_err = err_flux
    log_Ub_err = abs(np.log(y + Ub_err) - log_y)
    log_Lb_err = abs(np.log(y - Lb_err) - log_y)

    # Use appropriate error bound
    log_err = np.where(model > log_y, log_Ub_err, log_Lb_err)
    if not np.all(np.isfinite(log_err)):
        raise ValueError("Non-finite log(error) values detected.")

    # Compute likelihood
    sigma2 = log_err**2
    likelihood = -0.5 * np.sum((log_y - log_model) ** 2 / sigma2)

    # Debugging output
    if np.isnan(likelihood):
        warnings.warn("NaN detected in log-likelihood.")
    return likelihood


def log_prior(theta, param_names):
    """Check priors with error testing."""
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
    """Log-probability function with checks."""
    lp = log_prior(theta, param_names)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, err_flux, param_names, fixed_params, xi_N, d_L, z)


# Sampling diagnostic function
def sampling_diagnostics(sampler):
    """Summarize and plot diagnostics for the sampler."""
    # Acceptance fraction
    print(f"Mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")
    print("Acceptance fraction range:",
          f"{np.min(sampler.acceptance_fraction):.3f} - {np.max(sampler.acceptance_fraction):.3f}")

    # Plot trace for each parameter
    samples = sampler.get_chain()
    num_params = samples.shape[2]
    fig, axes = plt.subplots(num_params, figsize=(10, 7), sharex=True)
    for i in range(num_params):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_ylabel(f"Param {i}")
    axes[-1].set_xlabel("Step number")
    plt.tight_layout()
    plt.show()

# You can now integrate these functions into your testing workflow to validate the model and sampler.

