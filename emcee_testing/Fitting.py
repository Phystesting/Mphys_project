import numpy as np
import afterglowpy as grb
import emcee
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import corner
import os
from multiprocessing import Pool
import time

def ag_py(t,thetaObs,thetaCore,thetaWing,n0,p,epsilon_e,epsilon_B,E0):
    Z = {'jetType':     grb.jet.Gaussian,     # Gaussian jet your discrepancy
		 'specType':    grb.jet.SimpleSpec,   # Basic Synchrotron Emission Spectrum 

		 'thetaObs':    thetaObs,   # Viewing angle in radians -known
		 'E0':          10**E0, # Isotropic-equivalent energy in erg
		 'thetaCore':   thetaCore,    # Half-opening angle in radians
		 'thetaWing':   thetaWing,    # Outer truncation angle
		 'n0':          n0,    # circumburst density in cm^{-3}
		 'p':           p,    # electron energy distribution index
		 'epsilon_e':   epsilon_e,    # epsilon_e
		 'epsilon_B':   epsilon_B,   # epsilon_B
		 'xi_N':        1.0,    # Fraction of electrons accelerated
		 'd_L':         1.36e26, # Luminosity distance in cm -known
		 'z':           0.01}   # redshift -known

	# Calculate flux in a single X-ray band (all times have same frequency)
    nu = np.empty(t.shape)
    nu[:] = 1.0e18 #x-ray
    nu = np.empty(t.shape)
    nu[:] = 1.0e18 #x-ray

    # Calculate!
    return np.log(grb.fluxDensity(t, nu, **Z))
    
# Define the residual function to minimize
def residual(params, t, flux_observed):
    thetaObs, thetaCore, thetaWing, n0, p, epsilon_e, epsilon_B, E0 = params
    flux_predicted = ag_py(t, thetaObs, thetaCore, thetaWing, n0, p, epsilon_e, epsilon_B, E0)
    if not np.all(np.isfinite(flux_predicted)):
        print(E0, thetaObs, thetaCore, thetaWing, n0, p, epsilon_e, epsilon_B)
        raise ValueError("ag_py returned non-finite values.")
    return np.sum((flux_predicted - flux_observed)**2)  # Sum of squared errors

# input data
t_data, ydata = np.genfromtxt('../data/test_curve.txt',delimiter=',',skip_header=11, unpack=True)
flux_observed = np.log(ydata)

# Initial guess for parameters [thetaObs, thetaCore, thetaWing, n0, p, epsilon_e, epsilon_B, E0]
initial_guess = [0.4, 0.05, 0.4, 1e-3, 2.3, 0.1, 0.01, 53.0]

# Bounds for parameters (optional)
bounds = [(0, np.pi*0.5),  # thetaObs bounds
          (0, np.pi*0.5),  # thetaCore bounds
          (0, np.pi*0.5),  # thetaWing bounds
          (1e-3, 1e1),  # n0 bounds
          (2.1, 3.0),  # p bounds
          (1e-5, 1.0),  # epsilon_e bounds
          (1e-5, 1.0),  # epsilon_B bounds
          (51.0, 54.0)]  # E0 bounds (log10(E0))

# Define the constraint: thetaCore < thetaWing
epsilon = 1e-4  # Small positive margin to avoid numerical instability
def constraint_theta(params):
    thetaCore = params[1]
    thetaWing = params[2]
    return thetaWing - thetaCore # We want thetaWing > thetaCore

# Define the constraints for minimize
constraints = ({
    'type': 'ineq',  # Inequality constraint: constraint >= 0
    'fun': constraint_theta
})

# Perform the minimization

result = minimize(residual, initial_guess, args=(t_data, flux_observed), bounds=bounds,constraints=constraints) #,constraints=constraints,method='trust-constr'

# Extract the optimized parameters
optimized_params = result.x

print("Optimized parameters:", optimized_params)