import numpy as np
import matplotlib.pyplot as plt
import afterglowpy as grb
import emcee
import corner
import os
from multiprocessing import Pool

os.environ["OMP_NUM_THREADS"] = "1"

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

#initial parameter guesses
thetaObs = 0.4
thetaCore = 0.04
thetaWing = 0.06
n0 = 1.0e-4
p = 2.36
epsilon_e = 0.01
epsilon_B = 0.001
E0 = 53


guess = [thetaObs, thetaCore, thetaWing, n0, p, epsilon_e, epsilon_B, E0]

xdata, ydata = np.genfromtxt('../data/test_curve.txt',delimiter=',',skip_header=11, unpack=True)
xlog,ylog = np.log(xdata),np.log(ydata)

# Define the log likelihood function
def log_likelihood(theta, xdata, ydata, yerr):
    # Unpack the parameters from theta
    thetaObs, thetaCore, thetaWing, n0, p, epsilon_e, epsilon_B, E0 = theta
    
    # Compute the model flux
    model = ag_py(xdata, thetaObs, thetaCore, thetaWing, n0, p, epsilon_e, epsilon_B, E0)
    
    # Calculate the residuals (difference between model and data)
    sigma2 = yerr ** 2
    return -0.5 * np.sum((ydata - model) ** 2 / sigma2 + np.log(sigma2))

# Define the log prior function
def log_prior(theta):
    # Unpack parameters
    thetaObs, thetaCore, thetaWing, n0, p, epsilon_e, epsilon_B, E0 = theta

    # Apply bounds as priors
    if (0.0 < thetaObs < np.pi/2 and 0.0 < thetaCore < 0.2 and 0.0 < thetaWing < 0.4 and
        0.0 < n0 < 1e3 and 1.8 < p < 3.0 and 0.0 < epsilon_e < 0.1 and 
        0.0 < epsilon_B < 0.01 and 51 < E0 < 54):
        return 0.0  # log(1) for uniform prior
    return -np.inf  # log(0) for values outside bounds

# Define the log posterior function (sum of log prior and log likelihood)
def log_probability(theta, xdata, ydata, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, xdata, ydata, yerr)

# Initial parameter guesses (from least-squares fit or previous knowledge)
initial = [thetaObs, thetaCore, thetaWing, n0, p, epsilon_e, epsilon_B, E0]
nwalkers = 32  # Number of MCMC walkers
ndim = len(initial)  # Number of parameters to fit

# Add a small random perturbation to the initial guess to initialize walkers
pos = initial + 1e-4 * np.random.randn(nwalkers, ndim)


with Pool() as pool:
    # Define the MCMC sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,pool=pool, args=(xdata, ylog, np.ones_like(ylog)))

    # Run the MCMC chain
    nsteps = 5000  # Number of steps for MCMC to run
    sampler.run_mcmc(pos, nsteps, progress=True)

# Obtain the flat chain (discard burn-in)
flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)

# Plot the corner plot using the 'corner' library
labels = [r"$\theta_{\rm Obs}$", r"$\theta_{\rm Core}$", r"$\theta_{\rm Wing}$", 
          r"$n_0$", r"$p$", r"$\epsilon_e$", r"$\epsilon_B$", r"$E_0$"]
fig = corner.corner(flat_samples, labels=labels, truths=initial)

# Save the corner plot
fig.savefig('corner_plot.png')
plt.show()