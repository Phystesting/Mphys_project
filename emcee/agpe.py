import numpy as np
import afterglowpy as grb
import emcee
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import corner
import os
from multiprocessing import pool
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
    
def log_likelihood(theta, x, y, yerr):
    E0, thetaObs = theta
    thetaCore = 0.05
    thetaWing = 0.4
    n0 = 1.0e-3
    p = 2.3
    epsilon_e = 0.1
    epsilon_B = 0.0001
    model = ag_py(x,thetaObs,thetaCore,thetaWing,n0,p,epsilon_e,epsilon_B,E0)
    sigma2 = yerr**2
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

x, ydata = np.genfromtxt('../data/test_curve.txt',delimiter=',',skip_header=11, unpack=True)
y = np.log(ydata)
yerr = y*0.001

# Define bounds: E0 > 0 and log(f) unrestricted
bounds = [(0, None), (0, None)]  # E0 bounded to be positive, log(f) unbounded


E0 = 51
thetaObs = 0.0
np.random.seed(42)
nll = lambda *args: -log_likelihood(*args)
initial = np.array([E0,thetaObs]) + 0.1 * np.random.randn(2)
soln = minimize(nll, initial, args=(x, y, yerr), bounds=bounds)
E0_ml, thetaObs_ml = soln.x


thetaCore = 0.05
thetaWing = 0.4
n0 = 1.0e-3
p = 2.3
epsilon_e = 0.1
epsilon_B = 0.0001
y_ml = ag_py(x,thetaObs_ml,thetaCore,thetaWing,n0,p,epsilon_e,epsilon_B,E0_ml)

print("Maximum likelihood estimates:")
print("E0 = {0:.3f}".format(E0_ml))
print("thetaObs = {0:.3f}".format(thetaObs_ml))

def log_prior(theta):
    E0, thetaObs = theta
    if -51 < E0 < 54 and 0 < thetaObs < np.pi*0.5:
        return 0.0
    return -np.inf
    
def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)


pos = soln.x + 1e-4 * np.random.randn(32, 2)
nwalkers, ndim = pos.shape

sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability, args=(x, y, yerr)
)
sampler.run_mcmc(pos, 10, progress=True);

#tau = sampler.get_autocorr_time()
#print(tau)

fig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["E0", "thetaObs"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");
fig.savefig('test2.png')
plt.close(fig)


flat_samples = sampler.get_chain(flat=True)
fig2 = corner.corner(
    flat_samples, truths=[53, 0.01]
)


fig2.savefig('test3.png')
plt.close(fig2)
