import numpy as np
import afterglowpy as grb
import emcee
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import corner
import os
from multiprocessing import Pool
import time

#set up afterglowpy as a function suitable for fitting
def ag_py(t,thetaObs,thetaCore,thetaWing,n0,p,epsilon_e,epsilon_B,E0):
    Z = {'jetType':     grb.jet.Gaussian,     # Gaussian jet your discrepancy
		 'specType':    grb.jet.SimpleSpec,   # Basic Synchrotron Emission Spectrum 

		 'thetaObs':    thetaObs,   # Viewing angle in radians -known
		 'E0':          10**E0, # Isotropic-equivalent energy in erg
		 'thetaCore':   thetaCore,    # Half-opening angle in radians
		 'thetaWing':   thetaCore + (0.4-thetaCore)*thetaWing,    # Outer truncation angle
		 'n0':          10**n0,    # circumburst density in cm^{-3}
		 'p':           p,    # electron energy distribution index
		 'epsilon_e':   10**epsilon_e,    # epsilon_e
		 'epsilon_B':   10**epsilon_B,   # epsilon_B
		 'xi_N':        1.0,    # Fraction of electrons accelerated
		 'd_L':         1.36e26, # Luminosity distance in cm -known
		 'z':           0.01}   # redshift -known

	# Calculate flux in a single X-ray band (all times have same frequency)
    nu = np.empty(t.shape)
    nu[:] = 1.0e18 #x-ray

	# Calculate!
    return np.log(grb.fluxDensity(t, nu, **Z))

# define fitting residual
def residual(theta,x,y):
    thetaObs,thetaCore,thetaWing,n0,p,epsilon_e,epsilon_B,E0 = theta
    #print(E0, thetaObs, thetaCore, thetaWing, n0, p, epsilon_e, epsilon_B)
    model = ag_py(x,thetaObs,thetaCore,thetaWing,n0,p,epsilon_e,epsilon_B,E0)
    if not np.all(np.isfinite(model)):
        print(E0, thetaObs, thetaCore, thetaWing, n0, p, epsilon_e, epsilon_B)
        raise ValueError("ag_py returned non-finite values.")
    #print(sum(((y - model)**2) / abs(model)))
    return sum(((y - model)**2) / abs(model))

# Import data    
x, ydata = np.genfromtxt('../data/test_curve.txt',delimiter=',',skip_header=11, unpack=True)
y = np.log(ydata)


# Define bounds: E0 > 0 and log(f) unrestricted
bounds = [(0, np.pi*0.5),(0.01, 0.4), (0.0, 1.0),(-4.0, 3.0), (2.1, 2.8),(-4.0, 0.0), (-4.0, 0.0), (51.0, 54.0)] 

# definine inital parameters
E0 = 54.0
thetaObs = 0.1
thetaCore = 0.3
thetaWing = 0.5
n0 = -1.0
p = 2.5
epsilon_e = -2.0
epsilon_B = -4.0
np.random.seed(42)
initial = np.array([thetaObs,thetaCore,thetaWing,n0,p,epsilon_e,epsilon_B,E0])


#minimize the residual
start = time.time()
soln = minimize(residual, initial, args=(x, y), bounds=bounds,method='SLSQP')
end = time.time()
serial_time = end - start
print("Serial took {0:.1f} seconds".format(serial_time))

thetaObs_ml,thetaCore_ml,thetaWing_ml,n0_ml,p_ml,epsilon_e_ml,epsilon_B_ml,E0_ml = soln.x

y_ml = ag_py(x,thetaObs_ml,thetaCore_ml,thetaWing_ml,n0_ml,p_ml,epsilon_e_ml,epsilon_B_ml,E0_ml)

print("Maximum likelihood estimates:")
print("E0 = {0:.3f}".format(E0_ml))
print("thetaObs = {0:.3f}".format(thetaObs_ml))
print("thetaCore = {0:.3f}".format(thetaCore_ml))
print("thetaWing = {0:.3f}".format(thetaCore_ml + (0.4-thetaCore_ml) * thetaWing_ml))
print("n0 = {0:.4f}".format(10**n0_ml))
print("p = {0:.3f}".format(p_ml))
print("epsilon_e = {0:.4f}".format(10**epsilon_e_ml))
print("epsilon_B = {0:.4f}".format(10**epsilon_B_ml))

Fnu = ag_py(x,thetaObs_ml,thetaCore_ml,thetaWing_ml,n0_ml,p_ml,epsilon_e_ml,epsilon_B_ml,E0_ml)



fig, ax = plt.subplots(1, 1)

ax.plot(x,y,'x')
ax.plot(x, Fnu)

ax.set(xscale='log', xlabel=r'$t$ (s)', ylabel=r'$F_\nu$[$10^{18}$ Hz] (mJy)')

fig.savefig('datafit.png')
plt.close(fig)


def log_prior(theta):
    E0, thetaObs,thetaCore,thetaWing,n0,p,epsilon_e,epsilon_B = theta
    if (51 < E0 < 54
        and 0.0 < thetaObs < np.pi*0.5 
        and 0.01 < thetaCore < 0.4 
        and 0.0 < thetaWing < 1.0
        and -5.0 < n0 < 3.0
        and 2.1 < p < 3.0
        and -5.0 < epsilon_e < 0.0
        and -5.0 < epsilon_B < 0.0):
        return 0.0
    return -np.inf
    
def log_probability(theta, x, y):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + residual(theta, x, y)


pos = soln.x + 1e-4 * np.random.randn(32, 8)
nwalkers, ndim = pos.shape

sampler = emcee.EnsembleSampler(
	nwalkers, ndim, log_probability, args=(x, y)
)
sampler.run_mcmc(pos, 100000, progress=True);

#tau = sampler.get_autocorr_time()
#print(tau)

fig, axes = plt.subplots(8, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["E0", "thetaObs","thetaCore","ThetaWing","n0","p","epsilon_e","epsilon_B"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");
fig.savefig('parameter_steps.png')
plt.close(fig)


flat_samples = sampler.get_chain(flat=True)
fig2 = corner.corner(
    flat_samples, truths=[53, 0.3,0.05,0.4,1e-3,2.3,0.1,0.0001]
)


fig2.savefig('corner_plots.png')
plt.close(fig2)