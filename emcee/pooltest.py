import numpy as np
import afterglowpy as grb
import emcee
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import corner
import os
from multiprocessing import Pool, TimeoutError
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

def log_likelihood(theta,x,y):
    return -0.5 * residual(theta,x,y)

# Import data    
x, ydata = np.genfromtxt('../data/test_curve.txt',delimiter=',',skip_header=11, unpack=True)
y = np.log(ydata)



def log_prior(theta):
    thetaObs,thetaCore,thetaWing,n0,p,epsilon_e,epsilon_B, E0  = theta
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
    return lp + log_likelihood(theta, x, y)


pos = [0.3,0.05,1.0,-3.0,2.3,-1.0,-4.0,53.0] + 1e-4 * np.random.randn(32, 8)
nwalkers, ndim = pos.shape
if __name__ == '__main__':
    # start 4 worker processes
    with Pool(processes=4) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability, args=(x, y),pool=pool
        )
        try:
            sampler.run_mcmc(pos, 5000, progress=True)
        except Exception as e:
            print(f"Error during sampling: {e}")
        finally:
            pool.close()
            pool.join()  # Ensure all worker processes are cleaned up
    
#tau = sampler.get_autocorr_time()
#print(tau)

    fig, axes = plt.subplots(8, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = ["thetaObs","thetaCore","ThetaWing","n0","p","epsilon_e","epsilon_B","E0"]
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
        flat_samples, labels=labels, truths=[0.3,0.05,1.0,-3.0,2.3,-1.0,-4.0,53.0]
    )


    fig2.savefig('probable_parameters.png')
    plt.close(fig2)