import numpy as np
import afterglowpy as grb
import emcee
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import corner
import os
from multiprocessing import Pool
import time

def ag_py(t, thetaObs, thetaCore, thetaWing, n0, p, epsilon_e, epsilon_B, E0):
    # Function to calculate the model
    Z = {
        'jetType': grb.jet.Gaussian,
        'specType': grb.jet.SimpleSpec,
        'thetaObs': thetaObs,
        'E0': 10**E0,
        'thetaCore': thetaCore,
        'thetaWing': thetaCore + (0.4 - thetaCore) * thetaWing,
        'n0': 10**n0,
        'p': p,
        'epsilon_e': 10**epsilon_e,
        'epsilon_B': 10**epsilon_B,
        'xi_N': 1.0,
        'd_L': 1.36e26,
        'z': 0.01
    }
    nu = np.empty(t.shape)
    nu[:] = 1.0e18
    return np.log(grb.fluxDensity(t, nu, **Z))

def residual(theta, x, y):
    # Function to calculate the residual
    thetaObs, thetaCore, thetaWing, n0, p, epsilon_e, epsilon_B, E0 = theta
    model = ag_py(x, thetaObs, thetaCore, thetaWing, n0, p, epsilon_e, epsilon_B, E0)
    if not np.all(np.isfinite(model)):
        raise ValueError("ag_py returned non-finite values.")
    return sum(((y - model) ** 2) / abs(model))

def log_likelihood(theta, x, y):
    # Function for log-likelihood calculation
    return -0.5 * residual(theta, x, y)

def log_prior(theta):
    # Function for log-prior calculation
    thetaObs, thetaCore, thetaWing, n0, p, epsilon_e, epsilon_B, E0 = theta
    if (51 < E0 < 54
        and 0.0 < thetaObs < np.pi * 0.5
        and 0.01 < thetaCore < 0.4
        and 0.0 < thetaWing < 1.0
        and -5.0 < n0 < 3.0
        and 2.1 < p < 3.0
        and -5.0 < epsilon_e < 0.0
        and -5.0 < epsilon_B < 0.0):
        return 0.0
    return -np.inf

def log_probability(theta, x, y):
    # Function for log-probability calculation
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y)

def main():
    # Import data and perform initial setup
    x, ydata = np.genfromtxt('../data/test_curve.txt', delimiter=',', skip_header=11, unpack=True)
    y = np.log(ydata)

    # Define initial parameter values
    initial = np.array([0.1, 0.3, 0.5, -1.0, 2.5, -2.0, -4.0, 54.0])
    bounds = [(0, np.pi * 0.5), (0.01, 0.4), (0.0, 1.0), (-4.0, 3.0), 
              (2.1, 2.8), (-4.0, 0.0), (-4.0, 0.0), (51.0, 54.0)]

    # Minimize the residual
    start = time.time()
    soln = minimize(residual, initial, args=(x, y), bounds=bounds,method='SLSQP')
    end = time.time()
    serial_time = end - start
    print("Most Probable parameters identified in {0:.1f} seconds".format(serial_time))
    thetaObs_ml,thetaCore_ml,thetaWing_ml,n0_ml,p_ml,epsilon_e_ml,epsilon_B_ml,E0_ml = soln.x
    print("Chi squared estimates:")
    print("E0 = {0:.3f}".format(E0_ml))
    print("thetaObs = {0:.3f}".format(thetaObs_ml))
    print("thetaCore = {0:.3f}".format(thetaCore_ml))
    print("thetaWing = {0:.3f}".format(thetaCore_ml + (0.4-thetaCore_ml) * thetaWing_ml))
    print("n0 = {0:.4f}".format(10**n0_ml))
    print("p = {0:.3f}".format(p_ml))
    print("epsilon_e = {0:.4f}".format(10**epsilon_e_ml))
    print("epsilon_B = {0:.4f}".format(10**epsilon_B_ml))
    
    pos = soln.x + 1e-4 * np.random.randn(32, 8)
    nwalkers, ndim = pos.shape
    print("starting sampling...")
    if __name__ == '__main__':
        # Set up the MCMC sampler
        with Pool(processes=4) as pool:
            
            try:
                sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y), pool=pool)
                sampler.run_mcmc(pos, 5000, progress=True)
            except KeyboardInterrupt:
                print("\nSampling interrupted! Shutting down pool...")
                pool.terminate()
            except Exception as e:
                print(f"Error during sampling: {e}")
            finally:
                pool.close()
                pool.join()
        
        # Plotting results
        fig, axes = plt.subplots(8, figsize=(10, 7), sharex=True)
        samples = sampler.get_chain()
        labels = ["thetaObs", "thetaCore", "ThetaWing", "n0", "p", "epsilon_e", "epsilon_B", "E0"]
        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
        axes[-1].set_xlabel("step number")
        fig.savefig('parameter_steps.png')
        plt.close(fig)

        flat_samples = sampler.get_chain(discard=100, thin=15,flat=True)
        fig2 = corner.corner(flat_samples, labels=labels, truths=[0.3, 0.05, 1.0, -3.0, 2.3, -1.0, -4.0, 53.0])
        fig2.savefig('probable_parameters.png')
        plt.close(fig2)

# Entry point
if __name__ == '__main__':
    main()