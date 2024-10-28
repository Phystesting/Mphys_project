import numpy as np
import afterglowpy as grb
import emcee
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import curve_fit
import corner
import os
import sys
import multiprocessing
from multiprocessing import Pool
import time
from functools import partial
import concurrent.futures

#create a function to stack the lists
def appendList(l,element):
    l.append(element)
    return l

def Flux(x, thetaObs, thetaCore, n0, p, epsilon_e, epsilon_B, E0,xi_N,d_L,z):
    # Function to calculate the model
    Z = {
        'jetType': grb.jet.TopHat,
        'specType': grb.jet.SimpleSpec,
        'thetaObs': thetaObs,
        'E0': 10**E0,
        'thetaCore': thetaCore,
        'n0': 10**n0,
        'p': p,
        'epsilon_e': 10**epsilon_e,
        'epsilon_B': 10**epsilon_B,
        'xi_N': xi_N,
        'd_L': d_L,
        'z': z
	}
    t = x[0]
    nu = x[1]
    Flux = [np.log(grb.fluxDensity(t[0], nu, **Z))]
    #print(Flux)
    for i in t[1:]:
        Flux = appendList(Flux,np.log(grb.fluxDensity(i, nu, **Z)))
        #print(Flux)
        #print('')
    return Flux

def log_likelihood(theta, x, y, yerr,xi_N,d_L,z):
    thetaObs, thetaCore, n0, p, epsilon_e, epsilon_B, E0 = theta


    model = Flux(x, thetaObs, thetaCore, n0, p, epsilon_e, epsilon_B, E0,xi_N,d_L,z)
    
    
    # Check if the model returns finite values
    if not np.all(np.isfinite(model)):
        print(thetaObs, thetaCore, n0, p, epsilon_e, epsilon_B, E0)
        raise ValueError("Model returned non-finite values.")
    model = np.array(model)
    y = np.array(y)
    yerr = np.array(yerr)
    
    # Create a mask for non-NaN values in y
    mask = ~np.isnan(y)
    # Apply the mask to y, yerr, and model
    y = y[mask]
    yerr = yerr[mask]
    model = model[mask]
    
    sigma2 = yerr**2 + model**2
    #print(-0.5 * np.sum((y - model) ** 2 / sigma2 ))
    #print(thetaObs, thetaCore, n0, p, epsilon_e, epsilon_B, E0)
    #print(" ")
    return -0.5 * np.sum((y - model) ** 2 / sigma2 )

def log_probability(theta,x,y,yerr,xi_N,d_L,z):
    # Function for log-probability calculation
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr,xi_N,d_L,z)

def log_prior(theta):
    # Function for log-prior calculation
    thetaObs, thetaCore, n0, p, epsilon_e, epsilon_B, E0 = theta
    if (45 < E0 < 57
        and 0.0 < thetaObs < 0.8
        and 0.01 < thetaCore < np.pi*0.5
        and -10.0 < n0 < 10.0
        and 2.1 < p < 5.0
        and -5.0 < epsilon_e < 0.0
        and -5.0 < epsilon_B < 0.0):
        return 0.0
    return -np.inf

def run_parallel_optimization(x, y, initial, yerr, xi_N=1.0, d_L=1.0e26, z=0.01):
    # Bounds for parameters
    bounds = [(0.0, 0.8), (0.01, np.pi*0.5), (-10.0, 10.0), 
              (2.1, 5.0), (-5.0, 0.0), (-5.0, 0.0), (45.0, 57.0)]
    S_bounds = [(0.0, 0.1), (0.01, 0.2), (-2, 2), 
              (2.1, 5.0), (-3.0, -1), (-5.0, -3), (45.0, 57.0)]
    # Wrapper for likelihood function with fixed xi_N, d_L, z
    likelihood_fixed = partial(log_likelihood, xi_N=xi_N, d_L=d_L, z=z)
    nll = lambda *args: -likelihood_fixed(*args)

    # Generate multiple initial starting points within bounds
    np.random.seed(0)  # For reproducibility
    initial_points = [
        [np.random.uniform(low, high) for (low, high) in S_bounds] 
        for _ in range(1)
    ]

    # Function to perform optimization for a single initial point
    def optimize_from_starting_point(starting_point):
        soln = minimize(nll, starting_point, args=(x, y, yerr), bounds=bounds, method='Nelder-mead')
        return soln
    start = time.time()
    # Run optimization in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(optimize_from_starting_point, init) for init in initial_points]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    end = time.time()
    serial_time = end - start
    print("Most Probable parameters identified in {0:.1f} seconds".format(serial_time))
    # Select the solution with the lowest function value (lowest residual)
    best_solution = min(results, key=lambda sol: sol.fun)

    # Extract the best parameters
    thetaObs_ml, thetaCore_ml, n0_ml, p_ml, epsilon_e_ml, epsilon_B_ml, E0_ml = best_solution.x

    print("Best parameters identified:")
    print(f"E0 = {10**E0_ml:.3e}")
    print(f"thetaObs = {thetaObs_ml:.3f}")
    print(f"thetaCore = {thetaCore_ml:.3f}")
    print(f"n0 = {10**n0_ml:.10f}")
    print(f"p = {p_ml:.3f}")
    print(f"epsilon_e = {10**epsilon_e_ml:.5f}")
    print(f"epsilon_B = {10**epsilon_B_ml:.5f}")
    print(f"Residual (negative log-likelihood) = {best_solution.fun:.5f}")
    
    return best_solution


def run_optimization(x, y, initial, yerr,xi_N=1.0,d_L=1.0e26,z=0.01):
    # Perform the parameter optimization
    bounds = [(0.0, 0.8), (0.01, np.pi*0.5), (-10.0, 10.0), 
        (2.1, 5.0), (-5.0, 0.0), (-5.0, 0.0), (45.0, 57.0)]
    # Use the wrapper function to fix datatype for the residual calculation
    likelihood_fixed = partial(log_likelihood,xi_N=xi_N,d_L=d_L,z=z)
    nll = lambda *args: -likelihood_fixed(*args)
    start = time.time()
    soln = minimize(nll, initial, args=(x, y, yerr), bounds=bounds, method='Nelder-mead')
    end = time.time()
    serial_time = end - start
    print("Most Probable parameters identified in {0:.1f} seconds".format(serial_time))
    thetaObs_ml, thetaCore_ml, n0_ml, p_ml, epsilon_e_ml, epsilon_B_ml, E0_ml = soln.x
    print("E0 = {0:.3e}".format(10**E0_ml))
    print("thetaObs = {0:.3f}".format(thetaObs_ml))
    print("thetaCore = {0:.3f}".format(thetaCore_ml))
    print("n0 = {0:.10f}".format(10**n0_ml))
    print("p = {0:.3f}".format(p_ml))
    print("epsilon_e = {0:.5f}".format(10**epsilon_e_ml))
    print("epsilon_B = {0:.5f}".format(10**epsilon_B_ml))
    return soln

def run_sampling(x, y, initial, yerr, genfile=0,xi_N=1.0,d_L=1.0e26,z=0.01, steps=100, nwalkers=32, processes=4, filename="./data/test_sample.h5"):
    ndim = len(initial)
    
    # Use the wrapper function to fix datatype for the MCMC sampling
    log_prob_fixed = partial(log_probability,xi_N=xi_N,d_L=d_L,z=z)
    total_cores = multiprocessing.cpu_count()
    
    if genfile == 1:
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
            soln = run_optimization(x, y, initial,yerr,xi_N,d_L,z)
            pos = soln.x + 1e-4 * np.random.randn(nwalkers, ndim)
            backend.reset(nwalkers, ndim)
    else:
        print("Finding optimal starting parameters...")
        soln = run_optimization(x, y, initial,yerr,xi_N,d_L,z)
        pos = soln.x + 1e-4 * np.random.randn(nwalkers, ndim)
    print("Utilizing {0:.1f}% of avaliable processes".format(100*processes/total_cores))
    print("Beginning sampling...")
    # Run the MCMC sampling with a multiprocessing pool
    with Pool(processes=processes) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_fixed, args=(x, y, yerr), pool=pool, backend=backend if genfile == 1 else None)
        sampler.run_mcmc(pos, steps, progress=True)
    return sampler
    
initial = np.array([0.0,0.1,0.0,2.5,-2,-4,51])

nu = np.geomspace(1e0, 1e20, num=6)
t = np.geomspace(1e-1, 1e3, num=30)

x = [t,nu]
F = np.array(Flux(x,0.1,0.1,0,2.3,-2,-4,53,1,1e26,0.01))
noise_level = abs(0.05*F)
F_noise = F + np.random.normal(0, noise_level, size=F.shape)
yerr = abs(0.05*F)
truth = [0.1,0.1,0.0,2.3,-2,-4,53]
"""
T, Nu = np.meshgrid(np.log10(t), np.log10(nu), indexing='ij')
print(F)
fig, ax = plt.subplots(1, 1)
ax.grid()

# Use pcolormesh for grid representation
c = ax.pcolormesh(T, Nu, F_noise, shading='gouraud', cmap='viridis')  # Adjust shading as needed

# Add color bar to indicate flux values
cbar = plt.colorbar(c, ax=ax)
cbar.set_label('Flux', labelpad=20)

ax.set_title('Spectral Time Flux')

# Set axes label
ax.set_xlabel('Time (t)', labelpad=20)
ax.set_ylabel('Frequency (nu)', labelpad=20)

# Save the figure
fig.savefig('./graph/3D.png')
plt.close(fig)
"""
#if __name__ == "__main__":
    #run_parallel_optimization(x,F_noise,initial,yerr)

if __name__ == "__main__":
    run_sampling(x,F,initial,yerr,steps=2000,processes=10,genfile=1,filename='./data/combo2.h5')


    file_path = './data/combo2.h5'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The HDF5 file '{file_path}' does not exist.")
    backend = emcee.backends.HDFBackend(file_path)
    if backend.iteration == 0:
        raise ValueError("The HDF5 file is empty. No data was stored.")
    reader = emcee.backends.HDFBackend(file_path)


    try:
        tau = reader.get_autocorr_time()
        burnin = int(2 * np.max(tau))
        thin = int(0.5 * np.min(tau))
    except emcee.autocorr.AutocorrError:
        print("Warning: Autocorrelation time estimation failed. Proceeding with current chain length.")
        burnin = len(reader.get_chain()) // 3  # Use some default burn-in, e.g., one-third of the chain length
        thin = 1  # No thinning
    labels = ["thetaObs", "thetaCore", "n0", "p", "epsilon_e", "epsilon_B", "E0"]
    # Plot the sampling results
    samples = reader.get_chain()
    fig, axes = plt.subplots(len(labels), figsize=(10, 7), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")
    fig.savefig('./graph/combo_steps2.png')
    plt.close(fig)
    flat_samples = reader.get_chain(discard=burnin,thin=thin,flat=True)#
    fig2 = corner.corner(flat_samples, labels=labels, truths=truth)
    fig2.savefig('./graph/combo_param2.png')
    plt.close(fig2)
