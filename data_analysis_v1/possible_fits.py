import numpy as np
import afterglowpy as grb
import emcee
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import curve_fit
import sampler_v4 as splr
import os

identifier = 'GRB2_GA'
# 0 [thetaCore, n0, p, epsilon_e, epsilon_B, E0]
# 1 [thetaCore, p, epsilon_e, epsilon_B, E0, n0]
labels_layout = 1
# Import samples
file_path = f'/data/PROJECTS/2024-25/cjc233/Large_data/{identifier}_samples.h5' 
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The HDF5 file '{file_path}' does not exist.")
backend = emcee.backends.HDFBackend(file_path)
if backend.iteration == 0:
    raise ValueError("The HDF5 file is empty. No data was stored.")
sampler = emcee.backends.HDFBackend(file_path)
# Define labels
if labels_layout == 0:
    base_labels = ["thetaCore", "n0", "p", "epsilon_e", "epsilon_B", "E0"]
else:
    base_labels = ["thetaCore", "p", "epsilon_e", "epsilon_B", "E0", "n0"]
ndim_base = len(base_labels)

# Check if thetaObs is fitted
if sampler.shape[1] == ndim_base + 1:  # thetaObs is included
    labels = base_labels + ["thetaObs"]
else:  # thetaObs is not included
    labels = base_labels

ndim = len(labels)

# Autocorrelation time
try:
    tau = sampler.get_autocorr_time()
    burnin = int(2 * np.max(tau))
    thin = int(0.5 * np.min(tau))
except emcee.autocorr.AutocorrError:
    print("Warning: Autocorrelation time estimation failed. Proceeding with current chain length.")
    burnin = len(sampler.get_chain()) // 3  # Use some default burn-in, e.g., one-third of the chain length
    thin = 1  # No thinning

flat_samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
mcmc = np.zeros([ndim, 3])
q = np.zeros([ndim, 2])

"""
# 16th, 50th, and 84th percentiles
for i in range(ndim):
    mcmc[i] = np.percentile(flat_samples[:, i], [16, 50, 84])
    q[i] = np.diff(mcmc[i])
"""
output_file = f"./data/{identifier}_results.txt"

# Open the file in append mode
with open(output_file, "w") as f:
    f.write(f"{identifier}\n")
    # Print results
    for i, label in enumerate(labels):
        if label in ["n0", "epsilon_e", "epsilon_B", "E0"]:  # Log-scaled parameters
            mcmc[i] = 10**np.percentile(flat_samples[:, i], [16, 50, 84])
            q[i] = abs(mcmc[i][1] - mcmc[i][2]), abs(mcmc[i][1] - mcmc[i][0])
            print(f"{label}: {mcmc[i][1]:.3e} +{q[i][0]:.3e} -{q[i][1]:.3e}")
            f.write(f"{label}: {mcmc[i][1]:.3e} +{q[i][0]:.3e} -{q[i][1]:.3e}\n")
        else:  # Linear-scaled parameters
            mcmc[i] = np.percentile(flat_samples[:, i], [16, 50, 84])
            q[i] = abs(mcmc[i][1] - mcmc[i][2]), abs(mcmc[i][1] - mcmc[i][0])
            print(f"{label}: {mcmc[i][1]:.3e} +{q[i][0]:.3e} -{q[i][1]:.3e}")
            f.write(f"{label}: {mcmc[i][1]:.3e} +{q[i][0]:.3e} -{q[i][1]:.3e}\n")
f.close()
# Import data
time, freq, flux, UB_err,LB_err = np.genfromtxt(f'../data_generation_v1/data/GRB2_control_data.csv', delimiter=',', skip_header=1, unpack=True)
flux_err = LB_err,UB_err
# Set values
t_uniform = np.geomspace(min(time), max(time), num=50)
jet_type = grb.jet.Gaussian
xi_N = 1.0
d_L = 1.34e+26
z = 0.01

# Get best-fit parameters
if labels_layout == 0:
    thetaCore, n0, p, epsilon_e, epsilon_B, E0 = mcmc[:6, 1]
else:
    thetaCore, p, epsilon_e, epsilon_B, E0, n0 = mcmc[:6, 1]
if "thetaObs" in labels:
    thetaObs = mcmc[6, 1]
else:
    thetaObs = 0.0  # Default value if not fitted

# Prepare the figure
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Find unique frequencies
unique_freqs = np.unique(freq)

# Loop over each unique frequency
for nu_value in unique_freqs:
    # Create a mask to filter the data for this frequency
    mask = freq == nu_value
    t_filtered = time[mask]
    flux_filtered = flux[mask]
    flux_err_filtered = LB_err[mask],UB_err[mask]

    # Plot the observational data for this frequency
    ax.errorbar(
        t_filtered, flux_filtered, yerr=flux_err_filtered, fmt='o', label=f'{nu_value:.2e} Hz'
    )

t_uniform = []
freq_uniform = []
for nu_idx, nu_value in enumerate(unique_freqs):
    samples = 50
    for i in range(samples):
        t_uniform.append(np.exp(np.log(min(time)) + i*(np.log(max(time))-np.log(min(time)))/samples))
        freq_uniform.append(nu_value)
t_uniform = np.array(t_uniform)
freq_uniform = np.array(freq_uniform)

# Prepare the input for Flux calculation
x_uniform = [t_uniform, freq_uniform]
best = np.array(splr.Flux(x_uniform,thetaCore, np.log10(n0), p, np.log10(epsilon_e), np.log10(epsilon_B), np.log10(E0),thetaObs, xi_N, d_L, z,jet_type))


# Plot the best-fit model
for nu_value in unique_freqs:
    # Mask the `best` values corresponding to this frequency
    mask = freq_uniform == nu_value
    t_best_filtered = t_uniform[mask]
    best_filtered = best[mask]

    # Plot the best-fit curve for this frequency
    ax.plot(t_best_filtered, best_filtered, '-', label=f'Model {nu_value:.2e} Hz',zorder=1)

# Draw random sample models for uncertainty visualization
inds = np.random.randint(len(flat_samples), size=100)

for ind in inds:
    sample = flat_samples[ind]
    if labels_layout == 0:
        thetaCore, n0, p, epsilon_e, epsilon_B, E0 = sample[:6]
    else:
        thetaCore, p, epsilon_e, epsilon_B, E0, n0 = sample[:6]
    if "thetaObs" in labels:
        thetaObs = sample[6]
    else:
        thetaObs = 0.0

    # Calculate flux for this sample
    y_sample = splr.Flux(x_uniform,thetaCore, n0, p, epsilon_e, epsilon_B, E0,thetaObs, xi_N, d_L, z,jet_type)
    y_sample = np.array(y_sample)

    # Plot the sampled model for each frequency
    for nu_value in unique_freqs:
        mask = freq_uniform == nu_value
        t_sample_filtered = t_uniform[mask]
        y_sample_filtered = y_sample[mask]

        ax.plot(
            t_sample_filtered, y_sample_filtered, alpha=0.1, color="gray",zorder=0
        )

# Add labels and legend
ax.set(
    xscale="log", yscale="log",
    xlabel=r"$t$ (s)", ylabel=r"$F_\nu$ (mJy)"
)
ax.legend()

# Save the figure
probable_plots_path = f'./graph/{identifier}_probable_plots.png'
fig.savefig(probable_plots_path)
plt.close(fig)

