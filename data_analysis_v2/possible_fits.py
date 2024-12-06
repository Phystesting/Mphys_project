import numpy as np
import afterglowpy as grb
import emcee
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import curve_fit
import sampler_v4 as splr
import os
import h5py  # Import h5py for file access
import matplotlib.cm as cm
import matplotlib.colors as mcolors

identifier = '170817'
fit_type = 'GA'

# Define default values for all parameters (used if not included in the fit)
default_params = {
    "thetaCore": 0.05,  # Example value (radians)
    "n0": 0,        # cm^-3
    "p": 2.33,          # Spectral index
    "epsilon_e": -1.0,  # Fraction of energy in electrons
    "epsilon_B": -4.0, # Fraction of energy in magnetic fields
    "E0": 52.,       # Isotropic equivalent energy (erg)
    "thetaObs": 0.0,   # Observation angle (radians)
}
jet_type = grb.jet.Gaussian
xi_N = 1.0
d_L = 1.327e+26
z = 0.099

# Path to HDF5 file
file_path = f'/data/PROJECTS/2024-25/cjc233/samples/{identifier}_{fit_type}_samples.h5'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The HDF5 file '{file_path}' does not exist.")

# Load backend and check for data
backend = emcee.backends.HDFBackend(file_path)
if backend.iteration == 0:
    raise ValueError("The HDF5 file is empty. No data was stored.")

# Read parameter labels from HDF5
with h5py.File(file_path, "r") as f:
    labels = list(f.attrs["param_names"])  # Read labels as a list

ndim = len(labels)

# Autocorrelation time and chain processing
try:
    tau = backend.get_autocorr_time()
    burnin = int(2 * np.max(tau))
    thin = int(0.5 * np.min(tau))
except emcee.autocorr.AutocorrError:
    print("Warning: Autocorrelation time estimation failed. Proceeding with default burn-in and thinning.")
    burnin = len(backend.get_chain()) // 3
    thin = 1

flat_samples = backend.get_chain(discard=burnin, thin=thin, flat=True)
mcmc = np.zeros([ndim, 3])
q = np.zeros([ndim, 2])

# Output file
output_file = f"./data/{identifier}/{fit_type}_results.txt"

# Process each parameter
with open(output_file, "w") as f:
    f.write(f"{identifier}\n")
    for i, label in enumerate(labels):
        if label in ["n0", "epsilon_e", "epsilon_B", "E0"]:  # Log-scaled parameters
            mcmc[i] = 10 ** np.percentile(flat_samples[:, i], [16, 50, 84])
        else:  # Linear-scaled parameters
            mcmc[i] = np.percentile(flat_samples[:, i], [16, 50, 84])
        q[i] = abs(mcmc[i][1] - mcmc[i][2]), abs(mcmc[i][1] - mcmc[i][0])
        print(f"{label}: {mcmc[i][1]:.3e} +{q[i][0]:.3e} -{q[i][1]:.3e}")
        f.write(f"{label}: {mcmc[i][1]:.3e} +{q[i][0]:.3e} -{q[i][1]:.3e}\n")

# Dynamically allocate parameters, using default values if not fitted
parameter_values = {}
for param, default_value in default_params.items():
    if param in labels:
        idx = labels.index(param)
        parameter_values[param] = mcmc[idx, 1]
    else:
        parameter_values[param] = default_value

# Import observational data
time, freq, flux, UB_err, LB_err = np.genfromtxt(f'../data_generation_v2/data/{identifier}_data.csv', delimiter=',', skip_header=1, unpack=True)
flux_err = LB_err, UB_err

t_uniform = np.geomspace(min(time), max(time), num=50)


# Prepare the figure
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Generate a colormap for unique frequencies
cmap = cm.get_cmap("viridis", len(unique_freqs))
colors = cmap(np.linspace(0, 1, len(unique_freqs)))

# Loop over each unique frequency
for idx, nu_value in enumerate(unique_freqs):
    mask = freq == nu_value
    t_filtered = time[mask]
    flux_filtered = flux[mask]
    flux_err_filtered = LB_err[mask], UB_err[mask]

    # Plot the observational data for this frequency
    ax.errorbar(
        t_filtered, flux_filtered, yerr=flux_err_filtered,
        fmt='.', color=colors[idx], label=f'Data: {nu_value:.2e} Hz'
    )

# Uniform grid for model
t_uniform = []
freq_uniform = []
for nu_idx, nu_value in enumerate(unique_freqs):
    samples = 50
    for i in range(samples):
        t_uniform.append(np.exp(np.log(min(time)) + i * (np.log(max(time)) - np.log(min(time))) / samples))
        freq_uniform.append(nu_value)
t_uniform = np.array(t_uniform)
freq_uniform = np.array(freq_uniform)

# Prepare the input for Flux calculation
x_uniform = [t_uniform, freq_uniform]
best = np.array(splr.Flux(x_uniform, **parameter_values, xi_N=xi_N, d_L=d_L, z=z, jet_type=jet_type))

# Plot the best-fit model
for idx, nu_value in enumerate(unique_freqs):
    mask = freq_uniform == nu_value
    t_best_filtered = t_uniform[mask]
    best_filtered = best[mask]

    ax.plot(
        t_best_filtered, best_filtered, '-', color=colors[idx],
        label=f'Fit: {nu_value:.2e} Hz', zorder=1
    )

# Draw random sample models for uncertainty visualization
fraction = 0.01  # Fraction of samples to use (e.g., 10%)
num_samples = int(len(flat_samples) * fraction)  # Compute the number of samples to use
inds = np.random.choice(len(flat_samples), size=num_samples, replace=False)  # Randomly select indices

for ind in inds:
    sample = flat_samples[ind]
    sample_values = {}
    for param, default_value in default_params.items():
        if param in labels:
            sample_values[param] = sample[labels.index(param)]
        else:
            sample_values[param] = default_value

    # Calculate flux for this sample
    y_sample = splr.Flux(x_uniform, **sample_values, xi_N=xi_N, d_L=d_L, z=z, jet_type=jet_type)
    y_sample = np.array(y_sample)

    # Plot the sampled model for each frequency
    for idx, nu_value in enumerate(unique_freqs):
        mask = freq_uniform == nu_value
        t_sample_filtered = t_uniform[mask]
        y_sample_filtered = y_sample[mask]

        ax.plot(
            t_sample_filtered, y_sample_filtered, alpha=0.1,
            color=colors[idx], zorder=0
        )

# Add custom legend entries
handles, labels = ax.get_legend_handles_labels()
data_handles = [h for h, l in zip(handles, labels) if "Data" in l]
fit_handles = [h for h, l in zip(handles, labels) if "Fit" in l]
legend_handles = data_handles + fit_handles

ax.legend(
    legend_handles,
    [f'Data: {l.split(":")[1].strip()}' if "Data" in l else f'Fit: {l.split(":")[1].strip()}' for l in labels],
    title="Legend", loc="upper left", fontsize=10, title_fontsize=12
)

# Add labels and save the figure
ax.set(
    xscale="log", yscale="log",
    xlabel=r"$t$ (s)", ylabel=r"$F_\nu$ (mJy)"
)

probable_plots_path = f'./graph/{identifier}_fits.png'
fig.savefig(probable_plots_path)
plt.close(fig)

