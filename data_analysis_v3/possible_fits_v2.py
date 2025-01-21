import numpy as np
import afterglowpy as grb
import emcee
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import curve_fit
import sampler_v6 as splr
import os
import h5py  # Import h5py for file access
import matplotlib.cm as cm
import matplotlib.colors as mcolors

identifier = '170817'
fit_type = 'GA'
extra=''

# Define default values for all parameters (used if not included in the fit)
default_params = {
    "thetaCore": 0.05,  # Example value (radians)
    "log_n0": 0,        # cm^-3
    "p": 2.33,          # Spectral index
    "log_epsilon_e": -1.0,  # Fraction of energy in electrons
    "log_epsilon_B": -4.0, # Fraction of energy in magnetic fields
    "log_E0": 52.,       # Isotropic equivalent energy (erg)
    "thetaObs": 0.0,   # Observation angle (radians)
}
if fit_type == 'GA':
    jet_type = grb.jet.Gaussian
else:
    jet_type = grb.jet.TopHat
xi_N = 1.0
z = 0.0099
d_L = 1.327e+26

# Path to HDF5 file
file_path = f'/data/PROJECTS/2024-25/cjc233/samples_v4/{identifier}_{fit_type}_samples.h5'
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
time, freq, flux, UB_err,LB_err = np.genfromtxt(f'../data_generation_v2/data/{identifier}_data.csv', delimiter=',', skip_header=1, unpack=True)
#UB_err = flux_err
#LB_err = flux_err
flux_err = LB_err, UB_err

t_uniform = np.geomspace(min(time), max(time), num=50)

# Prepare the figure
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Find unique frequencies
unique_freqs = np.unique(freq)

# Generate a colormap for unique frequencies
cmap = cm.get_cmap("viridis", len(unique_freqs))
colors = cmap(np.linspace(0, 1, len(unique_freqs)))

# Plot data points and fit lines for each frequency
for idx, nu_value in enumerate(unique_freqs):
    # Filter data for the current frequency
    mask = freq == nu_value
    t_filtered = time[mask]
    flux_filtered = flux[mask]
    flux_err_filtered = LB_err[mask], UB_err[mask]

    # Plot observational data for this frequency
    ax.errorbar(
        t_filtered, flux_filtered, yerr=flux_err_filtered,
        fmt='.', color=colors[idx], label=f'{nu_value:.2e} Hz'
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

# Plot the best-fit model for each frequency
for idx, nu_value in enumerate(unique_freqs):
    mask = freq_uniform == nu_value
    t_best_filtered = t_uniform[mask]
    best_filtered = best[mask]

    ax.plot(
        t_best_filtered, best_filtered, '-', color=colors[idx]
    )

# Draw random sample models for uncertainty visualization (in gray)

num_samples = 10  # Compute the number of samples to use
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

    # Plot the sampled model for each frequency (in gray)
    for idx, nu_value in enumerate(unique_freqs):
        mask = freq_uniform == nu_value
        t_sample_filtered = t_uniform[mask]
        y_sample_filtered = y_sample[mask]

        ax.plot(
            t_sample_filtered, y_sample_filtered, alpha=0.1,
            color='gray', zorder=0
        )

# Create a custom legend
legend_labels = [
    "Points: Observational Data",
    "Lines: Best-Fit Model"
]
legend_colors = [
    "black",  # General color for data points
    "black"   # General color for fit lines
]

# Add individual frequency colors to legend
for idx, nu_value in enumerate(unique_freqs):
    legend_labels.append(f"{nu_value:.2e} Hz")
    legend_colors.append(colors[idx])

# Create proxy artists for the legend
legend_handles = []
# Add handles for general data and fit descriptions
legend_handles.append(plt.Line2D([0], [0], color="black", marker='o', linestyle=''))
legend_handles.append(plt.Line2D([0], [0], color="black", linestyle='-'))

# Add handles for frequencies
for color in colors:
    legend_handles.append(plt.Line2D([0], [0], color=color, linestyle='-'))

# Add the legend to the plot
ax.legend(
    legend_handles, legend_labels,
    title="Legend",
    loc="upper left", fontsize=10, title_fontsize=12
)

# Add labels and save the figure
ax.set(
    xscale="log", yscale="log",
    xlabel=r"$t$ (s)", ylabel=r"$F_\nu$ (mJy)"
)

probable_plots_path = f'./graph/{identifier}/{identifier}_{fit_type}_fits.png'
fig.savefig(probable_plots_path)
plt.close(fig)
