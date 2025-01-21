import numpy as np
import afterglowpy as grb
import emcee
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import curve_fit
import sampler_v13 as splr
import os
import h5py
import matplotlib.cm as cm
import matplotlib.lines as mlines

identifier = '170817'
extra = '2'
extra2 = 'reduced3'
# Define default values for all parameters (used if not included in the fit)
default_params = {
    "thetaCore": 0.05,  # Example value (radians)
    "log_n0": 0,        # cm^-3
    "p": 2.33,          # Spectral index
    "log_epsilon_e": -1.0,  # Fraction of energy in electrons
    "log_epsilon_B": -4.0,  # Fraction of energy in magnetic fields
    "log_E0": 52.,       # Isotropic equivalent energy (erg)
    "thetaObs": 0.0,     # Observation angle (radians)
}

# Specify manual overrides for any parameters you want to change
manual_overrides = {}

xi_N = 1.0
z = 0.0099
d_L = 1.327e+26

def largest_decimal_places(num1, num2):
    def count_decimals(n):
        # Convert to string, strip unnecessary zeros, and count decimals
        s = f"{n:.1g}".rstrip('0').rstrip('.')
        if '.' in s:
            return len(s.split('.')[1])
        return 0
    
    return max(count_decimals(num1), count_decimals(num2))

# Path to HDF5 files for TopHat and Gaussian jet models
file_path_tophat = f'/data/PROJECTS/2024-25/cjc233/samples_v10/{identifier}_GA{extra2}_samples.h5'
file_path_gaussian = f'/data/PROJECTS/2024-25/cjc233/samples_v10/{identifier}_GA{extra}_samples.h5'

# Check if both HDF5 files exist
if not os.path.exists(file_path_tophat) or not os.path.exists(file_path_gaussian):
    raise FileNotFoundError(f"One or both of the HDF5 files '{file_path_tophat}' or '{file_path_gaussian}' do not exist.")

# Load backend and check for data for TopHat jet
backend_tophat = emcee.backends.HDFBackend(file_path_tophat)
if backend_tophat.iteration == 0:
    raise ValueError(f"The HDF5 file '{file_path_tophat}' is empty. No data was stored.")

# Load backend and check for data for Gaussian jet
backend_gaussian = emcee.backends.HDFBackend(file_path_gaussian)
if backend_gaussian.iteration == 0:
    raise ValueError(f"The HDF5 file '{file_path_gaussian}' is empty. No data was stored.")

# Read parameter labels from HDF5 for TopHat jet
with h5py.File(file_path_tophat, "r") as f:
    labels_tophat = list(f.attrs["param_names"])  # Read labels as a list

# Read parameter labels from HDF5 for Gaussian jet
with h5py.File(file_path_gaussian, "r") as f:
    labels_gaussian = list(f.attrs["param_names"])

ndim_tophat = len(labels_tophat)
ndim_gaussian = len(labels_gaussian)

# Autocorrelation time and chain processing for TopHat jet
try:
    tau_tophat = backend_tophat.get_autocorr_time()
    burnin_tophat = int(2 * np.max(tau_tophat))
    thin_tophat = int(0.5 * np.min(tau_tophat))
except emcee.autocorr.AutocorrError:
    print("Warning: Autocorrelation time estimation failed for TopHat. Proceeding with default burn-in and thinning.")
    burnin_tophat = len(backend_tophat.get_chain()) // 3
    thin_tophat = 1

# Autocorrelation time and chain processing for Gaussian jet
try:
    tau_gaussian = backend_gaussian.get_autocorr_time()
    burnin_gaussian = int(2 * np.max(tau_gaussian))
    thin_gaussian = int(0.5 * np.min(tau_gaussian))
except emcee.autocorr.AutocorrError:
    print("Warning: Autocorrelation time estimation failed for Gaussian. Proceeding with default burn-in and thinning.")
    burnin_gaussian = len(backend_gaussian.get_chain()) // 3
    thin_gaussian = 1

# Extract flat samples for both jet types
flat_samples_tophat = backend_tophat.get_chain(discard=burnin_tophat, thin=thin_tophat, flat=True)
flat_samples_gaussian = backend_gaussian.get_chain(discard=burnin_gaussian, thin=thin_gaussian, flat=True)

# Process the parameters for both jet models
mcmc_tophat = np.zeros([ndim_tophat, 3])
mcmc_gaussian = np.zeros([ndim_gaussian, 3])

output_file_TH = f"./data/{identifier}/{identifier}_GA{extra2}_results.txt"

# Process each parameter
with open(output_file_TH, "w") as f:
    # Process TopHat samples
    for i, label in enumerate(labels_tophat):
        mcmc_tophat[i] = np.percentile(flat_samples_tophat[:, i], [16, 50, 84])
        plus_error = mcmc_tophat[i][2] - mcmc_tophat[i][1]
        minus_error = mcmc_tophat[i][1] - mcmc_tophat[i][0]
        
        # Round errors to 1 significant figure
        rounded_plus_error = round(plus_error, -int(np.floor(np.log10(abs(plus_error)))))
        rounded_minus_error = round(minus_error, -int(np.floor(np.log10(abs(minus_error)))))
        
        dp = largest_decimal_places(plus_error, minus_error)
        
        # Check if rounded errors are the same
        if rounded_plus_error == rounded_minus_error:
            f.write(f"& ${mcmc_tophat[i][1]:.{dp}f} \\pm {rounded_plus_error:.1g}$")
        else:
            f.write(f"& ${mcmc_tophat[i][1]:.{dp}f} ^{{+{rounded_plus_error:.1g}}} _{{-{rounded_minus_error:.1g}}}$")


f.close()

output_file_GA = f"./data/{identifier}/{identifier}_GA{extra}_results.txt"

# Process each parameter
with open(output_file_GA, "w") as f:
    # Process TopHat samples
    for i, label in enumerate(labels_tophat):
        mcmc_gaussian[i] = np.percentile(flat_samples_gaussian[:, i], [16, 50, 84])
        plus_error = mcmc_gaussian[i][2] - mcmc_gaussian[i][1]
        minus_error = mcmc_gaussian[i][1] - mcmc_gaussian[i][0]
        
        # Round errors to 1 significant figure
        rounded_plus_error = round(plus_error, -int(np.floor(np.log10(abs(plus_error)))))
        rounded_minus_error = round(minus_error, -int(np.floor(np.log10(abs(minus_error)))))
        
        dp = largest_decimal_places(plus_error, minus_error)
        
        # Check if rounded errors are the same
        if rounded_plus_error == rounded_minus_error:
            f.write(f"& ${mcmc_gaussian[i][1]:.{dp}f} \\pm {rounded_plus_error:.1g}$")
        else:
            f.write(f"& ${mcmc_gaussian[i][1]:.{dp}f} ^{{+{rounded_plus_error:.1g}}} _{{-{rounded_minus_error:.1g}}}$")


f.close()
# Dynamically allocate parameters, using best-fit values or defaults
parameter_values_tophat = {param: mcmc_tophat[labels_tophat.index(param), 1] if param in labels_tophat else value for param, value in default_params.items()}
parameter_values_gaussian = {param: mcmc_gaussian[labels_gaussian.index(param), 1] if param in labels_gaussian else value for param, value in default_params.items()}
#UB_err, LB_err
# Import observational data
time, freq, flux, UB_err, LB_err = np.genfromtxt(
    f'../data_generation_v2/data/{identifier}_data.csv', delimiter=',', skip_header=1, unpack=True
)
flux_err = LB_err, UB_err

# Assuming previous code for loading data, model parameters, and flux calculation is already in place.

# Prepare the figure
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Find unique frequencies
unique_freqs = np.unique(freq)

# Generate a colormap for unique frequencies
cmap = cm.get_cmap("viridis", len(unique_freqs))
colors = cmap(np.linspace(0, 1, len(unique_freqs)))

# Uniform grid for model
t_uniform, freq_uniform = np.meshgrid(
    np.logspace(np.log10(min(time)), np.log10(max(time)), 50),
    unique_freqs,
    indexing="ij"
)
t_uniform, freq_uniform = t_uniform.flatten(), freq_uniform.flatten()
x_uniform = [t_uniform, freq_uniform]

# Calculate flux for both models
best_tophat = np.array(splr.Flux(x_uniform, **parameter_values_tophat, xi_N=xi_N, d_L=d_L, z=z, jet_type=grb.jet.Gaussian))
best_gaussian = np.array(splr.Flux(x_uniform, **parameter_values_gaussian, xi_N=xi_N, d_L=d_L, z=z, jet_type=grb.jet.Gaussian))

# Plot data points and fit lines for each frequency
for idx, nu_value in enumerate(unique_freqs):
    mask = freq == nu_value
    ax.errorbar(
        time[mask], flux[mask], yerr=(LB_err[mask], UB_err[mask]),
        fmt='.', color=colors[idx]
    )

    mask_fit = freq_uniform == nu_value
    ax.plot(t_uniform[mask_fit], best_tophat[mask_fit], '--', color=colors[idx])
    ax.plot(t_uniform[mask_fit], best_gaussian[mask_fit], '-', color=colors[idx])

# Create dummy handles for the frequencies in the legend
legend_handles = [
    mlines.Line2D([0], [0], color=colors[idx], lw=0, marker='o', markersize=10, label=f'{unique_freqs[idx]:.2e} Hz')
    for idx in range(len(unique_freqs))
]

# Add a reduced legend with frequency-color mapping
handles = [
    plt.Line2D([0], [0], linestyle='none', marker='.', color='black', label='Data Points'),
    plt.Line2D([0], [0], linestyle='--', color='black', label='Removed Lx'),
    plt.Line2D([0], [0], linestyle='-', color='black', label='With Lx'),
] + legend_handles  # Append frequency handles

ax.legend(handles=handles, loc="lower left", fontsize=8, title="Legend")
ax.set(
    xscale="log", yscale="log",
    xlabel=r"$t$ (s)", ylabel=r"$F_\nu$ (mJy)"
)

# Save the plot
probable_plots_path = f'./graph/{identifier}/{identifier}_{extra}_fits.png'
fig.savefig(probable_plots_path)
plt.close(fig)
