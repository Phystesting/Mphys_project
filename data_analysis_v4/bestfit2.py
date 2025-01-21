import numpy as np
import afterglowpy as grb
import emcee
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import sampler_v12 as splr
import os
import h5py
import matplotlib.cm as cm

identifier = 'GRB1'
fit_types = ['GA', 'TH']  # 'GA' for Gaussian, 'TH' for Top-Hat
extra_datasets = ['control','thinning','jetbreak_post', 'jetbreak_pre']  # Different datasets (e.g., pre-jet break, post-jet break)

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

# Specify manual overrides for any parameters you want to change
manual_overrides = {}

# Set jet type based on fit type
jet_types = {
    'GA': grb.jet.Gaussian,
    'TH': grb.jet.TopHat
}

xi_N = 1.0
z = 0.0099
d_L = 1.327e+26

# Generate a colormap for the datasets
cmap = cm.get_cmap("tab10", len(extra_datasets))  # Generate colors based on number of datasets
dataset_colors = cmap(np.linspace(0, 1, len(extra_datasets)))

# Prepare the figure
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Import control dataset for plotting observational data
control_file = f'../data_generation_v2/data/{identifier}_control.csv'
time, freq, flux, UB_err, LB_err = np.genfromtxt(control_file, delimiter=',', skip_header=1, unpack=True)
flux_err = LB_err, UB_err

# Find unique frequencies in control dataset
unique_freqs = np.unique(freq)

# Plot control data points
for idx, nu_value in enumerate(unique_freqs):
    mask = freq == nu_value
    t_filtered = time[mask]
    flux_filtered = flux[mask]
    flux_err_filtered = LB_err[mask], UB_err[mask]

    ax.errorbar(
        t_filtered, flux_filtered, yerr=flux_err_filtered,
        fmt='.', color='black', label=f'Data - {nu_value:.2e} Hz' if idx == 0 else ""
    )

# Loop over fit types (Gaussian and Top-Hat)
for fit_type in fit_types:
    jet_type = jet_types[fit_type]
    
    # Loop over datasets (post-jet break and pre-jet break)
    for dataset_idx, dataset in enumerate(extra_datasets):
        color = dataset_colors[dataset_idx]  # Automatically assign a color
        
        # Construct HDF5 file path for this combination of fit type and dataset
        file_path = f'/data/PROJECTS/2024-25/cjc233/samples_v10/{identifier}_{fit_type}{dataset}_samples.h5'
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The HDF5 file '{file_path}' does not exist.")
        
        # Load the backend
        backend = emcee.backends.HDFBackend(file_path)
        if backend.iteration == 0:
            raise ValueError(f"The HDF5 file '{file_path}' is empty. No data was stored.")

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

        # Process each parameter
        for i, label in enumerate(labels):
            if label in ["n0", "epsilon_e", "epsilon_B", "E0"]:  # Log-scaled parameters
                mcmc[i] = 10 ** np.percentile(flat_samples[:, i], [16, 50, 84])
            else:  # Linear-scaled parameters
                mcmc[i] = np.percentile(flat_samples[:, i], [16, 50, 84])
            q[i] = abs(mcmc[i][1] - mcmc[i][2]), abs(mcmc[i][1] - mcmc[i][0])

        # Dynamically allocate parameters, using best-fit values, manual overrides, or defaults
        parameter_values = {}
        for param, default_value in default_params.items():
            if param in manual_overrides:  # Manual override has the highest priority
                parameter_values[param] = manual_overrides[param]
            elif param in labels:  # Use best-fit value from MCMC results if available
                idx = labels.index(param)
                parameter_values[param] = mcmc[idx, 1]
            else:  # Use default value
                parameter_values[param] = default_value

        # Uniform grid for model
        t_uniform = []
        freq_uniform = []
        for nu_idx, nu_value in enumerate(unique_freqs):
            samples = 50
            for i in range(samples + 1):
                t_uniform.append(np.exp(np.log(min(time)) + i * (np.log(max(time)) - np.log(min(time))) / samples))
                freq_uniform.append(nu_value)
        t_uniform = np.array(t_uniform)
        freq_uniform = np.array(freq_uniform)

        # Prepare input for Flux calculation
        x_uniform = [t_uniform, freq_uniform]

        # Calculate the best-fit model
        best = np.array(splr.Flux(x_uniform, **parameter_values, xi_N=xi_N, d_L=d_L, z=z, jet_type=jet_type))

        # Plot best-fit model for each frequency
        for idx, nu_value in enumerate(unique_freqs):
            mask = freq_uniform == nu_value
            t_best_filtered = t_uniform[mask]
            ax.plot(
                t_best_filtered, best[mask], '--' if fit_type == 'TH' else '-',
                color=color, label=f'{fit_type} Fit - {dataset}' if idx == 0 else ""
            )

# Finalize plot
ax.set(
    xscale="log", yscale="log",
    xlabel=r"$t$ (s)", ylabel=r"$F_\nu$ (mJy)",
    title="Fits for Gaussian and Top-Hat Models (Data: Control)"
)
ax.legend(loc="lower left")

probable_plots_path = f'./graph/{identifier}/{identifier}_combined_fits.png'
fig.savefig(probable_plots_path)
plt.close(fig)
