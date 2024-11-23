import sampler_v3 as splr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

# Define your fixed parameters separately
fixed_params = {}

# Define your initial guesses for the fitting parameters
initial = {
    "thetaCore": 0.1,
    "p": 2.33,
    "log_epsilon_e": -1.0,
    "log_epsilon_B": -3.0,
    "log_E0": 53.0,
    "log_n0": 3.0,
    "thetaObs": 0.4,
}

# Define inputs
z = 0.1
d_L = 1.43e+27
xi_N = 1.0
nwalkers = 32
processes = 40
steps = 600000
filename = '/data/PROJECTS/2024-25/cjc233/Large_data/170817_samples.h5'

# Unpack data
time_values, freq_values, flux_values, UB_err, LB_err = np.genfromtxt(
    '../data_generation_v1/data/GRB1_control_data.csv', delimiter=',', skip_header=1, unpack=True
)
flux_err = LB_err, UB_err

# Run optimization to get the best-fit parameters
result = splr.run_optimization(
    [time_values, freq_values], flux_values, initial, fixed_params, flux_err, xi_N, d_L, z
)

# Extract best-fit parameters from the result
best_fit_params = result.x
param_names = list(initial.keys())
best_fit_dict = dict(zip(param_names, best_fit_params))

# Unique frequencies for plotting
unique_freqs = np.unique(freq_values)

# Prepare the colormap
colormap = get_cmap("viridis")

# Create the plot
plt.figure(figsize=(10, 6))

for idx, nu_value in enumerate(unique_freqs):
    # Filter data for this frequency
    time_filtered = np.array([time_values[i] for i in range(len(time_values)) if freq_values[i] == nu_value])
    flux_filtered = np.array([flux_values[i] for i in range(len(flux_values)) if freq_values[i] == nu_value])
    UB_filtered = np.array([UB_err[i] for i in range(len(UB_err)) if freq_values[i] == nu_value])
    LB_filtered = np.array([LB_err[i] for i in range(len(LB_err)) if freq_values[i] == nu_value])

    # Check for matching lengths
    if len(LB_filtered) == len(flux_filtered) and len(UB_filtered) == len(flux_filtered):
        # Assign a unique color
        color = colormap(idx / len(unique_freqs))

        # Plot the data with error bars
        plt.errorbar(
            time_filtered,
            flux_filtered,
            yerr=(LB_filtered, UB_filtered),
            fmt=".",
            label=f"Frequency: {nu_value:.3e} Hz",
            color=color,
        )

        # Calculate the best-fit model flux
        model_flux = splr.Flux(
            [time_filtered, nu_value * np.ones_like(time_filtered)],
            **best_fit_dict,
            xi_N=xi_N,
            d_L=d_L,
            z=z,
        )

        # Plot the best-fit curve
        plt.plot(
            time_filtered,
            model_flux,
            label=f"Best fit (Freq: {nu_value:.3e} Hz)",
            linestyle=".",
            color=color,
        )
    else:
        print(f"Error: Data length mismatch for frequency {nu_value}")

# Add labels and title
plt.xlabel("Time (s)")
plt.ylabel("Flux (mJy)")
plt.title("Flux vs Time with Best-Fit Model")

# Set logarithmic axes
plt.xscale("log")
plt.yscale("log")

# Show legend
plt.legend()

# Show grid
plt.grid(True)

# Save the plot
plt.savefig("./graph/test_figure_best_fit.png")
# Display the plot
plt.show()

