import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import afterglowpy as grb
import matplotlib.cm as cm

# Set seed for reproducibility
seed = 65647437836358831880808032086803839625
rng = np.random.default_rng(seed)

def Flux(x, thetaCore, log_n0, p, log_epsilon_e, log_epsilon_B, log_E0, thetaObs, xi_N, d_L, z, jetType):
    # Function to calculate the model
    Z = {
        'jetType': jetType,  # Parameterize jet type
        'specType': grb.jet.SimpleSpec,
        'thetaObs': thetaObs,
        'E0': 10**log_E0,
        'thetaCore': thetaCore,
        'thetaWing': 4*thetaCore,
        'n0': 10**log_n0,
        'p': p,
        'epsilon_e': 10**log_epsilon_e,
        'epsilon_B': 10**log_epsilon_B,
        'xi_N': xi_N,
        'd_L': d_L,
        'z': z,
    }
    t = x[0]
    nu = x[1]

    try:
        Flux = grb.fluxDensity(t, nu, **Z)
        # Check if all elements of Flux are finite
        if isinstance(Flux, np.ndarray):
            if not np.all(np.isfinite(Flux)):
                raise ValueError("Flux computation returned non-finite values.")
        elif not np.isfinite(Flux):
            raise ValueError("Flux computation returned a non-finite value.")
    except Exception as e:
        print(f"Error in fluxDensity computation: {e}")
        return np.full_like(t, 1e-300)  # Return a very small flux value

    return Flux

# Define the parameters
log_E0 = 53
thetaCore = 0.3
thetaObs = 0.4
thetaWing =  4*thetaCore
log_n0 = 3.0
p = 2.33
log_epsilon_e = -1.0
log_epsilon_B = -3.0
xi_N = 1.0
d_L = 1.43e+27
z = 0.1

# Path to your input CSV file
data_file = './data/combined_statistics.csv'

# Load the generated data from CSV
data = pd.read_csv('./data/GRB1_control_data.csv')

# Extract time, frequency, flux, and error bounds
time_values = data['Time (s)'].values
freq_values = data['Frequency (Hz)'].values
flux_values = data['Flux (mJy)'].values
UB_err = data['UB Error'].values
LB_err = data['LB Error'].values

# Unique frequencies for plotting
unique_freqs = np.unique(freq_values)
# Create a colormap instance
colormap = cm.get_cmap('viridis', len(unique_freqs))  # Adjust 'viridis' to your preferred colormap

plt.figure(figsize=(10, 6))

# Loop through unique frequencies to plot data and models
for idx, nu_value in enumerate(unique_freqs):
    # Filter data for this frequency
    time_filtered = np.array([time_values[i] for i in range(len(time_values)) if freq_values[i] == nu_value])
    flux_filtered = np.array([flux_values[i] for i in range(len(flux_values)) if freq_values[i] == nu_value])
    UB_filtered = np.array([UB_err[i] for i in range(len(UB_err)) if freq_values[i] == nu_value])
    LB_filtered = np.array([LB_err[i] for i in range(len(LB_err)) if freq_values[i] == nu_value])

    if len(LB_filtered) == len(flux_filtered) and len(UB_filtered) == len(flux_filtered):
        # Assign a unique color
        color = colormap(idx / len(unique_freqs))

        # Plot the data with error bars
        plt.errorbar(
            time_filtered,
            flux_filtered,
            yerr=(LB_filtered, UB_filtered),
            fmt="o",
            label=f"Data (Freq: {nu_value:.3e} Hz)",
            color=color,
        )

        # Calculate the GaussianCore flux
        model_flux_gaussian = Flux(
            [time_filtered, nu_value * np.ones_like(time_filtered)],
            thetaCore, log_n0, p, log_epsilon_e, log_epsilon_B, log_E0, thetaObs, xi_N, d_L, z, grb.jet.GaussianCore
        )

        # Calculate the TopHat flux
        model_flux_tophat = Flux(
            [time_filtered, nu_value * np.ones_like(time_filtered)],
            thetaCore, log_n0, p, log_epsilon_e, log_epsilon_B, log_E0, thetaObs, xi_N, d_L, z, grb.jet.TopHat
        )

        # Plot the GaussianCore model
        plt.plot(
            time_filtered,
            model_flux_gaussian,
            label=f"GaussianCore (Freq: {nu_value:.3e} Hz)",
            linestyle="-",
            color=color,
            alpha=0.7,
        )

        # Plot the TopHat model
        plt.plot(
            time_filtered,
            model_flux_tophat,
            label=f"TopHat (Freq: {nu_value:.3e} Hz)",
            linestyle="--",
            color=color,
            alpha=0.7,
        )
    else:
        print(f"Error: Data length mismatch for frequency {nu_value}")

# Add labels and title
plt.xlabel("Time (s)")
plt.ylabel("Flux (mJy)")
plt.title("Flux vs Time for GaussianCore and TopHat Models")

# Set logarithmic axes
plt.xscale("log")
plt.yscale("log")

# Show legend
#plt.legend()

# Show grid
plt.grid(True)

# Save the plot
plt.savefig("./graph/gaussiancore_tophat_comparison.png")
# Display the plot
plt.show()

