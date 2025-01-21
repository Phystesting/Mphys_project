import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import afterglowpy as grb
import matplotlib.cm as cm
import fakedata_v6 as fd6
import removedata_v0 as rd0

# Initial parameters
identifier = 'GRB2'

# Path to your input CSV file
data_file = './data/combined_statistics.csv'

log_E0 = 54
thetaCore = 0.1
thetaObs = 0.3
log_n0 = 0.0
p = 2.3
log_epsilon_e = -0.5
log_epsilon_B = -3.
xi_N = 1.0
d_L = 1.34e+26
z = 0.01
theta = thetaCore, log_n0, p, log_epsilon_e, log_epsilon_B, log_E0, thetaObs, xi_N, d_L, z
mp = 1.67e-24
c = 3e10

Gamma = ((10**log_E0) / ((10**log_n0) * mp * (c**5)))**(1/8)
tbi = 2.95 * (((10**log_E0) / (1e53))**(1/3)) * ((10**log_n0)**(-1/3)) * ((thetaCore / 0.1)**(8/3)) * 86400
tbo = 24.9 * (((10**log_E0) / (1e53))**(1/3)) * ((10**log_n0)**(-1/3)) * (((thetaObs + 1.24 * thetaCore) / 0.5)**(8/3)) * 86400

tb = tbi if thetaObs < 1.01 * thetaCore else tbo

print(Gamma,tb)

# Generate realistic data
time_values, freq_values, flux_values, UB_err, LB_err = fd6.generate_realistic_data(
    frequencies=None, instruments=None, theta=theta, data_file=data_file, flux_cutoff=True
)

data = time_values, freq_values, flux_values, UB_err, LB_err

# Define the reduction functions and their arguments
reduction_functions = [
    ('jetbreak', rd0.jetbreak, {'jb_time': tb}),
    ('removeband_xray', rd0.removeband, {'band': 'x-rays'}),
    ('removeband_optical', rd0.removeband, {'band': 'optical'}),
    ('removeband_radio', rd0.removeband, {'band': 'radio'}),
    ('thinning', rd0.thinning, {'num': 2})
]

# Function to save data to a CSV file
def save_data_to_csv(data, identifier, suffix):
    time, freq, flux, UB, LB = data
    df = pd.DataFrame({
        'Time (s)': time,
        'Frequency (Hz)': freq,
        'Flux (mJy)': flux,
        'UB Error': UB,
        'LB Error': LB,
    })
    df.to_csv(f'./data/{identifier}_{suffix}.csv', index=False)

# Function to plot data
# Generate realistic data without flux cutoff
time_values_no_cutoff, freq_values_no_cutoff, flux_values_no_cutoff, UB_err_no_cutoff, LB_err_no_cutoff = fd6.generate_realistic_data(
    frequencies=None, instruments=None, theta=theta, data_file=data_file, flux_cutoff=False  # Remove flux cutoff here
)

data_no_cutoff = time_values_no_cutoff, freq_values_no_cutoff, flux_values_no_cutoff, UB_err_no_cutoff, LB_err_no_cutoff

# Function to plot data (modified to handle both cutoff and no cutoff)
def plot_data(data, identifier, suffix):
    time, freq, flux, UB, LB = data
    unique_freqs = np.unique(freq)
    colormap = cm.get_cmap('viridis', len(unique_freqs))
    plt.figure(figsize=(10, 6))

    for idx, nu_value in enumerate(unique_freqs):
        time_filtered = [time[i] for i in range(len(time)) if freq[i] == nu_value]
        flux_filtered = [flux[i] for i in range(len(flux)) if freq[i] == nu_value]
        UB_filtered = [UB[i] for i in range(len(UB)) if freq[i] == nu_value]
        LB_filtered = [LB[i] for i in range(len(LB)) if freq[i] == nu_value]

        if len(LB_filtered) == len(flux_filtered) and len(UB_filtered) == len(flux_filtered):
            color = colormap(idx / len(unique_freqs))
            plt.errorbar(
                time_filtered, flux_filtered,
                yerr=(LB_filtered, UB_filtered),
                fmt='.', label=f'Frequency: {nu_value:.3e} Hz', color=color
            )
        else:
            print(f"Error: Length mismatch for frequency {nu_value}")

    plt.xlabel('Time (s)')
    plt.ylabel('Flux (mJy)')
    plt.title(f'Flux vs Time ({suffix})')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(fontsize='x-small')
    plt.grid(True)
    plt.savefig(f'./graph/{identifier}_{suffix}.png')
    plt.close()

# Save and plot the unaltered data
save_data_to_csv(data, identifier, "control")
plot_data(data, identifier, "control")

# Save and plot the data without flux cutoff
save_data_to_csv(data_no_cutoff, identifier, "control_no_cutoff")
plot_data(data_no_cutoff, identifier, "control_no_cutoff")

# Process data through reduction functions
for suffix, func, kwargs in reduction_functions:
    if suffix == 'jetbreak':
        pre, post = func(data, **kwargs)
        
        # Save and plot pre-jetbreak data
        pre_suffix = f"{suffix}_pre"
        save_data_to_csv(pre, identifier, pre_suffix)
        plot_data(pre, identifier, pre_suffix)
        
        # Save and plot post-jetbreak data
        post_suffix = f"{suffix}_post"
        save_data_to_csv(post, identifier, post_suffix)
        plot_data(post, identifier, post_suffix)
    else:
        # For other functions, process as usual
        reduced_data = func(data, **kwargs)
        save_data_to_csv(reduced_data, identifier, suffix)
        plot_data(reduced_data, identifier, suffix)



