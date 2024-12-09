import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import afterglowpy as grb
import matplotlib.cm as cm
import fakedata_v6 as fd6


# Example of selecting frequencies and instruments
selected_frequencies = [2e14] 
selected_instruments = ['Chandra']

# Path to your input CSV file
data_file = './data/combined_statistics.csv'

log_E0 = 54
thetaCore = 0.1
thetaObs = 1.0
thetaWing =  4*thetaCore
log_n0 = 0.0
p = 2.3
log_epsilon_e = -1.0
log_epsilon_B = -3.0
xi_N = 1.0
d_L = 1.34e+26
z = 0.01
theta = thetaCore, log_n0, p, log_epsilon_e, log_epsilon_B, log_E0, thetaObs, xi_N, d_L, z
mp = 1.67e-24
c = 3e10
Gamma = ((10**log_E0)/((10**log_n0)*mp*(c**5)))**(1/8)
tbi = 2.95*(((10**log_E0)/(1e53))**(1/3))*((10**log_n0)**(-1/3))*((thetaCore/0.1)**(8/3))*86400
tbo = 24.9*(((10**log_E0)/(1e53))**(1/3))*((10**log_n0)**(-1/3))*(((thetaObs+1.24*thetaCore)/0.5)**(8/3))*86400

if thetaObs < 1.01*thetaCore:
    tb = tbi
else:
    tb = tbo
print(Gamma,np.log10(tb))

# Define the optical frequency range
optical_min_freq = 10**11
optical_max_freq = 10**16

# Generate the realistic data based on the selected frequencies or instruments
time_values, freq_values, flux_values, UB_err, LB_err = fd6.generate_realistic_data(
    frequencies=None, instruments=None, theta=theta, data_file=data_file, flux_cutoff=True
)

# Create a DataFrame to hold the generated data
generated_data = pd.DataFrame({
    'Time (s)': time_values,
    'Frequency (Hz)': freq_values,
    'Flux (mJy)': flux_values,
    'UB Error': UB_err,
    'LB Error': LB_err,
})

# Save the generated data to a CSV file
generated_data.to_csv('./data/GRB4_data.csv', index=False)

unique_freqs = np.unique(freq_values)
# Create a colormap instance
colormap = cm.get_cmap('viridis', len(unique_freqs))  # Adjust 'viridis' to your preferred colormap

plt.figure(figsize=(10, 6))

# Loop through unique frequencies to plot data
for idx, nu_value in enumerate(unique_freqs):
    # Get time and Flux values for the current frequency
    time_filtered = [time_values[i] for i in range(len(time_values)) if freq_values[i] == nu_value]
    flux_filtered = [flux_values[i] for i in range(len(flux_values)) if freq_values[i] == nu_value]
    UB_filtered = [UB_err[i] for i in range(len(UB_err)) if freq_values[i] == nu_value]
    LB_filtered = [LB_err[i] for i in range(len(LB_err)) if freq_values[i] == nu_value]
    
    if len(LB_filtered) == len(flux_filtered) and len(UB_filtered) == len(flux_filtered):
        # Assign a unique color from the colormap
        color = colormap(idx / len(unique_freqs))
        
        # Plot this frequency's data with error bars
        plt.errorbar(
            time_filtered, flux_filtered,
            yerr=(LB_filtered, UB_filtered),
            fmt='.', label=f'Frequency: {nu_value:.3e} Hz', color=color
        )
    else:
        print(f"Error: The lengths of LB_filtered, UB_filtered, and flux_filtered do not match for frequency {nu_value}")

# Adding labels and title
plt.xlabel('Time (s)')
plt.ylabel('Flux (mJy)')
plt.title('Flux vs Time for Different Frequencies')

# Make axis logarithmic
plt.xscale('log')
plt.yscale('log')

# Show legend
plt.legend(fontsize='x-small')

# Show grid
plt.grid(True)

# Save the plot
plt.savefig('./graph/GRB4_figure.png')
# Display the plot
plt.show()
