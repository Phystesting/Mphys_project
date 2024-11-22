import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sampler_v3 as splr

# Set seed for reproducibility
seed = 65647437836358831880808032086803839626
rng = np.random.default_rng(seed)

# Function to generate realistic data
# Function to generate realistic data
def generate_realistic_data(theta, data_file, frequencies=None, instruments=None, samples_per_band=10, uniform_time=False, flux_cutoff=True):
    time_min = -1
    time_max = 6
    # Unpack simulated GRB properties
    thetaCore, log_n0, p, log_epsilon_e, log_epsilon_B, log_E0, thetaObs, xi_N, d_L, z = theta
    # Load the input data from CSV file
    data = pd.read_csv(data_file)

    time_values = []
    freq_values = []
    flux_values = []
    UB_err = []
    LB_err = []
    
    xray_min = -1
    optical_min = 2
    radio_min = 4
    
    # If frequencies are provided, filter by frequency and use all available instruments for those frequencies
    if frequencies is not None:
        filtered_data = data[data['Frequency (Hz)'].isin(frequencies)]
    # If instruments are provided, filter by instrument and use all available frequencies for those instruments
    elif instruments is not None:
        filtered_data = data[data['Instrument'].isin(instruments)]
    else:
        filtered_data = data

    for _, row in filtered_data.iterrows():
        nu_value = row['Frequency (Hz)']
        instrument = row['Instrument']
        UB_err_mean = row['UB_err_mean']
        UB_err_var = row['UB_err_var']
        LB_err_mean = row['LB_err_mean']
        LB_err_var = row['LB_err_var']
        Min_Flux = row['Min_Flux']

        # Generate realistic time and flux values for each frequency and instrument
        for i in range(samples_per_band):
            if uniform_time == False:
                # Time distribution (uniform scale between time_min and time_max)
                if nu_value >= 10**16:
                    time_values.append(10**(rng.uniform(xray_min, time_max)))
                elif 10**10 <= nu_value < 10**16:
                    time_values.append(10**(rng.uniform(optical_min, time_max)))
                elif nu_value < 10**10:
                    time_values.append(10**(rng.uniform(radio_min, time_max)))
                else:
                    time_values.append(10**(rng.uniform(time_min, time_max)))
            else:
                time_values.append(10**(rng.uniform(time_min, time_max)))
            freq_values.append(nu_value)

            flux_values.append(splr.Flux([time_values[-1], freq_values[-1]], thetaCore, log_n0, p, log_epsilon_e, log_epsilon_B, log_E0, thetaObs, xi_N, d_L, z)[0])             
            
            # Generate Flux and errors based on instrument
            # Check if UB_err_var is NaN, and handle it accordingly
            if np.isnan(UB_err_var):
                UB_err.append(abs(rng.normal(UB_err_mean, 0.01)))
            else:
                UB_err.append(abs(rng.normal(UB_err_mean, UB_err_var)))

            # Check if LB_err_var is NaN, and handle it accordingly
            if np.isnan(LB_err_var):
                LB_err.append(abs(rng.normal(LB_err_mean, 0.01)))
            else:
                LB_err.append(abs(rng.normal(LB_err_mean, LB_err_var)))
            
            apply_shift_up = rng.choice([True, False])  # Randomly choose whether to shift up or down

            if apply_shift_up:
                # Apply a positive shift based on UB_err
                flux_values[-1] += flux_values[-1] * rng.normal(0, UB_err[-1])
            else:
                # Apply a negative shift based on LB_err
                flux_values[-1] -= flux_values[-1] * rng.normal(0, LB_err[-1])
            
            # Adjust error bounds by the flux scale
            UB_err[-1] = abs(UB_err[-1] * flux_values[-1])
            LB_err[-1] = abs(LB_err[-1] * flux_values[-1])
            
            if (flux_values[-1] < Min_Flux) & (flux_cutoff==True):
                flux_values.pop()
                time_values.pop()
                freq_values.pop()
                UB_err.pop()
                LB_err.pop()
            
    return time_values, freq_values, flux_values, UB_err, LB_err



# Example of selecting frequencies and instruments
selected_frequencies = [3000000000.0, 3400000000.0] 
selected_instruments = ['Chandra']

# Path to your input CSV file
data_file = './data/combined_statistics.csv'

thetaCore = 0.1
log_n0 = 0.0
p = 2.33
log_epsilon_e = -1.0
log_epsilon_B = -3.0
log_E0 = 51.0
thetaObs = 0.0
xi_N = 1.0
d_L = 1.43e+27
z = 0.1
theta = thetaCore, log_n0, p, log_epsilon_e, log_epsilon_B, log_E0, thetaObs, xi_N, d_L, z

# Generate the realistic data based on the selected frequencies or instruments
time_values, freq_values, flux_values, UB_err, LB_err = generate_realistic_data(
    frequencies=None, instruments=None,theta=theta, data_file=data_file,uniform_time=False,flux_cutoff=True
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
generated_data.to_csv('./data/control_data.csv', index=False)

# Create a plot for the generated data
plt.figure(figsize=(10, 6))

# Loop through unique frequencies to plot data
unique_freqs = np.unique(freq_values)
for nu_value in unique_freqs:
    # Get time and Flux values for the current frequency
    time_filtered = [time_values[i] for i in range(len(time_values)) if freq_values[i] == nu_value]
    flux_filtered = [flux_values[i] for i in range(len(flux_values)) if freq_values[i] == nu_value]
    UB_filtered = [UB_err[i] for i in range(len(UB_err)) if freq_values[i] == nu_value]
    LB_filtered = [LB_err[i] for i in range(len(LB_err)) if freq_values[i] == nu_value]
    if len(LB_filtered) == len(flux_filtered) and len(UB_filtered) == len(flux_filtered):
        # Plot this frequency's data with error bars
        plt.errorbar(time_filtered, flux_filtered, yerr=(LB_filtered, UB_filtered), fmt='.', label=f'Frequency: {nu_value:.3e} Hz')
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
plt.legend()

# Show grid
plt.grid(True)

# Save the plot
plt.savefig('./graph/control_figure.png')
# Display the plot
plt.show()

