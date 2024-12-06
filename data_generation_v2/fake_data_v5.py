import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import afterglowpy as grb
import matplotlib.cm as cm

# Set seed for reproducibility
seed = 65647437836358831880808032086803839625
rng = np.random.default_rng(seed)

def Flux(x, thetaCore, log_n0, p, log_epsilon_e, log_epsilon_B, log_E0, thetaObs, xi_N, d_L, z):
    # Function to calculate the model
    Z = {
        'jetType': grb.jet.Gaussian,
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

def estimate_errors(data, target_freq):
    """
    Estimate errors for a given frequency based on the nearest available frequencies in the dataset.
    """
    # Sort the data by frequency
    sorted_data = data.sort_values('Frequency (Hz)')
    
    # Find the nearest frequencies
    lower_band = sorted_data[sorted_data['Frequency (Hz)'] < target_freq].tail(1)
    upper_band = sorted_data[sorted_data['Frequency (Hz)'] > target_freq].head(1)
    
    # If both bounds are available, interpolate
    if not lower_band.empty and not upper_band.empty:
        f1, f2 = lower_band['Frequency (Hz)'].values[0], upper_band['Frequency (Hz)'].values[0]
        UB1, UB2 = lower_band['UB_err_mean'].values[0], upper_band['UB_err_mean'].values[0]
        LB1, LB2 = lower_band['LB_err_mean'].values[0], upper_band['LB_err_mean'].values[0]
        
        UB_err_mean = UB1 + (UB2 - UB1) * (target_freq - f1) / (f2 - f1)
        LB_err_mean = LB1 + (LB2 - LB1) * (target_freq - f1) / (f2 - f1)
    elif not lower_band.empty:  # Use only the lower bound
        UB_err_mean = lower_band['UB_err_mean'].values[0]
        LB_err_mean = lower_band['LB_err_mean'].values[0]
    elif not upper_band.empty:  # Use only the upper bound
        UB_err_mean = upper_band['UB_err_mean'].values[0]
        LB_err_mean = upper_band['LB_err_mean'].values[0]
    else:  # Default fallback
        UB_err_mean = LB_err_mean = 0.01  # A default small error value
    
    return UB_err_mean, LB_err_mean

# Function to generate realistic data
def generate_realistic_data(theta, data_file, frequencies=None, instruments=None, flux_cutoff=True):
    # Unpack simulated GRB properties
    thetaCore, log_n0, p, log_epsilon_e, log_epsilon_B, log_E0, thetaObs, xi_N, d_L, z = theta
    
    # Load the input data from CSV file
    data = pd.read_csv(data_file)

    time_values = []
    freq_values = []
    flux_values = []
    UB_err = []
    LB_err = []

    if frequencies:
        for freq in frequencies:
            if freq not in data['Frequency (Hz)'].values:
                # Find the closest matching frequency band
                sorted_data = data.sort_values('Frequency (Hz)')
                lower_band = sorted_data[sorted_data['Frequency (Hz)'] < freq].tail(1)
                upper_band = sorted_data[sorted_data['Frequency (Hz)'] > freq].head(1)

                # Use Min_Flux from the closest frequency band
                if not lower_band.empty:
                    similar_band = lower_band
                elif not upper_band.empty:
                    similar_band = upper_band
                else:
                    similar_band = None
                
                if similar_band is not None:
                    min_flux = similar_band['Min_Flux'].values[0]
                else:
                    min_flux = 1e-10  # Default fallback
                
                # Estimate errors for the new frequency
                UB_err_mean, LB_err_mean = estimate_errors(data, freq)
                estimates = pd.DataFrame([{
                    'Frequency (Hz)': freq,
                    'Instrument': 'Estimated',
                    'UB_err_mean': UB_err_mean,
                    'UB_err_var': np.nan,  # Assume no variance
                    'LB_err_mean': LB_err_mean,
                    'LB_err_var': np.nan,  # Assume no variance,
                    'Min_Flux': min_flux  # Use Min_Flux from similar band
                }])
                data = pd.concat([data, estimates], ignore_index=True)
    
    # Filter the data for selected frequencies or instruments
    if instruments:
        filtered_data = data[data['Instrument'].isin(instruments)]
    else:
        filtered_data = data
    
    for _, row in filtered_data.iterrows():
        nu_value = row['Frequency (Hz)']
        UB_err_mean = row['UB_err_mean']
        UB_err_var = row['UB_err_var']
        LB_err_mean = row['LB_err_mean']
        LB_err_var = row['LB_err_var']
        Min_Flux = row['Min_Flux']

        # Generate time samples as a geometric progression
        samples_per_band = rng.integers(5, 30)
        t = np.geomspace(1e1, 1e10, samples_per_band)  # Example time range

        # Generate realistic time and flux values for each frequency and instrument
        for i in range(samples_per_band):
            time_values.append(t[i])
            freq_values.append(nu_value)

            # Calculate flux using the sampler
            flux_values.append(Flux([t[i], nu_value], thetaCore, log_n0, p, log_epsilon_e, log_epsilon_B, log_E0, thetaObs, xi_N, d_L, z)[0])             
            
            # Generate Flux and errors based on instrument
            UB_err.append(abs(UB_err_mean) if np.isnan(UB_err_var) else abs(rng.normal(UB_err_mean, UB_err_var)))
            LB_err.append(abs(LB_err_mean) if np.isnan(LB_err_var) else abs(rng.normal(LB_err_mean, LB_err_var)))

            # Apply flux cutoff if required
            if (flux_values[-1] < Min_Flux) & (flux_cutoff == True):
                flux_values.pop()
                time_values.pop()
                freq_values.pop()
                UB_err.pop()
                LB_err.pop()

    return time_values, freq_values, flux_values, UB_err, LB_err




# Example of selecting frequencies and instruments
selected_frequencies = [2e14] 
selected_instruments = ['Chandra']

# Path to your input CSV file
data_file = './data/combined_statistics.csv'

log_E0 = 53
thetaCore = 0.1
thetaObs = 0.8
thetaWing =  4*thetaCore
log_n0 = -1.0
p = 2.33
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

# Function to introduce gaps in optical data
def introduce_optical_gaps(time_values, freq_values, flux_values, UB_err, LB_err, optical_min_freq, optical_max_freq):
    optical_indices = [
        i for i, freq in enumerate(freq_values)
        if optical_min_freq <= freq <= optical_max_freq
    ]
    
    if not optical_indices:
        return time_values, freq_values, flux_values, UB_err, LB_err

    # Define the mean and standard deviation for the gap length (in seconds)
    mean_gap_duration = 3 * 3600  # 3 hours in seconds
    std_gap_duration = 1 * 3600  # 1 hours in seconds

    # Define minimum spacing between gap starts (in seconds)
    min_spacing = 2 * 3600  # 4 hours in seconds

    # Determine the number of gaps (5-10 randomly)
    num_gaps = rng.integers(0, 4)
    print(num_gaps)
    # Select random start times for the gaps with enforced spacing
    all_optical_times = np.array([time_values[i] for i in optical_indices])
    min_time, max_time = all_optical_times.min(), all_optical_times.max()
    
    gap_starts = []
    attempts = 0  # To avoid infinite loop if constraints can't be met
    
    while len(gap_starts) < num_gaps and attempts < 1000:
        gap_start_candidate = np.exp(rng.uniform(np.log(min_time), np.log(max_time - mean_gap_duration)))
        
        # Check if the candidate satisfies the minimum spacing condition
        if all(abs(gap_start_candidate - existing_start) >= min_spacing for existing_start in gap_starts):
            gap_starts.append(gap_start_candidate)
        attempts += 1
    
    if attempts >= 1000:
        print("Warning: Maximum attempts reached while generating gap starts.")

    # Create gaps
    for gap_start in gap_starts:
        gap_duration = abs(rng.normal(mean_gap_duration, std_gap_duration))
        gap_end = gap_start + gap_duration

        # Remove data within this gap
        to_remove = [
            i for i in optical_indices
            if gap_start <= time_values[i] <= gap_end
        ]

        # Remove the indices from all lists
        for idx in sorted(to_remove, reverse=True):
            del time_values[idx]
            del freq_values[idx]
            del flux_values[idx]
            del UB_err[idx]
            del LB_err[idx]

    return time_values, freq_values, flux_values, UB_err, LB_err


# Define the optical frequency range
optical_min_freq = 10**11
optical_max_freq = 10**16

# Generate the realistic data based on the selected frequencies or instruments
time_values, freq_values, flux_values, UB_err, LB_err = generate_realistic_data(
    frequencies=selected_frequencies, instruments=None, theta=theta, data_file=data_file, flux_cutoff=True
)

# Introduce gaps for optical frequencies
time_values, freq_values, flux_values, UB_err, LB_err = introduce_optical_gaps(
    time_values, freq_values, flux_values, UB_err, LB_err, optical_min_freq, optical_max_freq
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
generated_data.to_csv('./data/GRB10_data.csv', index=False)

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
#plt.legend()

# Show grid
plt.grid(True)

# Save the plot
plt.savefig('./graph/GRB10_figure.png')
# Display the plot
plt.show()



