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

# Function to generate realistic data
# Function to generate realistic data
def generate_realistic_data(theta, data_file, frequencies=None, instruments=None, flux_cutoff=True):
    time_min = -1
    time_max = 1e10
    # Unpack simulated GRB properties
    thetaCore, log_n0, p, log_epsilon_e, log_epsilon_B, log_E0, thetaObs, xi_N, d_L, z = theta
    # Load the input data from CSV file
    data = pd.read_csv(data_file)

    time_values = []
    freq_values = []
    flux_values = []
    UB_err = []
    LB_err = []
    
    xray_min = 1e1
    optical_min = 1e2
    radio_min = 1e4
    
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
        print(nu_value,instrument)
        # Determine the minimum time based on the frequency range
        if nu_value > 10**16:
            base_min_time = xray_min
        elif 10**11 < nu_value <= 10**16:
            base_min_time = optical_min
        elif nu_value <= 10**11:
            base_min_time = radio_min
        else:
            base_min_time = optical_min  # Default case
        
        # Apply Gaussian variation to the starting point
        adjusted_min_time = abs(rng.normal(base_min_time, base_min_time * 0.2))  # 10% standard deviation
        
        # Generate time samples as a geometric progression starting from the adjusted time
        samples_per_band = rng.integers(5, 30) if nu_value > 10**11 else rng.integers(2, 10)
        t = np.geomspace(adjusted_min_time, time_max, samples_per_band)
        
        # Add gaps to the optical and radio bands
        num_gaps = rng.integers(0, 10)
        gapstart = []
        gapend = []
        for j in range(num_gaps):
            gaplength = abs(rng.normal(10*3600,3*3600))
            gapstart.append(10**rng.uniform(np.log10(base_min_time),8))
            gapend.append(gapstart[-1] + gaplength)
            
        print(gapstart,gapend)
        # Generate realistic time and flux values for each frequency and instrument
        for i in range(samples_per_band):
            time_values.append(t[i])
            freq_values.append(nu_value)

            # Calculate flux using the sampler
            flux_values.append(Flux([t[i], nu_value], thetaCore, log_n0, p, log_epsilon_e, log_epsilon_B, log_E0, thetaObs, xi_N, d_L, z)[0])             
            
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
            
            # Remove data within gaps
            for k in range(len(gapstart)):
                if (gapstart[k] < time_values[-1] < gapend[k]) & (freq_values[-1] < 1e16):
                    #print(time_values[-1])
                    flux_values.pop()
                    time_values.pop()
                    freq_values.pop()
                    UB_err.pop()
                    LB_err.pop()                  
                    
            # Apply flux cutoff if required
            if (flux_values[-1] < Min_Flux) & (flux_cutoff==True):
                flux_values.pop()
                time_values.pop()
                freq_values.pop()
                UB_err.pop()
                LB_err.pop()
            
    return time_values, freq_values, flux_values, UB_err, LB_err
    
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


def datagaps(time_values, freq_values, flux_values, UB_err, LB_err):
    
    # Define the mean and standard deviation for the gap length (in seconds)
    mean_gap_duration = 3 * 3600  # 3 hours in seconds
    std_gap_duration = 1 * 3600  # 1 hours in seconds

    # Define minimum spacing between gap starts (in seconds)
    min_spacing = 2 * 3600  # 4 hours in seconds

    # Determine the number of gaps (5-10 randomly)
    num_gaps = rng.integers(0, 4)
    mask = freq_values < 1e16
    tmax = max(time_values[mask])
    tmin = min(time_values[mask])

