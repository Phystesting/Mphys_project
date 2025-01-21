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
        return np.full_like(t, 1e-300)  # Return a very small flux value

    return Flux

# Function to generate realistic data
def generate_realistic_data(theta, data_file, frequencies=None, instruments=None, flux_cutoff=True):
    time_min = -1
    time_max = 1e8
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
            
        # Generate realistic time and flux values for each frequency and instrument
        for i in range(samples_per_band):
            time_values.append(t[i])
            freq_values.append(nu_value)

            # Calculate flux using the sampler
            flux_values.append(Flux([t[i], nu_value], thetaCore, log_n0, p, log_epsilon_e, log_epsilon_B, log_E0, thetaObs, xi_N, d_L, z)[0])             
            
            # Generate Flux and errors based on instrument
            # Check if UB_err_var is NaN, and handle it accordingly
            if np.isnan(UB_err_var):
                UB_err.append(UB_err_mean)
            else:
                UB_err.append(abs(rng.normal(UB_err_mean, UB_err_var)))

            # Check if LB_err_var is NaN, and handle it accordingly
            if np.isnan(LB_err_var):
                LB_err.append(LB_err_mean)
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

