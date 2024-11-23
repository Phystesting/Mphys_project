import sampler_v3 as splr
import numpy as np
import matplotlib.pyplot as plt

# Define your fixed parameters separately
fixed_params = {

}

# Define your initial guesses for the fitting parameters
initial = {
    "thetaCore": 0.1,
    "p": 2.14,
    "log_epsilon_e": -1.0,
    "log_epsilon_B": -3.0,
    "log_E0": 51.0,
    "log_n0": -4.0,
    "thetaObs": 0.0,
}

# Define inputs
z = 0.0099
d_L = 1.327e+26
xi_N = 1.0
nwalkers = 32
processes = 40
steps = 600000
filename = '/data/PROJECTS/2024-25/cjc233/Large_data/170817_samples.h5'

# Unpack data
time_values, freq_values, flux_values, UB_err, LB_err = np.genfromtxt('../data_generation_v1/data/170817_data.csv',delimiter=',',skip_header=1,unpack=True)
flux_err = LB_err, UB_err

result = splr.run_optimization([time_values,freq_values], flux_values, initial,fixed_params, flux_err, xi_N, d_L, z)


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
plt.legend()

# Show grid
plt.grid(True)

# Save the plot
plt.savefig('./graph/test_figure.png')
# Display the plot
plt.show()

#if __name__ == "__main__":
    #splr.run_sampling(x=[time,freq], y=flux, initial=initial,fixed_params=fixed_params, err_flux=flux_err, xi_N=xi_N, d_L=d_L, z=z, nwalkers=nwalkers, steps=steps, processes=processes,filename=filename)

