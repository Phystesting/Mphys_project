import numpy as np
import matplotlib.pyplot as plt

#unpack data
time, freq, flux, err = np.genfromtxt('./data/990510.csv',delimiter=',',skip_header=1,unpack=True)

# Get unique times and frequencies, and map them to indices
u_time, time_indices = np.unique(time, return_inverse=True)
u_freq, freq_indices = np.unique(freq, return_inverse=True)

# Initialize an empty array with NaN values
flux_arr = np.full((len(u_time), len(u_freq)), np.nan)
err_arr = np.full((len(u_time),len(u_freq)), np.nan)

# Populate the arrays using indices
flux_arr[time_indices, freq_indices] = flux
err_arr[time_indices, freq_indices] = err
print(err_arr)

for freq_idx, nu_value in enumerate(u_freq):
    plt.errorbar(np.log10(u_time),np.log10(flux_arr[:,freq_idx]),yerr=abs(np.log10(flux_arr[:,freq_idx])*err_arr[:,freq_idx]/flux_arr[:,freq_idx]),fmt='.', label=f'{nu_value:.2e} Hz')
plt.legend()
plt.show()
