import numpy as np
import matplotlib.pyplot as plt
import sampler as splr

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
#print(len(u_time))
initial = [0.1,1,2.3,-1,-2,53]
final_err = [err_arr,np.zeros((len(u_time),len(u_freq))),np.zeros((len(u_time),len(u_freq)))]
splr.run_optimization([u_time, u_freq],flux_arr, initial, final_err, thetaObs=0.0, xi_N=1.0, d_L=9.461e26, z=1.619)

for freq_idx, nu_value in enumerate(u_freq):                plt.errorbar(np.log10(u_time),np.log10(flux_arr[:,freq_idx]),yerr=abs(err_arr[:,freq_idx]/flux_arr[:,freq_idx]),fmt='.', label=f'{nu_value:.2e} Hz')
plt.legend()
plt.show()

