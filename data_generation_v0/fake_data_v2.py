import numpy as np
import afterglowpy as grb
import emcee
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import curve_fit
import corner
import os
import sys
import multiprocessing
from multiprocessing import Pool
import time
from functools import partial
import concurrent.futures
import sampler as splr
import random as rnmb

seed = 65647437836358831880808032086803839626
rng = np.random.default_rng(seed)


#representing data from VLT ANTU (Optical) and FORS1 (Broad-Range)
# try generation in 3 bands x-ray, optical, radio

nu = np.array([4.8e09,8.6e09,3.72e14,4.55e14,5.45e14,1.08e18])
freq = []
time = []

"""
#realistic time
for nu_idx, nu_value in enumerate(nu):
    if nu_value > 1e16:
        # High frequency
        samples = rng.integers(10,30)
    elif 1e10 < nu_value < 1e16:
        # Optical, UV, and Infrared
        samples = rng.integers(25,50)  
    elif nu_value < 1e10:
        # Radio
        samples = rng.integers(3,10)
        
    for i in range(samples):
        if nu_value > 1e16:
            time.append(10**(rng.normal(4.8,0.2)))
        elif 1e10 < nu_value < 1e16:
            time.append(10**(rng.normal(5,0.3)))
        elif nu_value < 1e10:
            time.append(10**(rng.normal(5.5,0.2)))
        freq.append(nu_value)
"""
#uniform time
for nu_idx, nu_value in enumerate(nu):
    samples = 25
    for i in range(samples):
        time.append(np.exp(12 + i*(18.274-12)/samples))
        freq.append(nu_value)


#

F_start = splr.Flux([time,freq],0.1,0.0,2.33,-1,-3,51,0.0,1.0,4.88e28,2.0)


err = []
displacement = []



for freq_idx, freq_value in enumerate(freq):
    if freq_value > 1e16:
        # High frequency error generation ~ 0.1-0.2
        err.append(abs(rng.normal(0.15, 0.05)))
        displacement.append(abs(rng.normal(0.0, 0.1)))
        
    elif 1e10 < freq_value < 1e16:
        # Optical, UV, and Infrared generation error ~ 0.02-0.2
        err.append(abs(rng.normal(0.11, 0.05)))
        displacement.append(abs(rng.normal(0.0, 0.1)))
        
    elif freq_value < 1e10:
        # Radio generation error ~ 0.1-0.7
        err.append(abs(rng.normal(0.4, 0.15)))
        displacement.append(abs(rng.normal(0.0, 0.35)))


# Generate F_err by adding displacement directly to F
F = F_start + (F_start*displacement)

err = F*err

# Assuming freq, time, and F are already defined as per the previous part of the code

data = open('./data/post_jb_data.csv', 'w')
data.write('time (s),Frequency (Hz),Flux (mJy),Flux error\n')
for i in range(len(F)):
    data.write(f"{time[i]},{freq[i]},{F[i]},{err[i]}\n")
data.close()

# Create a figure and axis
plt.figure(figsize=(10, 6))

# Loop through unique frequencies to plot data
unique_freqs = np.unique(freq)
# Plot each frequency with its corresponding data
for nu_value in unique_freqs:
    # Get time and F values for the current frequency
    time_values = [time[i] for i in range(len(time)) if freq[i] == nu_value]
    F_values = [F[i] for i in range(len(F)) if freq[i] == nu_value]
    err_values = [err[i] for i in range(len(err)) if freq[i] == nu_value]
    
    # Plot this frequency's data
    plt.errorbar(time_values, F_values,yerr=err_values,fmt='.', label=f'Frequency: {nu_value:.3e} Hz')

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

plt.savefig('./graph/post_jb_data.png')
# Display the plot
plt.show()

# Split the data in half
"""
# Find the midpoint in the logarithmic time range
log_time = np.log(time)
midpoint_log_time = (min(log_time) + max(log_time)) / 2
midpoint_time = np.exp(midpoint_log_time)


# Split data based on the midpoint time
first_half_indices = [i for i in range(len(time)) if time[i] <= midpoint_time]
second_half_indices = [i for i in range(len(time)) if time[i] > midpoint_time]

time_first_half = [time[i] for i in first_half_indices]
freq_first_half = [freq[i] for i in first_half_indices]
F_first_half = [F[i] for i in first_half_indices]
err_first_half = [err[i] for i in first_half_indices]

time_second_half = [time[i] for i in second_half_indices]
freq_second_half = [freq[i] for i in second_half_indices]
F_second_half = [F[i] for i in second_half_indices]
err_second_half = [err[i] for i in second_half_indices]

# Save the first half to a CSV file
data_first_half = open('./data/data_first_half.csv', 'w')
data_first_half.write('time (s),Frequency (Hz),Flux (mJy),Flux error\n')
for i in range(len(time_first_half)):
    data_first_half.write(f"{time_first_half[i]},{freq_first_half[i]},{F_first_half[i]},{err_first_half[i]}\n")
data_first_half.close()

# Save the second half to a CSV file
data_second_half = open('./data/data_second_half.csv', 'w')
data_second_half.write('time (s),Frequency (Hz),Flux (mJy),Flux error\n')
for i in range(len(time_second_half)):
    data_second_half.write(f"{time_second_half[i]},{freq_second_half[i]},{F_second_half[i]},{err_second_half[i]}\n")
data_second_half.close()

# Create the first plot for the first half
plt.figure(figsize=(10, 6))

unique_freqs_first_half = np.unique(freq_first_half)
for nu_value in unique_freqs_first_half:
    time_values = [time_first_half[i] for i in range(len(time_first_half)) if freq_first_half[i] == nu_value]
    F_values = [F_first_half[i] for i in range(len(F_first_half)) if freq_first_half[i] == nu_value]
    err_values = [err_first_half[i] for i in range(len(err_first_half)) if freq_first_half[i] == nu_value]
    
    plt.errorbar(time_values, F_values, yerr=err_values, fmt='.', label=f'Frequency: {nu_value:.3e} Hz')

plt.xlabel('Time (s)')
plt.ylabel('Flux (mJy)')
plt.title('Flux vs Time (First Half)')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.savefig('./graph/data_first_half.png')
plt.show()

# Create the second plot for the second half
plt.figure(figsize=(10, 6))

unique_freqs_second_half = np.unique(freq_second_half)
for nu_value in unique_freqs_second_half:
    time_values = [time_second_half[i] for i in range(len(time_second_half)) if freq_second_half[i] == nu_value]
    F_values = [F_second_half[i] for i in range(len(F_second_half)) if freq_second_half[i] == nu_value]
    err_values = [err_second_half[i] for i in range(len(err_second_half)) if freq_second_half[i] == nu_value]
    
    plt.errorbar(time_values, F_values, yerr=err_values, fmt='.', label=f'Frequency: {nu_value:.3e} Hz')

plt.xlabel('Time (s)')
plt.ylabel('Flux (mJy)')
plt.title('Flux vs Time (Second Half)')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.savefig('./graph/data_second_half.png')
plt.show()
"""




