import numpy as np
import csv

# Input and output file paths
input_file = './data/170817_data.csv'
output_file = './data/170817_data_reduced.csv'

# Time threshold (seconds)
time_threshold = 4e7

# Load the data
time, freq, flux, UB_err, LB_err = np.genfromtxt(
    input_file, delimiter=',', skip_header=1, unpack=True
)

# Apply the filter: keep rows where time <= 4e7
mask = time <= time_threshold
time_reduced = time[mask]
freq_reduced = freq[mask]
flux_reduced = flux[mask]
UB_err_reduced = UB_err[mask]
LB_err_reduced = LB_err[mask]

# Save the reduced dataset to a new CSV file
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Write header
    writer.writerow(['time', 'freq', 'flux', 'UB_err', 'LB_err'])
    # Write the filtered data
    writer.writerows(zip(time_reduced, freq_reduced, flux_reduced, UB_err_reduced, LB_err_reduced))

print(f"Reduced dataset saved to {output_file}.")
