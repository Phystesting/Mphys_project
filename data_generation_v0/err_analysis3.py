import afterglowpy as grb
import emcee
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
import re


band_dict = {
    2.418e17: "1 keV",
    1.209e+18: "5 keV",
    1.555e15: "UVW2",
    1.335e15: "UVM2",
    1.153e15: "UVW1",
    8.652e14: "U",
    8.443e14: "u",
    6.826e14: "B",
    6.389e14: "g",
    5.483e14: "V",
    4.862e14: "r",
    4.008e14: "i",
    3.356e14: "z",
    2.398e14: "J",
    1.851e14: "H",
    1.414e14: "Ks",
    1.000e10: "10 GHz",
    6.000e09: "6 GHz",
}

GRB = pd.read_csv('./data/GRB170817_data.csv')

#t,nu,mJy,upperlimit,Instrument,Counts,Background,fltr = np.loadtxt('./data/GRB170817.csv',delimiter=',',unpack=True)

def separate(string):
    # Use regular expression to capture numbers and operators, including cases like +- or -+
    numbers = re.findall(r'[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?', string) #
    i = 0
    while i < len(numbers):
        numbers[i] = float(numbers[i])
        i = i + 1
    return numbers

flux = np.full(len(GRB['t']), np.nan)
freqs = np.full(len(GRB['t']), np.nan)
time = np.full(len(GRB['t']), np.nan)
err = np.full(len(GRB['t']), np.nan)
UB_err = np.full(len(GRB['t']), np.nan)
LB_err = np.full(len(GRB['t']), np.nan)

# Iterate through the GRB DataFrame
for i in range(len(GRB['t'])):
    flux_w_err = separate(GRB['mJy'][i])
    
    # Handle flux and error cases
    if len(flux_w_err) == 1:
        flux[i] = flux_w_err[0]
        UB_err[i] = 0
        LB_err[i] = flux[i]
    elif len(flux_w_err) == 2:
        flux[i], err[i] = flux_w_err
        UB_err[i] = err[i]
        LB_err[i] = err[i]
    else:
        flux[i], UB_err[i], LB_err[i] = flux_w_err
    
    # Assign frequency
    if np.isfinite(GRB['nu'][i]):
        freqs[i] = GRB['nu'][i]
    else:
        # Match filter name to band_dict values to retrieve the key (frequency)
        filter_name = GRB['filter'][i]
        matching_freq = [k for k, v in band_dict.items() if v == filter_name]
        if matching_freq:
            freqs[i] = matching_freq[0]
        else:
            print(f"Warning: Filter '{filter_name}' not found in band_dict.")
            freq[i] = np.nan
    time[i] = GRB['t'][i]

UB_err = abs(UB_err)
LB_err = abs(LB_err)


t_filtered = []
freq_filtered = []
flux_filtered = []
Ub_filtered = []
Lb_filtered = []
for i in range(len(time)):
    if UB_err[i] != 0:
        t_filtered.append(time[i])
        freq_filtered.append(freqs[i])
        flux_filtered.append(flux[i])
        Ub_filtered.append(UB_err[i])
        Lb_filtered.append(LB_err[i])

t_filtered = np.array(t_filtered)
freq_filtered = np.array(freq_filtered)
flux_filtered = np.array(flux_filtered)
Ub_filtered = np.array(Ub_filtered)
Lb_filtered = np.array(Lb_filtered)

output_file = f"./data/170817_data.csv"
print(len(t_filtered))
# Open the file in append mode
with open(output_file, "w") as f:
    f.write("time (s), freq (Hz), flux (mJy), UB_err, LB_err\n")
    j = 0
    while j < len(flux_filtered):
        f.write(f"{t_filtered[j]},{freq_filtered[j]},{flux_filtered[j]},{Ub_filtered[j]},{Lb_filtered[j]}\n")
        j = j + 1



# Ensure 'Instrument' column exists and corresponds to the processed frequencies
if 'Instrument' in GRB.columns:
    # Use the filled frequencies and Instrument column to create combinations
    freq_instrument_combos = pd.DataFrame({
        'Frequency (Hz)': freq_filtered,
        'Instrument': GRB['Instrument'][:len(freq_filtered)]
    })

    # Drop duplicates to find unique combinations
    freq_instrument_combos = freq_instrument_combos.drop_duplicates()

    # Handle NaN values just in case
    freq_instrument_combos = freq_instrument_combos.dropna()

    # Sort for better readability
    freq_instrument_combos = freq_instrument_combos.sort_values(by='Frequency (Hz)')

    # Save or print the unique combinations
    output_combo_file = "./data/freq_instrument_combos.csv"
    freq_instrument_combos.to_csv(output_combo_file, index=False)
    print(freq_instrument_combos)
else:
    print("Error: 'Instrument' column not found in GRB data.")

# Create a DataFrame for frequency, instrument, and errors
error_analysis_df = pd.DataFrame({
    'Frequency (Hz)': freq_filtered,
    'Instrument': GRB['Instrument'][:len(freq_filtered)],
    'Flux': flux_filtered,
    'UB_err': Ub_filtered/flux_filtered,
    'LB_err': Lb_filtered/flux_filtered
})

# Group by frequency-instrument combination and calculate mean, variance, error, and minimum flux
combined_stats = (
    error_analysis_df
    .groupby(['Frequency (Hz)', 'Instrument'], as_index=False)
    .agg({
        'UB_err': ['mean', 'var'],  # Calculate mean and variance for UB_err
        'LB_err': ['mean', 'var'],  # Calculate mean and variance for LB_err
        'Flux': 'min'               # Calculate minimum flux for each group
    })
)

# Flatten multi-level column names for readability
combined_stats.columns = ['Frequency (Hz)', 'Instrument', 
                           'UB_err_mean', 'UB_err_var', 
                           'LB_err_mean', 'LB_err_var', 
                           'Min_Flux']

# Save or print the results
output_combined_stats_file = "./data/combined_statistics.csv"
combined_stats.to_csv(output_combined_stats_file, index=False)
print(combined_stats)


unique_freqs = np.unique(freq_filtered)


fig, ax = plt.subplots(1, 1, figsize=(10, 6))

for i in range(len(unique_freqs)):
    mask = freq_filtered == unique_freqs[i]
    ax.errorbar(t_filtered[mask],flux_filtered[mask],yerr=(Lb_filtered[mask],Ub_filtered[mask]),fmt='.')
ax.set(
    xscale="log", yscale="log",
    xlabel=r"$t$ (s)", ylabel=r"$F_\nu$ (mJy)"
)
plt.show()

