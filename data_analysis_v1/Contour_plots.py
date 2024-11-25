import numpy as np
import corner
import matplotlib.pyplot as plt
import os
import emcee

# Define the identifier variable (you can set it to whatever you want)
identifier = "GRB2_CO"  # Set your identifier here

file_path = f'/data/PROJECTS/2024-25/cjc233/Large_data/{identifier}_samples.h5'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The HDF5 file '{file_path}' does not exist.")
backend = emcee.backends.HDFBackend(file_path)
if backend.iteration == 0:
    raise ValueError("The HDF5 file is empty. No data was stored.")
reader = emcee.backends.HDFBackend(file_path)

try:
    tau = reader.get_autocorr_time()
    burnin = int(2 * np.max(tau))
    thin = int(0.5 * np.min(tau))
except emcee.autocorr.AutocorrError:
    print("Warning: Autocorrelation time estimation failed. Proceeding with current chain length.")
    burnin = len(reader.get_chain()) // 3  # Use some default burn-in, e.g., one-third of the chain length
    thin = 1  # No thinning

#base_labels = ["thetaCore", "n0", "p", "epsilon_e", "epsilon_B", "E0"]
base_labels = ["thetaCore", "p", "epsilon_e", "epsilon_B", "E0", "n0"]
ndim_base = len(base_labels)

# Check if thetaObs is fitted
if reader.shape[1] == ndim_base + 1:  # thetaObs is included
    labels = base_labels + ["thetaObs"]
else:  # thetaObs is not included
    labels = base_labels

ndim = len(labels)

# Plot the sampling results
samples = reader.get_chain()
fig, axes = plt.subplots(len(labels), figsize=(10, 7), sharex=True)
for i, ax in enumerate(axes):
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)
axes[-1].set_xlabel("step number")

# Save the figure with the identifier
sampling_plot_path = f'./graph/{identifier}_steps.png'
fig.savefig(sampling_plot_path)
plt.close(fig)

# Flatten the chain and plot the corner plot
flat_samples = reader.get_chain(discard=burnin, thin=thin, flat=True)
truth = None  # You can define truth values here if available
fig2 = corner.corner(flat_samples, labels=labels, truths=truth)

# Save the corner plot with the identifier
corner_plot_path = f'./graph/{identifier}_contour.png'
fig2.savefig(corner_plot_path)
plt.close(fig2)

