import matplotlib.pyplot as plt
import sampler_v3 as splr
import numpy as np
from matplotlib.colors import LogNorm

thetaCore = 0.1
log_n0 = 0.0
p = 2.33
log_epsilon_e = -1.0
log_epsilon_B = -3.0
log_E0 = 51.0
thetaObs = 0.0
xi_N = 1.0
d_L = 1.43e+27
z = 0.1

unique_nu = np.geomspace(1e8, 1e20, 100)
unique_t = np.geomspace(1e1, 1e8, 100)
t = []
nu = []
for nu_value in unique_nu:
    for t_value in unique_t:
        t.append(t_value)
        nu.append(nu_value)

F = splr.Flux([t, nu], thetaCore, log_n0, p, log_epsilon_e, log_epsilon_B, log_E0, thetaObs, xi_N, d_L, z)

# Reshape the data
t = np.array(t).reshape(len(unique_nu), len(unique_t))
nu = np.array(nu).reshape(len(unique_nu), len(unique_t))
F_values = F.reshape(len(unique_nu), len(unique_t))

# Compute the numerical derivative along the frequency axis
dF_dnu = np.gradient(np.log10(F_values), np.log10(unique_nu), axis=0)

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 6), constrained_layout=True)

# Plot the flux
im1 = axes[0].pcolormesh(t, nu, F_values, shading='auto', cmap='viridis', norm=LogNorm())
axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Frequency (Hz)')
axes[0].set_title('Flux (F)')
cbar1 = fig.colorbar(im1, ax=axes[0])
cbar1.set_label('Flux (mJy)')

# Plot the slope
im2 = axes[1].pcolormesh(t, nu, dF_dnu, shading='auto', cmap='plasma')
axes[1].set_xscale('log')
axes[1].set_yscale('log')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Frequency (Hz)')
axes[1].set_title('Slope of Flux (|dlog(F)/dlog(ν)|)')
cbar2 = fig.colorbar(im2, ax=axes[1])
cbar2.set_label('|dF/dlog(ν)')



plt.show()

