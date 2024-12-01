import matplotlib.pyplot as plt
import afterglowpy as grb
import numpy as np
from matplotlib.colors import LogNorm
import time

def Flux(x, thetaCore, log_n0, p, log_epsilon_e, log_epsilon_B, log_E0, thetaObs, xi_N, d_L, z):
    # Function to calculate the model
    Z = {
        'jetType': grb.jet.GaussianCore,
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

log_E0 = 52
thetaCore = 0.05
thetaObs = 0.1
thetaWing =  4*thetaCore
log_n0 = 0.0
p = 2.33
log_epsilon_e = -0.5
log_epsilon_B = -2.0
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
start = time.time()
F = Flux([t, nu], thetaCore, log_n0, p, log_epsilon_e, log_epsilon_B, log_E0, thetaObs, xi_N, d_L, z)
end = time.time()
serial_time = end - start
print(serial_time)

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

plt.savefig('./graph/GRB3_colour.png')

plt.show()

