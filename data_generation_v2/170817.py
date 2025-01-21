import matplotlib.pyplot as plt
import numpy as np
import afterglowpy as grb

# Function to calculate the model
def Flux(x, thetaCore, log_n0, p, log_epsilon_e, log_epsilon_B, log_E0, thetaObs, xi_N, d_L, z):
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


# Load data points for GW170817
time, freq, flux, Ub_err, Lb_err = np.genfromtxt('./data/170817_data_reduced.csv', delimiter=',', unpack=True, skip_header=1)
unique_freq = np.unique(freq)

# Set parameters for the fake light curve
thetaCore = 0.055      # core angle in radians
log_n0 = -2.3          # log10 of ambient density (cm^-3)
p = 2.16               # electron distribution power-law index
log_epsilon_e = -3.1    # log10 of epsilon_e
log_epsilon_B = -3.6    # log10 of epsilon_B
log_E0 = 54           # log10 of isotropic energy (ergs)
thetaObs = 0.33       # observer angle in radians
xi_N = 1.0            # fraction of electrons accelerated
z = 0.0099
d_L = 1.327e+26       # redshift

# Generate a fake light curve
fake_time = np.logspace(np.log10(time.min()), np.log10(time.max()), 100)  # Time grid
fake_flux = []
for nu in unique_freq:
    x = [fake_time, np.full_like(fake_time, nu)]
    model_flux = Flux(x, thetaCore, log_n0, p, log_epsilon_e, log_epsilon_B, log_E0, thetaObs, xi_N, d_L, z)
    fake_flux.append((nu, model_flux))

# Plotting
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Plot data points
for nu in unique_freq:
    mask = freq == nu
    ax.errorbar(time[mask], flux[mask], yerr=(Ub_err[mask], Lb_err[mask]), fmt='.', label=f"Data {nu:.1e} Hz")

# Plot fake light curve
for nu, model_flux in fake_flux:
    ax.plot(fake_time, model_flux, label=f"Model {nu:.1e} Hz", linestyle='--')

ax.set(
    xscale="log", yscale="log",
    xlabel=r"$t$ (s)", ylabel=r"$F_\nu$ (mJy)"
)
ax.legend()
plt.show()
