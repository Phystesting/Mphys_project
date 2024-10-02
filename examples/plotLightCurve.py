import numpy as np
import matplotlib.pyplot as plt
import afterglowpy as grb
import tempfile

# For convenience, place arguments into a dict.
Z = {'jetType':     grb.jet.Gaussian,     # Gaussian jet your discrepancy
     'specType':    grb.jet.SimpleSpec,   # Basic Synchrotron Emission Spectrum your discrepancy
     'counterjet': True,

     'thetaObs':    0.3,   # Viewing angle in radians -known
     'E0':          1.0e53, # Isotropic-equivalent energy in erg
     'thetaCore':   0.005,    # Half-opening angle in radians
     'thetaWing':   0.4,    # Outer truncation angle
     'n0':          1.0e-3,    # circumburst density in cm^{-3}
     'p':           2.2,    # electron energy distribution index
     'epsilon_e':   0.1,    # epsilon_e
     'epsilon_B':   0.0001,   # epsilon_B
     'xi_N':        1.0,    # Fraction of electrons accelerated
     'd_L':         1.36e26, # Luminosity distance in cm -known
     'z':           0.01}   # redshift -known
     
Z2 = {'jetType':     grb.jet.Gaussian,     # Gaussian jet your discrepancy
     'specType':    grb.jet.SimpleSpec,   # Basic Synchrotron Emission Spectrum your discrepancy
     'counterjet': True,

     'thetaObs':    0.3,   # Viewing angle in radians -known
     'E0':          1.0e53, # Isotropic-equivalent energy in erg
     'thetaCore':   0.05,    # Half-opening angle in radians
     'thetaWing':   0.4,    # Outer truncation angle
     'n0':          1.0e-3,    # circumburst density in cm^{-3}
     'p':           2.2,    # electron energy distribution index
     'epsilon_e':   0.1,    # epsilon_e
     'epsilon_B':   0.0001,   # epsilon_B
     'xi_N':        1.0,    # Fraction of electrons accelerated
     'd_L':         1.36e26, # Luminosity distance in cm -known
     'z':           0.01}   # redshift -known

# Space time points geometrically, from 10^3 s to 10^7 s
# Time and Frequencies
ta = 1.0e-1 * grb.day2sec
tb = 1.0e3 * grb.day2sec
t = np.geomspace(ta, tb, num=100)

# Calculate flux in a single X-ray band (all times have same frequency)
nu = np.empty(t.shape)
nu[:] = 1.0e18 #x-ray
nu = np.empty(t.shape)
nu[:] = 1.0e18 #x-ray

# Calculate!

Fnu = grb.fluxDensity(t, nu, **Z)
Fnu2 = grb.fluxDensity(t, nu, **Z2)
tday = t * grb.sec2day
# Write to a file

filename = tempfile.NamedTemporaryFile(prefix="LightCurve_", suffix=".txt", dir="./data", delete=False)
print(f'saved to --{filename.name}')
with open(f'{filename.name}', 'w') as f:
    f.write("# nu " + str(nu) + '\n')
    f.write("# t(s)     Fnu(mJy)\n")
    for i in range(len(t)):
        f.write("{0:.6e} {1:.6e}\n".format(t[i], Fnu[i]))

# Plot!


fig, ax = plt.subplots(1, 1)

ax.plot(tday, Fnu)
ax.plot(tday, Fnu2)

ax.set(xscale='log', xlabel=r'$t$ (s)',
       yscale='log', ylabel=r'$F_\nu$[$10^{18}$ Hz] (mJy)')

fig.tight_layout()
graphname = tempfile.NamedTemporaryFile(prefix="LightCurve_", suffix=".png", dir="./graph", delete=False)
print(f'saved to --{graphname.name}')
fig.savefig(f'{graphname.name}')
plt.close(fig)
