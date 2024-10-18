import numpy as np
import matplotlib.pyplot as plt
import afterglowpy as grb
import tempfile

# For convenience, place arguments into a dict.
Z = {'jetType':     grb.jet.Gaussian,     # Gaussian jet your discrepancy
     'specType':    grb.jet.SimpleSpec,   # Basic Synchrotron Emission Spectrum your discrepancy
     'counterjet': True,

     'thetaObs':    0.559,   # Viewing angle in radians -known
     'E0':          7.485e52, # Isotropic-equivalent energy in erg
     'thetaCore':   0.09,    # Half-opening angle in radians
     'thetaWing':   1.079,    # Outer truncation angle
     'n0':          0.07714,    # circumburst density in cm^{-3}
     'p':           2.372,    # electron energy distribution index
     'epsilon_e':   0.1472,    # epsilon_e
     'epsilon_B':   0.00004,   # epsilon_B
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


# Calculate!

Fnu = grb.fluxDensity(t, nu, **Z)

tday = t * grb.sec2day

# Plot!


fig, ax = plt.subplots(1, 1)

ax.plot(tday, Fnu)

ax.set(xscale='log', xlabel=r'$t$ (s)',
       yscale='log', ylabel=r'$F_\nu$[$10^{18}$ Hz] (mJy)')

fig.tight_layout()
graphname = tempfile.NamedTemporaryFile(prefix="LightCurve_", suffix=".png", dir="./graph", delete=False)
print(f'saved to --{graphname.name}')
fig.savefig(f'{graphname.name}')
plt.close(fig)
