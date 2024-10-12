import numpy as np
import afterglowpy as grb
import emcee
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import corner
import os
from multiprocessing import Pool
import time

#set up afterglowpy as a function suitable for fitting
def ag_py(t,thetaObs,thetaCore,thetaWing,n0,p,epsilon_e,epsilon_B,E0):
    Z = {'jetType':     grb.jet.Gaussian,     # Gaussian jet your discrepancy
		 'specType':    grb.jet.SimpleSpec,   # Basic Synchrotron Emission Spectrum 

		 'thetaObs':    thetaObs,   # Viewing angle in radians -known
		 'E0':          10**E0, # Isotropic-equivalent energy in erg
		 'thetaCore':   thetaCore,    # Half-opening angle in radians
		 'thetaWing':   thetaCore + (0.4-thetaCore)*thetaWing,    # Outer truncation angle
		 'n0':          10**n0,    # circumburst density in cm^{-3}
		 'p':           p,    # electron energy distribution index
		 'epsilon_e':   10**epsilon_e,    # epsilon_e
		 'epsilon_B':   10**epsilon_B,   # epsilon_B
		 'xi_N':        1.0,    # Fraction of electrons accelerated
		 'd_L':         1.36e26, # Luminosity distance in cm -known
		 'z':           0.01}   # redshift -known

	# Calculate flux in a single X-ray band (all times have same frequency)
    nu = np.empty(t.shape)
    nu[:] = 1.0e18 #x-ray

	# Calculate!
    return np.log(grb.fluxDensity(t, nu, **Z))

# define fitting residual
def residual(theta,x,y):
    thetaObs,thetaCore,thetaWing,n0,p,epsilon_e,epsilon_B,E0 = theta
    #print(E0, thetaObs, thetaCore, thetaWing, n0, p, epsilon_e, epsilon_B)
    model = ag_py(x,thetaObs,thetaCore,thetaWing,n0,p,epsilon_e,epsilon_B,E0)
    if not np.all(np.isfinite(model)):
        print(E0, thetaObs, thetaCore, thetaWing, n0, p, epsilon_e, epsilon_B)
        raise ValueError("ag_py returned non-finite values.")
    #print(sum(((y - model)**2) / abs(model)))
    return sum(((y - model)**2) / abs(model))

# Import data    
x, ydata = np.genfromtxt('../data/test_curve.txt',delimiter=',',skip_header=11, unpack=True)
y = np.log(ydata)


# Define bounds: E0 > 0 and log(f) unrestricted
bounds = [(0, np.pi*0.5),(0.01, 0.4), (0.0, 1.0),(-4.0, 3.0), (2.1, 2.8),(-4.0, 0.0), (-4.0, 0.0), (51.0, 54.0)] 

# definine inital parameters
E0 = 54.0
thetaObs = 0.1
thetaCore = 0.3
thetaWing = 0.5
n0 = -1.0
p = 2.5
epsilon_e = -2.0
epsilon_B = -4.0
np.random.seed(42)
initial = np.array([thetaObs,thetaCore,thetaWing,n0,p,epsilon_e,epsilon_B,E0])


#minimize the residual
start = time.time()
soln = minimize(residual, initial, args=(x, y), bounds=bounds,method='SLSQP')
end = time.time()
serial_time = end - start
print("Serial took {0:.1f} seconds".format(serial_time))

thetaObs_ml,thetaCore_ml,thetaWing_ml,n0_ml,p_ml,epsilon_e_ml,epsilon_B_ml,E0_ml = soln.x

y_ml = ag_py(x,thetaObs_ml,thetaCore_ml,thetaWing_ml,n0_ml,p_ml,epsilon_e_ml,epsilon_B_ml,E0_ml)

print("Maximum likelihood estimates:")
print("E0 = {0:.3f}".format(E0_ml))
print("thetaObs = {0:.3f}".format(thetaObs_ml))
print("thetaCore = {0:.3f}".format(thetaCore_ml))
print("thetaWing = {0:.3f}".format(thetaCore_ml + (0.4-thetaCore_ml) * thetaWing_ml))
print("n0 = {0:.4f}".format(10**n0_ml))
print("p = {0:.3f}".format(p_ml))
print("epsilon_e = {0:.4f}".format(10**epsilon_e_ml))
print("epsilon_B = {0:.4f}".format(10**epsilon_B_ml))

Fnu = ag_py(x,thetaObs_ml,thetaCore_ml,thetaWing_ml,n0_ml,p_ml,epsilon_e_ml,epsilon_B_ml,E0_ml)



fig, ax = plt.subplots(1, 1)

ax.plot(x,y,'x')
ax.plot(x, Fnu)

ax.set(xscale='log', xlabel=r'$t$ (s)', ylabel=r'$F_\nu$[$10^{18}$ Hz] (mJy)')

fig.savefig('datafit.png')
plt.close(fig)
