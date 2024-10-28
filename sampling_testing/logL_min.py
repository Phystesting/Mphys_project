import numpy as np
import afterglowpy as grb
import emcee
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import corner
import os
from multiprocessing import Pool
import time

def ag_py(t, thetaObs, thetaCore, thetaWing, n0, p, epsilon_e, epsilon_B, E0):
    # Function to calculate the model
    Z = {
        'jetType': grb.jet.Gaussian,
        'specType': grb.jet.SimpleSpec,
        'thetaObs': thetaObs,
        'E0': 10**E0,
        'thetaCore': thetaCore,
        'thetaWing': thetaCore + (np.pi*0.5 - thetaCore) * thetaWing,
        'n0': 10**n0,
        'p': p,
        'epsilon_e': 10**epsilon_e,
        'epsilon_B': 10**epsilon_B,
        'xi_N': 1.0,
        'd_L': 1.36e26,
        'z': 0.01
    }
    nu = np.empty(t.shape)
    nu[:] = 1.0e18
    return np.log(grb.fluxDensity(t, nu, **Z))

def log_likelihood(theta,x,y):
    # Function for log-likelihood calculation
    
    thetaObs, thetaCore, thetaWing, n0, p, epsilon_e, epsilon_B, E0 = theta
    model = ag_py(x, thetaObs, thetaCore, thetaWing, n0, p, epsilon_e, epsilon_B, E0)
    if not np.all(np.isfinite(model)):
        print( thetaObs, thetaCore, thetaWing, n0, p, epsilon_e, epsilon_B, E0)
        raise ValueError("ag_py returned non-finite values.")
    sigma2 = model**2
    return -0.5 * np.sum(((y - model) ** 2 / sigma2))#+0.01*np.log(sigma2)



# Import data and perform initial setup
x, ydata = np.genfromtxt('../data/test_curve.txt', delimiter=',', skip_header=11, unpack=True)
y = np.log(ydata)

# Define initial parameter values
initial = np.array([0.1, 0.01, 0.5, -1.0, 2.1, -2.0, -4.0, 54.0])
bounds = [(0, 0.8), (0.01, np.pi*0.5), (0.0, 1.0), (-10.0, 10.0), 
        (2.1, 5.0), (-5.0, 0.0), (-5.0, 0.0), (45.0, 57.0)]

# Minimize the residual
nll = lambda *args: -log_likelihood(*args)
start = time.time()
soln = minimize(nll, initial, args=(x, y), bounds=bounds,method='SLSQP')
end = time.time()
serial_time = end - start
print("Most Probable parameters identified in {0:.1f} seconds".format(serial_time))
thetaObs_ml,thetaCore_ml,thetaWing_ml,n0_ml,p_ml,epsilon_e_ml,epsilon_B_ml,E0_ml = soln.x
print("Chi squared estimates:")
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

fig.savefig('./graph/ChiFit.png')
plt.close(fig)