import numpy as np
import matplotlib.pyplot as plt
import afterglowpy as grb
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
import corner
import os
from multiprocessing import Pool
import emcee

#imbed the flux calcuation within a function

def ag_py(t,thetaObs,thetaCore,thetaWing,n0,p,epsilon_e,epsilon_B,E0):
	Z = {'jetType':     grb.jet.Gaussian,     # Gaussian jet your discrepancy
		 'specType':    grb.jet.SimpleSpec,   # Basic Synchrotron Emission Spectrum 

		 'thetaObs':    thetaObs,   # Viewing angle in radians -known
		 'E0':          10**E0, # Isotropic-equivalent energy in erg
		 'thetaCore':   thetaCore,    # Half-opening angle in radians
		 'thetaWing':   thetaWing,    # Outer truncation angle
		 'n0':          n0,    # circumburst density in cm^{-3}
		 'p':           p,    # electron energy distribution index
		 'epsilon_e':   epsilon_e,    # epsilon_e
		 'epsilon_B':   epsilon_B,   # epsilon_B
		 'xi_N':        1.0,    # Fraction of electrons accelerated
		 'd_L':         1.36e26, # Luminosity distance in cm -known
		 'z':           0.01}   # redshift -known

	# Calculate flux in a single X-ray band (all times have same frequency)
	nu = np.empty(t.shape)
	nu[:] = 1.0e18 #x-ray
	nu = np.empty(t.shape)
	nu[:] = 1.0e18 #x-ray

	# Calculate!

	return np.log(grb.fluxDensity(t, nu, **Z))

#wrapper for scaling with priority values
def fitwrapper(coeffs, *args):
    xdata,ydata,prio = args
    return prio*(ag_py(xdata, *coeffs)-ydata)



#initial parameter guesses
thetaObs = 0.4
thetaCore = 0.04
thetaWing = 0.06
n0 = 1.0e-4
p = 2.36
epsilon_e = 0.01
epsilon_B = 0.001
E0 = 53


guess = [thetaObs, thetaCore, thetaWing, n0, p, epsilon_e, epsilon_B, E0]

#set bounds for fitting parameters

b = ((0.0,0.0,0.0,0.0,1.8,0.0,0.0,51),(np.pi*0.5,0.2,0.4,1e3,3.0,0.1,0.01,54))

# import generated data
xdata, ydata = np.genfromtxt('../data/test_curve.txt',delimiter=',',skip_header=11, unpack=True)
xlog,ylog = np.log(xdata),np.log(ydata)

#locate peak in data
peak = np.array(ydata).argmax()

#allocate priority of fitting
n = len(xdata)
prio = np.ones(n)
#prio[0] = 10
#prio[peak] = 1000
#prio[-1] = 10000

#calculate least squares fit parameters
out = least_squares(fitwrapper, x0=guess, bounds=b, args=(xdata,ylog,prio))
p = out.x


#calculated flux curve based off fitted parameters
Fnu = ag_py(xdata,p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7])

print(f'thetaObs: {p[0]}\n thetaCore: {p[1]}\n thetaWing: {p[2]}\n n0: {p[3]}\n p: {p[4]}\n epsilon_e: {p[5]}\n epsilon_B: {p[6]}\n E0: {p[7]}')

# Define the log likelihood
def log_likelihood(coeffs, xdata, ydata, prio):
    model = ag_py(xdata, *coeffs)
    return -0.5 * np.sum(prio * (model - ydata)**2)

# Define the log prior
def log_prior(coeffs):
    thetaObs, thetaCore, thetaWing, n0, p, epsilon_e, epsilon_B, E0 = coeffs
    if 0.0 < thetaObs < np.pi*0.5 and 0.0 < thetaCore < 0.2 and 0.0 < thetaWing < 0.4 and 0.0 < n0 < 1e3 and 1.8 < p < 3.0 and 0.0 < epsilon_e < 0.1 and 0.0 < epsilon_B < 0.01 and 51.0 < E0 < 54.0:
        return 0.0
    return -np.inf

# Define the log probability function
def log_probability(coeffs, xdata, ydata, prio):
    lp = log_prior(coeffs)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(coeffs, xdata, ydata, prio)

# Initialize walkers
ndim = len(guess)  # Number of parameters to fit
nwalkers = 32  # Number of walkers (you can increase this)

# Initial guess for the walkers (slightly perturbed from the initial guess)
pos = guess + 1e-4 * np.random.randn(nwalkers, ndim)

# Run the MCMC sampler
nsteps = 1000  # Number of steps (can increase for more accurate results)
with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,Pool=pool, args=(xdata, ylog, prio))
    sampler.run_mcmc(pos, nsteps, progress=True)

# Get the chain of samples
samples = sampler.get_chain(discard=100, thin=15, flat=True)

# Generate corner plot
fig = corner.corner(samples, labels=["thetaObs", "thetaCore", "thetaWing", "n0", "p", "epsilon_e", "epsilon_B", "E0"],
                    truths=[thetaObs, thetaCore, thetaWing, n0, p, epsilon_e, epsilon_B, E0])
plt.savefig('corner_plot.png')
plt.show()
