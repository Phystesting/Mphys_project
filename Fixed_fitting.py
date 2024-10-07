import numpy as np
import matplotlib.pyplot as plt
import afterglowpy as grb
from scipy.optimize import curve_fit
from scipy.optimize import least_squares

#imbed the flux calcuation within a function

def ag_py(t,thetaObs,thetaCore,thetaWing,n0,p,epsilon_e,epsilon_B):
	Z = {'jetType':     grb.jet.Gaussian,     # Gaussian jet your discrepancy
		 'specType':    grb.jet.SimpleSpec,   # Basic Synchrotron Emission Spectrum 

		 'thetaObs':    thetaObs,   # Viewing angle in radians -known
		 'E0':          1.0e53, # Isotropic-equivalent energy in erg
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

# Space time points geometrically, from 10^3 s to 10^7 s
# Time and Frequencies
ta = 1.0e-1 * grb.day2sec
tb = 1.0e3 * grb.day2sec
t = np.geomspace(ta, tb, num=100)

#initial parameter guesses
thetaObs = 0.4
thetaCore = 0.04
thetaWing = 0.06
n0 = 1.0e-4
p = 2.36
epsilon_e = 0.01
epsilon_B = 0.001


guess = [thetaObs,thetaCore,thetaWing,n0,p,epsilon_e,epsilon_B]

#set bounds for fitting parameters

b = ((0,0.0,0.0,0.0,1.8,0.0,0.0),(np.pi*0.5,0.2,0.4,0.1,3.0,0.1,0.01))

# import generated data
xdata, ydata = np.genfromtxt('./data/test_curve.txt',delimiter=',',skip_header=11, unpack=True)
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
Fnu = ag_py(t,p[0],p[1],p[2],p[3],p[4],p[5],p[6])

print(f'thetaObs: {p[0]}\n thetaCore: {p[1]}\n thetaWing: {p[2]}\n n0: {p[3]}\n p: {p[4]}\n epsilon_e: {p[5]}\n epsilon_B: {p[6]}')

tday = t * grb.sec2day

fig, ax = plt.subplots(1, 1)

ax.plot(tday,ylog,'x')
ax.plot(tday, Fnu)

ax.set(xscale='log', xlabel=r'$t$ (s)', ylabel=r'$F_\nu$[$10^{18}$ Hz] (mJy)')

fig.savefig('datafit.png')
plt.close(fig)
