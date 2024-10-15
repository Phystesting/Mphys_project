import numpy as np
import matplotlib.pyplot as plt
import afterglowpy as grb
from scipy.optimize import curve_fit
from scipy.optimize import least_squares

#imbed the flux calcuation within a function

def ag_py(t,thetaObs,E0,thetaCore,thetaWing,n0,p,epsilon_e,epsilon_B,xi_N,d_L,z):
	Z = {'jetType':     grb.jet.Gaussian,     # Gaussian jet your discrepancy
		 'specType':    grb.jet.SimpleSpec,   # Basic Synchrotron Emission Spectrum 

		 'thetaObs':    thetaObs,   # Viewing angle in radians -known
		 'E0':          E0*1.0e53, # Isotropic-equivalent energy in erg
		 'thetaCore':   thetaCore,    # Half-opening angle in radians
		 'thetaWing':   thetaWing,    # Outer truncation angle
		 'n0':          n0,    # circumburst density in cm^{-3}
		 'p':           p,    # electron energy distribution index
		 'epsilon_e':   epsilon_e,    # epsilon_e
		 'epsilon_B':   epsilon_B,   # epsilon_B
		 'xi_N':        xi_N,    # Fraction of electrons accelerated
		 'd_L':         d_L*1.36e26, # Luminosity distance in cm -known
		 'z':           z}   # redshift -known

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
thetaObs = 1.4
E0 = 10
thetaCore = 0.02
thetaWing = 0.6
n0 = 1.0e-5
p = 2.1
epsilon_e = 0.54
epsilon_B = 0.0231
xi_N = 0.8
d_L = 0.1
z = 0.3
b_lim = np.zeros((4,2))

guess = [thetaObs,E0,thetaCore,thetaWing,n0,p,epsilon_e,epsilon_B,xi_N,d_L,z]

#set bounds for fitting parameters

b = ((0,1e-2,0.0,0.0,0.0,1.8,0.0,0.0,0.0,1.0e-2,0),(np.pi*0.5,10.0,np.pi*0.25,np.pi*0.5,0.1,3.0,1.0,1.0,1.0,100.0,1.0))

# import generated data and produce a log version
xdata, ydata = np.genfromtxt('./data/test_curve.txt',delimiter=',',skip_header=11, unpack=True)
logx,logy = np.log(xdata),np.log(ydata)

#locate peak in data
peak = np.array(ydata).argmax()

#allocate priority of fitting
n = len(xdata)
prio = np.ones(n)
#prio[0] = 1000
#prio[peak] = 1000
#prio[-1] = 1000

#calculate least squares fit parameters
out = least_squares(fitwrapper, x0=guess, bounds=b, args=(xdata,logy,prio))
p = out.x


#calculated flux curve based off fitted parameters
Fnu = ag_py(t,p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10])

print(f'thetaObs: {p[0]}\n E0: {p[1]*1.0e53}\n thetaCore: {p[2]}\n thetaWing: {p[3]}\n n0: {p[4]}\n p: {p[5]}\n epsilon_e: {p[6]}\n epsilon_B: {p[7]}\n xi_N: {p[8]}\n d_L: {p[9]*1.36e26}\n z: {p[10]}')

tday = t * grb.sec2day

fig, ax = plt.subplots(1, 1)

ax.plot(tday,logy,'x')
ax.plot(tday, Fnu)

ax.set(xscale='log', xlabel=r'$t$ (s)', ylabel=r'$F_\nu$[$10^{18}$ Hz] (mJy)')

fig.savefig('datafit.png')
plt.close(fig)
