import numpy as np
import afterglowpy as grb
import emcee
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import curve_fit
import corner
import os
import sys
import multiprocessing
from multiprocessing import Pool
import time
from functools import partial
import concurrent.futures
import sampler as splr
import random as rnmb

#start by generating a filled out uniform data rectangle over a reasonable bounds
t = np.geomspace(1e4,1e6,50)
nu = np.array([4.8e09,8.6e09,3.72e14,4.55e14,5.45e14,1.08e18])
x = [t,nu]

F_start = np.array(splr.Flux(x,0.095,1.0,2.134,-0.1,-4,53,0.0,1.0,39.755*9.461e26,1.619))
F = np.full((len(t), len(nu)), np.nan)
err = np.zeros((len(t),len(nu)))

# try generation in 3 bands x-ray, optical, radio
"""
#larger errors at larger times
for t_idx, t_value in enumerate(t):
    err[t_idx,:] = err[t_idx,:] + t_value/max(t)
"""
#thinning the data set detection threshold for each freq band 
#high frequency produces early time data

#medium frequency produces middle time with a normal distribution fall off

#low frequency produces late time data set detection threshold for each freq band 


for nu_idx, nu_value in enumerate(nu):
    if nu_value > 1e16:
        # High frequency
        samples = rnmb.randint(1,int(0.5*len(t)))
    elif 1e10 < nu_value < 1e16:
        # Optical, UV, and Infrared
        samples = rnmb.randint(1,len(t)-1)  
    elif nu_value < 1e10:
        # Radio
        samples = rnmb.randint(1,int(0.3*len(t)))
        
    for i in range(samples):
        if nu_value > 1e16:
            index = rnmb.randint(0,int(0.5*len(t)))
        elif 1e10 < nu_value < 1e16:
            index = rnmb.randint(0,len(t)-1)
        elif nu_value < 1e10:
            index = rnmb.randint(int(0.5*len(t)),len(t)-1)
        
        F[index, nu_idx] = F_start[index,nu_idx]

displacement = np.zeros_like(F)


for nu_idx, nu_value in enumerate(nu):
    if nu_value > 1e16:
        # High frequency error generation ~ 0.1-0.2
        err[:, nu_idx] += np.random.normal(0.15, 0.05, size=len(t))
        displacement[:, nu_idx] = np.random.normal(0.0, 0.1, size=len(t))
        
    elif 1e10 < nu_value < 1e16:
        # Optical, UV, and Infrared generation error ~ 0.02-0.2
        err[:, nu_idx] += np.random.normal(0.11, 0.05, size=len(t))
        displacement[:, nu_idx] = np.random.normal(0.0, 0.1, size=len(t))
        
    elif nu_value < 1e10:
        # Radio generation error ~ 0.1-0.7
        err[:, nu_idx] += np.random.normal(0.4, 0.15, size=len(t))
        displacement[:, nu_idx] = np.random.normal(0.0, 0.35, size=len(t))


# Generate F_err by adding displacement directly to F
F_err = F + displacement


for freq_idx, nu_value in enumerate(nu):
    plt.errorbar(np.log10(t),F_err[:,freq_idx],yerr=abs(err[:,freq_idx]),fmt='.', label=f'{nu_value:.2e} Hz')
plt.legend()
plt.show()

