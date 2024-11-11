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
nu = np.geomspace(1e9,1e20,4)
x = [t,nu]

F = np.array(splr.Flux(x,0.197,0,2.422,-1,-2,50,0.0,1.0,9.461e26,1.619))

err = np.zeros((len(t),len(nu)))

# try generation in 3 bands x-ray, optical, radio
#larger errors at larger times
for t_idx, t_value in enumerate(t):
    err[t_idx,:] = err[t_idx,:] + t_value/max(t)

#optical generation error ~ 0.02-0.2

for nu_idx, nu_value in enumerate(nu):
    #x-ray generation error ~ 0.1-0.2
    if nu_value > 1e16:
        err[:,nu_idx] = err[:,nu_idx] + float(rnmb.randint(100,200))/1000
    #optical UV and Infrared generation error ~ 0.02-0.2
    if 1e10 < nu_value < 1e16:
        err[:,nu_idx] = err[:,nu_idx] + float(rnmb.randint(20,200))/1000
    #radio generation error ~ 0.1-0.7
    if nu_value < 1e10:
        err[:,nu_idx] = err[:,nu_idx] + float(rnmb.randint(100,700))/1000

F_err = F + err*rnmb.randint(-1,1)

print(rnmb.randint(-1,1))
print(rnmb.randint(-1,1))
print(rnmb.randint(-1,1))
print(rnmb.randint(-1,1))
print(rnmb.randint(-1,1))
for freq_idx, nu_value in enumerate(nu):
    plt.errorbar(np.log10(t),F[:,freq_idx],yerr=err[:,freq_idx],fmt='.', label=f'{nu_value:.2e} Hz')
plt.legend()
plt.show()

