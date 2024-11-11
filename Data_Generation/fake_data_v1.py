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

F = np.array(splr.Flux(x,0.29,3,5,-1,-5,57,0.0,1.0,9.461e26,1.619))

err = np.zeros((len(t),len(nu)))

# try generation in 3 bands x-ray, optical, radio
"""
#larger errors at larger times
for t_idx, t_value in enumerate(t):
    err[t_idx,:] = err[t_idx,:] + t_value/max(t)
"""
#optical generation error ~ 0.02-0.2

# Define mean and standard deviation for each range based on the comments
optical_uv_ir_mean, optical_uv_ir_std = 0.11, 0.05   # Mean ~0.11, SD to keep it roughly within 0.02-0.2
radio_mean, radio_std = 0.4, 0.15                    # Mean ~0.4, SD to keep it roughly within 0.1-0.7
high_freq_mean, high_freq_std = 0.15, 0.05           # Mean ~0.15, SD to stay in ~0.1-0.2

for nu_idx, nu_value in enumerate(nu):
    if nu_value > 1e16:
        # High frequency error generation ~ 0.1-0.2
        err[:, nu_idx] += np.random.normal(high_freq_mean, high_freq_std)
        
    elif 1e10 < nu_value < 1e16:
        # Optical, UV, and Infrared generation error ~ 0.02-0.2
        error_value = np.random.normal(optical_uv_ir_mean, optical_uv_ir_std)
        err[:, nu_idx] += error_value
        print(error_value)
        
    elif nu_value < 1e10:
        # Radio generation error ~ 0.1-0.7
        err[:, nu_idx] += np.random.normal(radio_mean, radio_std)

displacement = np.zeros_like(F)
optical_uv_ir_mean, optical_uv_ir_std = 0.0, 0.2   # Mean ~0.11, SD to keep it roughly within 0.02-0.2
radio_mean, radio_std = 0.0, 0.7                   # Mean ~0.4, SD to keep it roughly within 0.1-0.7
high_freq_mean, high_freq_std = 0.0, 0.2           # Mean ~0.15, SD to stay in ~0.1-0.2

# Populate displacement based on frequency bands
for nu_idx, nu_value in enumerate(nu):
    if nu_value > 1e16:
        # High frequency displacement ~ 0.1-0.2
        displacement[:, nu_idx] = np.random.normal(high_freq_mean, high_freq_std, size=len(t))
        
    elif 1e10 < nu_value < 1e16:
        # Optical, UV, and Infrared displacement ~ 0.02-0.2
        displacement[:, nu_idx] = np.random.normal(optical_uv_ir_mean, optical_uv_ir_std, size=len(t))
        
    elif nu_value < 1e10:
        # Radio displacement ~ 0.1-0.7
        displacement[:, nu_idx] = np.random.normal(radio_mean, radio_std, size=len(t))

# Generate F_err by adding displacement directly to F
F_err = F + displacement


for freq_idx, nu_value in enumerate(nu):
    plt.errorbar(np.log10(t),F_err[:,freq_idx],yerr=err[:,freq_idx],fmt='.', label=f'{nu_value:.2e} Hz')
plt.legend()
plt.show()

