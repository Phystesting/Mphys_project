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

mid_mean, mid_std = 100.0, 50.0   # Mean ~0.11, SD to keep it roughly within 0.02-0.2
radio_mean, radio_std = 5.0, 5.0                   # Mean ~0.4, SD to keep it roughly within 0.1-0.7
high_mean, high_std = 20.0, 10.0  

for nu_idx, nu_value in enumerate(nu):
    if nu_value > 1e16:
        # High frequency
        samples = round(np.random.normal(0.3*len(t), 0.15*len(t)))
        
        
    elif 1e10 < nu_value < 1e16:
        # Optical, UV, and Infrared
        samples = round(np.random.normal(0.5*len(t), 0.25*len(t)))
        
    elif nu_value < 1e10:
        # Radio
        samples = round(np.random.normal(0.1*len(t), 0.05*len(t)))
    samples = max(0, min(samples, len(F_start[:, nu_idx]) - 1))    
    print(samples)
    for i in range(samples):
        if nu_value > 1e16:
            index = round(np.random.normal(0.8*len(F_start[:,nu_idx]), 0.5*len(F_start[:,nu_idx])))
            
        
        
        elif 1e10 < nu_value < 1e16:
            index = round(np.random.normal(0.5*len(F_start[:,nu_idx]), 0.5*len(F_start[:,nu_idx])))
            if index > len(F_start[:,nu_idx])-1:
                index = len(F_start[:,nu_idx]) - 1
        
        elif nu_value < 1e10:
            index = round(np.random.normal(0.2*len(F_start[:,nu_idx]), 0.5*len(F_start[:,nu_idx])))
        
        if index < 0:
                index = 0
        elif index > len(F_start[:,nu_idx])-1:
                index = len(F_start[:,nu_idx]) - 1
                
        index = max(0, min(index, len(F_start[:, nu_idx]) - 1))
        F[index, nu_idx] = F_start[index,nu_idx]

#thinning the data set detection threshold for each freq band 
#high frequency produces early time data

#medium frequency produces middle time with a normal distribution fall off

#low frequency produces late time data set detection threshold for each freq band 


