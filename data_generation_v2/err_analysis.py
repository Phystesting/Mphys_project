import afterglowpy as grb
import emcee
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
import re

time, freq, flux, err = np.genfromtxt('./data/990510_data.csv',delimiter=',',unpack=True,skip_header=1)

unique_freq = np.unique(freq)

min_flux = []
mean_err = []
err_var = []

for nu_vals in unique_freq:
    mask = freq == nu_vals
    min_flux.append(min(flux[mask]))
    mean_err.append(np.mean(err[mask]))
    err_var.append(np.var(err[mask]))
i=0
while i < len(unique_freq):
    print(unique_freq[i],mean_err[i],err_var[i],mean_err[i],err_var[i],min_flux[i])
    i = i + 1
