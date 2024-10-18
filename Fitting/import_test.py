import numpy as np
import afterglowpy as grb
import emcee
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import curve_fit
import corner
import os
from multiprocessing import Pool
import time
import tfreq_sampler as ts
import Jet_sampler as js

initial = np.array([0.1,0.1,0.0,2.3,-1,-3,53])
truth = [0.1,0.1,0.0,2.3,-1,-3,53]


x, ydata = np.genfromtxt('./data/test_curve.txt', delimiter=',', skip_header=11, unpack=True)
y = np.log(ydata)
x2, ydata2 = np.genfromtxt('./data/test_spec.txt', delimiter=',', skip_header=2, unpack=True)
y2 = np.log(ydata2)
yerr = abs(y*0.05)
yerr2 = abs(y2*0.05)
if __name__ == '__main__':
    
    sampler = js.run_sampling(x,y,initial,yerr=yerr,processes=10,filename='./data/time1000_noError.h5',steps=1000,datatype=0,fixed=1e18,d_L=1e28,z=0.55)
    #sampler = js.run_sampling(x2,y2,initial,processes=10,filename='./data/spec5000_noError.h5',steps=1000,datatype=1,fixed=86400,d_L=1e28,z=0.55)
    #js.optimize(x2,y2,initial,datatype=1,fixed=86400,d_L=1e28,z=0.55)
    
    file_path = './data/time1000_noError.h5'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The HDF5 file '{file_path}' does not exist.")
    backend = emcee.backends.HDFBackend(file_path)
    if backend.iteration == 0:
        raise ValueError("The HDF5 file is empty. No data was stored.")
    reader = emcee.backends.HDFBackend('./data/time1000_noError.h5')
    js.plot_results(reader,truth)
    
    #js.fit(x,y,yerr,reader,1e18,1,1e28,0.55)
