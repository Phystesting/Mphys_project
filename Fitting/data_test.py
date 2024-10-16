import numpy as np
import afterglowpy as grb
import emcee
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import corner
import os
from multiprocessing import Pool
import time
import param_sampler as ps

x, ydata = np.genfromtxt('../data/test_curve.txt', delimiter=',', skip_header=11, unpack=True)
y = np.log(ydata)

reader = emcee.backends.HDFBackend('./data/test_sample.h5')
truth=[0.3, 0.05, 1.0, -3.0, 2.3, -1.0, -4.0, 53.0]
ps.fit(x,y,reader)
