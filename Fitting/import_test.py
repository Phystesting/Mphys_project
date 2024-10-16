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

initial = np.array([0.1, 0.3, 0.5, -1.0, 2.5, -2.0, -4.0, 54.0])

x, ydata = np.genfromtxt('../data/test_curve.txt', delimiter=',', skip_header=11, unpack=True)
y = np.log(ydata)

sampler = ps.run_sampling(x,y,initial)

