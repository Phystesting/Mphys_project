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


# try generation in 3 bands x-ray, optical, radio
#larger errors at larger times
#x-ray generation error ~ 0.1-0.2

#optical generation error ~ 0.02-0.2

#radio generation error ~ 0.1-0.7

