import sampler_v5 as splr
import numpy as np
import afterglowpy as grb

# Define your fixed parameters separately
fixed_params = {
    
}

# Define your initial guesses for the fitting parameters
initial = {
    "thetaCore": 0.05,
    "p": 2.33,
    "log_epsilon_e": -3.0,
    "log_epsilon_B": -3.0,
    "log_E0": 54.0,
    "log_n0": 0.0,
    "thetaObs": 0.4,

}

# Define inputs
z = 0.01
d_L = 1.34e+26
xi_N = 1.0
nwalkers = 32
processes = 40
steps = 30000
jet_type = grb.jet.TopHat
filename = '/data/PROJECTS/2024-25/cjc233/Large_data/GRB9_samples.h5'

# Unpack data
time, freq, flux, Ub_err, Lb_err = np.genfromtxt('../data_generation_v1/data/GRB9_data.csv',delimiter=',',skip_header=1,unpack=True)
flux_err = Lb_err, Ub_err

#splr.run_optimization([time,freq], flux, initial,fixed_params, flux_err, xi_N, d_L, z)

if __name__ == "__main__":
    splr.run_sampling(x=[time,freq], y=flux, initial=initial,fixed_params=fixed_params, err_flux=flux_err, xi_N=xi_N, d_L=d_L, z=z, nwalkers=nwalkers, steps=steps, processes=processes,filename=filename,jet_type=jet_type)