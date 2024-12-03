import sampler_v4 as splr
import numpy as np
import afterglowpy as grb

# Define your fixed parameters separately
fixed_params = {
    "thetaObs": 0.4, #based off ryan paper
}

# Define your initial guesses for the fitting parameters
initial = {
    "thetaCore": 0.1,
    "p": 2.14,
    "log_epsilon_e": -1.0,
    "log_epsilon_B": -3.0,
    "log_E0": 51.0,
    "log_n0": -4.0,

}

# Define inputs
z = 0.0099
d_L = 1.327e+26
xi_N = 1.0
nwalkers = 32
processes = 40
steps = 50000
jet_type = grb.jet.Gaussian
filename = '/data/PROJECTS/2024-25/cjc233/Large_data/170817_GACO_samples.h5'

# Unpack data
time, freq, flux, UB_err, LB_err = np.genfromtxt('../data_generation_v1/data/170817_data.csv',delimiter=',',skip_header=1,unpack=True)
flux_err = LB_err, UB_err

#result = splr.run_optimization([time_values,freq_values], flux_values, initial,fixed_params, flux_err, xi_N, d_L, z)


if __name__ == "__main__":
    splr.run_sampling(x=[time,freq], y=flux, initial=initial,fixed_params=fixed_params, err_flux=flux_err, xi_N=xi_N, d_L=d_L, z=z, nwalkers=nwalkers, steps=steps, processes=processes,filename=filename,jet_type=jet_type)

