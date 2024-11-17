import sampler_v2 as splr
import numpy as np

# Define your fixed parameters separately
fixed_params = {
    "thetaObs": 0.0,
}

# Define your initial guesses for the fitting parameters
initial = {
    "thetaCore": 0.1,
    "p": 2.33,
    "log_epsilon_e": -1.0,
    "log_epsilon_B": -3.0,
    "log_E0": 51.0,
    "log_n0": 0.0,

}

# Define inputs
z = 2.0
d_L = 4.88e28
xi_N = 1.0
nwalkers = 32
processes = 40
steps = 600000
filename = '/data/PROJECTS/2024-25/cjc233/Large_data/post_jb_samples.h5'

# Unpack data
time, freq, flux, flux_err = np.genfromtxt('../Data_Generation/data/post_jb_data.csv',delimiter=',',skip_header=1,unpack=True)
#splr.run_optimization([time,freq], flux, initial,fixed_params, flux_err, xi_N, d_L, z)

if __name__ == "__main__":
    splr.run_sampling(x=[time,freq], y=flux, initial=initial,fixed_params=fixed_params, err_flux=flux_err, xi_N=xi_N, d_L=d_L, z=z, nwalkers=nwalkers, steps=steps, processes=processes,filename=filename)

