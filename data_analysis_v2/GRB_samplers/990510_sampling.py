import sampler_v7 as splr
import numpy as np
import afterglowpy as grb

# Define your fixed parameters separately
fixed_params = {

}

# Define your initial guesses for the fitting parameters
initial = {
    "thetaCore": 0.1,
    "p": 2.33,
    "log_epsilon_e": -1.0,
    "log_epsilon_B": -3.0,
    "log_E0": 51.0,
    "log_n0": 0.0,
    "thetaObs": 0.0,
}

# Define inputs
identifier = '990510' # The GRB identifier
fit_type = 'TH' # GA or TH for gaussian or tophat
z = 1.619
d_L = 3.76e28
xi_N = 1.0
nwalkers = 32
processes = 40
steps = 30000
if fit_type == 'GA':
    jet_type = grb.jet.Gaussian
else:
    jet_type = grb.jet.TopHat
# File location to generate the samples in
filename = f'/data/PROJECTS/2024-25/cjc233/samples/{identifier}T_{fit_type}_samples.h5'
datafile = f'../../data_generation_v2/data/{identifier}_data.csv'
# Unpack data
time, freq, flux, flux_err = np.genfromtxt(datafile,delimiter=',',skip_header=1,unpack=True)
flux_err = flux_err, flux_err

#result = splr.run_optimization([time_values,freq_values], flux_values, initial,fixed_params, flux_err, xi_N, d_L, z)


if __name__ == "__main__":
    splr.run_sampling(x=[time,freq], y=flux, initial=initial,fixed_params=fixed_params, err_flux=flux_err, xi_N=xi_N, d_L=d_L, z=z, nwalkers=nwalkers, steps=steps, processes=processes,filename=filename,jet_type=jet_type)

