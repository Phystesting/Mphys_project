import sampler_v7 as splr
import numpy as np
import afterglowpy as grb

# Define your fixed parameters separately
fixed_params = {

}

# Define your initial guesses for the fitting parameters
initial = {
    "thetaCore": 0.1,
    "p": 2.3,
    "log_epsilon_e": -1.0,
    "log_epsilon_B": -3.0,
    "log_E0": 54.0,
    "log_n0": 2.0,
    "thetaObs": 0.0,
}

# Define inputs
identifier = 'GRB1' # The GRB identifier
fit_type = 'GA' # GA or TH for gaussian or tophat
z = 0.01
d_L = 1.34e26
xi_N = 1.0
nwalkers = 32
processes = 40
steps = 50000
if fit_type == 'GA':
    jet_type = grb.jet.Gaussian
else:
    jet_type = grb.jet.TopHat
# File location to generate the samples in
filename = f'/data/PROJECTS/2024-25/cjc233/samples/{identifier}_{fit_type}_samples.h5'
datafile = f'../../data_generation_v2/data/{identifier}_data.csv'
# Unpack data
time, freq, flux, UB_err, LB_err = np.genfromtxt(datafile,delimiter=',',skip_header=1,unpack=True)
flux_err = LB_err, UB_err

#result = splr.run_optimization([time_values,freq_values], flux_values, initial,fixed_params, flux_err, xi_N, d_L, z)


if __name__ == "__main__":
    splr.run_sampling(x=[time,freq], y=flux, initial=initial,fixed_params=fixed_params, err_flux=flux_err, xi_N=xi_N, d_L=d_L, z=z, nwalkers=nwalkers, steps=steps, processes=processes,filename=filename,jet_type=jet_type)

