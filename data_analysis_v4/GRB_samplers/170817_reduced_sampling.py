import sampler_v13 as splr
import numpy as np
import afterglowpy as grb

# Define your fixed parameters separately
fixed_params = {

}

# Define your initial guesses for the fitting parameters
initial = {
    "thetaCore": 0.06,
    "p": 2.3,
    "log_epsilon_e": -1.0,
    "log_epsilon_B": -4.0,
    "log_E0": 52.0,
    "log_n0": -3,
    "thetaObs": 0.4,
}

# Define inputs
identifier = '170817' # The GRB identifier
extra = 'reduced3'
z = 0.0099
d_L = 1.327e+26
xi_N = 1.0
nwalkers = 32
processes = 40
steps = 500000

jet_type_GA = grb.jet.Gaussian
jet_type_TH = grb.jet.TopHat

# File location to generate the samples in
filenameGA = f'/data/PROJECTS/2024-25/cjc233/samples_v10/{identifier}_GA{extra}_samples.h5'
filenameTH = f'/data/PROJECTS/2024-25/cjc233/samples_v10/{identifier}_TH{extra}_samples.h5'
datafile = f'../../data_generation_v2/data/{identifier}_data_reduced.csv'
# Unpack data
time, freq, flux, UB_err, LB_err = np.genfromtxt(datafile,delimiter=',',skip_header=1,unpack=True)
flux_err = LB_err, UB_err

#result = splr.run_optimization([time,freq], flux, initial,fixed_params, flux_err, xi_N, d_L, z,jet_type_GA)


if __name__ == "__main__":
    #splr.run_sampling(x=[time,freq], y=flux, initial=initial,fixed_params=fixed_params, err_flux=flux_err, xi_N=xi_N, d_L=d_L, z=z, nwalkers=nwalkers, steps=steps, processes=processes,filename=filenameTH,jet_type=jet_type_TH)
    splr.run_sampling(x=[time,freq], y=flux, initial=initial,fixed_params=fixed_params, err_flux=flux_err, xi_N=xi_N, d_L=d_L, z=z, nwalkers=nwalkers, steps=steps, processes=processes,filename=filenameGA,jet_type=jet_type_GA)
    
