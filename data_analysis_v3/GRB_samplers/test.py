import sampler_v9 as splr
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
    "thetaObs": 0.5,
}

# Define inputs
identifier = 'test' # The GRB identifier
extra = '' # GA or TH for gaussian or tophat
z = 0.01
d_L = 1.34e26
xi_N = 1.0
nwalkers = 32
processes = 40
steps = 10

nu = np.geomspace(1e0, 1e20, num=7)
t = np.geomspace(1e-1 * grb.day2sec, 1e3 * grb.day2sec, num=17)
time = []
freq = []
for i in range(len(nu)):
    for j in range(len(t)):
        time.append(t[j])
        freq.append(nu[i])

#print(nu[6])
F = np.array(splr.Flux([time,freq],0.1,0,2.3,-2,-4,53,0.0,1,1.34e26,0.01,jet_type=grb.jet.TopHat))
x = (time,freq)

#F2 = np.array(Flux(x,0.1,0,2.3,-2,-4,53,0.0,1,1e26,0.01))
noise_level = abs(0.05*F)
flux = F + np.random.normal(0, noise_level, size=F.shape)
err_flux = abs(0.05*F)
truth = [0.1,0.0,2.3,-2,-4,53]

jet_type_GA = grb.jet.Gaussian
jet_type_TH = grb.jet.TopHat

# File location to generate the samples in
filenameGA = f'/data/PROJECTS/2024-25/cjc233/samples_v4/{identifier}_GA{extra}_samples.h5'
filenameTH = f'/data/PROJECTS/2024-25/cjc233/samples_v4/{identifier}_TH{extra}_samples.h5'
datafile = f'../../data_generation_v2/data/{identifier}_data.csv'
# Unpack data

flux_err = err_flux, err_flux

#result = splr.run_optimization([time,freq], flux, initial,fixed_params, flux_err, xi_N, d_L, z,jet_type=jet_type_TH)


if __name__ == "__main__":
    splr.run_sampling(x=[time,freq], y=flux, initial=initial,fixed_params=fixed_params, err_flux=flux_err, xi_N=xi_N, d_L=d_L, z=z, nwalkers=nwalkers, steps=steps, processes=processes,filename=filenameTH,jet_type=jet_type_TH)
    #splr.run_sampling(x=[time,freq], y=flux, initial=initial,fixed_params=fixed_params, err_flux=flux_err, xi_N=xi_N, d_L=d_L, z=z, nwalkers=nwalkers, steps=steps, processes=processes,filename=filenameGA,jet_type=jet_type_GA)
    
