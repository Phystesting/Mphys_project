import numpy as np
import afterglowpy as grb
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG, filename="test_debug.log", filemode="w", 
                    format="%(asctime)s - %(levelname)s - %(message)s")



def test_flux_with_parameters_from_file():
    # Load parameters from the CSV file using np.genfromtxt
    t, nu, flux, err = np.genfromtxt('./data/990510_data.csv', delimiter=',', unpack=True, skip_header=1)  # Skip header if present
    
    # Direct values (removed log values and used actual values directly)
    thetaCore = 0.057710979281019446352
    n0 = 0.27480008718307863402
    p = 2.10168577798454164
    epsilon_e = 0.70932637777379936583
    epsilon_B = 0.0016597263929861490878
    E0 = 2.2415777985416490e+53
    thetaObs = 0.032394329041707636829
    jet_type = grb.jet.Gaussian
    z = 1.619
    d_L = 3.76e28
    xi_N = 1.0

    logging.info(f"Testing with parameters: thetaCore={thetaCore}, n0={n0}, p={p}, "
                 f"epsilon_e={epsilon_e}, epsilon_B={epsilon_B}, E0={E0}, "
                 f"thetaObs={thetaObs}, d_L={d_L}, z={z}, xi_N={xi_N}, jet_type={jet_type}")
    
    flux = grb.fluxDensity(t, nu, 
                           jetType=jet_type,
                           specType=grb.jet.SimpleSpec,
                           thetaObs=thetaObs,
                           E0=E0,
                           thetaCore=thetaCore,
                           thetaWing=4 * thetaCore,
                           n0=n0,
                           p=p,
                           epsilon_e=epsilon_e,
                           epsilon_B=epsilon_B,
                           xi_N=xi_N,
                           d_L=d_L,
                           z=z)
    print(flux)
    # Run the flux calculation
    try:
        for time, freq in zip(t, nu):
            flux = grb.fluxDensity(time, freq, 
                                   jetType=jet_type,
                                   specType=grb.jet.SimpleSpec,
                                   thetaObs=thetaObs,
                                   E0=E0,
                                   thetaCore=thetaCore,
                                   thetaWing=4 * thetaCore,
                                   n0=n0,
                                   p=p,
                                   epsilon_e=epsilon_e,
                                   epsilon_B=epsilon_B,
                                   xi_N=xi_N,
                                   d_L=d_L,
                                   z=z)
            if not np.isfinite(flux):
                logging.error(f"Non-finite flux value detected at time={time}, freq={freq}, flux={flux}")
            else:
                logging.debug(f"Computed flux: time={time}, freq={freq}, flux={flux}")
    except Exception as e:
        logging.error(f"Exception occurred: {e}", exc_info=True)

if __name__ == "__main__":
    test_flux_with_parameters_from_file()
