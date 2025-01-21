import numpy as np
import matplotlib.pyplot as plt
import afterglowpy as grb

def test_flux_functions():
    # Define test input parameters
    thetaCore = 0.1
    log_n0 = -1
    p = 2.2
    log_epsilon_e = -1
    log_epsilon_B = -2
    log_E0 = 52
    thetaObs = 0.2
    xi_N = 1.0
    z = 0.0099
    d_L = 1.327e+26
    jet_type = grb.jet.Gaussian

    # Load data from CSV file
    time, freq, flux, Ub_err, Lb_err = np.genfromtxt(
        './data/170817_data.csv', delimiter=',', unpack=True, skip_header=1)
    x = [time, freq]

    def get_changes(lst):
        if not lst:
            return []
        unique_values, indices = np.unique(lst, return_index=True)
        return unique_values, sorted(indices.tolist())

    # First Flux function
    def Flux1(x, thetaCore, log_n0, p, log_epsilon_e, log_epsilon_B, log_E0, thetaObs, xi_N, d_L, z, jet_type):
        Z = {
            'jetType': jet_type,
            'specType': grb.jet.SimpleSpec,
            'thetaObs': thetaObs,
            'E0': 10**log_E0,
            'thetaCore': thetaCore,
            'thetaWing': 4 * thetaCore,
            'n0': 10**log_n0,
            'p': p,
            'epsilon_e': 10**log_epsilon_e,
            'epsilon_B': 10**log_epsilon_B,
            'xi_N': xi_N,
            'd_L': d_L,
            'z': z,
        }

        t = np.array(x[0])
        nu = np.array(x[1])

        unique_freq, segment_indices = get_changes(nu.tolist())
        segment_indices.append(len(nu))

        Flux = []

        for i in range(len(unique_freq)):
            start_idx = segment_indices[i]
            end_idx = segment_indices[i + 1]
            mask = slice(start_idx, end_idx)
            flux_segment = grb.fluxDensity(t[mask], unique_freq[i], **Z)
            Flux.extend(flux_segment)

        return np.array(Flux)

    # Second Flux function
    def Flux2(x, thetaCore, log_n0, p, log_epsilon_e, log_epsilon_B, log_E0, thetaObs, xi_N, d_L, z, jet_type):
        Z = {
            'jetType': jet_type,
            'specType': grb.jet.SimpleSpec,
            'thetaObs': thetaObs,
            'E0': 10**log_E0,
            'thetaCore': thetaCore,
            'thetaWing': 4 * thetaCore,
            'n0': 10**log_n0,
            'p': p,
            'epsilon_e': 10**log_epsilon_e,
            'epsilon_B': 10**log_epsilon_B,
            'xi_N': xi_N,
            'd_L': d_L,
            'z': z,
        }

        t = x[0]
        nu = x[1]

        try:
            Flux = grb.fluxDensity(t, nu, **Z)
            if isinstance(Flux, np.ndarray) and not np.all(np.isfinite(Flux)):
                raise ValueError("Flux computation returned non-finite values.")
            elif not isinstance(Flux, np.ndarray) and not np.isfinite(Flux):
                raise ValueError("Flux computation returned a non-finite value.")
        except Exception as e:
            print(f"Error in fluxDensity computation: {e}")
            return np.full_like(t, 1e-300)

        return Flux

    # Direct single-point computation
    def DirectFlux(x, thetaCore, log_n0, p, log_epsilon_e, log_epsilon_B, log_E0, thetaObs, xi_N, d_L, z, jet_type):
        Z = {
            'jetType': jet_type,
            'specType': grb.jet.SimpleSpec,
            'thetaObs': thetaObs,
            'E0': 10**log_E0,
            'thetaCore': thetaCore,
            'thetaWing': 4 * thetaCore,
            'n0': 10**log_n0,
            'p': p,
            'epsilon_e': 10**log_epsilon_e,
            'epsilon_B': 10**log_epsilon_B,
            'xi_N': xi_N,
            'd_L': d_L,
            'z': z,
        }

        t = x[0]
        nu = x[1]
        Flux = []
        for ti, nui in zip(t, nu):
            Flux.append(grb.fluxDensity(ti, nui, **Z)[0])
        return np.array(Flux)

    # Calculate flux using all methods
    flux1 = Flux1(x, thetaCore, log_n0, p, log_epsilon_e, log_epsilon_B, log_E0, thetaObs, xi_N, d_L, z, jet_type)
    flux2 = Flux2(x, thetaCore, log_n0, p, log_epsilon_e, log_epsilon_B, log_E0, thetaObs, xi_N, d_L, z, jet_type)
    direct_flux = DirectFlux(x, thetaCore, log_n0, p, log_epsilon_e, log_epsilon_B, log_E0, thetaObs, xi_N, d_L, z, jet_type)

    # Compare all methods
    print("Flux1:", flux1)
    print("Flux2:", flux2)
    print("DirectFlux:", direct_flux)

    # Plot results for comparison
    plt.figure(figsize=(10, 6))
    plt.loglog(time, flux1, label="Flux1 (Batch)", linestyle="--")
    plt.loglog(time, flux2, label="Flux2 (Batch)", linestyle="-")
    plt.loglog(time, direct_flux, label="Direct Flux (Point-by-Point)", linestyle=":")
    plt.xlabel("Time (s)")
    plt.ylabel("Flux Density")
    plt.title("Comparison of Flux Calculation Methods")
    plt.legend()
    plt.grid(which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()

# Run the test
test_flux_functions()
