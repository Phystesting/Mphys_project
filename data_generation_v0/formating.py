import afterglowpy as grb
import emcee
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool

band_dict = {
    2.418e17: "1 keV",
    1.555e15: "UVW2",
    1.335e15: "UVM2",
    1.153e15: "UVW1",
    8.652e14: "U",
    8.443e14: "u",
    6.826e14: "B",
    6.389e14: "g",
    5.483e14: "V",
    4.862e14: "r",
    4.008e14: "i",
    3.356e14: "z",
    2.398e14: "J",
    1.851e14: "H",
    1.414e14: "Ks",
    1.000e10: "10 GHz",
    6.000e09: "6 GHz",
}


def interpret(band_dict, data):
    # Corrects for days
    if "days" in data.columns:
        data["t"] = data["days"] * 86400
    elif "seconds" in data.columns:
        data["t"] = data["seconds"]
    elif "t_delta" in data.columns:
        data["t"] = data["t_delta"]
    # Renames filter column if needed
    if "Filter" in data.columns:
        data["filter"] = data["Filter"]
    elif "Band" in data.columns:
        data["filter"] = data["Band"]
    elif "band" in data.columns:
        data["filter"] = data["band"]
    # Figures out the flux correction
    if "microJy" in data.columns:
        data["flux"] = data["microJy"]
        flux_correct = 1e-3
    elif "Jy" in data.columns:
        data["flux"] = data["Jy"]
        flux_correct = 1e3
    elif "mJy" in data.columns:
        data["flux"] = data["mJy"]
        flux_correct = 1
    elif "mag" in data.columns:
        data["flux"] = data["mag"]
        flux_correct = "mag"
    # Loops over dataframe and grabs errors and upper limits
    freq, new_flux, err = [], [], []
    for i in range(data.shape[0]):
        try:
            # Try to find the frequency corresponding to the filter
            freq.append(
                list(band_dict.keys())[
                    list(band_dict.values()).index(data.iloc[i]["filter"])
                ]
            )
        except:
            freq.append("Unknown")

        flux = data.iloc[i]["flux"]
        
        # Handling error column when it exists
        if "err" in data.columns:
            error = float(data.iloc[i]["err"])
            if "<" in flux:  # Upper limit case
                new_flux.append("UL")
                err.append(0)
            elif ">" in flux:  # Upper limit case
                new_flux.append("UL")
                err.append(0)
            else:
                flux_value = float(flux)
                new_flux.append(flux_value)
                err.append(float(error))
        else:
            # Flux is directly given as '±' or other error representation
            if "<" in flux:  # Upper limit case
                new_flux.append("UL")
                err.append(0)
            elif ">" in flux:  # Upper limit case
                new_flux.append("UL")
                err.append(0)
            elif "±" in flux:  # Flux with '±' error
                splt = flux.split("±")
                new_flux.append(float(splt[0]))  # Flux value
                err.append(float(splt[1]))  # Error value
            elif "+-" in flux:  # Alternative error notation '+-' in flux
                splt = flux.split("+-")
                new_flux.append(float(splt[0]))  # Flux value
                err.append(float(splt[1]))  # Error value
            elif "+" in flux or "-" in flux:  # Handle cases like '1.0 +0.2 -0.1'
                splt = flux.split("+")
                new_flux.append(float(splt[0]))  # Flux value
                err_splt = splt[1].split("-")
                err.append(max([float(splt[0]), float(err_splt[1])]))  # Error value
            else:
                new_flux.append(flux)
                err.append(0)

    # Now we process the flux and errors
    for i in range(len(new_flux)):
        if new_flux[i] != "UL":
            if flux_correct != "mag":
                new_flux[i] = new_flux[i] * flux_correct
                err[i] = err[i] * flux_correct
            else:
                temp_flux = 1e3 * 3631 * 10 ** (float(new_flux[i]) / -2.5)
                max_flux = 1e3 * 3631 * 10 ** (float(new_flux[i] - err[i]) / -2.5)
                min_flux = 1e3 * 3631 * 10 ** (float(new_flux[i] + err[i]) / -2.5)
                new_flux[i] = 1e3 * 3631 * 10 ** (float(new_flux[i]) / -2.5)
                err[i] = max([max_flux - temp_flux, temp_flux - min_flux])

    # Add the frequency and error columns to the data
    data["frequency"] = freq
    data["flux"] = new_flux
    data["err"] = err

    # Remove rows with unknown flux or frequency
    data = data.loc[(data["flux"] != "UL") & (data["frequency"] != "Unknown")]

    # Return the data with the added flux and error information
    data = data[["t", "frequency", "flux", "err"]].astype(np.float64, errors='ignore')

    return data

    
data_in = pd.read_csv('./data/GRB170817.csv',delimiter=',')
data_out = interpret(band_dict,data_in)
print(data_out)
