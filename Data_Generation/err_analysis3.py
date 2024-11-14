import afterglowpy as grb
import emcee
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
import re


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

t,nu,mJy,upperlimit,Instrument,Counts,Background,fltr = np.loadtxt('./data/GRB170817.csv',delimiter=',',unpack=True)

def separate(string):
    # Use regular expression to capture numbers and operators, including cases like +- or -+
    numbers_and_operators = re.findall(r'[+-]?\d*\.?\d+e?-?\d*|\d+', string)
    
    # Separate out the numbers and operators
    numbers = []
    operators = []
    
    # Split the input into numbers and operators
    i = 0
    while i < len(numbers_and_operators):
        if numbers_and_operators[i] in ['+', '-', '+-', '-+']:
            operators.append(numbers_and_operators[i])
        else:
            numbers.append(numbers_and_operators[i])
        i += 1
    
    # Calculate the average of the last two numbers
    if len(numbers) > 1:
        avg_last_two = (float(numbers[-1]) + float(numbers[-2])) / 2
        return numbers, operators, avg_last_two
    else:
        return numbers, operators, None

print(mJy)

#print(separate(mJy[1]))

