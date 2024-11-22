import numpy as np

#unpack data
time, freq, flux, err = np.genfromtxt('./data/990510.csv',delimiter=',',skip_header=1,unpack=True)

def get_indices(element, lst):
    return [i for i in range(len(lst)) if lst[i] == element]

u_time = list(set(time))
u_time.sort()
u_freq = list(set(freq))
u_freq.sort()

#create an empty array
flux_arr = np.empty((len(u_time),len(u_freq)))
flux_arr[:] = np.nan

#fill in the data rectangle
i = 0
j = 0
for t in u_time:
    for f in u_freq:
        mask = (freq == f) & (time == t)
        f = flux[mask]
        if len(f) == 1:
            flux_arr[i, j] = f[0]
        j = j + 1
    j = 0
    i = i + 1
print(flux_arr)
