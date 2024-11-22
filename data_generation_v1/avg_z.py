import numpy as np

GRB,peakF,z,dL,E0,tj,thetaCore,Eg,ref,note = np.genfromtxt('./data/avg_z.txt',delimiter=',',unpack=True)

print(np.mean(z),np.mean(thetaCore))
