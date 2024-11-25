import numpy as np
import sampler_v3 as splr


t =np.geomspace(1e1,1e8,50)
nu = np.geomspace(1e9,1e20,50)
t_val = []
nu_val = []
for nu_value in nu:
	for t_value in t:
		t_val.append(t_value)
		nu_val.append(nu_value)

splr.Flux([t_val,nu_val],0.25683026,-4.25685193,2.16967589,-0.14414304,-4.22091356,51.74597245,0.39250542,1.0,0.0099,1.327e26)
