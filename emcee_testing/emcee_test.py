import numpy as np
import emcee
import matplotlib.pyplot as plt
import corner

def log_prob(x, ivar):
    return -0.5 * np.sum(ivar * x ** 2)

ndim, nwalkers = 5, 100
ivar = 1. / np.random.rand(ndim)
p0 = np.random.randn(nwalkers, ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=[ivar])
sampler.run_mcmc(p0, 10000)

samples = sampler.get_chain(flat=True)
import corner

fig = corner.corner(
    samples, truths=[ivar]
);
plt.savefig('emcee.png')