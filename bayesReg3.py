#181208 bayesReg3.py
# Attempt to perform bayesian regression as per Probabilistic programming in Python using PyMC3
# contained in the /Users/mdcair/Dropbox/DB Research/Bayesian Research

import numpy as np 
import matplotlib.pyplot as plt

np.random.seed(123)

# True parameter values 
alpha, sigma = 1, 1 
beta = [1, 2.5]

size = 100

# Predictor variable 
X1 = np.linspace(0, 1, size) 
X2 = np.linspace(0,.2, size)

# Simulate outcome variable
Y = alpha + beta[0]*X1 + beta[1]*X2 + np.random.randn(size)*sigma

# print(Y)

from pymc3 import Model, Normal, HalfNormal

basic_model = Model()

with basic_model:
    # Priors for unknown model parameters 
    alpha = Normal('alpha', mu=0, sd=10) 
    beta = Normal('beta', mu=0, sd=10, shape=2) 
    sigma = HalfNormal('sigma', sd=1)

    # Expected value of outcome 
    mu = alpha + beta[0]*X1 + beta[1]*X2
    # Likelihood (sampling distribution) of observations 
    Y_obs = Normal('Y_obs', mu=mu, sd=sigma, observed=Y)

from pymc3 import find_MAP 
map_estimate = find_MAP(model=basic_model) 
print(map_estimate)

from scipy import optimize
map_estimate = find_MAP(model=basic_model, fmin=optimize.fmin_powell)

print(map_estimate)

from pymc3 import NUTS, sample

with basic_model:
	# obtain starting values via MAP 
	start = find_MAP(fmin=optimize.fmin_powell)

	# instantiate sampler 
	step = NUTS(scaling=start)

	# draw 2000 posterior samples 
	trace = sample(2000, step, start=start)

trace['alpha'][-5:]

from pymc3 import traceplot

traceplot(trace)

from pymc3 import summary

summary(trace['alpha'])