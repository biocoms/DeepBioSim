import numpy as np

from mks_test import mkstest

# Generate two samples from a 5D Normal distribution
n = 100
d = 5
mu = np.zeros(d)
sigma = np.eye(d)
X = np.random.multivariate_normal(mu, sigma, n)
Y = np.random.multivariate_normal(mu, sigma, n)
mkstest(X, Y, alpha=0.05, verbose=True)
