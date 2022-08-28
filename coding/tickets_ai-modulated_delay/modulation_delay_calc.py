#The purpose of this file is to create rolling values of delays that are suitable for 

import numpy as np
from scipy.stats import poisson

EPSILON = 1e-8

#uses poisson distribution to calculate weights for various delay values
def discount_delay_calc(delay_vals, iteration, var=1., cutoff = 1e-6):
	masses = poisson.pmf(iteration, np.asarray(delay_vals)/var) + EPSILON
	weighted = masses/np.sum(masses)
	weighted = (weighted > cutoff) * weighted
	return weighted/np.sum(weighted)