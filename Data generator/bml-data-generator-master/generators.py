from scipy.stats import norm, poisson, gamma
import numpy as np


def generate_gaussian(y_exact, sigma2):
    return norm.rvs(loc=y_exact, scale=np.sqrt(sigma2), size=len(y_exact))


def generate_poisson(y_exact):
    y_exact = np.exp(y_exact)
    return poisson.rvs(y_exact, size=len(y_exact))


def generate_gamma(y_exact):
    alpha = np.mean(y_exact)**2 / np.var(y_exact)
    beta = abs(np.mean(y_exact)) / np.var(y_exact)
    return gamma.rvs(a=alpha, scale=1/beta, size=len(y_exact))