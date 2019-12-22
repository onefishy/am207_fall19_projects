import autograd.numpy as np


eps = 1e-7


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logit(x):
    return np.log(eps + x / (1 - x))


def affine(Z, slope, intercept):
    """Predicting observed variables given the latent variables.
    """
    return np.matmul(slope, Z.T).T + intercept


relu = lambda x: np.maximum(x, 0)
