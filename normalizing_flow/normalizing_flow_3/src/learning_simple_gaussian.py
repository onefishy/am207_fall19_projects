"""Trying to learn a simple mixture of Gaussians (1D and 2D)
"""
from autograd import numpy as np
import autograd.numpy.random as npr


import matplotlib.pyplot as plt

from normflows import flows, distributions, transformations, optimization, plotting


rs = npr.RandomState(0)


def sample_gaussian(mu, Sigma_diag, n_samples):
    """Sample `n_samples` points from a Gaussian mixture model.

    Arguments:
        mu {np.ndarray} -- means for gaussians
            shape: (n_mixtures, dim)
        Sigma_diag {np.ndarray} -- diagonals for covariance matrices
            shape: (n_mixtures, dim)

    Returns:
        samples {np.ndarray} -- samples from the gaussian
            shape: (n_samples, dim)
    """
    Sigma = np.diag(Sigma_diag)
    samps = rs.multivariate_normal(mu, Sigma, size=n_samples)
    return samps


def get_gaussian_samples(n_samples=10000):
    """Simple GMM with two modes
    """
    mu = np.array([4])
    Sigma_diag = np.array([1])
    samps = sample_gaussian(mu, Sigma_diag, n_samples)
    return samps


def f_true(Z):
    """Defining a simple affine transformation to transform Z into X.
    """
    slope_true = 2
    int_true = 3.5
    mus = transformations.affine(Z, 2, 3.5)
    Sigma = np.array([[1]])
    X = np.zeros_like(Z)
    for i in range(Z.shape[0]):
        X[i, :] = np.random.multivariate_normal(mus[i], Sigma, size=1)
    return X


def f_pred(Z, slope, intercept):
    """Predicting observed variables given the latent variables.
    """
    mus = transformations.affine(Z, slope, intercept)
    Sigma = np.array([[1]])  # Assuming the likelihood has variance 1
    X = np.zeros_like(Z)
    for i in range(Z.shape[0]):
        X[i, :] = rs.multivariate_normal(mus[i], Sigma, size=1)
    return X


def sample_from_pz(mu, log_sigma_diag, W, U, b, K, num_samples):
    Sigma = np.diag(np.exp(log_sigma_diag))
    dim_z = len(mu)

    Z = np.zeros((K + 1, num_samples, dim_z))
    z = rs.multivariate_normal(mu, Sigma, num_samples)

    Z[0] = z
    for k in range(K):
        Z[k + 1] = flows.planar_flow(z, W[k], U[k], b[k])
    return Z


def make_unpack_params(D, K):
    """Create variational objective, gradient, and parameter unpacking function

    Arguments:
        D {int} -- dimension of latent variables
        K {int} -- number of flows

    Returns
        'unpack_params'
    """
    def unpack_phi(phi):
        mu0 = phi[:D]
        log_sigma_diag0 = phi[D:2 * D]
        W = phi[2 * D:2 * D + K * D].reshape(K, D)
        U = phi[2 * D + K * D:2 * (D + K * D)].reshape(K, D)
        b = phi[-K:]

        return mu0, log_sigma_diag0, W, U, b

    def unpack_theta(theta):
        mu_z = theta[:D].flatten()
        log_sigma_diag_pz = theta[D: 2 * D].flatten()
        A = theta[-(D ** 2 + 2 * D):-2 * D].reshape(D, D)
        B = theta[-2 * D:-D]
        log_sigma_diag_lklhd = theta[-D:]

        return mu_z, log_sigma_diag_pz, A, B, log_sigma_diag_lklhd

    def unpack_params(params):
        phi = params[:2 * D * (1 + K) + K]
        theta = params[2 * D * (1 + K) + K:]

        phi = unpack_phi(phi)
        theta = unpack_theta(theta)

        return phi, theta

    return unpack_params

def get_init_params(D, K):
    # --- Initializing --- #
    # phi
    init_mu0 = np.zeros(D)
    init_log_sigma0 = np.zeros(D)
    init_W = np.ones((K, D))
    init_U = np.ones((K, D))
    init_b = np.zeros(K)

    init_phi = np.concatenate([
            init_mu0,
            init_log_sigma0,
            init_W.flatten(),
            init_U.flatten(),
            init_b
        ])

    # theta
    init_mu_z = np.zeros(D)
    init_log_sigma_z = np.zeros(D)
    init_A = np.eye(D)
    init_B = np.zeros(D)
    init_log_sigma_lklhd = np.zeros(D)  # Assuming diagonal covariance for likelihood

    init_theta = np.concatenate([
            init_mu_z.flatten(),
            init_log_sigma_z.flatten(),
            init_A.flatten(),
            init_B,
            init_log_sigma_lklhd
        ])

    init_params = np.concatenate((init_phi, init_theta))

    return init_params


def logp(X, Z, theta):
    """Joint likelihood for Gaussian mixture model
    """
    # Maybe reshape these bois
    mu_z, log_sigma_diag_z, A, B, log_sigma_diag_lklhd = theta
    log_prob_z = distributions.log_mvn(Z, mu_z, log_sigma_diag_z)

    mu_x = np.matmul(A, Z.T).T + B
    log_prob_x = distributions.log_mvn(X, mu_x, log_sigma_diag_lklhd)

    return log_prob_x + log_prob_z


def run_optimization(X, K, D, init_params, unpack_params, max_iter=10000, N=1000, step_size=1e-3):
    return optimization.optimize(logp, X, D, K, N,
                                 init_params, unpack_params, max_iter, step_size,
                                 verbose=True)


def main():
    K = 1
    D = 1
    n_samples = 500

    Z = get_gaussian_samples(n_samples=n_samples)
    X = f_true(Z)

    unpack_params = make_unpack_params(D, K)
    init_params = get_init_params(D, K)
    phi, theta = run_optimization(X, K, D, init_params, unpack_params,
                                  max_iter=10000, N=n_samples, step_size=1e-4)

    print(f"Variational params: {phi}")
    print(f"Generative params: {theta}")

    Zhat = sample_from_pz(*phi, K, n_samples)

    fig, axs = plt.subplots(ncols=2, nrows=2, sharex=True)
    plotting.plot_samples(Z, axs[0, 0])
    axs[0, 0].set_title('Latent')
    plotting.plot_samples(Zhat[-1, :], axs[1, 0])
    axs[1, 0].set_title('Variational latent')

    #TODO: Plot Xhat
    mu_z, log_sigma_diag_pz, A, B, log_sigma_diag_lklhd = theta
    Xhat = f_pred(Zhat[-1, :], A, B)

    plotting.plot_samples(X, axs[0, 1])
    axs[0, 1].set_title('Observed')
    plotting.plot_samples(Xhat, axs[1, 1])
    axs[1, 1].set_title("Variational observed")

    plt.show()


if __name__ == '__main__':
    main()
