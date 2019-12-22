from tqdm import tqdm

from autograd import numpy as np
from autograd.numpy import random as npr
from autograd import grad
from autograd.misc.optimizers import adam, rmsprop

from .flows import planar_flow
from .utils import clear_figs, get_samples_from_params, compare_reconstruction, make_batch_iter
from .config import results


rs = npr.RandomState(0)
# clear_figs()


def gradient_create(F, D, N, unpack_params):
    """Create variational objective, gradient, and parameter unpacking function

    Arguments:
        F {callable} -- Energy function (to be minimized)
        D {int} -- dimension of latent variables
        N {int} -- Number of samples to draw
        unpack_params {callable} -- Parameter unpacking function

    Returns
        'variational_objective', 'gradient', 'unpack_params'
    """

    def variational_objective(params, t):
        phi, theta = unpack_params(params)
        free_energy = F(phi, theta, t)
        return free_energy

    gradient = grad(variational_objective)

    return variational_objective, gradient


def optimize(logp, X, D, K, N, init_params,
             unpack_params, encode, decode,
             max_iter, batch_size, step_size, verbose=True):
    """Run the optimization for a mixture of Gaussians

    Arguments:
        logp {callable} -- Joint log-density of Z and X
        X {np.ndarray} -- Observed data
        D {int} -- Dimension of Z
        G {int} -- Number of Gaussians in GMM
        N {int} -- Number of samples to draw
        K {int} -- Number of flows
        max_iter {int} -- Maximum iterations of optimization
        step_size {float} -- Learning rate for optimizer
    """
    batch_iter = make_batch_iter(X, batch_size, max_iter)
    def logq0(z):
        """Just a standard Gaussian
        """
        return -D / 2 * np.log(2 * np.pi) - 0.5 * np.sum(z ** 2, axis=0)

    def hprime(x):
        return 1 - np.tanh(x) ** 2

    logdet_jac = lambda w, z, b: np.sum(w * hprime(np.sum(w * z, axis=1) + b).reshape(-1, 1), axis=1)

    def F(phi, theta, t):
        eps = 1e-7
        Xbatch = batch_iter(t)
        z0 = rs.randn(Xbatch.shape[0], D)
        mu0, log_sigma_diag0, W, U, B = encode(phi, Xbatch)
        cooling_max = np.min(np.array([max_iter / 4, 10000]))
        beta_t = np.min(np.array([1, 0.001 + t / cooling_max]))

        sd = np.sqrt(eps + np.exp(log_sigma_diag0))
        zk = z0 * sd + mu0
        # Unsure if this should be z0 or z1 (after adding back in mean and sd)
        first = np.mean(logq0(z0))


        running_sum = 0.
        for k in range(K):
            w, u, b = W[k], U[k], B[k]
            running_sum = running_sum + np.log(eps + np.abs(1. + np.sum(u * logdet_jac(w, zk, b).reshape(-1, 1), axis=1)))
            zk = planar_flow(zk, w, u, b)
        third = np.mean(running_sum)

        logits = decode(theta, zk)
        # second = np.mean(logp(Xbatch, zk, logits))
        second = np.mean(logp(Xbatch, zk, logits)) * beta_t  # Play with temperature

        # return -second - third
        return first - second - third

    objective, gradient = gradient_create(F, D, N, unpack_params)
    pbar = tqdm(total=max_iter)

    def callback(params, t, g):
        pbar.update()
        if verbose:
            if t % 100 == 0:
                grad_mag = np.linalg.norm(gradient(params, t))
                tqdm.write(f"Iteration {t}; objective: {objective(params, t)} gradient mag: {grad_mag:.3f}")
            if t % 200 == 0:
                try:
                    Xtrue = X[101].reshape(1, -1)
                    phi, theta = unpack_params(params)
                    compare_reconstruction(phi, theta, Xtrue, encode, decode, K, t)
                except ValueError:
                    tqdm.write("nan gradient")
                    tqdm.write(str(params))
                    tqdm.write(str(unpack_params(params)))

            if t == max_iter - 1:
                variational_lower_bound = objective(params, t)
                with (results / 'free_energy.txt').open('a') as f:
                    f.write(f'\n{K} flows: {variational_lower_bound}')

    variational_params = adam(gradient, init_params, step_size=step_size, callback=callback, num_iters=max_iter)
    pbar.close()

    return unpack_params(variational_params)
