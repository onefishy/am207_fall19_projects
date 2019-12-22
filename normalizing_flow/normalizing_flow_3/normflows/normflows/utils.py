import autograd.numpy as np
import autograd.numpy.random as npr
import matplotlib.pyplot as plt
from tqdm import tqdm

from .config import figs, rs, figname
from .distributions import make_samples_z, sample_from_pz
from .transformations import affine
from .plotting import plot_mnist


def clear_figs():
    [f.unlink() for f in figs.glob('*')]


def get_samples_from_params(phi, theta, X, K):
    N, D = X.shape
    Zhat = make_samples_z(*phi, K)
    ZK = Zhat[K, :, :]
    mu_z, log_sigma_diag_pz, logit_pi, A, B = theta
    Xhat = affine(ZK, A, B) + rs.randn(N, D)
    return Xhat, ZK


def compare_reconstruction(phi, theta, x_true, encode, decode, K, t):
    eps = 1e-7
    mu0, log_sigma_diag0, W, U, b = encode(phi, x_true)
    K = W.shape[0]
    z = sample_from_pz(mu0, log_sigma_diag0, W, U, b, K)
    logits = decode(theta, z)
    xhat = npr.binomial(1, logits + eps)

    x_true_im = x_true.reshape(28, 28)
    xhat_im = xhat.reshape(28, 28)

    plot_mnist(x_true_im, xhat_im)
    plt.savefig(figname.format(K, t))
    plt.close()


def make_batch_iter(X, batch_size, max_iter):
    N, D = X.shape
    n_batches = int(np.ceil(N / batch_size))
    n_epochs = int(np.ceil(max_iter / n_batches))


    idx = np.arange(N)
    batch_sched = []
    for i in range(n_epochs):
        idx_shuffled = np.random.permutation(idx)
        batches = np.array_split(idx_shuffled, n_batches)
        batch_sched.append(batches)

    def get_batch(t):
        epoch = int(np.floor(t / n_batches))
        batch = t % n_batches
        epoch_idx = batch_sched[epoch]
        idx_batch = epoch_idx[batch]
        return X[idx_batch].reshape(len(idx_batch), D)
    return get_batch
