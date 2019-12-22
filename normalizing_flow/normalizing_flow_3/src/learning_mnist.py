"""
Building a VAE for the MNIST dataset
"""
import autograd.numpy as np
from mlxtend.data import loadlocal_mnist

from normflows import (config, optimization,
                       distributions, nn_models, transformations)


K = 4
dim_z = 40
dim_x = 28 * 28
width = 64
n_hidden = 3
activation_fn = transformations.relu
activation_fn_type = 'relu'

encoder_architecture = {
    'width': width,
    'hidden_layers': n_hidden,
    'input_dim': dim_x,
    'output_dim': 2 * dim_z + 2 * dim_z * K + 1 * K,
    'activation_fn_type': activation_fn_type,
    'activation_fn_params': '',
    'activation_fn': activation_fn
}
decoder_architecture = {
    'width': width,
    'hidden_layers': n_hidden,
    'input_dim': dim_z,
    'output_dim': dim_x,
    'activation_fn_type': activation_fn_type,
    'activation_fn_params': '',
    'activation_fn': activation_fn,
    'output_activation_fn': transformations.sigmoid
}


encoder = nn_models.Feedforward(architecture=encoder_architecture)
decoder = nn_models.Feedforward(architecture=decoder_architecture)


def load_data():
    X, y = loadlocal_mnist(
        images_path=str(config.mnist / 'train-images-idx3-ubyte'),
        labels_path=str(config.mnist / 'train-labels-idx1-ubyte'))

    keep_digits = np.isin(y, [0, 1, 4, 7])
    X = X[keep_digits]
    y = y[keep_digits]
    X = X / 255
    X = (X >= 0.5).astype(int)  # Binarizing
    return X, y


def make_unpack_params():
    """Make parameter unpacking functions (this is where the NN is called)
    """
    def encode(weights, X):
        N = X.shape[0]
        phi = encoder.forward(weights.reshape(1, -1), X.T)[0]
        mu0 = phi[:dim_z].reshape(N, dim_z)
        log_sigma_diag0 = phi[dim_z:2 * dim_z].reshape(N, dim_z)
        W = phi[2 * dim_z:2 * dim_z + K * dim_z].reshape(K, N, dim_z)
        U = phi[2 * dim_z + K * dim_z:2 * dim_z + 2 * K * dim_z].reshape(K, N, dim_z)
        b = phi[-K:].reshape(K, N)

        return mu0, log_sigma_diag0, W, U, b

    def decode(weights, Z):
        logits = decoder.forward(weights.reshape(1, -1), Z.T)[0]
        return logits.T

    def unpack_params(params):
        phi = params[:encoder.D]
        theta = params[encoder.D:]
        return phi, theta

    return unpack_params, encode, decode


def get_init_params():
    init_weights = np.random.randn(encoder.D + decoder.D) * 0.05

    return init_weights


def logp(X, Z, logits):
    """Joint likelihood for MNIST

    :param X: np.ndarray -- Data (N, dim_x)
    :param Z: np.ndarray -- Latent variables (N, dim_z)
    :param logits: np.ndarray -- logits for bernoulli distribution (N, dim_x)
    :return: np.ndarray -- Log-joint probability assuming p(z) is a unit Gaussian
    """
    log_prob_z = distributions.log_std_norm(Z)
    log_prob_x = distributions.log_bern_mult(X, logits)
    return log_prob_x + log_prob_z


def run_optimization(X, init_params, unpack_params, encode, decode,
                     max_iter=20000, batch_size=256, N=None, step_size=1e-4):
    if not N:
        N = X.shape[0]
    else:
        idx = np.random.randint(X.shape[0], size=N)
        X = X[idx]
    return optimization.optimize(logp, X, dim_z, K, N,
                                 init_params, unpack_params, encode, decode,
                                 max_iter, batch_size, step_size,
                                 verbose=True)


def main():
    X, y = load_data()
    unpack_params, encode, decode = make_unpack_params()
    init_params = get_init_params()

    phi, theta = run_optimization(X, init_params, unpack_params, encode, decode,
                                  max_iter=10000, batch_size=128, N=2000, step_size=1e-3)
    for arr, name in [(phi, 'phi'), (theta, 'theta')]:
        np.save(config.models / "mnist40d" / f"weights_{name}_{K}.npy", arr)
    print("DONE")

if __name__ == '__main__':
    main()
