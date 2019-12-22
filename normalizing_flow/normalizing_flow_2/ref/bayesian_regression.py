from autograd import numpy as np
from autograd import scipy as sp
from autograd import grad

from ref.inference import Inference

class Bayesian_Regression:
    def __init__(self, params, forward, Sigma_W, sigma_y, random=None, weights=None):
        self.params = params
        self.forward = forward
        if random is not None:
            self.random = random
        else:
            self.random = np.random.RandomState(0)
        self.Sigma_W = Sigma_W
        self.sigma_y = sigma_y
        self.D = len(Sigma_W)

        self.posterior_samples = None
        self.posterior_predictive_samples = None
        self.log_prior = None
        self.log_lklhd = None

    def make_model(self, x_train, y_train, N):

        Sigma_W_inv = np.linalg.inv(self.Sigma_W)
        Sigma_W_det = np.linalg.det(self.Sigma_W)

        self.variational_dim = self.D

        def log_prior(W,
                      Sigma_W_det=Sigma_W_det,
                      Sigma_W_inv=Sigma_W_inv):
            assert len(W.shape) == 2 and W.shape[1] == self.variational_dim
            S = len(W)

            constant_W = -0.5 * (self.D * np.log(2 * np.pi) + np.log(Sigma_W_det))
            exponential_W = -0.5 * np.diag(np.dot(np.dot(W, Sigma_W_inv), W.T))
            assert exponential_W.shape == (S, )
            log_p_W = constant_W + exponential_W
            return log_p_W

        def log_lklhd(W,
                      x_train=x_train,
                      y_train=y_train):
            assert len(W.shape) == 2 and W.shape[1] == self.variational_dim
            S = W.shape[0]
            constant = (-np.log(self.sigma_y) - 0.5 * np.log(2 * np.pi)) * N
            exponential = -0.5 * self.sigma_y**-2 * np.sum((y_train.reshape((1, self.params['D_in'], N)) - self.forward(W, x_train))**2, axis=2).flatten()
            assert exponential.shape == (S, )
            return constant + exponential

        prior_var = np.diag(self.Sigma_W)
        return log_prior, log_lklhd, prior_var

    def fit(self, data_train, N, data_validate=None):

        x_train = data_train[:, 0].reshape((1, self.params['D_in'], N))
        y_train = data_train[:, 1].reshape((1, self.params['D_out'], N))

        self.log_prior, self.log_lklhd, self.prior_var = self.make_model(x_train, y_train, N)

    def sample_posterior(self, method='vi', alpha=0.5, params=None):
        inference = Inference(method, self.log_lklhd, self.log_prior, self.variational_dim, alpha=alpha, prior_var=self.prior_var, params=params, random=self.random)
        trace = inference.sample()
        self.posterior_samples = inference.samples
        return trace

    def sample_posterior_predictive(self, x, samples=100, S=100, trace=None):
        ''' Produce samples from posterior predictive given samples from posterior over weights '''
        x = x.reshape((1, self.params['D_in'], -1))
        N = x.shape[2]
        if trace is not None:
            posterior_samples = trace[self.random.choice(np.arange(len(trace)), samples)][:, :self.D]
        else:
            posterior_samples = self.posterior_samples[self.random.choice(np.arange(len(self.posterior_samples)), samples)][:, :self.D]
        self.posterior_predictive_samples = self.forward(posterior_samples, x.reshape((self.params['D_in'], -1)))
        self.posterior_predictive_samples += self.random.normal(0, self.sigma_y**0.5, size=self.posterior_predictive_samples.shape)
