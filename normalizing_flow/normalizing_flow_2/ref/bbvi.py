from autograd.misc.optimizers import adam, sgd
from autograd import numpy as np
from autograd import grad
from autograd.numpy import random as npr
import autograd.scipy.stats.multivariate_normal as mvn
import autograd.scipy.stats.norm as norm

import matplotlib.pyplot as plt


class BBVI:
    ''' BBVI with mean field approximation and reparameterization trick'''
    def __init__(self, log_probability, dimension, S=500, random=None, analytic_entropy=True, softplus=True):
        #instantiate random number generator
        if random is not None:
            self.random = random
        else:
            self.random = np.random.RandomState(0)
        self.D = dimension #dimension of support of target distribution
        self.S = S #number of monte carlo samples for computing expectation
        self.target = log_probability #target distribution
        self.unpack_params, self.variational_objective, self.gradient = self.make_variational_objective(analytic_entropy, softplus, log_probability)
        self.check_point = 100 #number of iterations before outputing optimization stats
        self.ELBO = np.empty((1, 1)) #elbo values over optimization
        self.variational_params = np.empty((1, 2 * self.D)) #variational parmeeters over optimization


    def make_variational_objective(self, analytic_entropy, softplus, log_probability):

        if softplus:
            def unpack_params(params):
                mean, parametrized_var = params[:self.D], params[self.D:]
                var = np.log(1 + np.exp(parametrized_var))
                return mean, var
        else:
            def unpack_params(params):
                mean, parametrized_var = params[:self.D], params[self.D:]
                var = np.exp(parametrized_var)
                return mean, var

        if analytic_entropy:
            def entropy(var):
                    ''' Gaussian entropy '''
                    return 0.5 * np.sum(np.log(2 * np.pi * np.e) + np.log(var))

            def variational_objective(params, t):
                ''' Varational objective = H[q] + E_q[target] '''
                mean, var = unpack_params(params) #unpack var parameters
                samples = self.random.randn(self.S, self.D) * var + mean #sample from q using reparametrization
                lower_bound = entropy(var) +  np.mean(log_probability(samples, t)) #ELBO
                return -lower_bound
        else:
            def log_gaussian_pdf(samples, mean, var):
                assert samples.shape == (self.S, self.D)
                assert mean.shape == (1, self.D)

                Sigma = np.diag(var)
                Sigma_det = np.linalg.det(Sigma)
                Sigma_inv = np.linalg.det(Sigma)

                constant = -0.5 * (self.D * np.log(2 * np.pi) + np.log(Sigma_det))
                dist_to_mean = samples - mean
                exponential = -0.5 * np.diag(np.dot(np.dot(dist_to_mean, Sigma_inv), dist_to_mean.T))
                return constant + exponential

            def variational_objective(params, t):
                mean, var = unpack_params(params) #unpack var parameters
                samples = self.random.randn(self.S, self.D) * var + mean #sample from q using reparametrization
                lower_bound = np.mean(log_probability(samples, t) - log_gaussian_pdf(samples, mean.reshape((1, self.D)), var)) #ELBO
                return -lower_bound

        return unpack_params, variational_objective, grad(variational_objective)

    def call_back(self, params, iteration, g):
        ''' Actions per optimization step '''
        elbo = -self.variational_objective(params, iteration)
        self.ELBO = np.vstack((self.ELBO, elbo))
        mean, var = self.unpack_params(params)
        self.variational_params = np.vstack((self.variational_params, np.hstack((mean, var)).reshape((1, -1))))
        if self.verbose and iteration % self.check_point == 0:
            print("Iteration {} lower bound {}; gradient mag: {}".format(iteration, elbo, np.linalg.norm(self.gradient(params, iteration))))

    def debug_call_back(self, params, iteration):
        ''' Actions per optimization step '''
        elbo = -self.variational_objective(params, iteration)
        self.ELBO = np.vstack((self.ELBO, elbo))
        self.variational_params = np.vstack((self.variational_params, params.reshape((1, -1))))
        if iteration % self.check_point == 0:
            print("Iteration {} lower bound {}; gradient mag: {}".format(iteration, elbo, np.linalg.norm(self.gradient(params, iteration))))

    def fit(self, step_size=1e-2, max_iteration=5000, check_point=None, params_init=None, call_back=None, verbose=True, optimizer='adam', mass=None, reset=True):
        ''' Optimization of the variational objective '''
        if check_point is not None:
            self.check_point = check_point

        if params_init is None:
            mean_init = self.random.normal(0, 0.1, size=self.D)
            parametrized_var_init = self.random.normal(0, 0.1, size=self.D)
            params_init = np.concatenate([mean_init, parametrized_var_init])

        assert len(params_init) == 2 * self.D

        self.verbose = verbose

        if call_back is None:
            call_back = self.call_back

        if reset:
            self.ELBO = np.empty((1, 1))
            self.variational_params = np.empty((1, 2 * self.D))

        if optimizer == 'adam':
            adam(self.gradient, params_init, step_size=step_size, num_iters=max_iteration, callback=call_back)
        elif optimizer == 'sgd':
            if mass is None:
                mass = 1e-16
            sgd(self.gradient, params_init, step_size=step_size, num_iters=max_iteration, callback=call_back, mass=mass)
        elif optimizer == 'debug':
            params = params_init
            for i in range(max_iteration):
                params -= step_size * self.gradient(params, i)
                self.debug_call_back(params, i)

        self.variational_params = self.variational_params[1:]
        self.ELBO = self.ELBO[1:]
