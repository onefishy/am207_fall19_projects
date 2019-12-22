from autograd import numpy as np
from autograd import scipy as sp
from autograd import grad
from ref.alpha_hmc import HMC
from ref.bbvi import BBVI
#from bbalpha import BBALPHA
from autograd.misc.optimizers import adam, sgd

import matplotlib.pyplot as plt


class Inference:
    def __init__(self, method, log_lklhd, log_prior, D, alpha=0.5, prior_var=None, params=None, random=None):
        self.method = method
        self.params = params
        if random is not None:
            self.random = random
        else:
            self.random = np.random.RandomState(0)
        self.sampler = self.make_sampler(log_lklhd, log_prior, prior_var, D, params, alpha=alpha)
        self.samples = None

    def make_vi_sampler(self, log_lklhd, log_prior, D, params, alpha=0.5, prior_var=None, method='bbb'):
        step_size = 0.01
        max_iteration = 5000
        check_point = 100
        params_init = None
        call_back = None
        verbose = True
        mass = None
        optimizer = 'adam'
        random_restarts = 5
        total_samples = 5000
        S = 500

        if 'step_size' in params.keys():
            step_size = params['step_size']
        if 'max_iteration' in params.keys():
            max_iteration = params['max_iteration']
        if 'params_init' in params.keys():
            params_init = params['params_init']
        if 'call_back' in params.keys():
            call_back = params['call_back']
        if 'verbose' in params.keys():
            verbose = params['verbose']
        if 'mass' in params.keys():
            mass = params['mass']
        if 'S' in params.keys():
            S = params['S']
        if 'optimizer' in params.keys():
            optimizer = params['optimizer']
        if 'check_point' in params.keys():
            check_point = params['check_point']
        if 'random_restarts' in params.keys():
            random_restarts = params['random_restarts']

        def joint(W, t):
            return log_lklhd(W) + log_prior(W)

        def sampler(step_size=step_size,
                    max_iteration=max_iteration,
                    check_point=check_point,
                    params_init=params_init,
                    call_back=call_back,
                    optimizer=optimizer,
                    verbose=verbose,
                    mass=mass):

            if method == 'bbb':
                print('using bbb')
                variational_sampler = BBVI(joint, D, S=S, random=self.random)
            #elif method == 'bbalpha':
            #    print('using bbalpha')
            #    variational_sampler = BBALPHA(log_lklhd, prior_var, D, alpha=alpha, S=S, random=self.random)

            optimal_ELBO = -1e16
            optimal_var_params = None
            variational_mu = np.zeros(D)
            variational_Sigma = np.eye(D)
            ELBO = None

            for i in range(random_restarts):

                variational_sampler.fit(step_size=step_size,
                                 max_iteration=max_iteration,
                                 check_point=check_point,
                                 params_init=params_init,
                                 call_back=call_back,
                                 verbose=verbose,
                                 optimizer=optimizer,
                                 mass=mass)

                local_opt = np.max(variational_sampler.ELBO[-100:])

                if local_opt > optimal_ELBO:
                    optimal_ELBO = local_opt
                    opt_param_index = np.argmax(variational_sampler.ELBO[-100:])
                    optimal_var_params  = variational_sampler.variational_params[-100:][opt_param_index]
                    variational_mu = optimal_var_params[:D]
                    variational_Sigma = np.diag(optimal_var_params[D:])
                    ELBO = variational_sampler.ELBO

                params_init = None

            posterior_samples = self.random.multivariate_normal(variational_mu, variational_Sigma, size=total_samples).reshape((-1, D))
            optimal_var_means, optimal_var_var = variational_sampler.unpack_params(optimal_var_params)

            return posterior_samples, (variational_sampler.variational_params, optimal_ELBO, ELBO, optimal_var_means, optimal_var_var)

        return sampler

    def make_hmc_sampler(self, log_lklhd, log_prior, D, params):

        def potential_energy(W):
            return -1 * (log_lklhd(W) + log_prior(W))[0]
        def kinetic_energy_distribution(n):
            return self.random.normal(0, 1, size=D)
        def kinetic_energy(W):
            return np.sum(W**2) / 2.0

        position_init = self.random.normal(0, 1, size=(1, D))
        step_size = 1e-3
        leapfrog_steps = 50
        total_samples = 5000
        burn_in = 0.1
        thinning_factor = 1

        if params is not None:
            if 'M' in params.keys():
                def energy(W):
                    return np.dot(np.dot(W.T, params['M']), W) / 2.0
                kinetic_energy = energy

            if 'position_init' in params.keys():
                position_init = params['position_init']
                assert position_init.shape == (1, D)

            if 'step_size' in params.keys():
                step_size = params['step_size']

            if 'leapfrog_steps' in params.keys():
                leapfrog_steps = params['leapfrog_steps']

            if 'total_samples' in params.keys():
                total_samples = params['total_samples']

            if 'burn_in' in params.keys():
                burn_in = params['burn_in']

            if 'thinning_factor' in params.keys():
                thinning_factor = params['thinning_factor']

            if 'alpha' in params.keys():
                HMC_sampler = Alpha_HMC(potential_energy, kinetic_energy,
                                        kinetic_energy_distribution,
                                        random=self.random,
                                        alpha=params['alpha'])
            else:
                HMC_sampler = HMC(potential_energy, kinetic_energy,
                                  kinetic_energy_distribution,
                                  random=self.random)

        else:
            HMC_sampler = HMC(potential_energy, kinetic_energy,
                              kinetic_energy_distribution,
                              random=self.random)

        def sampler(position_init=position_init,
                    step_size=step_size,
                    leapfrog_steps=leapfrog_steps,
                    total_samples=total_samples,
                    burn_in=burn_in,
                    thinning_factor=thinning_factor):

            HMC_sampler.sample(position_init=position_init,
                               step_size=step_size,
                               leapfrog_steps=leapfrog_steps,
                               total_samples=total_samples,
                               burn_in=burn_in,
                               thinning_factor=thinning_factor)

            return HMC_sampler.trace[::thinning_factor], (HMC_sampler.trace, HMC_sampler.potential_energy_trace)

        return sampler

    def make_sampler(self, log_lklhd, log_prior, prior_var, D, params, alpha=0.5):

        sampler = None

        if self.method == 'hmc':
            sampler = self.make_hmc_sampler(log_lklhd, log_prior, D, params)

        if self.method == 'vi':
            sampler = self.make_vi_sampler(log_lklhd, log_prior, D, params, method='bbb')

        if self.method == 'bbalpha':
            sampler = self.make_vi_sampler(log_lklhd, log_prior, D, params, alpha=alpha, prior_var=prior_var, method='bbalpha')

        return sampler

    def sample(self):
        self.samples, trace = self.sampler()
        return trace
