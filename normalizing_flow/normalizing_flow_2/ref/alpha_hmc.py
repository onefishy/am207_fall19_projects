from autograd import numpy as np
from autograd import scipy as sp
from autograd import grad

class HMC:

    def __init__(self, potential_energy, kinetic_energy, kinetic_energy_distribution, random=None, diagnostic_mode=False):
        if random is not None:
            self.random = random
        else:
            self.random = np.random.RandomState(0)
        self.D = np.max(kinetic_energy_distribution(1).shape)
        self.potential_energy = potential_energy
        self.kinetic_energy = kinetic_energy
        self.total_energy = lambda position, momentum: potential_energy(position) + kinetic_energy(momentum)

        self.sample_momentum = lambda n: kinetic_energy_distribution(n).reshape((1, self.D))
        self.grad_potential_energy = grad(potential_energy)

        self.params = {'step_size': 0.1,
                       'leapfrog_steps': 10,
                       'total_samples': 1000,
                       'burn_in': 0.1,
                       'thinning_factor': 1,
                       'diagnostic_mode': diagnostic_mode}

        self.accepts = 0.
        self.iterations = 0.
        self.trace = np.empty((1, self.D))
        self.potential_energy_trace = np.empty((1,))

        assert self.sample_momentum(1).shape == (1, self.D)
        assert isinstance(self.potential_energy(self.sample_momentum(1)), float)
        assert isinstance(self.kinetic_energy(self.sample_momentum(1)), float)
        assert isinstance(self.total_energy(self.sample_momentum(1), self.sample_momentum(1)), float)
        assert self.grad_potential_energy(self.sample_momentum(1)).shape == (1, self.D)


    def leap_frog(self, position_init, momentum_init):
        # initialize position
        position = position_init

        # half step update of momentum
        momentum = momentum_init - self.params['step_size'] * self.grad_potential_energy(position_init) / 2

        # full leap frog steps
        for _ in range(self.params['leapfrog_steps'] - 1):
            position += self.params['step_size'] * momentum
            momentum -= self.params['step_size'] * self.grad_potential_energy(position)
            assert not np.any(np.isnan(position))
            assert not np.any(np.isnan(momentum))

        # full step update of position
        position_proposal = position #+ self.params['step_size'] * momentum
        # half step update of momentum
        momentum_proposal = momentum - self.params['step_size'] * self.grad_potential_energy(position) / 2


        return position_proposal, momentum_proposal

    def hmc(self, position_current, momentum_current):
        ### Refresh momentum
        momentum_current = self.sample_momentum(1)

        ### Simulate Hamiltonian dynamics using Leap Frog
        position_proposal, momentum_proposal = self.leap_frog(position_current, momentum_current)

        # compute total energy in current position and proposal position
        current_total_energy = self.total_energy(position_current, momentum_current)
        proposal_total_energy = self.total_energy(position_proposal, momentum_proposal)

        ### Output for diganostic mode
        if self.params['diagnostic_mode']:
            print('potential energy change:',
                  self.potential_energy(position_current),
                  self.potential_energy(position_proposal))
            print('kinetic energy change:',
                  self.kinetic_energy(momentum_current),
                  self.kinetic_energy(momentum_proposal))
            print('total enregy change:',
                  current_total_energy,
                  proposal_total_energy)
            print('\n\n')

        ### Metropolis Hastings Step
        # comute accept probability
        accept_prob = np.min([1, np.exp(current_total_energy - proposal_total_energy)])
        # accept proposal with accept probability
        if self.random.rand() < accept_prob:
            self.accepts += 1.
            position_current = np.copy(position_proposal)
            momentum_current = momentum_proposal

        return position_current, momentum_current

    def tuning(self, burn_in_period, position_init, momentum_init):
        ### Determine check point
        if self.params['diagnostic_mode']:
            check_point = 10
        else:
            check_point = 100

        ### Initialize position and momentum
        position_current = position_init
        momentum_current = momentum_init

        ### Tune step size param during burn-in period
        for i in range(burn_in_period):
            ### Checks accept rate at check point iterations and adjusts step size
            if i % check_point == 0 and i > 0:
                accept_rate = self.accepts / i
                print('HMC {}: accept rate of {} with step size {}'.format(i, accept_rate * 100., self.params['step_size']))

                if accept_rate < 0.5:
                    self.params['step_size'] *= 0.95
                if accept_rate > 0.8:
                    self.params['step_size'] *= 1.05

            ### perform one HMC step
            position_current, momentum_current = self.hmc(position_current, momentum_current)

        ### Reset number of accepts
        self.accepts = 0

        return position_current, momentum_current

    def run_hmc(self, check_point, position_init, momentum_init):
        ### Initialize position and momentum
        position_current = position_init
        momentum_current = momentum_init

        ### Perform multiple HMC steps
        for i in range(self.params['total_samples']):
            self.iterations += 1
            ### output accept rate at check point iterations
            if i % check_point == 0 and i > 0:
                accept_rate = self.accepts * 100. / i
                print('HMC {}: accept rate of {}'.format(i, accept_rate))

            position_current, momentum_current = self.hmc(position_current, momentum_current)

            # add sample to trace
            if i % self.params['thinning_factor'] == 0:
                self.trace = np.vstack((self.trace, position_current))
                self.potential_energy_trace = np.vstack((self.potential_energy_trace,
                                                         self.potential_energy(position_current)))

        self.trace = self.trace[1:]

    def sample(self, position_init=None, step_size=None, leapfrog_steps=None,
               total_samples=None, burn_in=None, thinning_factor=None, check_point=200,
               alpha=None, diagnostic_mode=None):

        ### Sample random initial momentum
        momentum_init = self.sample_momentum(1)

        ### Set model parameters
        if position_init is None:
            position_init = self.random.normal(0, 1, size=momentum_init.shape)
        else:
            assert position_init.shape == (1, self.D)
        if step_size is not None:
            self.params['step_size'] = step_size
        if leapfrog_steps is not None:
            self.params['leapfrog_steps'] = leapfrog_steps
        if total_samples is not None:
            self.params['total_samples'] = total_samples
        if burn_in is not None:
            self.params['burn_in'] = burn_in
        if thinning_factor is not None:
            self.params['thinning_factor'] = thinning_factor
        if diagnostic_mode is not None:
            self.params['diagnostic_mode'] = diagnostic_mode

        ### Tune parameters during burn-in period
        burn_in_period = int(self.params['burn_in'] * self.params['total_samples'])
        position_current, momentum_current = self.tuning(burn_in_period, position_init, momentum_init)
        ### Obtain samples from HMC using optimized parameters
        self.run_hmc(check_point, position_current, momentum_current)
        self.trace = self.trace[::self.params['thinning_factor']]

