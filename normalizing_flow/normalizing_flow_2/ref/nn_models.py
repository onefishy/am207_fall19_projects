from autograd import numpy as np
from autograd import scipy as sp
from autograd import grad
from autograd.misc.optimizers import adam, sgd

class Feedforward:
    def __init__(self, architecture, random=None, weights=None):
        self.params = {'H': architecture['width'],
                       'L': architecture['hidden_layers'],
                       'D_in': architecture['input_dim'],
                       'D_out': architecture['output_dim'],
                       'activation_type': architecture['activation_fn_type'],
                       'activation_params': architecture['activation_fn_params']}

        self.D = (  (architecture['input_dim'] * architecture['width'] + architecture['width'])
                  + (architecture['output_dim'] * architecture['width'] + architecture['output_dim'])
                  + (architecture['hidden_layers'] - 1) * (architecture['width']**2 + architecture['width'])
                 )

        if random is not None:
            self.random = random
        else:
            self.random = np.random.RandomState(0)

        self.h = architecture['activation_fn']

        if weights is None:
            self.weights = self.random.normal(0, 1, size=(1, self.D))
        else:
            self.weights = weights

        self.objective_trace = np.empty((1, 1))
        self.weight_trace = np.empty((1, self.D))


    def forward(self, weights, x):
        ''' Forward pass given weights and input '''
        H = self.params['H']
        D_in = self.params['D_in']
        D_out = self.params['D_out']

        assert weights.shape[1] == self.D

        if len(x.shape) == 2:
            assert x.shape[0] == D_in
            x = x.reshape((1, D_in, -1))
        else:
            assert x.shape[1] == D_in

        weights = weights.T


        #input to first hidden layer
        W = weights[:H * D_in].T.reshape((-1, H, D_in))
        b = weights[H * D_in:H * D_in + H].T.reshape((-1, H, 1))
        input = self.h(np.matmul(W, x) + b)
        index = H * D_in + H

        assert input.shape[1] == H

        #additional hidden layers
        for _ in range(self.params['L'] - 1):
            before = index
            W = weights[index:index + H * H].T.reshape((-1, H, H))
            index += H * H
            b = weights[index:index + H].T.reshape((-1, H, 1))
            index += H
            output = np.matmul(W, input) + b
            input = self.h(output)

            assert input.shape[1] == H

        #output layer
        W = weights[index:index + H * D_out].T.reshape((-1, D_out, H))
        b = weights[index + H * D_out:].T.reshape((-1, D_out, 1))
        output = np.matmul(W, input) + b
        assert output.shape[1] == self.params['D_out']

        return output

    def make_objective(self, x_train, y_train, reg_param):

        def objective(W, t):
            squared_error = np.linalg.norm(y_train - self.forward(W, x_train), axis=1)**2
            if reg_param is None:
                sum_error = np.sum(squared_error)
                return sum_error
            else:
                mean_error = np.mean(squared_error) + reg_param * np.linalg.norm(W)
                return mean_error

        return objective, grad(objective)

    def fit(self, x_train, y_train, reg_param=None, params=None):

        assert x_train.shape[0] == self.params['D_in']
        assert y_train.shape[0] == self.params['D_out']

        ### make objective function for training
        self.objective, self.gradient = self.make_objective(x_train, y_train, reg_param)

        ### set up optimization
        step_size = 0.01
        max_iteration = 5000
        check_point = 100
        weights_init = self.weights.reshape((1, -1))
        mass = None
        optimizer = 'adam'
        random_restarts = 5

        if 'step_size' in params.keys():
            step_size = params['step_size']
        if 'max_iteration' in params.keys():
            max_iteration = params['max_iteration']
        if 'check_point' in params.keys():
            self.check_point = params['check_point']
        if 'init' in params.keys():
            weights_init = params['init']
        if 'call_back' in params.keys():
            call_back = params['call_back']
        if 'mass' in params.keys():
            mass = params['mass']
        if 'optimizer' in params.keys():
            optimizer = params['optimizer']
        if 'random_restarts' in params.keys():
            random_restarts = params['random_restarts']

        def call_back(weights, iteration, g):
            ''' Actions per optimization step '''
            objective = self.objective(weights, iteration)
            self.objective_trace = np.vstack((self.objective_trace, objective))
            self.weight_trace = np.vstack((self.weight_trace, weights))
            if iteration % check_point == 0:
                print("Iteration {} lower bound {}; gradient mag: {}".format(iteration, objective, np.linalg.norm(self.gradient(weights, iteration))))

        ### train with random restarts
        optimal_obj = 1e16
        optimal_weights = self.weights

        for i in range(random_restarts):
            if optimizer == 'adam':
                adam(self.gradient, weights_init, step_size=step_size, num_iters=max_iteration, callback=call_back)
            local_opt = np.min(self.objective_trace[-100:])
            if local_opt < optimal_obj:
                opt_index = np.argmin(self.objective_trace[-100:])
                self.weights = self.weight_trace[-100:][opt_index].reshape((1, -1))
            weights_init = self.random.normal(0, 1, size=(1, self.D))

        self.objective_trace = self.objective_trace[1:]
        self.weight_trace = self.weight_trace[1:]

class Noisy_Feedforward:
    def __init__(self, architecture, random=None, weights=None):
        self.params = {'H': architecture['width'],
                       'L': architecture['hidden_layers'],
                       'D_in': architecture['input_dim'],
                       'D_noise': architecture['noise_dim'],
                       'D_out': architecture['output_dim'],
                       'activation_type': architecture['activation_fn_type'],
                       'activation_params': architecture['activation_fn_params']}

        self.D = (((architecture['input_dim'] + architecture['noise_dim']) * architecture['width'] + architecture['width'])
                  + (architecture['output_dim'] * architecture['width'] + architecture['output_dim'])
                 + (architecture['hidden_layers'] - 1) * (architecture['width']**2 + architecture['width'])
                 )
        self.D_noise = architecture['noise_dim']


        if random is not None:
            self.random = random
        else:
            self.random = np.random.RandomState(0)

        self.h = architecture['activation_fn']

        if weights is None:
            self.weights = self.random.normal(0, 1, size=(1, self.D))
        else:
            self.weights = weights

        self.objective_trace = np.empty((1, 1))


    def forward(self, weights, x, z):
        ''' Forward pass given weights and input '''
        H = self.params['H']
        D_in = self.params['D_in']
        D_out = self.params['D_out']
        D_noise = self.params['D_noise']
        assert len(weights.shape) == 2 and len(x.shape) == 3 and len(z.shape) == 3
        assert weights.shape[1] == self.D and x.shape[1] == D_in and z.shape[1] == D_noise
        weights = weights.T
        noisy_x = np.concatenate((x, z), axis=1)
        assert len(noisy_x.shape) == 3 and noisy_x.shape[0] == x.shape[0]

        #input to first hidden layer
        W = weights[:H * (D_in + D_noise)].T.reshape((-1, H, (D_in + D_noise)))
        b = weights[H * (D_in + D_noise):H * (D_in + D_noise) + H].T.reshape((-1, H, 1))
        input = self.h(np.matmul(W, noisy_x) + b)
        index = H * (D_in + D_noise) + H
        assert input.shape[1] == H

        #additional hidden layers
        for _ in range(self.params['L'] - 1):
            before = index
            W = weights[index:index + H * H].T.reshape((-1, H, H))
            index += H * H
            b = weights[index:index + H].T.reshape((-1, H, 1))
            index += H
            output = np.matmul(W, input) + b
            input = self.h(output)

            assert input.shape[1] == H

        #output layer
        W = weights[index:index + H * D_out].T.reshape((-1, D_out, H))
        b = weights[index + H * D_out:].T.reshape((-1, D_out, 1))
        output = np.matmul(W, input) + b
        assert output.shape[1] == self.params['D_out']

        return output

    def make_objective(self, x_train, y_train, W_reg_param, z_reg_param):
        assert len(x_train.shape) == 2 and x_train.shape[0] == self.D_noise
        self.N = x_train.shape[1]
        self.z = np.zeros((1, self.D_noise, self.N))
        self.param_trace = np.empty((1, self.D + self.D_noise * self.N))

        def unpack_params(params):
            params = params.reshape((1, -1))
            assert params.shape[1] == self.D + self.D_noise * self.N
            W, z = params[:, :self.D], params[:, self.D:]
            return W, z

        def objective(params, t):
            W, z = unpack_params(params)
            squared_error = np.linalg.norm(y_train - self.forward(W.reshape((1, self.D)), x_train.reshape((1, self.D_noise, -1)), z.reshape((1, self.D_noise, -1))), axis=1)**2

            if W_reg_param is None and z_reg_param is None:
                sum_error = np.mean(squared_error)
                return sum_error
            elif W_reg_param is not None:
                mean_error = np.mean(squared_error) + W_reg_param * np.linalg.norm(W)
                return mean_error
            elif z_reg_param is not None:
                mean_error = np.mean(squared_error) + z_reg_param * np.linalg.norm(z)
                return mean_error
            else:
                mean_error = np.mean(squared_error) + W_reg_param * np.linalg.norm(W) + z_reg_param * np.linalg.norm(z)
                return mean_error

        return unpack_params, objective, grad(objective)

    def fit(self, x_train, y_train, W_reg_param=None, z_reg_param=None, params=None):

        assert x_train.shape[0] == self.params['D_in']
        assert y_train.shape[0] == self.params['D_out']

        ### make objective function for training
        self.unpack_params, self.objective, self.gradient = self.make_objective(x_train, y_train, W_reg_param, z_reg_param)

        ### set up optimization
        step_size = 0.01
        max_iteration = 5000
        check_point = 100
        params_init = self.random.normal(0, 1, size=(1, self.D + self.D_noise * self.N))
        mass = None
        optimizer = 'adam'
        random_restarts = 5

        if 'step_size' in params.keys():
            step_size = params['step_size']
        if 'max_iteration' in params.keys():
            max_iteration = params['max_iteration']
        if 'check_point' in params.keys():
            self.check_point = params['check_point']
        if 'init' in params.keys():
            weights_init = params['init']
        if 'call_back' in params.keys():
            call_back = params['call_back']
        if 'mass' in params.keys():
            mass = params['mass']
        if 'optimizer' in params.keys():
            optimizer = params['optimizer']
        if 'random_restarts' in params.keys():
            random_restarts = params['random_restarts']

        def call_back(params, iteration, g):
            ''' Actions per optimization step '''
            objective = self.objective(params, iteration)
            self.objective_trace = np.vstack((self.objective_trace, objective))
            self.param_trace = np.vstack((self.param_trace, params))
            if iteration % check_point == 0:
                print("Iteration {} lower bound {}; gradient mag: {}".format(iteration, objective, np.linalg.norm(self.gradient(params, iteration))))

        ### train with random restarts
        optimal_obj = 1e16
        optimal_params = params_init

        for i in range(random_restarts):
            if optimizer == 'adam':
                adam(self.gradient, params_init, step_size=step_size, num_iters=max_iteration, callback=call_back)
            local_opt = np.min(self.objective_trace[-100:])
            if local_opt < optimal_obj:
                opt_index = np.argmin(self.objective_trace[-100:])
                self.weights, self.z = self.unpack_params(self.param_trace[-100:][opt_index])
                self.weights = self.weights.reshape((1, -1))
                self.z = self.z.reshape((1, self.D_noise, self.N))
            params_init = self.random.normal(0, 1, size=(1, self.D + self.D_noise * self.N))

        self.objective_trace = self.objective_trace[1:]
        self.param_trace = self.param_trace[1:]

class Additive_Noisy_Feedforward(Noisy_Feedforward):
    # def __init__(self, architecture, random=None, weights=None):
    #     Noisy_Feedforward.__init__(self, architecture, random=None, weights=None)
    def __init__(self, architecture, random=None, weights=None):
        self.params = {'H': architecture['width'],
                       'L': architecture['hidden_layers'],
                       'D_in': architecture['input_dim'],
                       'D_noise': architecture['noise_dim'],
                       'D_out': architecture['output_dim'],
                       'activation_type': architecture['activation_fn_type'],
                       'activation_params': architecture['activation_fn_params']}

        self.D = (((architecture['input_dim'] + architecture['noise_dim']) * architecture['width'] + architecture['width'])
                  + (architecture['output_dim'] * architecture['width'] + architecture['output_dim'])
                  + (architecture['hidden_layers'] - 1) * (architecture['width']**2 + architecture['width'])
                 )
        self.D_noise = architecture['noise_dim']


        if random is not None:
            self.random = random
        else:
            self.random = np.random.RandomState(0)

        self.h = architecture['activation_fn']

        if weights is None:
            self.weights = self.random.normal(0, 1, size=(1, self.D))
        else:
            self.weights = weights

        self.objective_trace = np.empty((1, 1))

    def var_forward(self, weights, x, z):
        ''' Forward pass given weights and input '''
        H = self.params['H']
        D_in = self.params['D_in']
        D_out = self.params['D_out']
        D_noise = self.params['D_noise']
        assert len(weights.shape) == 2 and len(x.shape) == 3 and len(z.shape) == 3
        assert weights.shape[1] == self.D and x.shape[1] == D_in and z.shape[1] == D_noise
        weights = weights.T
        noisy_x = np.concatenate((x, z), axis=1)
        assert len(noisy_x.shape) == 3 and noisy_x.shape[0] == x.shape[0]

        #input to first hidden layer
        W = weights[:H * (D_in + D_noise)].T.reshape((-1, H, (D_in + D_noise)))
        b = weights[H * (D_in + D_noise):H * (D_in + D_noise) + H].T.reshape((-1, H, 1))
        input = self.h(np.matmul(W, noisy_x) + b)
        index = H * (D_in + D_noise) + H
        assert input.shape[1] == H

        #additional hidden layers
        for _ in range(self.params['L'] - 1):
            before = index
            W = weights[index:index + H * H].T.reshape((-1, H, H))
            index += H * H
            b = weights[index:index + H].T.reshape((-1, H, 1))
            index += H
            output = np.matmul(W, input) + b
            input = self.h(output)

            assert input.shape[1] == H

        #output layer
        W = weights[index:index + H * D_out].T.reshape((-1, D_out, H))
        b = weights[index + H * D_out:].T.reshape((-1, D_out, 1))
        output = np.matmul(W, input) + b
        assert output.shape[1] == self.params['D_out']

        return output

    def forward(self, weights, x, z):
        # return Noisy_Feedforward.forward(self, weights, x + z, np.zeros(z.shape))
        return self.var_forward(weights, x + z, np.zeros(z.shape))

    def make_objective(self, x_train, y_train, W_reg_param, z_reg_param):
        assert len(x_train.shape) == 2 and x_train.shape[0] == self.D_noise
        self.N = x_train.shape[1]
        self.z = np.zeros((1, self.D_noise, self.N))
        self.param_trace = np.empty((1, self.D + self.D_noise * self.N))

        def unpack_params(params):
            params = params.reshape((1, -1))
            assert params.shape[1] == self.D + self.D_noise * self.N
            W, z = params[:, :self.D], params[:, self.D:]
            return W, z

        def objective(params, t):
            W, z = unpack_params(params)
            squared_error = np.linalg.norm(y_train - self.forward(W.reshape((1, self.D)), x_train.reshape((1, self.D_noise, self.N)), z.reshape((1, self.D_noise, self.N))), axis=1)**2
            if W_reg_param is None and z_reg_param is None:
                sum_error = np.mean(squared_error)
                return sum_error
            elif W_reg_param is not None:
                mean_error = np.mean(squared_error) + W_reg_param * np.linalg.norm(W)
                return mean_error
            elif z_reg_param is not None:
                mean_error = np.mean(squared_error) + z_reg_param * np.linalg.norm(z)
                return mean_error
            else:
                mean_error = np.mean(squared_error) + W_reg_param * np.linalg.norm(W) + z_reg_param * np.linalg.norm(z)
                return mean_error

        return unpack_params, objective, grad(objective)

    def fit(self, x_train, y_train, W_reg_param=None, z_reg_param=None, params=None):

        assert x_train.shape[0] == self.params['D_in']
        assert y_train.shape[0] == self.params['D_out']

        ### make objective function for training
        self.unpack_params, self.objective, self.gradient = self.make_objective(x_train, y_train, W_reg_param, z_reg_param)

        ### set up optimization
        step_size = 0.01
        max_iteration = 5000
        check_point = 100
        params_init = self.random.normal(0, 1, size=(1, self.D + self.D_noise * self.N))
        mass = None
        optimizer = 'adam'
        random_restarts = 5

        if 'step_size' in params.keys():
            step_size = params['step_size']
        if 'max_iteration' in params.keys():
            max_iteration = params['max_iteration']
        if 'check_point' in params.keys():
            self.check_point = params['check_point']
        if 'init' in params.keys():
            weights_init = params['init']
        if 'call_back' in params.keys():
            call_back = params['call_back']
        if 'mass' in params.keys():
            mass = params['mass']
        if 'optimizer' in params.keys():
            optimizer = params['optimizer']
        if 'random_restarts' in params.keys():
            random_restarts = params['random_restarts']

        def call_back(params, iteration, g):
            ''' Actions per optimization step '''
            objective = self.objective(params, iteration)
            self.objective_trace = np.vstack((self.objective_trace, objective))
            self.param_trace = np.vstack((self.param_trace, params))
            if iteration % check_point == 0:
                print("Iteration {} lower bound {}; gradient mag: {}".format(iteration, objective, np.linalg.norm(self.gradient(params, iteration))))

        ### train with random restarts
        optimal_obj = 1e16
        optimal_params = params_init

        for i in range(random_restarts):
            if optimizer == 'adam':
                adam(self.gradient, params_init, step_size=step_size, num_iters=max_iteration, callback=call_back)
            local_opt = np.min(self.objective_trace[-100:])
            if local_opt < optimal_obj:
                opt_index = np.argmin(self.objective_trace[-100:])
                self.weights, self.z = self.unpack_params(self.param_trace[-100:][opt_index])
                self.weights = self.weights.reshape((1, -1))
                self.z = self.z.reshape((1, self.D_noise, self.N))
            params_init = self.random.normal(0, 1, size=(1, self.D + self.D_noise * self.N))

        self.objective_trace = self.objective_trace[1:]
        self.param_trace = self.param_trace[1:]
