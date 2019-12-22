import autograd
import autograd.numpy as np
import autograd.scipy as sp
import autograd.misc.optimizers
from autograd import grad
import numpy
import time
import os
import sys
from tqdm import tqdm
from matplotlib import pyplot as plt
from autograd.misc.optimizers import adam
from scripts.flows import adam_solve, energy_bound

class Feedforward:
    def __init__(self, architecture, random=None, weights=None):
        self.params = {'H': architecture['width'],
                       'L': architecture['hidden_layers'],
                       'D_in': architecture['input_dim'],
                       'D_out': architecture['output_dim'],
                       'activation_type': architecture['activation_fn_type'],
                       'activation_params': architecture['activation_fn_params']}

        self.D = (  (architecture['input_dim'] * architecture['width'] + 
                     architecture['width'])
                  + (architecture['output_dim'] * architecture['width'] + 
                     architecture['output_dim'])
                  + (architecture['hidden_layers'] - 1) * 
                    (architecture['width']**2 + architecture['width'])
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

    def make_objective(self, x_train, y_train, reg_param=None):
        ''' Make objective functions: depending on whether or not you want to apply l2 regularization '''

        if reg_param is None:

            def objective(W, t):
                squared_error = np.linalg.norm(y_train - self.forward(W, x_train), axis=1)**2
                sum_error = np.sum(squared_error)
                return sum_error

            return objective, grad(objective)

        else:

            def objective(W, t):
                squared_error = np.linalg.norm(y_train - self.forward(W, x_train), axis=1)**2
                mean_error = np.mean(squared_error) + reg_param * np.linalg.norm(W)
                return mean_error

            return objective, grad(objective)

    def fit(self, x_train, y_train, params, reg_param=None):
        ''' Wrapper for MLE through gradient descent '''
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


def log_prior(W, D, Sigma_W_det, Sigma_W_inv):
    #assert len(W.shape) == 2 and W.shape[1] == self.variational_dim
    S = len(W)

    constant_W = np.zeros(S)
    for i in range(S):
        constant_W[i] = -0.5 * (D * np.log(2 * np.pi) + np.log(Sigma_W_det[i]))

    exponential_W = np.zeros(S)
    for i in range(S):
        #print(W[i].shape)
        #print(Sigma_W_inv.shape)
        #print(np.dot(np.dot(W[i][np.newaxis,], Sigma_W_inv), W[i]).shape)
        #print(-0.5 * np.diag(np.dot(np.dot(W[i][np.newaxis,],Sigma_W_inv),W[i].T)))
        exponential_W[i] = -0.5 * np.diag(np.dot(np.dot(W[i][np.newaxis,], Sigma_W_inv), W[i].T))._value
    assert exponential_W.shape == (S, )
    #print(constant_W.shape, exponential_W.shape)
    log_p_W = constant_W + exponential_W
    return log_p_W

def log_lklhd(W, x_train, y_train, bnn, sigma_y=0.5):
    #assert len(W.shape) == 2 and W.shape[1] == self.variational_dim
    S = W.shape[0]
    constant = (-np.log(sigma_y) - 0.5 * np.log(2 * np.pi)) * len(W)
    exponential = []
    for i in range(len(W)):
        exponential.append(np.sum(-0.5 * sigma_y**-2 * np.sum(y_train) - 
                                  bnn.forward(W[i][np.newaxis,], x_train[np.newaxis,])[0]**2))

    exponential = np.array(exponential)
    assert exponential.shape == (S, )
    return constant + exponential


def log_joint(W, t, x, y, sigma_w_det, sigma_w_inv, bnn, D):
    return log_lklhd(W, x, y, bnn) + log_prior(W, D, sigma_w_det, sigma_w_inv)

#def log_joint(w, x, y, nn, mu=0, sig1=2., sig2=0.5, N=16):
#
#    # Prior is set, I think?
#    sig1_mat = sig1**2 * np.eye(N)
#    mu = nn.forward(w, x[np.newaxis,])[0][0]
#    prior = sum(np.log(sp.stats.multivariate_normal.pdf(w, np.zeros(N), sig1_mat)))
#
#    # Likelihood, probably good?
#    likelihood = 0
#    for i in range(len(y)):
#        print("DATA: {}, MEAN: {}, SIGMA: {}".format(y[i], mu[i], sig2**2))
#        print(np.log(sp.stats.norm.pdf(y[i], mu[i], sig2**2)))
#        print()
#        likelihood += np.log(sp.stats.norm.pdf(y[i], mu[i], sig2**2))
#    print("LIKELIHOOD: {}".format(likelihood))
#    return prior + likelihood


def posterior_predictive(bnn, output, x, y):
    #weight_dists = [np.random.normal(flowed_means[i], flowed_vars[i]) 
    #                for i in range(len(flowed_means))]

    #D = len(flowed_means)
    xs = np.linspace(min(x)-0.5, max(x)+0.5, 200)
    fig, ax = plt.subplots()
    ax.scatter(x, y, color='k', label="True Data")
    pred_samp = []
    for i in range(500):
        #print(flowed_std[i])
        #sampled_weights = np.array([np.random.normal(flowed_means[i], flowed_std[i])
        #                            for i in range(D)])[np.newaxis,]
        sampled_weights = np.copy(output[np.random.randint(len(output))])
        pred = bnn.forward(sampled_weights[np.newaxis,], xs[np.newaxis,])
        pred_samp.append(pred[0][0])

    low = numpy.percentile(pred_samp, 2.5, axis=0)
    high = numpy.percentile(pred_samp, 97.5, axis=0)
    ax.fill_between(xs, low, high, alpha=0.3, label="95% Predictive Interval")
    ax.plot(xs, np.mean(pred_samp, axis=0), color='r', label="Prediction Mean")
    ax.legend(loc='best')
    plt.savefig("./nn_fit/fit.png")
    plt.show()


if __name__ == '__main__':
    # Read in data
    data = np.loadtxt("./HW7_data.csv", delimiter=',')
    x = data[:,0]
    y = data[:,1]

    ###define rbf activation function
    alpha = 1
    c = 0
    h = lambda x: np.exp(-alpha * (x - c)**2)
    
    ###neural network model design choices
    width = 5
    hidden_layers = 1
    input_dim = 1
    output_dim = 1
    
    architecture = {'width': width,
                   'hidden_layers': hidden_layers,
                   'input_dim': input_dim,
                   'output_dim': output_dim,
                   'activation_fn_type': 'rbf',
                   'activation_fn_params': 'c=0, alpha=1',
                   'activation_fn': h}
    
    #set random state to make the experiments replicable
    rand_state = 0
    random = np.random.RandomState(rand_state)
    
    #instantiate a Feedforward neural network object
    bnn = Feedforward(architecture, random=random)

    # Flow parameters
    num_flows = 3
    num_samples = 100
    h = np.tanh
    D = 2*bnn.D + 1
    flow_params = [1.] * D
    lambda_flows = np.array([np.array(flow_params)]*num_flows)

    # Generate samples
    samples = np.random.multivariate_normal([0]*bnn.D, np.eye(bnn.D), num_samples)
    print(samples.shape)

    # Energy bound from paper
    grad_energy_bound = autograd.grad(energy_bound)

    # Joint prob of BNN
    #joint_prob = lambda w: log_joint(w, x, y, bnn)
    joint_prob = lambda w: log_joint(w, 0, x, y, 5*np.ones(num_samples), 
                              1/5*np.eye(bnn.D), bnn, bnn.D)

    # Fit
    output = adam_solve(lambda_flows, grad_energy_bound, samples, joint_prob, h,
                        m=3000, step_size=0.01, bnn=True)

    # Gather parameters
    #flowed_means = np.mean(output, axis=0)
    #flowed_std = np.std(output, axis=0)

    # Predict and plot
    #posterior_predictive(bnn, flowed_means, flowed_std, x, y)
    posterior_predictive(bnn, output, x, y)
    
