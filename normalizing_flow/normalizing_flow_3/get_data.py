from autograd import numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.misc.optimizers import adam, sgd, rmsprop
from autograd import scipy as sp
import pandas as pd
import numpy
import scipy
import matplotlib.pyplot as plt
import statsmodels.api as sm
from normflows import flows
from tqdm import tqdm
import sys

sigmoid = lambda x: 1/(1+np.exp(-x))
w1 = lambda x: np.sin(2 * np.pi * x/4)
w2 = lambda x: 3*np.exp(-0.5*((x-1)/0.6)**2)
w3 = lambda x: 3*sigmoid((x-1)/0.3)

def trial1(z):
    z1, z2 = z[:, 0], z[:, 1]
    norm = np.sqrt(z1 ** 2 + z2 ** 2)
    exp1 = np.exp(-0.5 * ((z1 - 2) / 0.8) ** 2)
    exp2 = np.exp(-0.5 * ((z1 + 2) / 0.8) ** 2)
    u = 0.5 * ((norm - 4) / 0.4) ** 2 - np.log(exp1 + exp2)
    return np.exp(-u)

def p1(z):
    '''Apply posterior p_1 to bivariate z.'''
    z1, z2 = z[:, 0], z[:, 1]
    first = (np.linalg.norm(z, 2, 1) - 2)/0.4
    exp1 = np.exp(-0.5*((z1 - 2)/0.6)**2)
    exp2 = np.exp(-0.5*((z1 + 2)/0.6)**2)
    u = 0.5*first**2 - np.log(exp1 + exp2)
    return np.exp(-u)

def p2(z):
    '''Apply posterior p_2 to bivariate z.'''
    z1, z2 = z[:, 0], z[:, 1]
    u = 0.5 * ((z2 - w1(z1))/0.4) ** 2
    dummy = np.ones(u.shape) * 1e7
    u = np.where(np.abs(z1) <= 4, u, dummy)
    return np.exp(-u)

def p3(z):
    '''Apply posterior p_3 to bivariate z.'''
    z1, z2 = z[:, 0], z[:, 1]
    exp1 = np.exp(-0.5*((z2 - w1(z1))/0.35)**2)
    exp2 = np.exp(-0.5*((z2 - w1(z1) + w2(z1))/0.35)**2)
    u = - np.log(exp1 + exp2)
    return np.exp(-u)

def p4(z):
    '''Apply posterior p_4 to bivariate z.'''
    z1, z2 = z[:, 0], z[:, 1]
    exp1 = np.exp(-0.5*((z2 - w1(z1))/0.4)**2)
    exp2 = np.exp(-0.5*((z2 - w1(z1) + w3(z1))/0.35)**2)
    u = - np.log(exp1 + exp2)
    return np.exp(-u)

def gmm(z):
    gauss_uni = lambda x, mu, sigma: 1/np.sqrt(2*np.pi*sigma**2) * np.exp(-0.5*((x-mu)/sigma)**2)
    pi = np.array([0.3, 0.7])
    mu = np.array([-1, 3])
    sigma = np.array([1 , 1])
    return pi[0]*gauss_uni(z, mu[0], sigma[0]) + pi[1]*gauss_uni(z, mu[1], sigma[1])

m = lambda x: -1 + np.log(1 + np.exp(x))
h = lambda x: np.tanh(x)
h_prime = lambda x: 1 - np.tanh(x)**2

def gradient_create(target, eps, dim_z, num_samples, K):

    def unpack_params(params):
        W = params[:K*dim_z].reshape(K,dim_z)
        U = params[K*dim_z:2*K*dim_z].reshape(K,dim_z)
        B = params[-K:]
        return W,U,B

    def other_metrics(params, t):
        W,U,B = unpack_params(params)
        z0 = np.random.multivariate_normal(np.zeros(dim_z), np.eye(dim_z),
                num_samples)
        z_prev = z0
        sum_log_det_jacob = 0.
        for k in range(K):
            w, u, b = W[k], U[k], B[k]
            u_hat = (m(np.dot(w,u)) - np.dot(w,u)) * (w / np.linalg.norm(w)) + u
            affine = np.outer(h_prime(np.matmul(z_prev, w) + b), w)
            sum_log_det_jacob += np.log(eps + np.abs(1 + np.matmul(affine, u)))
            z_prev = z_prev + np.outer(h(np.matmul(z_prev, w) + b), u_hat)
        z_K = z_prev
        log_q_K = -0.5 * np.sum(np.log(2*np.pi) + z0**2, 1) - sum_log_det_jacob
        log_p = np.log(eps + target(z_K))
        return np.mean(log_q_K), np.mean(log_p)

    def variational_objective(params, t):
        """Provides a stochastic estimate of the variational lower bound."""
        W,U,B = unpack_params(params)
        z0 = np.random.multivariate_normal(np.zeros(dim_z), np.eye(dim_z),
                num_samples)
        z_prev = z0
        sum_log_det_jacob = 0.
        for k in range(K):
            w, u, b = W[k], U[k], B[k]
            u_hat = (m(np.dot(w,u)) - np.dot(w,u)) * (w / np.linalg.norm(w)) + u
            affine = np.outer(h_prime(np.matmul(z_prev, w) + b), w)
            sum_log_det_jacob += np.log(eps + np.abs(1 + np.matmul(affine, u)))
            z_prev = z_prev + np.outer(h(np.matmul(z_prev, w) + b), u_hat)
        z_K = z_prev
        log_q_K = -0.5 * np.sum(np.log(2*np.pi) + z0**2, 1) - sum_log_det_jacob
        log_p = np.log(eps + target(z_K))
        return np.mean(log_q_K - log_p)

    gradient = grad(variational_objective)

    return variational_objective, gradient, unpack_params, other_metrics

def optimise(func, num_samples, num_iter, lr, K, dim_z=2):

    objective, gradient, unpack_params, other_metrics = gradient_create(func, 1e-7, dim_z, num_samples, K)
    init_W = 0.1*np.ones((K, dim_z))
    init_U = 0.1*np.ones((K, dim_z))
    init_b = 0.1*np.ones((K))

    init_params = np.concatenate((init_W.flatten(), init_U.flatten(), init_b.flatten()))
    
    def callback(params, t, g):
        if t%100 == 0:
            energy = objective(params, t)
            joint, entropy = other_metrics(params, t)
            print("Iteration {}; Energy: {}; Joint: {}; Entropy: {}".format(t, energy, joint, entropy))
        if (t+1)%num_iter == 0:
            print("\nFINAL METRICS\n")
            joint, entropy = other_metrics(params, t)
            print("Free energy: ", objective(params, t))
            print("Joint: ", joint)
            print("Entropy: ", entropy)

    variational_params = rmsprop(gradient, init_params, callback, num_iter, lr)
    
    return variational_params

num_samples=100
K=int(sys.argv[1])
num_iter=int(sys.argv[2])
lr=float(sys.argv[3])
variational_params = optimise(p1, num_samples, num_iter, lr, K)