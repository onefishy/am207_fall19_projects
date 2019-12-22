from autograd import numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.misc.optimizers import adam, sgd, rmsprop
#from autograd import scipy as sp
import numpy
import matplotlib.pyplot as plt
import sys

def trial1(z):
    z1, z2 = z[:, 0], z[:, 1]
    norm = np.sqrt(z1 ** 2 + z2 ** 2)
    exp1 = np.exp(-0.5 * ((z1 - 2) / 0.8) ** 2)
    exp2 = np.exp(-0.5 * ((z1 + 2) / 0.8) ** 2)
    u = 0.5 * ((norm - 4) / 0.4) ** 2 - np.log(exp1 + exp2)
    return np.exp(-u)

def gmm(z):
    gauss_uni = lambda x, mu, sigma: 1/np.sqrt(2*np.pi*sigma**2) * np.exp(-0.5*((x-mu)/sigma)**2)
    pi = np.array([0.5, 0.5])
    mu = np.array([-1, 1])
    sigma = np.array([0.5 ,0.5])
    return pi[0]*gauss_uni(z, mu[0], sigma[0]) + pi[1]*gauss_uni(z, mu[1], sigma[1])

def p1(z):
    z1, z2 = z[:, 0], z[:, 1]
    first = (np.linalg.norm(z, 2, 1) - 2)/0.4
    exp1 = np.exp(-0.5*((z1 - 2)/0.6)**2)
    exp2 = np.exp(-0.5*((z1 + 2)/0.6)**2)
    u = 0.5*first**2 - np.log(exp1 + exp2)
    return np.exp(-u)

def p2(z):
    z1, z2 = z[:, 0], z[:, 1]
    w1 = lambda x: np.sin(2 * np.pi * x/4)
    u = 0.5 * ((z2 - w1(z1))/0.4) ** 2
    dummy = np.ones(u.shape) * 1e7
    u = np.where(np.abs(z1) <= 4, u, dummy)
    return np.exp(-u)

m = lambda x: -1 + np.log(1 + np.exp(x))
h = lambda x: np.tanh(x)
h_prime = lambda x: 1 - np.tanh(x)**2


def gradient_create(target, eps, dim_z, num_samples, K):

    def unpack_params(params):
        W = params[:K*dim_z].reshape(K,dim_z)
        U = params[K*dim_z:2*K*dim_z].reshape(K,dim_z)
        B = params[-K:]
        return W,U,B

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

    return variational_objective, gradient, unpack_params

K = 4
dim_z = 1
num_samples = 1000
num_iter = 10000
func = gmm

objective, gradient, unpack_params = gradient_create(func, 1e-7, dim_z, num_samples, K)

objectives = []
def callback(params, t, g):
    if t%100 == 0:
        print("Iteration {}; Gradient mag: {}; Objective: {}".format(t,
            np.linalg.norm(gradient(params, t)), objective(params, t)))
        objectives.append(objective(params, t))
    if t%5000 == 0:
        W, U, B = unpack_params(params)
        z0 = np.random.randn(num_samples, dim_z)
        z_prev = z0
        for k in range(K):
            w, u, b = W[k], U[k], B[k]
            u_hat = (m(np.dot(w,u)) - np.dot(w,u)) * (w / np.linalg.norm(w)) + u
            z_prev = z_prev + np.outer(h(np.matmul(z_prev, w) + b), u_hat)
        z_K = z_prev
        plt.figure(figsize=(5,4))
        plt.hist(z_K, 100, density=True)
        plt.show()


init_W = 1*np.ones((K, dim_z))
init_U = 1*np.ones((K, dim_z))
init_b = 1*np.ones((K))
init_params = np.concatenate((init_W.flatten(), init_U.flatten(), init_b.flatten()))

variational_params = sgd(gradient, init_params, callback, num_iter, 5e-4)


W, U, B = unpack_params(variational_params)
z0 = np.random.randn(num_samples, dim_z)
z_prev = z0
for k in range(K):
    w, u, b = W[k], U[k], B[k]
    u_hat = (m(np.dot(w,u)) - np.dot(w,u)) * (w / np.linalg.norm(w)) + u
    z_prev = z_prev + np.outer(h(np.matmul(z_prev, w) + b), u_hat)
z_K = z_prev

plt.figure(figsize=(10,8))
plt.plot(objectives)
plt.show()

# fig,ax=plt.subplots(1,1,figsize = (10,8))
# nbins = 100
# x, y = z0[:, 0], z0[:, 1]
# xi, yi = numpy.mgrid[-4:4:nbins*1j, -4:4:nbins*1j]
# zi = np.array([func(np.vstack([xi.flatten(), yi.flatten()])[:,i].reshape(-1,2)) for i in range(nbins**2)])
# ax.pcolormesh(xi, yi, zi.reshape(xi.shape))
# ax.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.Reds_r)
# plt.scatter(z_K[:,0], z_K[:,1], alpha=0.2)
# plt.xlim([-4, 4])
# plt.ylim([-4, 4])
# # plt.savefig('results/'+func.__name__+'/'+func.__name__+'_'+str(K)+'_'+str(num_iter)+'.png')
# plt.show()

plt.figure(figsize=(10,8))
samples = np.linspace(-3, 3, 601)
z = np.array([gmm(samples[i]) for i in range(samples.shape[0])])
idx = np.argsort(samples)
plt.plot(samples[idx], z[idx], label='p')
plt.hist(z_K, 100, label='q', density=True)
plt.legend()
plt.show()
