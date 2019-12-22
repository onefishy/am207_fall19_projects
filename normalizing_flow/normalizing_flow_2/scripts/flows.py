import autograd
import autograd.numpy as np
import autograd.scipy as sp
import autograd.misc.optimizers
import numpy
import time
import os
import sys
from tqdm import tqdm
from matplotlib import pyplot as plt
from autograd.misc.optimizers import adam
#from pynverse import inversefunc

e_bound = []
joint_probs = []
flow_probs = []
grad_norms = []
m = 10000
start = time.time()
q_0_mu = np.array([0,0])
q_0_sigma = 1
D = q_0_mu.shape[0]
def callback(x, i, g, num_samples=100):
    '''
        Callback function used in Adam solver. Has functionality to plot intermediate steps
        and do progress bar
    '''
    grad_norms.append(np.linalg.norm(g))
    if(i%10 == 0):
        left = '['
        right = ']'
        eq = '=' * int(20*i/m)
        blank = ' ' * int(np.ceil(20*(1 - i/m)))
        sys.stdout.write("{0}{1}{2}{3}  {4:.3f}%  {5:.2f}s\r".format(
                         left, eq, blank, right, 100*i/m, time.time()-start))
        sys.stdout.flush()
    if(i==(m-1)):
        sys.stdout.write("{}\r".format(' '*50))
        sys.stdout.flush()
        print("[{}]  100%  {}".format(20*'=', time.time() - start))
    if(i%10 == 0):
        if(i==0):
            leading_zeros = int(np.log(m)/np.log(10))
        elif(i==1000):
            leading_zeros = int(np.log(m)/np.log(10)) - int(np.log(i)/np.log(10)) - 1
        # else:
        #     leading_zeros = int(np.log(m)/np.log(10)) - int(np.log(i)/np.log(10))
        # zeros = '0'*leading_zeros
        # new_samples = np.random.randn(num_samples)[:,np.newaxis]
        # #new_samples = np.random.uniform(-1, 1, num_samples)[:,np.newaxis]
        # #new_samples = np.random.multivariate_normal(q_0_mu, q_0_sigma*np.eye(D), num_samples)
        # flowed_samples = flow_samples(x, new_samples, np.tanh)
        # fig, ax = plt.subplots()
        # ax.hist(flowed_samples, bins=int(np.sqrt(num_samples)), density=True)
        # #ax = setup_plot(u1)
        # #ax.scatter(flowed_samples[:,0], flowed_samples[:,1], alpha=0.4)
        # #ax.set(xlim=(-4,4), ylim=(-4,4), title="{} Flows, Iteration {}".format(20, i))
        # #plt.savefig("./data_fit_1d/{}{}.png".format(zeros, i))
        # plt.close()


###
#   Toy density functions
###
def w1(z):
    return np.sin(2*np.pi*z/4)

def w2(z):
    return 3*np.exp(-1/2*((z-1)/0.6)**2)

def u1(z, N=1):
    exp_factor = 1/2*((np.linalg.norm(z, axis=2) - 2)/0.4)**2 - \
          np.log(np.exp(-1/2*((z[:,:,0] - 2)/0.6)**2) + np.exp(-1/2*((z[:,:,0] + 2)/0.6)**2))
    return N * np.exp(-exp_factor)


def u2(z, N=1):
    exp_factor = 1/2*((z[:,:,1] - np.sin(2*np.pi*z[:,:,0]/4))/0.4)**2
    return np.exp(-exp_factor)


def u3(z, N=1):
    exp_factor = -np.log(np.exp(-1/2*((z[:,:,1] - w1(z[:,:,0]))/0.35)**2) + \
                  np.exp(-1/2*((z[:,:,1] - w1(z[:,:,0]) + w2(z[:,:,0]))/0.35)**2))
    return np.exp(-exp_factor)


def setup_plot(u_func):
    '''
        Function used to set up plot of target density, returns axis object for additional
        plotting
    '''
    try:
        X, Y = numpy.mgrid[-4:4:0.05, -4:4:0.05]
        dat = np.dstack((X, Y))
        U_z1 = u_func(dat)
        
        fig, ax = plt.subplots()
        ax.contourf(X, Y, U_z1, cmap='Reds', levels=15)
    except (TypeError, ValueError):
        plt.close()
        x = np.linspace(-8, 8, 1000)
        fig, ax = plt.subplots()
        ax.plot(x, u_func(x), label="Target Distribution")
        ax.set(title="Comparison of Target Density and Flowed Samples")
    return ax


def plot_shape():
    '''
        Simply plots target density
    '''
    ax = setup_plot(u_func)
    plt.show()


def plot_shape_samples(samples, u_func):
    '''
        plots target density and samples
    '''
    ax = setup_plot(u_func)
    ax.scatter(samples[:, 0], samples[:, 1], alpha=.5)
    plt.show()


def flow_once(lambda_flow, z, h):
    '''
        Flow one planar tranfsormation flow
    '''
    D = (lambda_flow.shape[0]-1)//2
    return z + np.dot(h(np.dot(z, lambda_flow[D:2*D].reshape(-1, 1))+lambda_flow[-1]), \
           lambda_flow[:D].reshape(1, -1))


# lambda u, w, b
def flow_samples(lambda_flows, z, h):
    '''
        Transform sample through multiple flows
    '''
    D = (lambda_flows.shape[1]-1)//2
    for lambda_flow in lambda_flows:
        z = flow_once(lambda_flow, z, h)
    return z


# Psi
def psi(lambda_flow, z, h):
    '''
        Computes log-det-jacobian according to formula from the paper
    '''
    D = (lambda_flow.shape[0]-1)//2
    return (1-h(np.dot(z, lambda_flow[D:2*D].reshape(-1, 1))+lambda_flow[-1])**2) * \
            lambda_flow[D:2*D]


# Calculate energy bound
def energy_bound(lambda_flows, z, h, u_func, beta=1., bnn=False):
    '''
        Energy bound formula from the paper. We exclude the initial sampling contribution
        because it is independent of flow parameters.
    '''
    D = (lambda_flows.shape[1]-1)//2
    #initial_exp = np.mean(np.log(sp.stats.norm.pdf(z, loc=q_0_mu, scale=np.sqrt(q_0_sigma))))
    initial_exp = 0
    if(bnn):
        joint_exp = beta*np.mean(u_func(flow_samples(lambda_flows, z, h)))
    else:
        joint_exp = beta*np.mean(np.log(u_func(flow_samples(lambda_flows, z, h
                                                            ).reshape(1, -1, 2))))
    #print("JOINT EXP: {}".format(joint_exp))

    # log-det-jacobian contribution from the paper
    flow_exp = 0
    for k, lambda_flow in enumerate(lambda_flows):
        flow_exp = flow_exp + \
                   np.mean(np.log(np.abs(1 + np.dot(psi(lambda_flow, z, h), lambda_flow[:D]))))
        z = flow_once(lambda_flow, z, h)

    # Store probabilities for plotting and analysis
    e_bound.append((initial_exp - joint_exp - flow_exp)._value)
    joint_probs.append(joint_exp._value)
    flow_probs.append(flow_exp._value)
    return initial_exp - joint_exp - flow_exp


def get_joint_exp(lambda_flows, z, h, u_func):
    '''
        Get joint contribution to energy for gradient descent according to formula
        from the paper
    '''
    return np.mean(np.log(u_func(flow_samples(lambda_flows, z, h).reshape(1, -1, 2))))


def get_flow_exp(lambda_flows, z, h):
    '''
        Get flow contribution to energy function for gradient descent
    '''
    D = (lambda_flows.shape[1]-1)//2
    flow_exp = 0
    for lambda_flow in lambda_flows:
        flow_exp = flow_exp + \
                   np.mean(np.log(np.abs(1 + np.dot(psi(lambda_flow, z, h), lambda_flow[:D]))))
        z = flow_once(lambda_flow, z, h)
    return flow_exp


def gradient_descent(m, lambda_flows, grad_energy_bound, samples):
    '''
        Gradient descent for finding parameters. This may not work anymore since switching over
        to the Adam optimizer.
    '''
    energy_hist = np.empty(m)
    joint_hist = np.empty(m)
    flow_hist = np.empty(m)
    lambda_hist = np.empty((m, *lambda_flows.shape))
    samples_flowed = samples
    for i in tqdm(range(m)):
        beta = min(1, 0.01+i/10000)
        samples_flowed = flow_samples(lambda_flows, samples, h)
        
        gradient = grad_energy_bound(lambda_flows, samples, h, beta)
        lambda_flows -= step_size*gradient
        #lambda_flows = autograd.misc.optimizers.adam(grad_energy_bound, lambda_flows)
        
        # Debug
        energy_hist[i] = energy_bound(lambda_flows, samples, h)
        joint_hist[i] = get_joint_exp(lambda_flows, samples, h)
        flow_hist[i] = get_flow_exp(lambda_flows, samples, h)
        lambda_hist[i] = lambda_flows
        
        # Plot
        if i % 20 == 0:
            if(i==0):
                leading_zeros = int(np.log(m)/np.log(10))
            elif(i==1000):
                leading_zeros = int(np.log(m)/np.log(10)) - int(np.log(i)/np.log(10)) - 1
            else:
                leading_zeros = int(np.log(m)/np.log(10)) - int(np.log(i)/np.log(10))
            zeros = '0'*leading_zeros

            ax = setup_plot(u_func)
            ax.scatter(samples_flowed[:, 0], samples_flowed[:, 1], alpha=.5)
            plt.savefig("./plots/{}{}.png".format(zeros, i))
            plt.close()


def adam_solve(lambda_flows, grad_energy_bound, samples, u_func, h, m=1000, step_size=0.001,
               bnn=False):
    '''
        Uses adam solver to optimize the energy bound
    '''
    output = np.copy(lambda_flows) # Copies to avoid changing initial conditions
    print("BEFORE LEARNING:\n{}".format(output))
    grad_energy_bound = autograd.grad(energy_bound)  # Autograd gradient of energy
    g_eb = lambda lambda_flows, i: grad_energy_bound(lambda_flows, samples, h, u_func, 
                                                     #beta= (0.1 + i/1000))
                                                     beta=min(2, i/1000), # Annealing
                                                     #beta=min(1, 0.01+i/10000),
                                                     bnn=bnn) # Annealing
    output = adam(g_eb, output, num_iters=m, callback=callback, step_size=step_size)
    print("\nAFTER LEARNING:\n{}".format(output))

    #samples = np.random.randn(30000)[:,np.newaxis] # Plot with more samples for better clarity
    q_0_mu = np.array([0,0])
    q_0_sigma = 1
    D = q_0_mu.shape[0]
    #samples = np.random.multivariate_normal(q_0_mu, q_0_sigma*np.eye(D), 20000)

    samples_flowed = flow_samples(output, samples, h)
    #np.savetxt("./data_fit_1d/flow_params.txt", output)
    np.savetxt("./nn_fit/flow_params.txt", output)
    if(bnn):
        np.savetxt("./nn_fit/energy_bound.txt", e_bound)
        fig, ax = plt.subplots()
        ax.plot(e_bound)
        ax.set(title="Energy Bound")
        plt.savefig("./nn_fit/energy_bound.png")
        plt.close()

        np.savetxt("./nn_fit/joint_probs.txt", joint_probs)
        fig, ax = plt.subplots()
        ax.plot(joint_probs)
        ax.set(title="Joint Probability")
        plt.savefig("./nn_fit/joint_probs.png")
        plt.close()

        np.savetxt("./nn_fit/flow_probs.txt", flow_probs)
        fig, ax = plt.subplots()
        ax.plot(flow_probs)
        ax.set(title="Flow Probs")
        plt.savefig("./nn_fit/flow_probs.png")
        plt.close()

        np.savetxt("./nn_fit/grad_norms.txt", grad_norms)
        fig, ax = plt.subplots()
        ax.plot(grad_norms)
        ax.set(title="Gradient Norms")
        plt.savefig("./nn_fit/grad_norms.png")
        plt.close()


    return samples_flowed


def shape_fit_2d(m, step_size, u_func, num_flows=8, num_samples=1000):
    # Parameters
    h = np.tanh
    
    q_0_mu = np.array([0,0])
    q_0_sigma = 1
    D = q_0_mu.shape[0]

    # 2D flows
    #lambda_flows = np.array([np.array([1., 0., 4., 5., 0.])]*num_flows)
    lambda_flows = np.array([np.array([1., 1., 0., 0., 0.])]*num_flows)

    # 2D samples
    samples = np.random.multivariate_normal(q_0_mu, q_0_sigma*np.eye(D), num_samples)

    # Gradient of energy function -> used to minimize energy
    grad_energy_bound = autograd.grad(energy_bound)

    #gradient_descent(m, lambda_flows, grad_energy_bound, samples)
    flowed_samples = adam_solve(lambda_flows, grad_energy_bound, samples,
                                u_func, h, m, step_size)

    os.system("cd ./plots/ ; convert -delay 10 -loop 0 *.png learning_flows.gif")
    # Plot Transformed samples
    ax = setup_plot(u_func)
    #print(samples_flowed.shape)
    ax.scatter(flowed_samples[:,0], flowed_samples[:,1], alpha=0.2)
    ax.set(title="{} Flows, {} Iterations, {} Step Size".format(num_flows, m, step_size),
           xlim=(-4,4), ylim=(-4,4))
    plt.savefig("./2d_plots/adam_fit_test.png")

    # Convert plots to gif or mp4
    #plot_str = "cd ./plots/ ; ffmpeg -pattern_type glob -i \"*.png\" -c:v "
    #plot_str += "libx264 -pix_fmt yuv420p -movflags +faststart learning_flows.mp4"
    #os.system(plot_str)


def shape_fit_1d(m, step_size, u_func, num_flows=8, num_samples=1000):
    # Parameters
    h = np.tanh
    
    q_0_mu = np.array([0,0])
    q_0_sigma = 10
    D = q_0_mu.shape[0]

    # flows
    #lambda_flows = np.array([np.array([1., 1., 1., 1., 0.])])
    #lambda_flows = np.array([np.array([1., 1., 0.])]*num_flows)
    lambda_flows = np.loadtxt("./data_fit_1d/flow_params.txt")

    # 1D samples
    samples = np.random.randn(num_samples)[:,np.newaxis]
    #samples = np.random.uniform(-1, 1, num_samples)[:,np.newaxis]
    
    start = time.time()
    grad_energy_bound = autograd.grad(energy_bound)

    # JOINT PROBABILITY IS NEW U_FUNC
    #print(energy_bound(lambda_flows, samples, h, u_func))

    #target = lambda x: (sp.stats.norm.pdf((x-2)) + sp.stats.norm.pdf((x+2)))/2

    #gradient_descent(m, lambda_flows, grad_energy_bound, samples)
    flowed_samples = adam_solve(lambda_flows, grad_energy_bound, samples,
                                u_func, h, m, step_size)

    # Plot Transformed samples
    ax = setup_plot(u_func)
    ax.hist(flowed_samples, bins=100, alpha=0.5, density=True, label="Transformed Samples")
    #plt.savefig("./plots/adam_fit_test.png")
    ax.legend(loc='best')
    plt.savefig("./data_fit_1d/adam_fit.png")

    # Convert plots to gif or mp4
    #os.system("cd ./plots/ ; convert -delay 10 -loop 0 *.png learning_flows.gif")
    #plot_str = "cd ./plots/ ; ffmpeg -pattern_type glob -i \"*.png\" -c:v "
    #plot_str += "libx264 -pix_fmt yuv420p -movflags +faststart learning_flows.mp4"
    #os.system(plot_str)


if __name__ == '__main__':
    m = 14000
    m = 10000
    m = 1
    u_func = u1   # 2D Shape fit
    #u_func = lambda x: (sp.stats.norm.pdf((x-4)) + sp.stats.norm.pdf((x+4)))/2 # 1D Shape fit
    #u_func = lambda x: sp.stats.gamma.pdf(x, 1)
    #u_func = lambda x: sp.stats.laplace.pdf(x, 4)
    u_func = lambda x: (1/2*np.exp(-np.abs(x-2)) + 1/2*np.exp(-np.abs(x)) + \
                        1/2*np.exp(-np.abs(x+2)))/3

    #target = lambda x: (sp.stats.norm.pdf((x-4)) + sp.stats.norm.pdf((x+4)))/2  # 1D Shape fit
    #u_func = lambda x, z: (sp.stats.norm.pdf(x-4-z) + sp.stats.norm.pdf(x-4-z))/2 * \
    #                      (sp.stats.norm.pdf(x-4) + sp.stats.norm.pdf(x-4))/2
    print("SAMPLING")

    num_samples = 20000
    num_samples = 2000
    #x_dat = sample(target, 1, -8, 8, num_samples)

    step_size = .0002
    num_flows = 20
    start = time.time()
    #shape_fit_2d(m, step_size, u_func, num_flows, num_samples)
    shape_fit_1d(m, step_size, u_func, num_flows, num_samples)
    
    fig, ax = plt.subplots(nrows=4, figsize=(8,20))
    ax[0].plot(grad_norms, label="Norm of gradient")
    ax[0].legend(loc='best')
    ax[1].plot(e_bound, label="Energy Bound")
    ax[1].legend(loc='best')
    ax[2].plot(joint_probs, label="Joint Probability")
    ax[2].legend(loc='best')
    ax[3].plot(flow_probs, label="Flow Probability")
    ax[3].legend(loc='best')
    #plt.savefig("./data_fit_1d/probabilities.png")
    plt.savefig("./2d_plots/probabilities.png")
    plt.show()

