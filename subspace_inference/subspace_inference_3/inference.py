#
# Bayesian inference utilities
#

import torch
import numpy as np
import math
from tqdm import tqdm
from utils import decay_lr, set_model_weights

def SWAG(model, dataloader, optimizer, criterion, epochs=3000,
         print_freq=1000, swag_start=2000, M=1e5, lr_ratio=1, verbose=False):
    '''Implementation of Stochastic Weight Averaging'''
    model.train()  # prep model layers for training

    # initialize first moment as vectors w/ length = num model parameters
    num_params = sum(param.numel() for param in model.parameters())
    first_moment = torch.zeros(num_params)

    # initialize deviation matrix 'A'
    A = torch.empty(0, num_params, dtype=torch.float32)
    lr_init = optimizer.defaults['lr']

    n_iterates = 0
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0

        # Implementation of learning rate decay from paper
        epoch_ratio = (epoch + 1) / swag_start
        lr = decay_lr(optimizer, epoch_ratio, lr_init=lr_init, lr_ratio=lr_ratio)

        for inputs, labels in dataloader:
            optimizer.zero_grad()  # clear gradients

            preds = model(inputs)  # perform a forward pass
            loss = criterion(preds, labels)  # compute the loss

            loss.backward()  # backpropagate
            optimizer.step()  # update the weights

            epoch_loss += loss.data.item() * inputs.shape[0]

        # Print output
        if (epoch % print_freq == 0 or epoch == epochs - 1) and verbose:
            print('Epoch %d | LR: %g | Loss: %.4f' % (epoch, lr, epoch_loss))

        # Average gradient weights
        if epoch > swag_start:
            # obtain a flattened vector of weights
            weights_list = [param.detach() for param in model.parameters()]
            w = torch.cat([w.contiguous().view(-1, 1) for w in weights_list]).view(-1)

            # update the first moment
            first_moment = (n_iterates * first_moment + w) / (n_iterates + 1)

            # update 'a' matrix  (following their code implementation)
            a = w - first_moment
            A = torch.cat((A, a.view(1, -1)), dim=0)

            # only store the last 'M' deviation vectors if memory limited
            if A.shape[1] > M:
                A = A[1:, :]

            n_iterates += 1

    return first_moment.double(), A.numpy()


class ESS():
    '''Elliptical Slice Sampling (ESS) for Bayesian inference'''

    def __init__(self, subspace, num_samples=1000, T=1, noise_var=1, loss_type='NLL', nu_scale=1):
        # Initialize parameters
        self.subspace = subspace
        self.num_samples = num_samples
        self.T = T
        self.noise_var = noise_var
        self.loss_type = loss_type
        self.nu_scale = nu_scale

        # Initialize samples
        print(f'Initializing {num_samples} samples...')
        self.samples = self.init_samples()

    def log_likelihood(self, z):
        # Project sample into model weight space
        z = torch.from_numpy(z)
        w = self.subspace.shift + z @ self.subspace.P.double()

        # Set the weights of the neural network
        model = self.subspace.model_info['model']
        set_model_weights(model, w)

        # Do a forward pass of the neural network
        loss = 0
        num_datapoints = 0
        for batch_num, (inputs, labels) in enumerate(self.subspace.model_info['dataloader']):
            num_datapoints += inputs.size(0)
            if self.loss_type == 'NLL':
                batch_loss = self.subspace.model_info['criterion'](model(inputs), labels, noise_var=self.noise_var)
            elif self.loss_type == 'MSE':  # MSE
                batch_loss = self.subspace.model_info['criterion'](model(inputs), labels)
            else:
                raise ('Loss type not recognized')

            loss += batch_loss

        # Return likelihood w/ MSE loss
        loss = loss / (batch_num + 1) * num_datapoints
        if self.loss_type == 'NLL':  # Negative log likelihood loss
            factor = self.T
        elif self.loss_type == 'MSE':  # MSE loss
            factor = self.T * 2 * self.noise_var
        else:
            raise('Loss type not recognized')

        return -loss.numpy() / factor

    def init_samples(self):
        k = self.subspace.rank  # dimension of subspace
        z = np.zeros(k)  # initialize z-sample
        subspace_samples = np.zeros((self.num_samples, k))
        for i in tqdm(range(self.num_samples)):
            nu = np.random.multivariate_normal(np.zeros(k), np.eye(k) * self.nu_scale)  # sample nu from its prior
            z, _ = elliptical_slice(initial_theta=z, prior=nu, lnpdf=self.log_likelihood)
            subspace_samples[i, :] = z

        # Project samples back into model weight space
        s = torch.from_numpy(subspace_samples).double()
        samples = self.subspace.shift + s @ self.subspace.P.double()

        return samples

    def sample(self):
        # Return a random sample from the batch
        ind = np.random.randint(self.num_samples)
        return self.samples[ind, :]

def elliptical_slice(initial_theta, prior, lnpdf,
                     cur_lnpdf=None, angle_range=None, **kwargs):
    """
    NAME:
       elliptical_slice
    PURPOSE:
       Markov chain update for a distribution with a Gaussian "prior" factored out
    INPUT:
       initial_theta - initial vector
       prior - cholesky decomposition of the covariance matrix
               (like what np.linalg.cholesky returns),
               or a sample from the prior
       lnpdf - function evaluating the log of the pdf to be sampled
       kwargs= parameters to pass to the pdf
       cur_lnpdf= value of lnpdf at initial_theta (optional)
       angle_range= Default 0: explore whole ellipse with break point at
                    first rejection. Set in (0,2*pi] to explore a bracket of
                    the specified width centred uniformly at random.
    OUTPUT:
       new_theta, new_lnpdf
    HISTORY:
       Originally written in matlab by Iain Murray (http://homepages.inf.ed.ac.uk/imurray2/pub/10ess/elliptical_slice.m)
       2012-02-24 - Written - Bovy (IAS)
    """
    D = len(initial_theta)
    if cur_lnpdf is None:
        cur_lnpdf = lnpdf(initial_theta, **kwargs)

    # Set up the ellipse and the slice threshold
    if len(prior.shape) == 1:  # prior = prior sample
        nu = prior
    else:  # prior = cholesky decomp
        if not prior.shape[0] == D or not prior.shape[1] == D:
            raise IOError("Prior must be given by a D-element sample or DxD chol(Sigma)")
        nu = np.dot(prior, np.random.normal(size=D))
    hh = math.log(np.random.uniform()) + cur_lnpdf

    # Set up a bracket of angles and pick a first proposal.
    # "phi = (theta'-theta)" is a change in angle.
    if angle_range is None or angle_range == 0.:
        # Bracket whole ellipse with both edges at first proposed point
        phi = np.random.uniform() * 2. * math.pi
        phi_min = phi - 2. * math.pi
        phi_max = phi
    else:
        # Randomly center bracket on current point
        phi_min = -angle_range * np.random.uniform()
        phi_max = phi_min + angle_range
        phi = np.random.uniform() * (phi_max - phi_min) + phi_min

    # Slice sampling loop
    while True:
        # Compute xx for proposed angle difference and check if it's on the slice
        xx_prop = initial_theta * math.cos(phi) + nu * math.sin(phi)
        cur_lnpdf = lnpdf(xx_prop, **kwargs)
        if cur_lnpdf > hh:
            # New point is on slice, ** EXIT LOOP **
            break
        # Shrink slice to rejected point
        if phi > 0:
            phi_max = phi
        elif phi < 0:
            phi_min = phi
        else:
            raise RuntimeError('BUG DETECTED: Shrunk to current position and still not acceptable.')
        # Propose new angle difference
        phi = np.random.uniform() * (phi_max - phi_min) + phi_min
    return (xx_prop, cur_lnpdf)