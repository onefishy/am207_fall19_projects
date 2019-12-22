#
# Utility functions
#

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

def unflatten_like(vector, likeTensorList):
    # Takes a flat torch.tensor and unflattens it to a list of torch.tensors (model params)
    # shaped like likeTensorList
    outList = []
    i=0
    for tensor in likeTensorList:
        n = tensor.numel()
        outList.append(vector[:,i:i+n].view(tensor.shape))
        i+=n
    return outList

def set_model_weights(model, weights):
    vec_list = unflatten_like(likeTensorList=list(model.parameters()), vector=weights.view(1,-1))
    for param, v in zip(model.parameters(), vec_list):
        param.detach_()  # I'm assuming this is so we don't update weights during inference
        param.mul_(0.0).add_(v.float())

def plot_predictive(data, trajectories, xs, mu=None, sigma=None, title=None, verbose=False, save_path=None,ax=None):
    sns.set()
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(data[:, 0], data[:, 1], facecolor='red', alpha=0.6, edgecolor="black")

    if mu is None:
        mu = np.mean(trajectories, axis=0)
    if sigma is None:
        sigma = np.std(trajectories, axis=0)

    if verbose:
        print('mu: ', mu)
        print('sigma: ', sigma)

    ax.plot(xs, mu, "-", lw=2., color='b')
    ax.plot(xs, mu - 3 * sigma, "-", lw=0.75, color='b')
    ax.plot(xs, mu + 3 * sigma, "-", lw=0.75, color='b')
    np.random.shuffle(trajectories)
    for traj in trajectories[:10]:
        ax.plot(xs, traj, "-", alpha=.5, color='b', lw=1.)

    ax.fill_between(xs, mu - 3 * sigma, mu + 3 * sigma, alpha=0.35, color='b')

    ax.set_xlim([np.min(xs), np.max(xs)])
    # ax.set_xlabel('X', fontsize=15)
    # ax.set_ylabel('Y', fontsize=15)
    if title:
        ax.set_title(title, fontsize=20)
    # ax.set_axis_off()
    ax.set_yticklabels([])
    if save_path is not None:
        plt.savefig(save_path, bbox_inches = 'tight', pad_inches = 0)


def preload_author_model(path="data/swag_checkpoint0-3000.pt"):
    # Load in data from authors
    swag_model = torch.load(path)['state_dict']
    swag_shift = swag_model['mean']
    swag_A = swag_model['subspace.cov_mat_sqrt'].numpy()

    model_weights = np.array([])
    for key, value in swag_model.items():
        if 'base_model' in key:
            model_weights = np.append(model_weights, value.numpy().flatten())

    preloaded = {'swag_shift': swag_shift, 'swag_A': swag_A, 'model_weights': model_weights}

    return preloaded


def preload_our_model(path="model_weights/pca_model.pt"):
    # Load in data from authors
    save_dict = torch.load(path)
    swag_model = save_dict['state_dict']
    swag_shift = save_dict['swag_shift']
    swag_A = save_dict['swag_A']

    model_weights = np.array([])
    for key, value in swag_model.items():
        model_weights = np.append(model_weights, value.numpy().flatten())

    preloaded = {'swag_shift': swag_shift, 'swag_A': swag_A, 'model_weights': model_weights}

    return preloaded


def decay_lr(optimizer, epoch_ratio, lr_init=1e-2, lr_ratio=.05, start=0.5):
    '''
    Decay the learning rate linearly from 'start' until 'min_lr'
    '''
    if epoch_ratio <= start:
        factor = 1.0
    else:
        factor = np.max([lr_ratio, 1.0 - (1.0 - lr_ratio) * (epoch_ratio - 0.5) / 0.4])

    lr = factor * lr_init
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def NLL_loss(inputs, targets, noise_var=0.5):
    mse = nn.MSELoss()
    return mse(inputs, targets)/(2*noise_var)

def get_criterion(loss='NLL'):
    if loss == 'NLL':
        return NLL_loss
    elif loss == 'MSE':
        return nn.MSELoss()
    else:
        raise('Loss function not recognized')