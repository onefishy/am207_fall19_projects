#
# Subspace construction
#

import torch
import numpy as np
from inference import SWAG
from utils import set_model_weights
from sklearn.utils.extmath import randomized_svd

class RandomSubspace():
    def __init__(self, model_info, rank=10, seed=1, preloaded=None,
                 lr_ratio=1, save=True, save_path='model_weights/random_model.pt'):
        self.model_info = model_info
        self.rank = rank
        self.lr_ratio = lr_ratio

        if preloaded is not None:
            print('Loading pretrained random weights...')
            self.shift = preloaded['swag_shift']

        else:
            print('Computing Random Subspace...')
            self.shift, _ = SWAG(model_info['model'], model_info['dataloader'],
                                 model_info['optimizer'], model_info['criterion'], lr_ratio=self.lr_ratio)
            if save:
                save_dict = {'state_dict': self.model_info['model'].state_dict(),
                             'swag_shift': self.shift,
                             'swag_A': None}
                torch.save(save_dict, save_path)

        self.num_parameters = self.shift.shape[0]
        torch.manual_seed(seed)
        self.P = torch.randn(rank, self.num_parameters)  # self.cov_factor in author's github
        self.P /= torch.norm(self.P, dim=1)[:, None]


# Generate PCA subspace
class PCASubspace():
    def __init__(self, model_info, rank=10, lr_ratio=.1, preloaded=None,
                 save=True, save_path='model_weights/pca_model.pt'):

        self.model_info = model_info
        self.rank = rank
        self.lr_ratio = lr_ratio

        if preloaded is not None:
            print('Loading pretrained PCA weights...')
            self.shift = preloaded['swag_shift']
            self.A = preloaded['swag_A']
            model_weights = torch.Tensor(preloaded['model_weights'])
            set_model_weights(self.model_info['model'], model_weights)

        else:
            print('Computing PCA Subspace...')
            self.shift, self.A = SWAG(model_info['model'], model_info['dataloader'],
                                      model_info['optimizer'], model_info['criterion'], lr_ratio=self.lr_ratio)

            if save:
                save_dict = {'state_dict': self.model_info['model'].state_dict(),
                             'swag_shift': self.shift,
                             'swag_A': self.A}
                torch.save(save_dict, save_path)

        # perform PCA on A matrix
        #  Note: in the code they do PCA on A/sqrt(k - 1) but in the paper they say to do PCA on just A
        _, s, Vt = randomized_svd(self.A / np.sqrt(self.rank - 1), n_components=self.rank, n_iter=5)
        self.P = torch.FloatTensor(s[:, None] * Vt)