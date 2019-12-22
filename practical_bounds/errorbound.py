from autograd import numpy as np
from autograd import scipy as sp
import scipy.stats as st
import numpy

class Errorbound():
    def __init__(self, mean, std, propose_family, log_joint_posterior, num_samples, t_dist_df=None):
        """
        Initiate a class to calculate ELBO, CUBO, error bounds for divergence metric and distance metric 
        
        Parameters
        ---------
        mean (array): array of mean of proposed distribution
        std (array): array of std of proposed distribution
        propose_family (str): "normal" or "t", indicate family of the proposed distribution
        t_dist_df (int): degree of freedom of proposed t distribution
        log_joint_posterior (density function): log density function of target posterior
        num_samples (int): number of samples
        
        Returns
        ------
        a class for error bound calculation
        """
        self.mean = mean
        self.std = std
        assert len(self.mean) == len(self.std) ## dimension of mean and standard deviation should be the same
        self.family = propose_family
        self.t_dist_df = t_dist_df
        self.log_joint_posterior = log_joint_posterior
        self.num_samples = num_samples
        
        # draw samples
        rs = np.random.RandomState()
        D = len(self.mean)
    
        if self.family=='normal':
            self.samples = rs.randn(self.num_samples, D) * self.std + self.mean
        if self.family=='t':
            assert self.t_dist_df is not None
            self.samples = rs.standard_t(df=self.t_dist_df, size=(self.num_samples, D)) * self.std + self.mean

    def c_2p(self):
        """
        Compute Cp for proposed distribution
        """
        p = 2
        samples_mean = self.samples.mean(axis=0, keepdims=True)
        samples_centered = self.samples - samples_mean
        samples_l2norm = np.sqrt(np.sum(samples_centered**2, axis=1))
        return 2*(np.mean(samples_l2norm**(2*p))**(1/(2*p)))
    
    def elbo(self):
        """
        Compute ELBO for proposed t distribution and the target posterior distribution
        
        Returns
        ------
        elbo lower bound of proposed t distribution and target posterior       
        """
        def gaussian_entropy(log_std):
            D = len(log_std)
            return 0.5 * D * (1.0 + np.log(2*np.pi)) + np.sum(log_std)
        
        def t_entropy(df, scale=None):
            ''' Taken from https://math.stackexchange.com/questions/2272184/differential-entropy-of-the-multivariate-student-t-distribution'''
            standard_entropy = ((df+1)/2)*(sp.special.digamma((df+1)/2) - sp.special.digamma(df/2)) + np.log(np.sqrt(df) * sp.special.beta(df/2, 1/2))
            if scale is None:
                return standard_entropy
            D = len(scale)
            return standard_entropy*D + np.log(scale).sum()

        if self.family == 'normal':
            lower_bound = np.mean(self.log_joint_posterior(self.samples)) + gaussian_entropy(np.log(self.std))
        elif self.family == 't':
            assert self.t_dist_df is not None
            lower_bound = np.mean(self.log_joint_posterior(self.samples)) + t_entropy(df=self.t_dist_df, scale=self.std) 
        return lower_bound
            
    def cubo_2(self):
        """
        Compute CUBO when alpha is 2 for proposed t distribution and the target posterior distribution
    
        Returns
        ------
        Chi upper bound of proposed t distribution and target posterior    
        """        
        joint_posterior = lambda x: np.exp(self.log_joint_posterior(x))
        if self.family=='normal':
            log_inner = self.log_joint_posterior(self.samples) - sp.stats.multivariate_normal.logpdf(self.samples, self.mean, np.diag(self.std**2))
        elif self.family=='t':
            assert self.t_dist_df is not None
            log_inner = self.log_joint_posterior(self.samples) - np.sum(sp.stats.t.logpdf(self.samples, df=self.t_dist_df, loc=self.mean, scale=self.std), axis=1)
    
        rescaled_inner = np.exp(log_inner - np.max(log_inner))**2
        upper_bound = 0.5*np.log(np.mean(rescaled_inner)) + np.max(log_inner)
        return upper_bound      
    
    def divergence_bound(self):
        """
        Compute error bound for divergence metrics when alpha is 2
    
        Returns
        ------
        Error bound for KL divergence and alpha divergence metrics 
        """
        return 2*(self.cubo_2() - self.elbo())
    
    def distance_bound(self):
        """
        Compute error bound for Wasserstein distance metric when p is 2
        
        Returns
        ------
        Error bound for Wasserstein distance metric 
        """        
        p=2
        return self.c_2p()*(np.exp(self.divergence_bound())-1)**(1/(2*p))
    
    def get_bounds(self, verbose=False):
        """
        Print CUBO, ELBO, error bound for divergence metrics, error bound for W2_bound if verbose is true
        Get error boundaries for different summary statistics
        
        Parameters
        ----------
        verbose(bool): Print intermidiate process or not
        
        Returns
        ------
        Error bounds for all kinds of summary statistics
        """
        elbo = self.elbo()
        cubo = self.cubo_2()
        Cp = self.c_2p()
        div_bound = self.divergence_bound()
        dis_bound = self.distance_bound()
        if verbose:
            print('cubo:', cubo)
            print('elbo:', elbo)
            print('div_bound:', div_bound)
            print('C_2p:', Cp)
            print('W2_bound:', dis_bound)
        if self.samples.shape[1] == 1:
            min_std = self.std[0]
        else:
            min_std = np.sqrt(np.cov(self.samples.flatten()))
        all_bounds = {'mean bound': dis_bound,
                 'std bound': 0.5*(np.sqrt(2) + np.sqrt(6))*dis_bound,
                 'variance bound': 2*np.sqrt(2)*min_std*dis_bound + (1+3*np.sqrt(2))*dis_bound**2}
        return all_bounds