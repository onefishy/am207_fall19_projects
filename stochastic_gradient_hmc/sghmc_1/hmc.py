### AM 207 Final Project
###
# some baseline code for original HMC only from:
# https://colindcarroll.com/2019/04/11/hamiltonian-monte-carlo-from-scratch/

from autograd import grad
import autograd.numpy as np
import scipy.stats as st
import time


def leapfrog(M, q, p, dVdq, path_len, step_size):
    """Leapfrog integrator for standard HMC and naive SGHMC

    Parameters
    ----------
    M : np.matrix
      Mass of the Euclidean-Gaussian kinetic energy of shape D x D
    q : np.floatX
      Initial position
    p : np.floatX
      Initial momentum
    dVdq : callable
      Gradient of the velocity
    path_len : float
      How long to integrate for
    step_size : float
      How long each integration step should be

    Returns
    -------
    q, p : np.floatX, np.floatX
      New position and momentum
    """
    q, p = np.copy(q), np.copy(p)
    Minv = np.linalg.inv(M)

    p -= step_size * dVdq(q) / 2  # half step
    for _ in range(int(path_len / step_size) - 1):
        q += step_size * np.dot(Minv, p)  # whole step
        p -= step_size * dVdq(q)  # whole step
    q += step_size * np.dot(Minv, p)  # whole step
    p -= step_size * dVdq(q) / 2  # half step

    # momentum flip at end
    return q, -p


def leapfrog_friction(M, C, V, q, p, dVdq, path_len, step_size):
    """Leapfrog integrator for Stochastic Gradient Hamiltonian Monte Carlo.
    Includes friction term per https://arxiv.org/abs/1402.4102

    Parameters
    ----------
    M : np.matrix
      Mass of the Euclidean-Gaussian kinetic energy of shape D x D
    C : matrix
      Upper bound parameter for friction term
    V: matrix
      Covariance of the stochastic gradient noise, such that B = V/2
    q : np.floatX
      Initial position
    p : np.floatX
      Initial momentum
    dVdq : callable
      Gradient of the velocity
    path_len : float
      How long to integrate for
    step_size : float
      How long each integration step should be

    Returns
    -------
    q, p : np.floatX, np.floatX
      New position and momentum
    """
    q, p = np.copy(q), np.copy(p)
    Minv = np.linalg.inv(M)
    B = 0.5 * V
    D = np.sqrt(2 * (C - B))
    D = np.diagonal(D)

    p -= step_size * (dVdq(q) + np.dot(C, np.dot(Minv, p)) - np.random.normal() * D) / 2  # half step
    for _ in range(int(path_len / step_size) - 1):
        q += step_size * np.dot(Minv, p)  # whole step
        p -= step_size * (dVdq(q) + np.dot(C, np.dot(Minv, p)) - np.random.normal() * D)  # whole step
    q += step_size * np.dot(Minv, p)  # whole steps
    p -= step_size * (dVdq(q) + np.dot(C, np.dot(Minv, p)) - np.random.normal() * D) / 2  # half step

    # momentum flip at end
    return q, -p


def hmc(M, n_samples, negative_log_prob, initial_position,
    path_len=1, step_size=0.5, diagnostics=False, mh=True):
    """Hamiltonian Monte Carlo sampling.

    Parameters
    ----------
    M : np.matrix
      Mass of the Euclidean-Gaussian kinetic energy of shape D x D
    n_samples : int
      Number of samples to return
    negative_log_prob : callable
      The negative log probability to sample from
    initial_position : np.array
      A place to start sampling from.
    path_len : float
      How long each integration path is. Smaller is faster and more correlated.
    step_size : float
      How long each integration step is. Smaller is slower and more accurate.
    debug : bool
      Flag to include debugging information like timing in the returned values
    mh : bool
      True to include the Metropolis-Hastings acceptance step

    Returns
    -------
    samples, diag : np.array, dict
      Array of length `n_samples`;
      Dictionary of diagnostics output (if diag=False, this is empty dict)
    """

    if diagnostics:
        times = []
        mhs = []
        starttime_total = time.time()

    diag = {}

    if diagnostics:
        starttime = time.time()

    # autograd magic
    dVdq = grad(negative_log_prob)

    if diagnostics:
        endtime = time.time()
        diag['times_grad'] = endtime - starttime

    # collect all our samples in a list
    samples = [initial_position]

    # Keep a single object for momentum resampling
    momentum = st.norm(0, 1)

    # If initial_position is a 10d vector and n_samples is 100, we want
    # 100 x 10 momentum draws. We can do this in one call to momentum.rvs, and
    # iterate over rows
    size = (n_samples,) + initial_position.shape[:1]

    for p0 in momentum.rvs(size=size):

        ### BEGIN leapfrog
        if diagnostics:
            starttime = time.time()

        # Integrate over our path to get a new position and momentum
        q_new, p_new = leapfrog(
            M,
            samples[-1],
            p0,
            dVdq,
            path_len=path_len,
            step_size=step_size,
        )
        if diagnostics:
            endtime = time.time()
            times.append(endtime - starttime)
        ### END leapfrog


        ### BEGIN MH step
        if diagnostics:
            starttime = time.time()

        # Check Metropolis acceptance criterion
        if mh == True:
            start_log_p = negative_log_prob(samples[-1]) - np.sum(momentum.logpdf(p0))
            new_log_p = negative_log_prob(q_new) - np.sum(momentum.logpdf(p_new))
            if np.log(np.random.rand()) < start_log_p - new_log_p:
                samples.append(q_new)
            else:
                samples.append(np.copy(samples[-1]))

        # Do not check Metropolis acceptance criterion
        else:
            samples.append(q_new)

        if diagnostics:
            endtime = time.time()
            mhs.append(endtime - starttime)
        ### END MH step

    if diagnostics:
        endtime_total = time.time()
        diag['total'] = endtime_total - starttime_total
        diag['times'] = np.array(times)
        diag['mhs'] = np.array(mhs)

    return np.array(samples[1:]), diag


def nsghmc(X, y, M, n_samples, negative_log_prob, initial_position,
    batch_size=1, path_len=1, step_size=0.5, diagnostics=False, mh=True):
    """Naive Stochastic Hamiltonian Monte Carlo sampling.

    Parameters
    ----------
    X : np.matrix
      Predictor data in matrix of dimensions N x D.
    y: np.array
      Response data in a vector of length N.
    M : np.matrix
      Mass of the Euclidean-Gaussian kinetic energy of shape D x D
    n_samples : int
      Number of samples to return
    negative_log_prob : callable
      The negative log probability to sample from. Should be a function taking
      three arguments: p(w, x_train, y_train) for the parameters, predictor data,
      and response data.
    initial_position : np.array
      A place to start sampling from.
    path_len : float
      How long each integration path is. Smaller is faster and more correlated.
    step_size : float
      How long each integration step is. Smaller is slower and more accurate.
    debug : bool
      Flag to include debugging information like timing in the returned values
    mh : bool
      True to include the Metropolis-Hastings acceptance step

    Returns
    -------
    samples, diag : np.array, dict
      Array of length `n_samples`;
      Dictionary of diagnostics output (if diag=False, this is empty dict)
    """

    diag = {}

    if diagnostics:
        starttime_total = time.time()
        times = []
        times_grad = []
        mhs = []

    # collect all our samples in a list
    samples = [initial_position]

    # Keep a single object for momentum resampling
    momentum = st.norm(0, 1)

    # If initial_position is a 10d vector and n_samples is 100, we want
    # 100 x 10 momentum draws. We can do this in one call to momentum.rvs, and
    # iterate over rows
    size = (n_samples,) + initial_position.shape[:1]

    # shuffle the data
    n = len(y)
    indices_shuffled = np.random.choice(n, n, replace=False)
    X = X[indices_shuffled,]
    y = y[indices_shuffled]

    batch_num = 0

    # iterate for samples
    for p0 in momentum.rvs(size=size):

        ### BEGIN batching
        if diagnostics:
            starttime = time.time()

        if (batch_num + 1) * batch_size > len(y):
            batch_num = 0
        # subset the data
        indices_subset = range(batch_num * batch_size, (batch_num + 1) * batch_size)
        X_sub = X[indices_subset,]
        y_sub = y[indices_subset]

        batch_num += 1

        # autograd stochastic gradient on batch magic
        dVdq = grad(lambda q: negative_log_prob(q, X_sub, y_sub))

        if diagnostics:
            endtime = time.time()
            times_grad.append(endtime - starttime)
        ### END batching


        ### BEGIN leapfrog
        if diagnostics:
            starttime = time.time()

        # Integrate over our path to get a new position and momentum
        q_new, p_new = leapfrog(
            M,
            samples[-1],
            p0,
            dVdq,
            path_len=path_len,
            step_size=step_size,
        )

        if diagnostics:
            endtime = time.time()
            times.append(endtime - starttime)
        ### END leapfrog


        ### BEGIN MH step
        if diagnostics:
            starttime = time.time()

        # Check Metropolis acceptance criterion
        if mh == True:
            start_log_p = negative_log_prob(samples[-1], X_sub, y_sub) - np.sum(momentum.logpdf(p0))
            new_log_p = negative_log_prob(q_new, X_sub, y_sub) - np.sum(momentum.logpdf(p_new))
            if np.log(np.random.rand()) < start_log_p - new_log_p:
                samples.append(q_new)
            else:
                samples.append(np.copy(samples[-1]))

        # Do not check Metropolis acceptance criterion
        else:
            samples.append(q_new)

        if diagnostics:
            endtime = time.time()
            mhs.append(endtime - starttime)
        ### END MH step


    if diagnostics:
        endtime_total = time.time()
        diag['total'] = endtime_total - starttime_total
        diag['times'] = np.array(times)
        diag['times_grad'] = np.array(times_grad)
        diag['mhs'] = np.array(mhs)

    return np.array(samples[1:]), diag


def sghmc(X, y, M, C, V, n_samples, negative_log_prob, initial_position,
    batch_size=1, path_len=1, step_size=0.5, diagnostics=False):
    """ Stochastic Hamiltonian Monte Carlo sampling.
    Based on https://arxiv.org/abs/1402.4102

    Parameters
    ----------
    X : np.matrix
      Predictor data in matrix of dimensions N x D.
    y: np.array
      Response data in a vector of length N.
    M : np.matrix
      Mass of the Euclidean-Gaussian kinetic energy of shape D x D
    C : matrix
      Upper bound parameter for friction term
    V: matrix
      Covariance of the stochastic gradient noise, such that B = V/2
    n_samples : int
      Number of samples to return
    negative_log_prob : callable
      The negative log probability to sample from
    initial_position : np.array
      A place to start sampling from.
    path_len : float
      How long each integration path is. Smaller is faster and more correlated.
    step_size : float
      How long each integration step is. Smaller is slower and more accurate.
    debug : bool
      Flag to include debugging information like timing in the returned values

    Returns
    -------
    samples, diag : np.array, dict
      Array of length `n_samples`;
      Dictionary of diagnostics output (if diag=False, this is empty dict)
    """
    diag = {}

    if diagnostics:
        times = []
        times_grad = []
        starttime_total = time.time()

    # collect all our samples in a list
    samples = [initial_position]

    # Keep a single object for momentum resampling
    momentum = st.norm(0, 1)

    # If initial_position is a 10d vector and n_samples is 100, we want
    # 100 x 10 momentum draws. We can do this in one call to momentum.rvs, and
    # iterate over rows
    size = (n_samples,) + initial_position.shape[:1]

    # shuffle the data
    n = len(y)
    indices_shuffled = np.random.choice(n, n, replace=False)
    X = X[indices_shuffled,]
    y = y[indices_shuffled]

    batch_num = 0

    # iterate for samples
    for p0 in momentum.rvs(size=size):

        ### BEGIN batching
        if diagnostics:
            starttime = time.time()

        if (batch_num + 1) * batch_size > len(y):
            batch_num = 0
        # subset the data
        indices_subset = range(batch_num * batch_size, (batch_num + 1) * batch_size)
        X_sub = X[indices_subset,]
        y_sub = y[indices_subset]

        batch_num += 1

        # autograd stochastic gradient on batch magic
        dVdq = grad(lambda q: negative_log_prob(q, X_sub, y_sub))

        if diagnostics:
            endtime = time.time()
            times_grad.append(endtime - starttime)
        ### END batching


        ### BEGIN leapfrog
        if diagnostics:
            starttime = time.time()

        # Integrate over our path to get a new position and momentum
        q_new, p_new = leapfrog_friction(
            M,
            C,
            V,
            samples[-1],
            p0,
            dVdq,
            path_len=path_len,
            step_size=step_size,
        )

        if diagnostics:
            endtime = time.time()
            times.append(endtime - starttime)
        ### END leapfrog

        samples.append(q_new)

    if diagnostics:
        endtime_total = time.time()
        diag['total'] = endtime_total - starttime_total
        diag['times'] = np.array(times)
        diag['times_grad'] = np.array(times_grad)

    return np.array(samples[1:]), diag


def hmc2(M, n_samples, U, gradU, initial_position,
    path_len=1, step_size=0.5, diagnostics=False, mh=True):
    """Hamiltonian Monte Carlo sampling for cases in which we specify a function U and gradient U

    Parameters
    ----------
    M : np.matrix
      Mass of the Euclidean-Gaussian kinetic energy of shape D x D
    n_samples : int
      Number of samples to return
    U : callable
      The negative log probability of the target distribution
    gradU : callable
      Gradient of the negative log of the target distribution
    initial_position : np.array
      A place to start sampling from.
    path_len : float
      How long each integration path is. Smaller is faster and more correlated.
    step_size : float
      How long each integration step is. Smaller is slower and more accurate.
    debug : bool
      Flag to include debugging information like timing in the returned values
    mh : bool
      True to include the Metropolis-Hastings acceptance step

    Returns
    -------
    samples, debug : np.array, dict
      Array of length `n_samples`;
      Dictionary of debugging output (if debug=False, this is empty dict)
    """

    # Dictionary to store diagnostics
    diag = {}

    # Start timer
    if diagnostics:
        times = []
        mhs = []
        starttime_total = time.time()

    # collect all our samples in a list
    samples = [initial_position]

    # Keep a single object for momentum resampling
    momentum = st.norm(0, 1)

    # If initial_position is a 10d vector and n_samples is 100, we want
    # 100 x 10 momentum draws. We can do this in one call to momentum.rvs, and
    # iterate over rows
    size = (n_samples,) + initial_position.shape[:1]
    momentums = momentum.rvs(size=size)
    samples_p = [momentums[0]]
    
    # Iterate over different momentum draws
    for p0 in momentums:

        ### BEGIN leapfrog
        if diagnostics:
            starttime = time.time()

        # Integrate over our path to get a new position and momentum
        q_new, p_new = leapfrog(
            M,
            samples[-1],
            p0,
            gradU,
            path_len=path_len,
            step_size=step_size,
        )

        if diagnostics:
            endtime = time.time()
            times.append(endtime - starttime)
        ### END leapfrog

        ### BEGIN MH step
        if diagnostics:
            starttime = time.time()

        # Check Metropolis acceptance criterion
        if mh == True:
            start_log_p = U(samples[-1]) - np.sum(momentum.logpdf(p0))
            new_log_p = U(q_new) - np.sum(momentum.logpdf(p_new))
            if np.log(np.random.rand()) < start_log_p - new_log_p:
                samples.append(q_new)
                samples_p.append(p_new)
            else:
                samples.append(np.copy(samples[-1]))
                samples_p.append(np.copy(samples[-1]))

        # Do not check Metropolis acceptance criterion
        else:
            samples.append(q_new)
            samples_p.append(p_new)

        if diagnostics:
            endttime = time.time()
            mhs.append(endtime - starttime)
        ### END MH step

    # End timer and add to diagnostics
    if diagnostics:
        endtime_total = time.time()
        diag['total'] = endtime_total - starttime_total
        diag['times'] = np.array(times)
        diag['mhs'] = np.array(mhs)

    return np.array(samples[1:]), np.array(samples_p[1:]), diag


def sghmc2(M, C, V, n_samples, U, gradU, initial_position,
    path_len=1, step_size=0.5, diagnostics=False):
    """Hamiltonian Monte Carlo sampling.

    Parameters
    ----------
    M : np.matrix
      Mass of the Euclidean-Gaussian kinetic energy of shape D x D
    C : matrix
      Upper bound parameter for friction term
    V: matrix
      Covariance of the stochastic gradient noise, such that B = V/2
    n_samples : int
      Number of samples to return
    U : callable
      The negative log probability of the target distribution
    gradU : callable
      Gradient of the negative log of the target distribution
    initial_position : np.array
      A place to start sampling from.
    path_len : float
      How long each integration path is. Smaller is faster and more correlated.
    step_size : float
      How long each integration step is. Smaller is slower and more accurate.
    diagnostics : bool
      Flag to include diagnostic information like timing in the returned values

    Returns
    -------
    samples, debug : np.array, dict
      Array of length `n_samples`;
      Dictionary of debugging output (if debug=False, this is empty dict)
    """

    # Dictionary to store diagnostics
    diag = {}

    # Start timer
    if diagnostics:
        times = []
        starttime_total = time.time()

    # collect all our samples in a list
    samples = [initial_position]

    # Keep a single object for momentum resampling
    momentum = st.norm(0, 1)

    # If initial_position is a 10d vector and n_samples is 100, we want
    # 100 x 10 momentum draws. We can do this in one call to momentum.rvs, and
    # iterate over rows
    size = (n_samples,) + initial_position.shape[:1]
    momentums = momentum.rvs(size=size)
    
    # Iterate over different momentum draws
    for p0 in momentums:

        ### BEGIN leapfrog
        if diagnostics:
            starttime = time.time()

        # Integrate over our path to get a new position and momentum
        q_new, p_new = leapfrog_friction(
            M,
            C,
            V,
            samples[-1],
            p0,
            gradU,
            path_len=path_len,
            step_size=step_size,
        )

        if diagnostics:
            endtime = time.time()
            times.append(endtime - starttime)
        ### END leapfrog

        samples.append(q_new)

    # End timer and add to diagnostics
    if diagnostics:
        endtime_total = time.time()
        diag['total'] = endtime_total - starttime_total
        diag['times'] = np.array(times)

    return np.array(samples[1:]), np.array(momentums), diag
