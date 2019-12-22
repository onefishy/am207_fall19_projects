def thaw(X, y, M, C, V, n_samples, negative_log_prob, initial_position,
    batch_size=1, path_len=1, step_size=0.5, diagnostics=False, mh=True, burnin_c=1.):
    """ Thawing sampling
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
    test = []
    frozen_coefs = {x for x in range(initial_position.shape[0])}
    iteration = 0
    last_change = 0

    # iterate for samples
    for p0 in momentum.rvs(size=size):
        # Check to see if we should thaw
        if iteration == 10:
            frozen_coefs = {x for x in range(initial_position.shape[0])}
            sign = [assess_slope(samples, 10, x) for x in frozen_coefs]
        elif iteration > 10 and len(frozen_coefs):
            for x in frozen_coefs:
                to_remove = set()
                slope = assess_slope(samples, 10, x)
                # Means the line has started to go the other direction
                if slope * sign[x] < 0:
                    to_remove.add(x)

            frozen_coefs = frozen_coefs.difference(to_remove)
            if len(frozen_coefs) == 0:
                frozen_coefs = {x for x in range(initial_position.shape[0])}
                burnin_c = min(np.exp(5 / (iteration - last_change + 1))+ burnin_c, C)
                # 10/(iteration - last_change + 1)
                last_change = iteration

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
            burnin_c,
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

        iteration += 1
        samples.append(q_new)

    if diagnostics:
        endtime_total = time.time()
        diag['total'] = endtime_total - starttime_total
        diag['times'] = np.array(times)
        diag['times_grad'] = np.array(times_grad)

    print(burnin_c)
    return np.array(samples[1:]), diag, test