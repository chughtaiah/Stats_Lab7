import math
import numpy as np


def bernoulli_log_likelihood(data, theta):
    """
    Compute the Bernoulli log-likelihood for binary data.

    Parameters
    ----------
    data : array-like
        Sequence of 0/1 observations.
    theta : float
        Bernoulli parameter, must satisfy 0 < theta < 1.

    Returns
    -------
    float
        Log-likelihood:
            sum_i [x_i log(theta) + (1-x_i) log(1-theta)]

    Requirements
    ------------
    - Raise ValueError if data is empty
    - Raise ValueError if theta is not in (0,1)
    - Raise ValueError if data contains values other than 0 and 1
    """
    raise NotImplementedError("Implement bernoulli_log_likelihood")


def bernoulli_mle_with_comparison(data, candidate_thetas=None):
    """
    Estimate the Bernoulli MLE and compare candidate theta values.

    Parameters
    ----------
    data : array-like
        Sequence of 0/1 observations.
    candidate_thetas : array-like or None
        Optional candidate theta values to compare using log-likelihood.
        If None, use [0.2, 0.5, 0.8].

    Returns
    -------
    dict
        A dictionary with:
        - 'mle': float
            The Bernoulli MLE
        - 'num_successes': int
        - 'num_failures': int
        - 'log_likelihoods': dict
            Mapping candidate theta -> log-likelihood
        - 'best_candidate': float
            Candidate theta with highest log-likelihood

    Requirements
    ------------
    - Validate data
    - Compute MLE analytically
    - Compute candidate log-likelihoods using bernoulli_log_likelihood
    - In case of ties in best candidate, return the first one encountered
    """
    raise NotImplementedError("Implement bernoulli_mle_with_comparison")


def poisson_log_likelihood(data, lam):
    """
    Compute the Poisson log-likelihood for count data.

    Parameters
    ----------
    data : array-like
        Sequence of nonnegative integer counts.
    lam : float
        Poisson rate, must satisfy lam > 0.

    Returns
    -------
    float
        Log-likelihood:
            sum_i [x_i log(lam) - lam - log(x_i!)]

    Requirements
    ------------
    - Raise ValueError if data is empty
    - Raise ValueError if lam <= 0
    - Raise ValueError if data contains negative or non-integer values

    Notes
    -----
    You may use math.lgamma(x + 1) for log(x!) since log(x!) = lgamma(x+1).
    """
    raise NotImplementedError("Implement poisson_log_likelihood")


def poisson_mle_analysis(data, candidate_lambdas=None):
    """
    Estimate the Poisson MLE and compare candidate lambda values.

    Parameters
    ----------
    data : array-like
        Sequence of nonnegative integer counts.
    candidate_lambdas : array-like or None
        Optional candidate lambdas to compare using log-likelihood.
        If None, use [1.0, 3.0, 5.0].

    Returns
    -------
    dict
        A dictionary with:
        - 'mle': float
            The Poisson MLE
        - 'sample_mean': float
        - 'total_count': int
        - 'n': int
        - 'log_likelihoods': dict
            Mapping candidate lambda -> log-likelihood
        - 'best_candidate': float
            Candidate lambda with highest log-likelihood

    Requirements
    ------------
    - Validate data
    - Compute MLE analytically
    - Compute candidate log-likelihoods using poisson_log_likelihood
    - In case of ties in best candidate, return the first one encountered
    """
    raise NotImplementedError("Implement poisson_mle_analysis")
