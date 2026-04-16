import math
import numpy as np
import pytest

from AI_stats_lab import (
    bernoulli_log_likelihood,
    bernoulli_mle_with_comparison,
    poisson_log_likelihood,
    poisson_mle_analysis,
)


# -----------------------------
# Bernoulli log-likelihood tests
# -----------------------------
def test_bernoulli_log_likelihood_basic():
    data = np.array([1, 1, 0, 1, 1])
    theta = 0.8
    expected = 4 * np.log(0.8) + 1 * np.log(0.2)
    result = bernoulli_log_likelihood(data, theta)
    assert np.isclose(result, expected, atol=1e-6)


def test_bernoulli_log_likelihood_other_theta():
    data = np.array([1, 0, 1, 0])
    theta = 0.5
    expected = 2 * np.log(0.5) + 2 * np.log(0.5)
    result = bernoulli_log_likelihood(data, theta)
    assert np.isclose(result, expected, atol=1e-6)


def test_bernoulli_log_likelihood_invalid_theta_low():
    with pytest.raises(ValueError):
        bernoulli_log_likelihood([1, 0, 1], 0.0)


def test_bernoulli_log_likelihood_invalid_theta_high():
    with pytest.raises(ValueError):
        bernoulli_log_likelihood([1, 0, 1], 1.0)


def test_bernoulli_log_likelihood_invalid_data():
    with pytest.raises(ValueError):
        bernoulli_log_likelihood([1, 2, 0], 0.5)


def test_bernoulli_log_likelihood_empty():
    with pytest.raises(ValueError):
        bernoulli_log_likelihood([], 0.5)


# -----------------------------
# Bernoulli MLE analysis tests
# -----------------------------
def test_bernoulli_mle_with_comparison_basic():
    data = np.array([1, 1, 0, 1, 1])
    result = bernoulli_mle_with_comparison(data, [0.2, 0.5, 0.8])

    assert isinstance(result, dict)
    assert np.isclose(result["mle"], 0.8, atol=1e-6)
    assert result["num_successes"] == 4
    assert result["num_failures"] == 1
    assert result["best_candidate"] == 0.8


def test_bernoulli_mle_with_comparison_all_zeros():
    data = np.array([0, 0, 0, 0, 0])
    result = bernoulli_mle_with_comparison(data, [0.1, 0.2, 0.3])

    assert np.isclose(result["mle"], 0.0, atol=1e-6)
    assert result["num_successes"] == 0
    assert result["num_failures"] == 5
    assert result["best_candidate"] == 0.1


def test_bernoulli_mle_with_comparison_default_candidates():
    data = np.array([1, 0, 1, 1, 0])
    result = bernoulli_mle_with_comparison(data)

    assert "log_likelihoods" in result
    assert set(result["log_likelihoods"].keys()) == {0.2, 0.5, 0.8}


# -----------------------------
# Poisson log-likelihood tests
# -----------------------------
def test_poisson_log_likelihood_basic():
    data = np.array([2, 3, 4])
    lam = 3.0
    expected = sum(x * np.log(lam) - lam - math.lgamma(x + 1) for x in data)
    result = poisson_log_likelihood(data, lam)
    assert np.isclose(result, expected, atol=1e-6)


def test_poisson_log_likelihood_invalid_lambda():
    with pytest.raises(ValueError):
        poisson_log_likelihood([1, 2, 3], 0.0)


def test_poisson_log_likelihood_negative_count():
    with pytest.raises(ValueError):
        poisson_log_likelihood([1, -1, 3], 2.0)


def test_poisson_log_likelihood_noninteger_count():
    with pytest.raises(ValueError):
        poisson_log_likelihood([1, 2.5, 3], 2.0)


def test_poisson_log_likelihood_empty():
    with pytest.raises(ValueError):
        poisson_log_likelihood([], 2.0)


# -----------------------------
# Poisson MLE analysis tests
# -----------------------------
def test_poisson_mle_analysis_basic():
    data = np.array([3, 4, 2, 6, 5])
    result = poisson_mle_analysis(data, [2.0, 4.0, 6.0])

    assert isinstance(result, dict)
    assert np.isclose(result["mle"], 4.0, atol=1e-6)
    assert np.isclose(result["sample_mean"], 4.0, atol=1e-6)
    assert result["total_count"] == 20
    assert result["n"] == 5
    assert result["best_candidate"] == 4.0


def test_poisson_mle_analysis_default_candidates():
    data = np.array([0, 1, 1, 2, 1])
    result = poisson_mle_analysis(data)

    assert "log_likelihoods" in result
    assert set(result["log_likelihoods"].keys()) == {1.0, 3.0, 5.0}


def test_poisson_mle_analysis_single_value():
    data = np.array([10])
    result = poisson_mle_analysis(data, [5.0, 10.0, 15.0])

    assert np.isclose(result["mle"], 10.0, atol=1e-6)
    assert result["best_candidate"] == 10.0
