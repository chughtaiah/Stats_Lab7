import numpy as np
import AI_stats_lab as amt


def test_uniform_analysis_structure():
    result = amt.uniform_analysis(6)
    assert isinstance(result, tuple)
    assert len(result) == 8


def test_uniform_analysis_theoretical_values():
    a = 6
    result = amt.uniform_analysis(a)

    theoretical_mean = result[0]
    theoretical_variance = result[1]

    assert abs(theoretical_mean - 3.0) < 1e-6
    assert abs(theoretical_variance - 3.0) < 1e-6


def test_uniform_analysis_sample_values():
    a = 6
    result = amt.uniform_analysis(a)

    sample_mean = result[2]
    sample_variance = result[3]

    assert abs(sample_mean - 3.0) < 0.1
    assert abs(sample_variance - 3.0) < 0.1


def test_uniform_analysis_errors():
    a = 6
    result = amt.uniform_analysis(a)

    mean_error = result[4]
    variance_error = result[5]

    assert mean_error >= 0
    assert variance_error >= 0
    assert mean_error < 0.1
    assert variance_error < 0.1


def test_uniform_analysis_transformation():
    a = 6
    result = amt.uniform_analysis(a)

    transformed_mean = result[6]
    transformed_variance = result[7]

    assert abs(transformed_mean - 7.0) < 1e-6
    assert abs(transformed_variance - 12.0) < 1e-6
