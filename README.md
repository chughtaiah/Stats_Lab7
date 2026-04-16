# AI Stats Lab

## Objective

Let X ~ Uniform(0, a)

You must compute theoretical and sample-based statistics and analyze the transformation:

Z = 2X + 1

---

## Task

Implement:

def uniform_analysis(a, n_samples=10000)

in AI_stats_lab.py

---

## Return Format

Return:

(
    theoretical_mean,
    theoretical_variance,
    sample_mean,
    sample_variance,
    mean_error,
    variance_error,
    transformed_mean,
    transformed_variance
)

---

## Formulas

E[X] = a / 2

Var(X) = a^2 / 12

E[X^2] = a^2 / 3

Var(X) = E[X^2] - (E[X])^2

---

## Transformation

Z = 2X + 1

E[Z] = 2E[X] + 1

Var(Z) = 4 Var(X)

---

## Errors

mean_error = abs(sample_mean - theoretical_mean)

variance_error = abs(sample_variance - theoretical_variance)

---

## Requirements

Use:

- numpy.random.uniform
- numpy.mean
- numpy.var

Do NOT:

- print anything
- change return structure

---

## Run Tests

pytest test_AIstats_lab.py -q

---

## Setup (GitHub Classroom)

pip install numpy pytest

---

## Example

For a = 6:

E[X] = 3  
Var(X) = 3  
E[Z] = 7  
Var(Z) = 12
