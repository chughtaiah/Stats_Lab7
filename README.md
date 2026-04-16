# AI Stats Lab
## Operations on a Single Random Variable

This lab is based on the topic **expected value, moments, variance, and transformations of a random variable** from the slides.

---

## Objective

Let:

\[
X \sim \text{Uniform}(0,a)
\]

You will analyze this random variable using both:

- theoretical formulas
- simulation

You will also study the transformed random variable:

\[
Z = 2X + 1
\]

---

## Task

Implement the function:

```python
def uniform_analysis(a, n_samples=10000):
