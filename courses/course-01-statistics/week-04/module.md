# Module 4: Joint Distributions and Independence

## Overview

Real-world phenomena involve multiple random variables that may be related. A customer's age affects their purchase amount. Temperature affects ice cream sales. This module covers joint distributions—how to model and analyze relationships between random variables—and the crucial concept of independence.

Understanding joint distributions is essential for multivariate analysis, correlation, and building ML models with multiple features.

---

## 1. Joint Distributions

### Discrete Case: Joint PMF

For two discrete random variables X and Y, the joint PMF is:

```
p(x, y) = P(X = x AND Y = y)
```

**Properties:**
1. p(x, y) ≥ 0 for all x, y
2. Σₓ Σᵧ p(x, y) = 1

### Example: Two Dice

Let X = first die, Y = second die. If fair and independent:

```
p(x, y) = 1/36 for all x, y ∈ {1,2,3,4,5,6}
```

What about Z = X + Y? We can find P(Z = z) by summing:
```
P(Z = 7) = P(X=1,Y=6) + P(X=2,Y=5) + ... + P(X=6,Y=1) = 6/36 = 1/6
```

### Continuous Case: Joint PDF

For continuous variables, the joint PDF f(x, y) satisfies:

```
P((X, Y) ∈ A) = ∫∫_A f(x, y) dx dy
```

**Properties:**
1. f(x, y) ≥ 0
2. ∫∫ f(x, y) dx dy = 1

---

## 2. Marginal Distributions

The **marginal distribution** of X is obtained by "integrating out" Y:

### Discrete
```
p_X(x) = P(X = x) = Σᵧ p(x, y)
```

### Continuous
```
f_X(x) = ∫ f(x, y) dy
```

### Example

| X\Y | Y=0 | Y=1 | p_X(x) |
|-----|-----|-----|--------|
| X=0 | 0.2 | 0.1 | 0.3 |
| X=1 | 0.3 | 0.4 | 0.7 |
| p_Y(y) | 0.5 | 0.5 | 1.0 |

```python
import numpy as np

# Joint distribution as a 2D array
joint = np.array([[0.2, 0.1],
                  [0.3, 0.4]])

# Marginals
p_X = joint.sum(axis=1)  # Sum across Y
p_Y = joint.sum(axis=0)  # Sum across X

print(f"p_X = {p_X}")  # [0.3, 0.7]
print(f"p_Y = {p_Y}")  # [0.5, 0.5]
```

---

## 3. Independence

### Definition

X and Y are **independent** if and only if:

```
P(X = x, Y = y) = P(X = x) × P(Y = y)  for all x, y
```

Equivalently: p(x, y) = p_X(x) × p_Y(y)

### Checking Independence

For the table above:
- P(X=0, Y=0) = 0.2
- P(X=0) × P(Y=0) = 0.3 × 0.5 = 0.15

Since 0.2 ≠ 0.15, X and Y are **not** independent.

### Independence in Practice

Two events are independent when knowing one tells you nothing about the other:
- Coin flips are independent
- Height and weight are NOT independent (correlated)
- Spam emails and containing "free" are NOT independent

```python
# Generate independent vs dependent data
np.random.seed(42)

# Independent
X_ind = np.random.normal(0, 1, 1000)
Y_ind = np.random.normal(0, 1, 1000)

# Dependent
X_dep = np.random.normal(0, 1, 1000)
Y_dep = 0.8 * X_dep + 0.6 * np.random.normal(0, 1, 1000)

print(f"Independent correlation: {np.corrcoef(X_ind, Y_ind)[0,1]:.3f}")  # ≈ 0
print(f"Dependent correlation: {np.corrcoef(X_dep, Y_dep)[0,1]:.3f}")    # ≈ 0.8
```

---

## 4. Covariance

### Definition

Covariance measures how two variables move together:

```
Cov(X, Y) = E[(X - μ_X)(Y - μ_Y)] = E[XY] - E[X]E[Y]
```

**Interpretation:**
- Cov(X, Y) > 0: X and Y tend to move in the same direction
- Cov(X, Y) < 0: X and Y tend to move in opposite directions
- Cov(X, Y) = 0: No linear relationship (but NOT necessarily independent!)

### Properties

1. Cov(X, X) = Var(X)
2. Cov(X, Y) = Cov(Y, X)
3. Cov(aX, Y) = a × Cov(X, Y)
4. Cov(X + Y, Z) = Cov(X, Z) + Cov(Y, Z)

### Independence Implies Zero Covariance

If X and Y are independent:
```
E[XY] = E[X] × E[Y]  →  Cov(X, Y) = 0
```

**Warning:** The converse is false! Zero covariance does not imply independence.

```python
# Counter-example: X and X² are not independent but can have Cov = 0
X = np.random.uniform(-1, 1, 10000)
Y = X ** 2

print(f"Cov(X, X²) = {np.cov(X, Y)[0,1]:.6f}")  # ≈ 0
# But clearly Y depends on X!
```

---

## 5. Correlation

### Pearson Correlation Coefficient

Correlation is covariance normalized to [-1, 1]:

```
ρ(X, Y) = Cov(X, Y) / (σ_X × σ_Y)
```

Also written as Corr(X, Y) or r.

**Interpretation:**
- ρ = 1: Perfect positive linear relationship
- ρ = -1: Perfect negative linear relationship
- ρ = 0: No linear relationship

### Correlation vs Causation

**Correlation does not imply causation!**

Examples of spurious correlations:
- Ice cream sales correlate with drowning deaths (both caused by summer)
- Shoe size correlates with math ability (both caused by age in children)

```python
import matplotlib.pyplot as plt

# Different correlations
def generate_correlated(rho, n=200):
    X = np.random.normal(0, 1, n)
    Y = rho * X + np.sqrt(1 - rho**2) * np.random.normal(0, 1, n)
    return X, Y

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, rho in zip(axes, [-0.9, -0.3, 0.3, 0.9]):
    X, Y = generate_correlated(rho)
    ax.scatter(X, Y, alpha=0.5)
    ax.set_title(f'ρ = {rho}')
plt.tight_layout()
```

---

## 6. Variance of Sums (Revisited)

### General Formula

For any X and Y:
```
Var(X + Y) = Var(X) + Var(Y) + 2×Cov(X, Y)
```

### Special Case: Independent Variables

If independent, Cov(X, Y) = 0, so:
```
Var(X + Y) = Var(X) + Var(Y)
```

### Portfolio Variance

In finance, for portfolio with weights w₁, w₂:
```
Var(w₁X + w₂Y) = w₁²Var(X) + w₂²Var(Y) + 2w₁w₂Cov(X, Y)
```

Diversification works when Cov(X, Y) < 0.

```python
# Portfolio example
returns_A = np.array([0.10, 0.05, -0.02, 0.08, 0.12])
returns_B = np.array([0.08, -0.01, 0.06, 0.04, 0.07])

var_A = np.var(returns_A)
var_B = np.var(returns_B)
cov_AB = np.cov(returns_A, returns_B)[0,1]

# 50-50 portfolio
w = 0.5
portfolio_var = w**2 * var_A + w**2 * var_B + 2 * w * w * cov_AB

print(f"Var(A) = {var_A:.6f}")
print(f"Var(B) = {var_B:.6f}")
print(f"Var(Portfolio) = {portfolio_var:.6f}")  # Less than average!
```

---

## 7. Multivariate Normal Distribution

### Definition

The multivariate normal extends the normal to multiple dimensions:

```
X ~ N(μ, Σ)
```

Where:
- μ = mean vector (n × 1)
- Σ = covariance matrix (n × n, symmetric, positive semi-definite)

### Covariance Matrix

For two variables:
```
Σ = [[Var(X),    Cov(X,Y)]
     [Cov(X,Y),  Var(Y)  ]]
```

### Generating Correlated Normal Variables

```python
from scipy.stats import multivariate_normal

# Mean and covariance
mu = [0, 0]
Sigma = [[1.0, 0.7],
         [0.7, 1.0]]  # Correlation ρ = 0.7

# Generate samples
mvn = multivariate_normal(mu, Sigma)
samples = mvn.rvs(1000)

print(f"Sample correlation: {np.corrcoef(samples.T)[0,1]:.3f}")  # ≈ 0.7
```

### Key Property

For multivariate normal, **uncorrelated ⟺ independent**

This is special to the normal distribution!

---

## 8. Conditional Expectation

### Definition

The expected value of X given Y = y:

```
E[X | Y = y] = Σₓ x × P(X = x | Y = y)
```

### Law of Total Expectation

```
E[X] = E[E[X | Y]]
```

The overall expectation equals the average of conditional expectations.

**Example:** Heights
```
E[Height] = P(Male) × E[Height|Male] + P(Female) × E[Height|Female]
```

---

## Key Takeaways

1. **Joint distributions** describe multiple random variables together

2. **Marginals** are obtained by summing/integrating out other variables

3. **Independence:** P(X,Y) = P(X)P(Y) for all values

4. **Covariance** measures linear co-movement: Cov(X,Y) = E[XY] - E[X]E[Y]

5. **Correlation** normalizes covariance to [-1, 1]

6. **Zero correlation ≠ independence** (except for multivariate normal)

7. **Var(X + Y) = Var(X) + Var(Y) + 2Cov(X,Y)**—simplifies when independent

---

## Connections to Future Modules

- **Module 5:** Conditional probability leads to Bayes' theorem
- **Module 6:** Sample covariance estimates true covariance
- **Module 10:** Bayesian inference conditions on observed data
- **Module 12:** Graphical models encode conditional independence

---

## Practice Problems

1. Given the joint PMF table, determine if X and Y are independent:
   | X\Y | 0 | 1 |
   |-----|---|---|
   | 0 | 0.1 | 0.2 |
   | 1 | 0.3 | 0.4 |

2. If Var(X) = 4, Var(Y) = 9, and Cov(X, Y) = 3, find Var(2X - 3Y).

3. Generate 1000 pairs from a bivariate normal with ρ = 0.5 and verify the correlation.

4. Prove that Cov(X, X) = Var(X).

5. Show that for constants a, b: Cov(aX, bY) = ab × Cov(X, Y).

---

## Further Reading

- Ross, S. *A First Course in Probability* - Chapter 6
- DeGroot & Schervish, *Probability and Statistics* - Chapter 4
- Bishop, *Pattern Recognition and ML* - Chapter 2 (Probability Distributions)
