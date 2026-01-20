# Module 3: Expectation and Variance

## Overview

While probability distributions tell us everything about a random variable, we often need simpler summaries. Expectation (mean) tells us the "center" of a distribution, while variance and standard deviation measure its "spread." These concepts are fundamental to understanding risk, making decisions, and evaluating machine learning models.

This module covers how to compute these quantities, their properties, and why they matter for data science.

---

## 1. Expected Value (Mean)

### Intuition

The expected value is the long-run average—if you repeated an experiment infinitely many times and averaged the outcomes, you'd get the expected value.

**Notation:** E[X], μ, or sometimes ⟨X⟩

### Discrete Random Variables

```
E[X] = Σ x × P(X = x) = Σ x × p(x)
```

**Example: Fair Die**
```
E[X] = 1×(1/6) + 2×(1/6) + 3×(1/6) + 4×(1/6) + 5×(1/6) + 6×(1/6)
     = (1 + 2 + 3 + 4 + 5 + 6) / 6
     = 21/6 = 3.5
```

Note: The expected value doesn't have to be a possible outcome!

### Continuous Random Variables

```
E[X] = ∫[-∞ to ∞] x × f(x) dx
```

**Example: Uniform(0, 1)**
```
E[X] = ∫[0 to 1] x × 1 dx = [x²/2] from 0 to 1 = 1/2
```

### Computing in Python

```python
import numpy as np
from scipy.stats import binom, norm

# Discrete: Binomial(10, 0.3)
X = binom(n=10, p=0.3)
print(f"E[X] = {X.mean():.2f}")  # 3.00

# Or manually:
x_vals = np.arange(11)
E_X = np.sum(x_vals * X.pmf(x_vals))
print(f"E[X] = {E_X:.2f}")  # 3.00

# Continuous: Normal(100, 15)
Y = norm(loc=100, scale=15)
print(f"E[Y] = {Y.mean():.2f}")  # 100.00
```

---

## 2. Expected Value of a Function

Sometimes we want E[g(X)] for some function g.

### The Law of the Unconscious Statistician (LOTUS)

```
E[g(X)] = Σ g(x) × P(X = x)     (discrete)
E[g(X)] = ∫ g(x) × f(x) dx      (continuous)
```

We don't need to find the distribution of g(X)—we can compute directly!

**Example:** E[X²] for a die
```
E[X²] = 1²×(1/6) + 2²×(1/6) + ... + 6²×(1/6)
      = (1 + 4 + 9 + 16 + 25 + 36) / 6
      = 91/6 ≈ 15.17
```

---

## 3. Properties of Expected Value

### Linearity of Expectation

This is the most important property:

```
E[aX + b] = a × E[X] + b
E[X + Y] = E[X] + E[Y]     (even if X and Y are NOT independent!)
```

**Example: Portfolio Return**

If stock A has expected return 8% and stock B has expected return 12%, a 50-50 portfolio has expected return:
```
E[0.5A + 0.5B] = 0.5 × 8% + 0.5 × 12% = 10%
```

### Linearity in Practice

```python
# Verify linearity with simulation
np.random.seed(42)
X = np.random.normal(10, 2, 10000)
Y = np.random.normal(5, 3, 10000)

print(f"E[X] = {X.mean():.3f}")           # ≈ 10
print(f"E[Y] = {Y.mean():.3f}")           # ≈ 5
print(f"E[X + Y] = {(X + Y).mean():.3f}") # ≈ 15
print(f"E[3X + 2] = {(3*X + 2).mean():.3f}") # ≈ 32
```

---

## 4. Variance

### Definition

Variance measures how spread out a distribution is:

```
Var(X) = E[(X - μ)²] = E[X²] - (E[X])²
```

**Notation:** Var(X), σ², or V(X)

The second form is often easier to compute:
```
Var(X) = E[X²] - (E[X])²
```

### Example: Fair Die

We already computed E[X] = 3.5 and E[X²] = 91/6

```
Var(X) = 91/6 - (3.5)² = 15.17 - 12.25 = 2.92
```

### Standard Deviation

The square root of variance, in the same units as X:

```
σ = SD(X) = √Var(X)
```

For our die: σ = √2.92 ≈ 1.71

---

## 5. Properties of Variance

### Scaling Property

```
Var(aX + b) = a² × Var(X)
```

Note: Constants shift the distribution but don't affect spread.

### Variance of Sums

For **independent** random variables:
```
Var(X + Y) = Var(X) + Var(Y)
```

For **dependent** variables:
```
Var(X + Y) = Var(X) + Var(Y) + 2×Cov(X, Y)
```

(We'll cover covariance in Module 4)

### Why Independence Matters

```python
# Independent variables
X = np.random.normal(0, 1, 10000)
Y = np.random.normal(0, 1, 10000)
print(f"Var(X) + Var(Y) = {X.var() + Y.var():.3f}")  # ≈ 2
print(f"Var(X + Y) = {(X + Y).var():.3f}")           # ≈ 2 ✓

# Dependent variables (Y depends on X)
Z = 0.5 * X + np.random.normal(0, 1, 10000)
print(f"Var(X) + Var(Z) = {X.var() + Z.var():.3f}")  # ≈ 2.25
print(f"Var(X + Z) = {(X + Z).var():.3f}")           # > 2.25 (positive correlation)
```

---

## 6. Standard Distributions: Mean and Variance

### Summary Table

| Distribution | E[X] | Var(X) |
|--------------|------|--------|
| Bernoulli(p) | p | p(1-p) |
| Binomial(n, p) | np | np(1-p) |
| Poisson(λ) | λ | λ |
| Geometric(p) | 1/p | (1-p)/p² |
| Uniform(a, b) | (a+b)/2 | (b-a)²/12 |
| Exponential(λ) | 1/λ | 1/λ² |
| Normal(μ, σ²) | μ | σ² |

### Deriving a Formula: Bernoulli

```
X ~ Bernoulli(p), so X ∈ {0, 1}

E[X] = 0×(1-p) + 1×p = p

E[X²] = 0²×(1-p) + 1²×p = p

Var(X) = E[X²] - (E[X])² = p - p² = p(1-p)
```

### Binomial as Sum of Bernoullis

If X = X₁ + X₂ + ... + Xₙ where each Xᵢ ~ Bernoulli(p):

```
E[X] = E[X₁] + ... + E[Xₙ] = np

Var(X) = Var(X₁) + ... + Var(Xₙ) = np(1-p)    (independence!)
```

---

## 7. Chebyshev's Inequality

How much of a distribution can be far from the mean?

```
P(|X - μ| ≥ kσ) ≤ 1/k²
```

**Interpretation:**
- At most 25% of values are more than 2σ from the mean
- At most 11% of values are more than 3σ from the mean

This holds for **any** distribution, regardless of shape!

```python
# Verify with simulation
from scipy.stats import expon

X = expon(scale=1)  # Exponential with mean=1, variance=1
samples = X.rvs(100000)
μ, σ = X.mean(), X.std()

# Chebyshev says at most 25% are > 2σ from mean
actual = np.mean(np.abs(samples - μ) >= 2 * σ)
print(f"P(|X - μ| ≥ 2σ): Chebyshev ≤ 0.25, Actual = {actual:.3f}")
```

---

## 8. Moments

### Raw Moments

The k-th raw moment is E[X^k]:
- 1st moment: E[X] (mean)
- 2nd moment: E[X²]

### Central Moments

The k-th central moment is E[(X - μ)^k]:
- 2nd central moment: Variance
- 3rd: Related to skewness (asymmetry)
- 4th: Related to kurtosis (tail heaviness)

### Skewness and Kurtosis

```
Skewness = E[(X - μ)³] / σ³

Kurtosis = E[(X - μ)⁴] / σ⁴
```

```python
from scipy.stats import skew, kurtosis

# Normal distribution has skewness=0, kurtosis=0 (excess)
normal_samples = np.random.normal(0, 1, 10000)
print(f"Normal: Skew = {skew(normal_samples):.3f}, Kurt = {kurtosis(normal_samples):.3f}")

# Exponential is right-skewed
exp_samples = np.random.exponential(1, 10000)
print(f"Exponential: Skew = {skew(exp_samples):.3f}, Kurt = {kurtosis(exp_samples):.3f}")
```

---

## 9. Practical Applications

### Risk Assessment

In finance, mean is expected return, variance is risk:

```python
# Two investments
A_returns = np.array([0.10, 0.12, 0.08, 0.11, 0.09])
B_returns = np.array([0.05, 0.20, -0.05, 0.15, 0.15])

print(f"Investment A: Mean = {A_returns.mean():.2%}, SD = {A_returns.std():.2%}")
print(f"Investment B: Mean = {B_returns.mean():.2%}, SD = {B_returns.std():.2%}")
# A has same mean but lower risk
```

### Model Evaluation

In ML, we care about expected loss:
```
Risk = E[L(Y, Ŷ)] 

where L is the loss function
```

### Quality Control

Manufacturing: products should be within μ ± 3σ

```python
# Widgets should be 10mm ± 0.5mm
widget_sizes = np.random.normal(10, 0.15, 1000)
defective = np.sum((widget_sizes < 9.5) | (widget_sizes > 10.5))
print(f"Defective rate: {defective/1000:.2%}")
```

---

## Key Takeaways

1. **Expected value is the long-run average**: E[X] = Σ x × P(X = x)

2. **Linearity is powerful**: E[X + Y] = E[X] + E[Y] always, even for dependent variables

3. **Variance measures spread**: Var(X) = E[X²] - (E[X])²

4. **Standard deviation has same units as X**: σ = √Var(X)

5. **For independent sums**: Var(X + Y) = Var(X) + Var(Y)

6. **Chebyshev's inequality**: At most 1/k² of probability is k standard deviations from mean

7. **Know the formulas** for common distributions (see table above)

---

## Connections to Future Modules

- **Module 4:** Covariance extends variance to pairs of variables
- **Module 6:** Sample mean and variance estimate these quantities
- **Module 7:** Confidence intervals use variance to quantify uncertainty
- **Module 9:** MLE often involves maximizing expected log-likelihood
- **Module 10:** Bayesian inference updates beliefs about means and variances

---

## Practice Problems

1. Calculate E[X] and Var(X) for a loaded die where P(6) = 1/2 and other faces have equal probability.

2. If X ~ Poisson(4), find P(|X - 4| ≥ 4) using (a) exact computation and (b) Chebyshev's inequality. Compare.

3. A roulette wheel has 18 red, 18 black, and 2 green slots. If you bet $1 on red (win $1 if red, lose $1 otherwise), what is E[winnings] and Var(winnings)?

4. Prove that Var(X) ≥ 0 for any random variable X.

5. If X₁, X₂, ..., X₁₀₀ are independent with E[Xᵢ] = 5 and Var(Xᵢ) = 4, find E[S] and Var(S) where S = X₁ + X₂ + ... + X₁₀₀.

---

## Further Reading

- Ross, S. *A First Course in Probability* - Chapters 4-5
- DeGroot & Schervish, *Probability and Statistics* - Chapter 4
- Casella & Berger, *Statistical Inference* - Chapter 2
