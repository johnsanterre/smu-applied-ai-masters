# Module 7: Point and Interval Estimation

## Overview

Estimation is the bridge between data and unknowns. Given a sample, how do we estimate population parameters? This module covers point estimates (single best guesses) and interval estimates (ranges that likely contain the true value). These concepts are fundamental for quantifying uncertainty in any data-driven field.

---

## 1. The Estimation Problem

### Setup

- **Population parameter θ:** Unknown quantity we want to learn (e.g., μ, σ², p)
- **Sample data:** X₁, X₂, ..., Xₙ drawn from the population
- **Estimator θ̂:** A function of the sample data that estimates θ

### Point vs Interval Estimation

- **Point estimate:** Single value (θ̂ = 42.5)
- **Interval estimate:** Range with confidence level (95% CI: [40.1, 44.9])

---

## 2. Properties of Point Estimators

### Bias

An estimator is **unbiased** if E[θ̂] = θ

```
Bias(θ̂) = E[θ̂] - θ
```

**Example:** Sample mean X̄ is unbiased for μ:
```
E[X̄] = μ  →  Bias = 0
```

### Variance

```
Var(θ̂) = E[(θ̂ - E[θ̂])²]
```

Lower variance = more precise estimator.

### Mean Squared Error (MSE)

```
MSE(θ̂) = E[(θ̂ - θ)²] = Var(θ̂) + [Bias(θ̂)]²
```

MSE captures both bias and variance—the **bias-variance tradeoff**.

### Consistency

An estimator is **consistent** if θ̂ → θ as n → ∞.

The sample mean is consistent: X̄ → μ by the Law of Large Numbers.

---

## 3. Common Point Estimators

### Estimating the Mean

```
μ̂ = X̄ = (1/n) Σ Xᵢ
```

Properties:
- Unbiased: E[X̄] = μ
- Variance: Var(X̄) = σ²/n
- Consistent

### Estimating Variance

**Sample variance:**
```
S² = (1/(n-1)) Σ (Xᵢ - X̄)²
```

Why n-1? Without Bessel's correction:
- E[(1/n) Σ (Xᵢ - X̄)²] = ((n-1)/n) σ² ≠ σ²

Using n-1 gives unbiased estimate: E[S²] = σ²

### Estimating Proportions

For binary outcomes (Bernoulli), estimate p with:
```
p̂ = (number of successes) / n = X̄
```

Unbiased: E[p̂] = p
Variance: Var(p̂) = p(1-p)/n

```python
import numpy as np

# Simulate estimation
np.random.seed(42)
true_mean = 100
true_std = 15

# Multiple samples
n_samples = 1000
sample_size = 30
means = []
variances = []

for _ in range(n_samples):
    sample = np.random.normal(true_mean, true_std, sample_size)
    means.append(sample.mean())
    variances.append(sample.var(ddof=1))  # Using n-1

print(f"True μ = {true_mean}, E[X̄] ≈ {np.mean(means):.2f}")
print(f"True σ² = {true_std**2}, E[S²] ≈ {np.mean(variances):.2f}")
```

---

## 4. Confidence Intervals

### Definition

A **95% confidence interval** is a random interval that contains the true parameter with probability 0.95.

**Interpretation (frequentist):** If we repeated the sampling process many times, 95% of the constructed intervals would contain the true parameter.

**NOT:** "There's a 95% probability that θ is in this interval." (That's a Bayesian statement.)

### General Form

```
Point estimate ± Margin of error
        θ̂     ±   (critical value) × SE(θ̂)
```

---

## 5. CI for the Mean (σ Known)

If X₁, ..., Xₙ ~ N(μ, σ²) with σ known:

```
X̄ ~ N(μ, σ²/n)

Standardized: Z = (X̄ - μ) / (σ/√n) ~ N(0,1)
```

### Construction

For 95% CI, we need z_{0.025} = 1.96:

```
P(-1.96 < Z < 1.96) = 0.95

P(-1.96 < (X̄ - μ)/(σ/√n) < 1.96) = 0.95

P(X̄ - 1.96(σ/√n) < μ < X̄ + 1.96(σ/√n)) = 0.95
```

**95% CI for μ:**
```
X̄ ± 1.96 × σ/√n
```

### Common Critical Values

| Confidence Level | z* |
|-----------------|-----|
| 90% | 1.645 |
| 95% | 1.960 |
| 99% | 2.576 |

```python
from scipy.stats import norm

def ci_known_variance(x_bar, sigma, n, confidence=0.95):
    z_star = norm.ppf(1 - (1 - confidence) / 2)
    margin = z_star * sigma / np.sqrt(n)
    return (x_bar - margin, x_bar + margin)

# Example
ci = ci_known_variance(x_bar=105, sigma=15, n=25)
print(f"95% CI: ({ci[0]:.2f}, {ci[1]:.2f})")  # (99.12, 110.88)
```

---

## 6. CI for the Mean (σ Unknown): The t-Distribution

When σ is unknown, we estimate it with S. The resulting distribution is NOT normal—it's the **t-distribution**.

### The t-Distribution

```
T = (X̄ - μ) / (S/√n) ~ t_{n-1}
```

Properties:
- Bell-shaped, symmetric
- Heavier tails than normal
- More uncertainty when estimating σ
- As n → ∞, t → N(0,1)

### CI Using t-Distribution

```
X̄ ± t*_{n-1} × S/√n
```

Where t* is the critical value from t-distribution with n-1 degrees of freedom.

```python
from scipy.stats import t

def ci_unknown_variance(sample, confidence=0.95):
    n = len(sample)
    x_bar = np.mean(sample)
    s = np.std(sample, ddof=1)
    
    t_star = t.ppf(1 - (1 - confidence) / 2, df=n-1)
    margin = t_star * s / np.sqrt(n)
    
    return (x_bar - margin, x_bar + margin)

# Example with real sample
sample = np.random.normal(100, 15, 25)
ci = ci_unknown_variance(sample)
print(f"Sample mean: {np.mean(sample):.2f}")
print(f"95% CI: ({ci[0]:.2f}, {ci[1]:.2f})")
```

### t vs z Critical Values

| df | t*_{0.025} | z*_{0.025} |
|----|-----------|------------|
| 5 | 2.571 | 1.960 |
| 10 | 2.228 | 1.960 |
| 30 | 2.042 | 1.960 |
| 100 | 1.984 | 1.960 |
| ∞ | 1.960 | 1.960 |

---

## 7. CI for Proportions

For large n, p̂ is approximately normal:

```
p̂ ~ approximately N(p, p(1-p)/n)
```

### Wald Confidence Interval

```
p̂ ± z* × √(p̂(1-p̂)/n)
```

**Example:** In a poll of 500 people, 280 support a policy.

```python
def ci_proportion(successes, n, confidence=0.95):
    p_hat = successes / n
    z_star = norm.ppf(1 - (1 - confidence) / 2)
    margin = z_star * np.sqrt(p_hat * (1 - p_hat) / n)
    return (p_hat - margin, p_hat + margin)

ci = ci_proportion(280, 500)
print(f"p̂ = {280/500:.3f}")
print(f"95% CI: ({ci[0]:.3f}, {ci[1]:.3f})")  # (0.517, 0.603)
```

---

## 8. Factors Affecting Interval Width

### 1. Sample Size (n)

Width ∝ 1/√n

To halve the width, quadruple the sample size.

### 2. Confidence Level

Higher confidence → wider interval

| Confidence | z* | Relative Width |
|------------|-----|----------------|
| 90% | 1.645 | 84% |
| 95% | 1.960 | 100% |
| 99% | 2.576 | 131% |

### 3. Population Variability (σ)

More variable population → wider interval (more uncertainty)

### Trade-offs

- Want narrow interval → need large n or accept lower confidence
- Want high confidence → accept wider interval or increase n

---

## 9. Sample Size Determination

### For Means

To achieve margin of error E with confidence (1-α):

```
n = (z*σ / E)²
```

### For Proportions

```
n = (z*/E)² × p(1-p)
```

Conservative (maximum variance): use p = 0.5

```python
def required_n_for_mean(sigma, E, confidence=0.95):
    z_star = norm.ppf(1 - (1 - confidence) / 2)
    return int(np.ceil((z_star * sigma / E) ** 2))

def required_n_for_proportion(E, confidence=0.95, p=0.5):
    z_star = norm.ppf(1 - (1 - confidence) / 2)
    return int(np.ceil((z_star / E) ** 2 * p * (1 - p)))

print(f"For mean (σ=15, E=2): n = {required_n_for_mean(15, 2)}")  # 217
print(f"For proportion (E=0.03): n = {required_n_for_proportion(0.03)}")  # 1068
```

---

## Key Takeaways

1. **Point estimates** give single best guesses; **interval estimates** quantify uncertainty

2. **Good estimators** are unbiased, low variance, and consistent

3. **MSE = Variance + Bias²** captures total estimation error

4. **95% CI interpretation:** 95% of such intervals contain the true parameter

5. **Use t-distribution** when σ is unknown (always in practice)

6. **Width ∝ 1/√n:** Quadruple n to halve the interval width

7. **Higher confidence = wider interval** (trade precision for certainty)

---

## Connections to Future Modules

- **Module 8:** Hypothesis testing uses CIs and sampling distributions
- **Module 9:** MLE provides another way to find point estimates
- **Module 10:** Bayesian intervals (credible intervals) have different interpretation

---

## Practice Problems

1. A sample of 50 measurements has mean 45.3 and std 8.2. Construct a 99% CI for the population mean.

2. In a survey of 400 customers, 120 prefer Product A. Find a 95% CI for the true proportion.

3. How large a sample is needed to estimate a mean within ±3 with 95% confidence if σ ≈ 20?

4. Explain why using z instead of t for small samples gives intervals that are too narrow.

5. Simulate 1000 95% CIs for μ when true μ=50. What fraction contain 50?

---

## Further Reading

- DeGroot & Schervish, *Probability and Statistics* - Chapter 8
- Casella & Berger, *Statistical Inference* - Chapter 9
- NIST Engineering Statistics Handbook - Confidence Intervals
