# Module 6: Sampling and the Central Limit Theorem

## Overview

The Central Limit Theorem (CLT) is one of the most important results in all of statistics. It explains why the normal distribution appears everywhere and why sample means are so useful. This module covers sampling, the behavior of sample statistics, and why averages of random variables tend toward normality.

Understanding the CLT is essential for inference, hypothesis testing, and interpreting machine learning results.

---

## 1. From Population to Sample

### Population vs Sample

- **Population:** The complete set of items we're interested in (often huge or infinite)
- **Sample:** A subset of the population that we actually observe

Examples:
| Population | Sample |
|------------|--------|
| All voters | Poll of 1000 |
| All website visits | Today's traffic |
| All possible predictions | Training set |

### Random Sampling

A **random sample** X₁, X₂, ..., Xₙ consists of independent, identically distributed (i.i.d.) random variables from the population.

**i.i.d. means:**
- **Independent:** Each observation doesn't affect others
- **Identically distributed:** All drawn from the same distribution

---

## 2. Sample Statistics

### Sample Mean

```
X̄ = (1/n) Σᵢ Xᵢ
```

The sample mean X̄ is itself a random variable—it varies from sample to sample.

### Sample Variance

```
S² = (1/(n-1)) Σᵢ (Xᵢ - X̄)²
```

Note: We divide by (n-1), not n. This is called **Bessel's correction** and makes S² an unbiased estimator of the population variance.

### Computing in Python

```python
import numpy as np

# Population: Normal(100, 15)
population_mean = 100
population_std = 15

# Take a sample
np.random.seed(42)
sample = np.random.normal(population_mean, population_std, size=30)

# Sample statistics
sample_mean = np.mean(sample)
sample_var = np.var(sample, ddof=1)  # ddof=1 for sample variance
sample_std = np.std(sample, ddof=1)

print(f"Sample mean: {sample_mean:.2f} (population: {population_mean})")
print(f"Sample std: {sample_std:.2f} (population: {population_std})")
```

---

## 3. Distribution of the Sample Mean

### Expected Value of X̄

```
E[X̄] = E[(1/n) Σ Xᵢ] = (1/n) Σ E[Xᵢ] = (1/n) × n × μ = μ
```

The sample mean is an **unbiased estimator** of the population mean.

### Variance of X̄

```
Var(X̄) = Var((1/n) Σ Xᵢ) = (1/n²) Σ Var(Xᵢ) = (1/n²) × n × σ² = σ²/n
```

**Key insight:** Variance decreases with sample size!

### Standard Error

The **standard error of the mean (SEM)** is:

```
SE(X̄) = σ / √n
```

This measures how much X̄ varies from sample to sample.

```python
# Demonstrate variance reduction
def simulate_sample_means(pop_mean, pop_std, sample_sizes, n_simulations=10000):
    results = {}
    for n in sample_sizes:
        sample_means = [np.random.normal(pop_mean, pop_std, n).mean() 
                       for _ in range(n_simulations)]
        results[n] = {
            'mean': np.mean(sample_means),
            'std': np.std(sample_means),
            'theoretical_se': pop_std / np.sqrt(n)
        }
    return results

results = simulate_sample_means(100, 15, [5, 10, 30, 100])
for n, r in results.items():
    print(f"n={n:3d}: Mean={r['mean']:.2f}, SE={r['std']:.2f} (theory: {r['theoretical_se']:.2f})")
```

Output:
```
n=  5: Mean=100.01, SE=6.71 (theory: 6.71)
n= 10: Mean=100.00, SE=4.74 (theory: 4.74)
n= 30: Mean=100.01, SE=2.74 (theory: 2.74)
n=100: Mean=100.00, SE=1.50 (theory: 1.50)
```

---

## 4. The Central Limit Theorem

### Statement

Let X₁, X₂, ..., Xₙ be i.i.d. random variables with mean μ and variance σ². Then as n → ∞:

```
(X̄ - μ) / (σ/√n) → N(0, 1)
```

Or equivalently:

```
X̄ ~ approximately N(μ, σ²/n)  for large n
```

### The Magic

**The CLT works regardless of the original distribution!** The population could be:
- Uniform
- Exponential
- Bimodal
- Discrete

As long as it has finite mean and variance, sample means approach normality.

### Visualization

```python
import matplotlib.pyplot as plt
from scipy.stats import expon, uniform

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

distributions = [
    ('Exponential', expon(scale=1)),
    ('Uniform', uniform(0, 1))
]
sample_sizes = [1, 5, 30, 100]

for row, (name, dist) in enumerate(distributions):
    for col, n in enumerate(sample_sizes):
        # Generate 10000 sample means of size n
        sample_means = [dist.rvs(n).mean() for _ in range(10000)]
        
        axes[row, col].hist(sample_means, bins=50, density=True, alpha=0.7)
        axes[row, col].set_title(f'{name}, n={n}')
        
plt.suptitle('Central Limit Theorem: Sample Means Become Normal')
plt.tight_layout()
```

---

## 5. How Large is "Large Enough"?

### Rules of Thumb

- **Symmetric distributions:** n ≥ 15-20 is usually enough
- **Moderately skewed:** n ≥ 30 is the classic rule
- **Heavily skewed:** n ≥ 50 or more

### The More Skewed, the More Samples Needed

```python
from scipy.stats import skew

# Very skewed distribution: Exponential
exp_samples = expon.rvs(size=10000)
print(f"Exponential skewness: {skew(exp_samples):.2f}")

# Need larger n for CLT to kick in
for n in [5, 10, 30, 50, 100]:
    sample_means = [expon.rvs(n).mean() for _ in range(10000)]
    print(f"n={n:3d}: Skewness of means = {skew(sample_means):.3f}")
```

---

## 6. Sum Version of CLT

The CLT also applies to sums:

```
Sₙ = X₁ + X₂ + ... + Xₙ ~ approximately N(nμ, nσ²)
```

Standardized:
```
(Sₙ - nμ) / (σ√n) → N(0, 1)
```

### Application: Insurance Claims

An insurance company has 1000 policies. Each claim is independently distributed with mean $500 and std $800.

What's the probability total claims exceed $600,000?

```python
from scipy.stats import norm

n = 1000
mu = 500
sigma = 800

# Sum is approximately normal
sum_mean = n * mu  # 500,000
sum_std = sigma * np.sqrt(n)  # 25,298

# P(Sum > 600,000)
p = 1 - norm.cdf(600000, sum_mean, sum_std)
print(f"P(Total Claims > $600,000) = {p:.4f}")  # ≈ 0.0000
```

---

## 7. Law of Large Numbers

### Statement

As sample size increases, the sample mean converges to the population mean:

```
X̄ₙ → μ  as n → ∞
```

### Difference from CLT

- **LLN:** X̄ₙ → μ (convergence to a constant)
- **CLT:** Describes the distribution of X̄ₙ around μ

### Visualization

```python
np.random.seed(42)

# Cumulative mean converges to true mean
true_mean = 0.5
samples = np.random.uniform(0, 1, 10000)
cumulative_means = np.cumsum(samples) / np.arange(1, 10001)

plt.figure(figsize=(10, 5))
plt.plot(cumulative_means)
plt.axhline(y=true_mean, color='r', linestyle='--', label='True Mean')
plt.xlabel('Sample Size')
plt.ylabel('Cumulative Mean')
plt.title('Law of Large Numbers')
plt.legend()
```

---

## 8. Practical Applications

### Polling and Surveys

Margin of error ≈ 2 × SE = 2σ/√n

For a poll of n=1000 with p=0.5:
```
SE = √(p(1-p)/n) = √(0.25/1000) ≈ 0.016
Margin of error ≈ ±3.2%
```

### A/B Testing

Comparing two sample means:
```
SE(X̄_A - X̄_B) = √(σ²_A/n_A + σ²_B/n_B)
```

### Sample Size Calculation

To achieve margin of error E with 95% confidence:
```
n = (1.96 × σ / E)²
```

```python
def required_sample_size(std, margin_of_error, confidence=0.95):
    from scipy.stats import norm
    z = norm.ppf(1 - (1 - confidence) / 2)
    n = (z * std / margin_of_error) ** 2
    return int(np.ceil(n))

# Example: Estimate mean with std=15, want margin of error ±2
n = required_sample_size(std=15, margin_of_error=2)
print(f"Required sample size: {n}")  # 217
```

---

## Key Takeaways

1. **Sample mean is unbiased:** E[X̄] = μ

2. **Variance decreases with n:** Var(X̄) = σ²/n

3. **Standard error:** SE = σ/√n measures precision

4. **CLT is universal:** Sample means become normal regardless of population distribution

5. **n ≥ 30** is often enough for CLT (more for skewed distributions)

6. **LLN:** X̄ → μ as n → ∞ (convergence to true mean)

7. **Margin of error ∝ 1/√n:** To halve error, quadruple sample size

---

## Connections to Future Modules

- **Module 7:** Confidence intervals use SE and CLT
- **Module 8:** Hypothesis tests rely on sampling distributions
- **Module 9:** MLE uses sample statistics
- **Module 13:** Monte Carlo methods exploit LLN

---

## Practice Problems

1. IQ scores have μ=100, σ=15. For a sample of n=36, what's the probability X̄ > 105?

2. An exponential distribution has mean 5. How large a sample is needed for the CLT approximation to be reasonable?

3. A coin has unknown probability p. How many flips are needed to estimate p within ±0.02 with 95% confidence?

4. Simulate the sampling distribution of X̄ for a bimodal population. At what n does it become approximately normal?

5. Prove that Var(X̄) = σ²/n using properties of variance.

---

## Further Reading

- Ross, S. *A First Course in Probability* - Chapter 8
- DeGroot & Schervish, *Probability and Statistics* - Chapter 6
- Feller, W. *An Introduction to Probability Theory* - Chapter 8
