# Module 8: Hypothesis Testing

## Overview

Hypothesis testing provides a framework for making decisions with data. Given evidence, should we reject a claim? Is a new treatment better than the standard? Does this feature improve the model? This module covers the logic of hypothesis testing, p-values, types of errors, and practical considerations that every data scientist must understand.

---

## 1. The Hypothesis Testing Framework

### Components

1. **Null hypothesis (H₀):** The default claim, often "no effect" or "no difference"
2. **Alternative hypothesis (H₁ or Hₐ):** What we're trying to show
3. **Test statistic:** A value computed from data
4. **Decision rule:** Based on the test statistic, reject or fail to reject H₀

### Example Setup

Testing whether a coin is fair:
- H₀: p = 0.5 (fair coin)
- H₁: p ≠ 0.5 (biased coin)

### One-Tailed vs Two-Tailed Tests

**Two-tailed (≠):** H₁: μ ≠ μ₀
```
Reject if test statistic too extreme in either direction
```

**One-tailed (<):** H₁: μ < μ₀
```
Reject only if test statistic extremely low
```

**One-tailed (>):** H₁: μ > μ₀
```
Reject only if test statistic extremely high
```

---

## 2. Type I and Type II Errors

### Error Types

|  | H₀ True | H₀ False |
|--|---------|----------|
| **Reject H₀** | Type I Error (α) | Correct ✓ |
| **Fail to Reject** | Correct ✓ | Type II Error (β) |

- **Type I Error (False Positive):** Rejecting H₀ when it's true
- **Type II Error (False Negative):** Failing to reject H₀ when it's false

### Significance Level (α)

The probability of Type I error we're willing to accept:
```
α = P(Reject H₀ | H₀ is true)
```

Common values: α = 0.05, 0.01, 0.10

### Power (1 - β)

```
Power = P(Reject H₀ | H₀ is false) = 1 - β
```

Higher power = better ability to detect real effects.

---

## 3. The Testing Procedure

### Step-by-Step

1. **State hypotheses:** H₀ and H₁
2. **Choose α:** Typically 0.05
3. **Compute test statistic:** From sample data
4. **Find p-value:** Probability of observing this extreme a result under H₀
5. **Make decision:** 
   - If p-value < α: Reject H₀
   - If p-value ≥ α: Fail to reject H₀

### Visual Interpretation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

fig, ax = plt.subplots(figsize=(10, 5))
x = np.linspace(-4, 4, 1000)
ax.plot(x, norm.pdf(x), 'b-', linewidth=2)

# Rejection regions for α = 0.05, two-tailed
alpha = 0.05
z_crit = norm.ppf(1 - alpha/2)

ax.fill_between(x[x > z_crit], norm.pdf(x[x > z_crit]), color='red', alpha=0.3)
ax.fill_between(x[x < -z_crit], norm.pdf(x[x < -z_crit]), color='red', alpha=0.3)

ax.axvline(z_crit, color='red', linestyle='--', label=f'Critical value = ±{z_crit:.2f}')
ax.axvline(-z_crit, color='red', linestyle='--')
ax.set_title('Rejection Regions for α = 0.05 (Two-Tailed)')
ax.legend()
```

---

## 4. Z-Test for Means (σ Known)

### Test Statistic

```
Z = (X̄ - μ₀) / (σ/√n)
```

Under H₀, Z ~ N(0, 1).

### Example

Claim: A factory produces bolts with mean length 10cm. A sample of 36 has X̄ = 10.3cm. Population σ = 0.6cm. Test at α = 0.05.

```python
from scipy.stats import norm

def z_test(x_bar, mu_0, sigma, n, alternative='two-sided'):
    z = (x_bar - mu_0) / (sigma / np.sqrt(n))
    
    if alternative == 'two-sided':
        p_value = 2 * (1 - norm.cdf(abs(z)))
    elif alternative == 'greater':
        p_value = 1 - norm.cdf(z)
    else:  # less
        p_value = norm.cdf(z)
    
    return z, p_value

z_stat, p_val = z_test(x_bar=10.3, mu_0=10, sigma=0.6, n=36)
print(f"Z = {z_stat:.2f}, p-value = {p_val:.4f}")
# Z = 3.00, p-value = 0.0027

if p_val < 0.05:
    print("Reject H₀: Mean differs from 10cm")
```

---

## 5. T-Test for Means (σ Unknown)

### Test Statistic

```
T = (X̄ - μ₀) / (S/√n)
```

Under H₀, T ~ t_{n-1}.

### One-Sample T-Test

```python
from scipy.stats import ttest_1samp

# Sample data
sample = np.array([52, 48, 55, 51, 49, 53, 50, 52, 47, 51])
mu_0 = 50

t_stat, p_value = ttest_1samp(sample, mu_0)
print(f"t = {t_stat:.3f}, p-value = {p_value:.4f}")

# Manual calculation
x_bar = sample.mean()
s = sample.std(ddof=1)
n = len(sample)
t_manual = (x_bar - mu_0) / (s / np.sqrt(n))
print(f"Manual t = {t_manual:.3f}")
```

### Two-Sample T-Test

Comparing means of two groups:

```python
from scipy.stats import ttest_ind

group_A = np.array([85, 90, 78, 92, 88, 76, 95])
group_B = np.array([78, 82, 85, 74, 80, 79, 83])

t_stat, p_value = ttest_ind(group_A, group_B)
print(f"t = {t_stat:.3f}, p-value = {p_value:.4f}")
```

---

## 6. Understanding P-Values

### Definition

The p-value is the probability of observing data as extreme or more extreme than what we observed, **assuming H₀ is true**.

```
p-value = P(data this extreme | H₀ true)
```

### What P-Values Are NOT

- NOT P(H₀ is true)
- NOT P(results due to chance)
- NOT the probability of making an error

### Interpretation Guidelines

| p-value | Interpretation |
|---------|---------------|
| < 0.01 | Strong evidence against H₀ |
| 0.01 - 0.05 | Moderate evidence against H₀ |
| 0.05 - 0.10 | Weak evidence against H₀ |
| > 0.10 | Little evidence against H₀ |

### The 0.05 Threshold

- Historical convention (Fisher)
- Arbitrary but widely used
- Many fields moving toward different standards
- Better: report exact p-value and effect size

---

## 7. Connection to Confidence Intervals

**Duality:** A 95% CI contains all values μ₀ that would not be rejected at α = 0.05.

```python
# If 95% CI is (48.2, 52.3)
# Then we wouldn't reject H₀: μ = 50 at α = 0.05
# But would reject H₀: μ = 48 at α = 0.05
```

Advantages of CIs:
- Show effect size, not just significance
- More informative than yes/no decision

---

## 8. Power Analysis

### Factors Affecting Power

1. **Effect size:** Larger effect = higher power
2. **Sample size:** Larger n = higher power
3. **Significance level:** Higher α = higher power (but more Type I errors)
4. **Variability:** Lower σ = higher power

### Computing Power

```python
from scipy.stats import norm

def power_z_test(mu_0, mu_1, sigma, n, alpha=0.05):
    """Power for one-tailed z-test (H₁: μ > μ₀)"""
    z_crit = norm.ppf(1 - alpha)
    
    # Under H₁, the test statistic has mean (μ₁ - μ₀)/(σ/√n)
    noncentrality = (mu_1 - mu_0) / (sigma / np.sqrt(n))
    
    power = 1 - norm.cdf(z_crit - noncentrality)
    return power

# Example: Detect μ = 52 when H₀: μ = 50, σ = 10
for n in [10, 30, 50, 100]:
    pwr = power_z_test(mu_0=50, mu_1=52, sigma=10, n=n)
    print(f"n = {n:3d}: Power = {pwr:.3f}")
```

### Sample Size for Desired Power

```python
from scipy.stats import norm

def sample_size_for_power(effect_size, sigma, power=0.80, alpha=0.05):
    """Sample size for two-tailed z-test"""
    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(power)
    n = ((z_alpha + z_beta) * sigma / effect_size) ** 2
    return int(np.ceil(n))

n_required = sample_size_for_power(effect_size=2, sigma=10)
print(f"Required n for 80% power: {n_required}")  # 196
```

---

## 9. Multiple Testing Problem

### The Issue

If you test 20 hypotheses at α = 0.05, you expect 1 false positive even if all nulls are true!

```
P(at least 1 false positive in 20 tests) = 1 - 0.95²⁰ ≈ 0.64
```

### Bonferroni Correction

Divide α by the number of tests:
```
α_adjusted = α / m
```

For 20 tests at α = 0.05: use α_adjusted = 0.0025

**Problem:** Very conservative—reduces power.

### False Discovery Rate (FDR)

Control the expected proportion of false positives among rejections:

```python
from scipy.stats import false_discovery_control

p_values = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.10, 0.20]
reject = false_discovery_control(p_values, method='bh')
print(f"Reject using BH: {reject}")
```

---

## 10. Practical Considerations

### Statistical vs Practical Significance

A result can be statistically significant but practically meaningless:
- Very large n can detect tiny, irrelevant effects
- Always report effect sizes alongside p-values

### Common Effect Size Measures

**Cohen's d (for means):**
```
d = (X̄₁ - X̄₂) / pooled_std
```

| d | Interpretation |
|---|---------------|
| 0.2 | Small |
| 0.5 | Medium |
| 0.8 | Large |

### Reporting Best Practices

Bad: "p < 0.05, significant"

Good: "Mean difference = 2.3 points (95% CI: 0.8 to 3.8), t(48) = 3.2, p = 0.002, d = 0.45"

---

## Key Takeaways

1. **H₀ is tested, not proven:** We reject or fail to reject; never "accept"

2. **Type I (α):** False positive; Type II (β): False negative

3. **p-value:** Probability of data this extreme if H₀ true (NOT probability H₀ is true)

4. **p < α → reject H₀**, but always consider effect size

5. **Power = 1 - β:** Ability to detect real effects; aim for ≥ 0.80

6. **Multiple testing** inflates false positives—use corrections

7. **Statistical significance ≠ practical importance**

---

## Connections to Future Modules

- **Module 9:** MLE provides estimates used in tests
- **Module 10:** Bayesian hypothesis testing provides alternatives
- *Course on ML:* A/B testing, model comparison

---

## Practice Problems

1. A sample of 25 has mean 82 and std 12. Test H₀: μ = 80 vs H₁: μ ≠ 80 at α = 0.05.

2. In an A/B test, group A (n=100) has mean 5.2, group B (n=100) has mean 4.8, pooled std = 1.5. Is the difference significant?

3. How many subjects do you need to detect an effect of d = 0.5 with 80% power at α = 0.05?

4. You run 50 hypothesis tests and find 5 significant results. Are you concerned about false positives?

5. A drug study finds p = 0.048. The effect is reducing headache duration by 2 minutes (from 60 to 58). Discuss.

---

## Further Reading

- Wasserstein & Lazar, "The ASA Statement on p-Values" (2016)
- Cohen, J. "The Earth is Round (p < .05)"
- Cumming, G. *Understanding the New Statistics*
