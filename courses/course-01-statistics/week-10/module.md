# Module 10: Bayesian Inference

## Overview

Bayesian inference is a framework for updating beliefs in light of evidence. Unlike frequentist methods that estimate fixed but unknown parameters, Bayesian methods treat parameters as random variables with probability distributions. This module covers priors, posteriors, conjugate families, and computational approaches to Bayesian analysis.

Bayesian thinking is increasingly important in modern machine learning, from uncertainty quantification to probabilistic programming.

---

## 1. The Bayesian Framework

### Core Idea

Instead of asking "What's the best estimate of θ?" we ask "What's the probability distribution of θ given the data?"

### Bayes' Theorem for Parameters

```
P(θ | data) = P(data | θ) × P(θ) / P(data)
```

Or more compactly:
```
Posterior ∝ Likelihood × Prior
```

### The Components

| Term | Symbol | Meaning |
|------|--------|---------|
| Prior | P(θ) | Belief about θ before seeing data |
| Likelihood | P(data \| θ) | Probability of data given θ |
| Posterior | P(θ \| data) | Updated belief after seeing data |
| Evidence | P(data) | Normalizing constant |

---

## 2. Prior Distributions

### Types of Priors

**Informative priors:** Encode existing knowledge
```
Example: We know μ is probably between 90-110, so use N(100, 5²)
```

**Weakly informative priors:** Regularize towards reasonable values
```
Example: Use N(0, 10²) to keep coefficients from exploding
```

**Non-informative priors:** Let data dominate
```
Example: Uniform(0, 1) for a probability
```

**Improper priors:** Don't integrate to 1 but can give proper posteriors
```
Example: p(θ) ∝ 1 for all real θ
```

### Choosing Priors

1. **Domain knowledge:** What do experts believe?
2. **Previous studies:** Use posterior from last study as prior
3. **Conjugate convenience:** Mathematically tractable
4. **Sensitivity analysis:** Check if conclusions change with different priors

---

## 3. Conjugate Priors

A prior is **conjugate** if the posterior is in the same family as the prior.

### Why Conjugacy Matters

- Closed-form posteriors
- No numerical integration needed
- Easy sequential updating

### Common Conjugate Pairs

| Likelihood | Prior | Posterior |
|------------|-------|-----------|
| Bernoulli/Binomial | Beta | Beta |
| Poisson | Gamma | Gamma |
| Normal (known σ) | Normal | Normal |
| Normal (known μ) | Inverse-Gamma | Inverse-Gamma |
| Multinomial | Dirichlet | Dirichlet |

---

## 4. Beta-Binomial Model

### Setup

- Data: k successes in n trials (Binomial likelihood)
- Prior: p ~ Beta(α, β)

### The Beta Distribution

```
Beta(α, β): f(p) ∝ p^(α-1) × (1-p)^(β-1)
```

Properties:
- E[p] = α / (α + β)
- Mode = (α - 1) / (α + β - 2) for α, β > 1
- Var[p] = αβ / [(α+β)²(α+β+1)]

### Posterior

```
p | data ~ Beta(α + k, β + n - k)
```

**Interpretation:** 
- Prior "pseudo-counts": α successes, β failures
- Posterior: add observed successes/failures

### Example

Prior: Beta(2, 2) (slightly favors p = 0.5)
Observe: 7 heads in 10 flips

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Prior parameters
alpha_prior, beta_prior = 2, 2

# Data
heads, flips = 7, 10

# Posterior parameters
alpha_post = alpha_prior + heads
beta_post = beta_prior + (flips - heads)

# Plot
p = np.linspace(0, 1, 1000)
plt.plot(p, beta.pdf(p, alpha_prior, beta_prior), 'b-', label='Prior')
plt.plot(p, beta.pdf(p, alpha_post, beta_post), 'r-', label='Posterior')
plt.axvline(heads/flips, color='g', linestyle='--', label='MLE')
plt.xlabel('p')
plt.ylabel('Density')
plt.legend()
plt.title('Bayesian Update: Beta-Binomial')
```

---

## 5. Normal-Normal Model

### Setup

- Data: x₁, ..., xₙ from N(μ, σ²) with σ² known
- Prior: μ ~ N(μ₀, τ²)

### Posterior

```
μ | data ~ N(μ_n, τ²_n)
```

Where:
```
τ²_n = 1 / (n/σ² + 1/τ²)
μ_n = τ²_n × (n×X̄/σ² + μ₀/τ²)
```

### Interpretation

The posterior mean is a weighted average:
```
μ_n = w × X̄ + (1-w) × μ₀
```

Where w depends on relative precisions (1/variance).

**More data → posterior closer to data**
**Stronger prior → posterior closer to prior**

```python
from scipy.stats import norm

def normal_posterior(x_bar, n, sigma_sq, mu_0, tau_sq):
    precision_data = n / sigma_sq
    precision_prior = 1 / tau_sq
    
    tau_post_sq = 1 / (precision_data + precision_prior)
    mu_post = tau_post_sq * (precision_data * x_bar + precision_prior * mu_0)
    
    return mu_post, np.sqrt(tau_post_sq)

# Example
mu_post, sigma_post = normal_posterior(
    x_bar=105, n=25, sigma_sq=225,  # Data: mean 105, n=25, σ=15
    mu_0=100, tau_sq=100  # Prior: N(100, 10)
)
print(f"Posterior: N({mu_post:.2f}, {sigma_post:.2f}²)")
```

---

## 6. Credible Intervals

### Definition

A 95% **credible interval** is an interval that contains θ with posterior probability 0.95:

```
P(a < θ < b | data) = 0.95
```

### Types

**Equal-tailed interval:** 2.5% in each tail

**Highest Posterior Density (HPD):** Shortest interval containing 95%

```python
# Credible interval for Beta posterior
from scipy.stats import beta

alpha_post, beta_post = 9, 5  # From earlier example

# Equal-tailed 95% CI
ci_lower = beta.ppf(0.025, alpha_post, beta_post)
ci_upper = beta.ppf(0.975, alpha_post, beta_post)
print(f"95% CI: ({ci_lower:.3f}, {ci_upper:.3f})")
```

### Interpretation Difference

- **Frequentist CI:** "95% of such intervals contain the true parameter"
- **Bayesian credible interval:** "There's a 95% probability the parameter is in this interval"

---

## 7. Bayesian Point Estimates

### Posterior Mean

```
θ̂ = E[θ | data] = ∫ θ × p(θ | data) dθ
```

Minimizes expected squared error.

### Posterior Mode (MAP)

```
θ̂_MAP = argmax p(θ | data)
```

With flat prior, MAP = MLE.

### Posterior Median

Robust to outliers in the posterior.

---

## 8. Sequential Updating

Bayesian inference naturally handles streaming data.

### Process

1. Start with prior P(θ)
2. Observe data batch → compute posterior
3. Posterior becomes prior for next batch
4. Repeat

### Example: Learning a Coin

```python
def sequential_beta_update(observations, alpha_init=1, beta_init=1):
    """Track posterior as data arrives"""
    alpha, beta_param = alpha_init, beta_init
    history = [(alpha, beta_param)]
    
    for obs in observations:
        if obs == 1:  # Heads
            alpha += 1
        else:  # Tails
            beta_param += 1
        history.append((alpha, beta_param))
    
    return history

# Observe: H, H, T, H, H, T, H
observations = [1, 1, 0, 1, 1, 0, 1]
history = sequential_beta_update(observations)

# Plot evolution
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
p = np.linspace(0, 1, 100)
for i, (ax, (a, b)) in enumerate(zip(axes.flat, history)):
    ax.plot(p, beta.pdf(p, a, b))
    ax.set_title(f'After {i} observations: Beta({a}, {b})')
plt.tight_layout()
```

---

## 9. Computational Methods

### When Conjugacy Fails

Most real problems don't have conjugate priors. Solutions:

### Grid Approximation

Discretize θ and compute posterior at each point:

```python
def grid_approximation(data, prior_func, likelihood_func, grid):
    """Simple grid approximation for 1D posterior"""
    prior = np.array([prior_func(theta) for theta in grid])
    likelihood = np.array([likelihood_func(theta, data) for theta in grid])
    
    posterior = prior * likelihood
    posterior = posterior / posterior.sum()  # Normalize
    
    return posterior

# Example: estimate mean with non-conjugate prior
data = [2.1, 2.5, 2.3, 2.8, 2.4]
grid = np.linspace(0, 5, 1000)

prior_func = lambda theta: 1  # Uniform
likelihood_func = lambda theta, data: np.prod(norm.pdf(data, theta, 0.3))

posterior = grid_approximation(data, prior_func, likelihood_func, grid)

plt.plot(grid, posterior)
plt.xlabel('θ')
plt.ylabel('P(θ | data)')
```

### Markov Chain Monte Carlo (MCMC)

Sample from the posterior (covered in Module 13).

---

## 10. Bayesian vs Frequentist

| Aspect | Frequentist | Bayesian |
|--------|-------------|----------|
| Parameters | Fixed, unknown | Random variables |
| Probability | Long-run frequency | Degree of belief |
| Prior information | Not formally used | Prior distribution |
| Inference | Point estimate + CI | Posterior distribution |
| Interpretation | Procedure properties | Direct probability statements |

### When to Use Bayesian

- Prior knowledge is available and relevant
- Want probability statements about parameters
- Sequential learning is needed
- Small sample sizes
- Complex hierarchical models

---

## Key Takeaways

1. **Posterior ∝ Likelihood × Prior**—the fundamental equation

2. **Prior encodes beliefs** before seeing data; posterior updates them

3. **Conjugate priors** give closed-form posteriors (Beta-Binomial, Normal-Normal)

4. **Credible intervals** have direct probability interpretation

5. **Sequential updating** is natural: old posterior = new prior

6. **Posterior mean minimizes squared error**; MAP with flat prior = MLE

7. **MCMC** enables Bayesian inference for complex models

---

## Connections to Future Modules

- **Module 12:** Graphical models often use Bayesian parameter estimation
- **Module 13:** MCMC methods for sampling from posteriors
- *Course on ML:* Bayesian optimization, Bayesian neural networks

---

## Practice Problems

1. Prior: Beta(1, 1). Observe 8 heads in 10 flips. Find posterior mean, mode, and 95% CI.

2. Use Normal-Normal model with prior N(170, 5²) for height. Observe: 175, 168, 172 (σ=6 known). Find posterior.

3. Show that with flat prior, the posterior mode equals the MLE.

4. Implement grid approximation for a Poisson model with Gamma prior.

5. A drug has prior efficacy p ~ Beta(2, 8). After trial with 15/20 successes, what's P(p > 0.5 | data)?

---

## Further Reading

- Kruschke, J. *Doing Bayesian Data Analysis*
- Gelman et al., *Bayesian Data Analysis*
- McElreath, R. *Statistical Rethinking*
