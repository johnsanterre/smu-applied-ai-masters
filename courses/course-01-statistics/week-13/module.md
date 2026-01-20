# Module 13: Monte Carlo Methods

## Overview

Monte Carlo methods use random sampling to solve problems that are analytically intractable. This module covers the fundamentals of Monte Carlo simulation, importance sampling, and Markov Chain Monte Carlo (MCMC)—the computational backbone of modern Bayesian statistics. These methods are essential for probabilistic programming and training complex models.

---

## 1. The Monte Carlo Idea

### Core Principle

Replace analytical computation with sampling:

```
E[f(X)] = ∫ f(x) p(x) dx ≈ (1/n) Σᵢ f(xᵢ)  where xᵢ ~ p(x)
```

**Law of Large Numbers guarantees convergence!**

### When to Use Monte Carlo

- High-dimensional integrals
- Complex posterior distributions
- Expectations that lack closed forms
- Simulating complex systems

---

## 2. Basic Monte Carlo Estimation

### Estimating Expectations

```python
import numpy as np
from scipy import stats

# Estimate E[X²] where X ~ Normal(0, 1)
# True answer: Var(X) = 1

np.random.seed(42)
n_samples = 10000
samples = np.random.normal(0, 1, n_samples)
estimate = np.mean(samples ** 2)
se = np.std(samples ** 2) / np.sqrt(n_samples)

print(f"Estimate: {estimate:.4f} ± {se:.4f}")
print(f"True value: 1.0000")
```

### Estimating Probabilities

```
P(A) = E[𝟙(X ∈ A)] ≈ (1/n) Σᵢ 𝟙(xᵢ ∈ A)
```

**Example:** Estimate P(|X| > 2) for X ~ N(0, 1)

```python
samples = np.random.normal(0, 1, 100000)
p_estimate = np.mean(np.abs(samples) > 2)
p_true = 2 * (1 - stats.norm.cdf(2))

print(f"Estimate: {p_estimate:.4f}")
print(f"True: {p_true:.4f}")
```

### Monte Carlo Integration

```
∫₀^1 f(x) dx ≈ (1/n) Σᵢ f(xᵢ)  where xᵢ ~ Uniform(0, 1)
```

**Example:** Estimate π using circle area

```python
def estimate_pi(n_samples):
    x = np.random.uniform(0, 1, n_samples)
    y = np.random.uniform(0, 1, n_samples)
    inside_circle = (x**2 + y**2) <= 1
    return 4 * np.mean(inside_circle)

for n in [100, 1000, 10000, 100000]:
    pi_est = estimate_pi(n)
    print(f"n={n:6d}: π ≈ {pi_est:.4f}")
```

---

## 3. Error Analysis

### Standard Error

```
SE = σ / √n
```

Error decreases as 1/√n regardless of dimension.

### Confidence Intervals

95% CI for Monte Carlo estimate:
```
Estimate ± 1.96 × SE
```

### How Many Samples?

To achieve precision ε with probability 1-α:
```
n = (z_{α/2} × σ / ε)²
```

---

## 4. Importance Sampling

### The Problem

What if we can't sample from p(x), or p(x) is near zero where f(x) is large?

### The Solution

Sample from a different distribution q(x) and reweight:

```
E_p[f(X)] = ∫ f(x) × (p(x)/q(x)) × q(x) dx = E_q[f(X) × w(X)]
```

Where w(x) = p(x)/q(x) are importance weights.

### Implementation

```python
def importance_sampling(f, p, q, q_sampler, n_samples):
    """
    Estimate E_p[f(X)] using samples from q
    
    f: function to estimate expectation of
    p: target density (up to constant)
    q: proposal density
    q_sampler: function that samples from q
    """
    samples = q_sampler(n_samples)
    weights = p(samples) / q(samples)
    values = f(samples)
    
    # Self-normalized importance sampling
    return np.sum(weights * values) / np.sum(weights)

# Example: Estimate E[X²] for truncated normal (X > 2)
def f(x): return x ** 2
def p(x): return stats.norm.pdf(x) * (x > 2)
def q(x): return stats.expon.pdf(x - 2)  # Shifted exponential
def q_sampler(n): return stats.expon.rvs(size=n) + 2

estimate = importance_sampling(f, p, q, q_sampler, 10000)
print(f"E[X² | X > 2] ≈ {estimate:.3f}")
```

---

## 5. The MCMC Revolution

### The Challenge

For complex posteriors, we often can't:
1. Sample directly
2. Evaluate the normalizing constant
3. Perform importance sampling (curse of dimensionality)

### The Insight

Construct a Markov chain whose stationary distribution is the target!

```
Run chain for long time → Samples from target distribution
```

---

## 6. Metropolis-Hastings Algorithm

### The Algorithm

1. Start at x₀
2. For t = 1, 2, ...:
   - Propose x' from proposal distribution q(x' | xₜ)
   - Compute acceptance probability:
     ```
     α = min(1, [p(x') × q(xₜ | x')] / [p(xₜ) × q(x' | xₜ)])
     ```
   - Accept x' with probability α; otherwise stay at xₜ

### Key Properties

- Only requires p(x) up to a constant (normalization cancels!)
- Chain converges to target distribution
- Works for any proposal (but efficiency varies)

### Implementation

```python
def metropolis_hastings(log_p, proposal, x0, n_samples, burn_in=1000):
    """
    Metropolis-Hastings sampler
    
    log_p: log of target density (up to constant)
    proposal: function(x) returning (x_new, log_forward, log_backward)
    x0: initial state
    n_samples: number of samples to return
    burn_in: samples to discard
    """
    x = x0
    samples = []
    log_p_x = log_p(x)
    
    for i in range(n_samples + burn_in):
        # Propose
        x_new, log_forward, log_backward = proposal(x)
        log_p_new = log_p(x_new)
        
        # Acceptance probability
        log_alpha = log_p_new - log_p_x + log_backward - log_forward
        
        # Accept/reject
        if np.log(np.random.random()) < log_alpha:
            x = x_new
            log_p_x = log_p_new
        
        if i >= burn_in:
            samples.append(x.copy())
    
    return np.array(samples)

# Example: Sample from bimodal distribution
def log_target(x):
    return np.logaddexp(stats.norm.logpdf(x, -2, 0.5),
                        stats.norm.logpdf(x, 2, 0.5))

def proposal(x):
    x_new = x + np.random.normal(0, 1)
    return x_new, 0, 0  # Symmetric proposal

samples = metropolis_hastings(log_target, proposal, x0=0, n_samples=10000)

plt.hist(samples, bins=50, density=True, alpha=0.7)
plt.title('MCMC Samples from Bimodal Distribution')
```

---

## 7. Gibbs Sampling

### The Idea

For multivariate distributions, sample each variable conditioned on the others:

```
Repeat:
    Sample x₁ ~ p(x₁ | x₂, x₃, ..., xₙ)
    Sample x₂ ~ p(x₂ | x₁, x₃, ..., xₙ)
    ...
    Sample xₙ ~ p(xₙ | x₁, x₂, ..., xₙ₋₁)
```

### Advantages

- No rejection—every step is accepted
- Natural for graphical models (conditional distributions follow from structure)

### Implementation

```python
def gibbs_sampler(conditionals, x0, n_samples, burn_in=1000):
    """
    Gibbs sampler for d-dimensional distribution
    
    conditionals: list of functions, each samples x_i | x_{-i}
    x0: initial state
    """
    x = x0.copy()
    d = len(x)
    samples = []
    
    for i in range(n_samples + burn_in):
        for j in range(d):
            x[j] = conditionals[j](x)
        
        if i >= burn_in:
            samples.append(x.copy())
    
    return np.array(samples)

# Example: Bivariate normal
def gibbs_bivariate_normal(n_samples, rho=0.8):
    """Sample from bivariate normal with correlation rho"""
    
    def sample_x1_given_x2(x):
        return np.random.normal(rho * x[1], np.sqrt(1 - rho**2))
    
    def sample_x2_given_x1(x):
        return np.random.normal(rho * x[0], np.sqrt(1 - rho**2))
    
    conditionals = [sample_x1_given_x2, sample_x2_given_x1]
    return gibbs_sampler(conditionals, [0, 0], n_samples)

samples = gibbs_bivariate_normal(5000)
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.1)
```

---

## 8. Diagnostics

### Trace Plots

```python
def plot_trace(samples, name='Parameter'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Trace plot
    axes[0].plot(samples)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel(name)
    axes[0].set_title('Trace Plot')
    
    # Histogram
    axes[1].hist(samples, bins=50, density=True)
    axes[1].set_xlabel(name)
    axes[1].set_title('Posterior Histogram')
    
    plt.tight_layout()
```

### Convergence Checks

**Gelman-Rubin R̂:** Compare variance within chains to variance between chains
```
R̂ ≈ 1 suggests convergence
R̂ > 1.1 suggests problems
```

**Effective Sample Size (ESS):** Accounts for autocorrelation
```
ESS = n / (1 + 2 × Σₖ ρ(k))
```

### Autocorrelation

```python
def plot_acf(samples, max_lag=50):
    acf = [np.corrcoef(samples[:-lag], samples[lag:])[0, 1] 
           for lag in range(1, max_lag)]
    plt.bar(range(1, max_lag), acf)
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.axhline(0, color='k', linewidth=0.5)
```

---

## 9. Modern MCMC Methods

### Hamiltonian Monte Carlo (HMC)

Uses gradient information to make distant proposals with high acceptance.

**Advantages:**
- Much more efficient for high dimensions
- Reduced random walk behavior

### No-U-Turn Sampler (NUTS)

Auto-tunes HMC parameters. Used by default in Stan and PyMC.

### Probabilistic Programming

Libraries that automate MCMC:
- **Stan:** Fast HMC
- **PyMC:** Python-based, uses NUTS
- **Pyro:** Deep learning + probabilistic programming

```python
# PyMC example
import pymc as pm

with pm.Model() as model:
    # Priors
    mu = pm.Normal('mu', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=1)
    
    # Likelihood
    y_obs = pm.Normal('y', mu=mu, sigma=sigma, observed=data)
    
    # Sample
    trace = pm.sample(1000, tune=1000)

pm.plot_posterior(trace)
```

---

## 10. Applications

| Application | Method |
|-------------|--------|
| Bayesian posterior | MCMC |
| High-dimensional integrals | MC integration |
| Option pricing | MC simulation |
| Physics simulations | MC methods |
| A/B test analysis | Bayesian MCMC |
| Language models | Sampling for generation |

---

## Key Takeaways

1. **Monte Carlo:** Estimate expectations via random sampling

2. **Error ∝ 1/√n:** Independent of dimension!

3. **Importance sampling:** Sample from q, reweight to get E_p

4. **MCMC:** Construct Markov chain with target as stationary distribution

5. **Metropolis-Hastings:** General-purpose, only needs p up to constant

6. **Gibbs sampling:** Sample each coordinate from conditional

7. **Diagnostics matter:** Check trace plots, R̂, ESS

8. **HMC/NUTS:** Modern efficient methods for high dimensions

---

## Connections to Other Modules

- **Module 10:** MCMC enables Bayesian inference for complex posteriors
- **Module 11:** MCMC is built on Markov chain theory
- **Module 12:** Gibbs sampling natural for graphical models

---

## Practice Problems

1. Estimate P(X + Y > 3) where X, Y ~ Exp(1) using 10,000 samples.

2. Use importance sampling to estimate E[X] for X ~ Normal(0, 1) truncated to [3, ∞).

3. Implement M-H to sample from Gamma(α=2, β=1) using Normal proposals.

4. Compare Gibbs sampling and M-H for sampling from a bivariate normal.

5. Run MCMC with 4 chains and compute R̂ for convergence diagnostics.

---

## Further Reading

- Robert & Casella, *Monte Carlo Statistical Methods*
- MacKay, *Information Theory, Inference, and Learning Algorithms* - Ch. 29
- Gelman et al., *Bayesian Data Analysis* - Chapters 10-12
- Stan Documentation: https://mc-stan.org/
