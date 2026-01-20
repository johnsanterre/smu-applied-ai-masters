# Module 9: Maximum Likelihood Estimation

## Overview

Maximum Likelihood Estimation (MLE) is one of the most important techniques in statistics and machine learning. Given observed data, MLE finds the parameter values that make the data most probable. This module covers the theory, computation, and applications of MLE—the foundation for logistic regression, neural network training, and countless other methods.

---

## 1. The Likelihood Function

### Setup

- **Data:** Observations x₁, x₂, ..., xₙ (assumed i.i.d.)
- **Model:** A parametric family of distributions f(x; θ)
- **Goal:** Find θ that best explains the observed data

### Definition

The **likelihood function** is the probability of the data viewed as a function of the parameter:

```
L(θ) = P(data | θ) = ∏ᵢ f(xᵢ; θ)
```

**Key insight:** Likelihood is NOT P(θ | data). It's P(data | θ)—same formula, different perspective.

### Example: Coin Flips

Observe: H, H, T, H, H (4 heads, 1 tail)

For a coin with parameter p:
```
L(p) = p⁴ × (1-p)¹ = p⁴(1-p)
```

```python
import numpy as np
import matplotlib.pyplot as plt

def likelihood_coin(p, heads=4, tails=1):
    return (p ** heads) * ((1 - p) ** tails)

p_values = np.linspace(0, 1, 100)
L = [likelihood_coin(p) for p in p_values]

plt.plot(p_values, L)
plt.xlabel('p')
plt.ylabel('L(p)')
plt.title('Likelihood Function for Coin Flips')
plt.axvline(0.8, color='r', linestyle='--', label='MLE')
plt.legend()
```

---

## 2. Maximum Likelihood Estimation

### The Idea

The MLE is the parameter value that maximizes the likelihood:

```
θ̂_MLE = argmax_θ L(θ)
```

**Interpretation:** The MLE is the parameter value that makes the observed data most probable.

### Log-Likelihood

In practice, we work with the log-likelihood:

```
ℓ(θ) = log L(θ) = Σᵢ log f(xᵢ; θ)
```

**Advantages:**
1. Products become sums (easier to work with)
2. Avoids numerical underflow
3. Same maximizer (log is monotonic)

---

## 3. Finding the MLE

### Analytical Approach

1. Write the log-likelihood ℓ(θ)
2. Take derivative: dℓ/dθ
3. Set to zero and solve: dℓ/dθ = 0
4. Verify it's a maximum (second derivative < 0)

### Example: Normal Distribution (Unknown μ, Known σ)

Data x₁, ..., xₙ from N(μ, σ²) with σ known:

```
ℓ(μ) = Σᵢ log[(1/√(2πσ²)) exp(-(xᵢ-μ)²/(2σ²))]
     = -n/2 log(2πσ²) - (1/(2σ²)) Σᵢ(xᵢ-μ)²
```

Taking derivative:
```
dℓ/dμ = (1/σ²) Σᵢ(xᵢ - μ) = 0
→ Σᵢ xᵢ = nμ
→ μ̂ = (1/n) Σᵢ xᵢ = X̄
```

**The MLE for μ is the sample mean!**

### Example: Normal Distribution (Both Unknown)

```
μ̂ = X̄
σ̂² = (1/n) Σᵢ(xᵢ - X̄)²
```

Note: The MLE for variance uses n, not n-1 (slightly biased).

---

## 4. MLE for Common Distributions

| Distribution | Parameter | MLE |
|--------------|-----------|-----|
| Bernoulli(p) | p | X̄ = (successes)/n |
| Poisson(λ) | λ | X̄ |
| Exponential(λ) | λ | 1/X̄ |
| Normal(μ, σ²) | μ | X̄ |
| Normal(μ, σ²) | σ² | (1/n)Σ(xᵢ-X̄)² |
| Uniform(0, θ) | θ | max(xᵢ) |

---

## 5. Properties of MLE

### Consistency

As n → ∞: θ̂_MLE → θ_true

### Asymptotic Normality

For large n:
```
θ̂_MLE ~ approximately N(θ, 1/I(θ))
```

Where I(θ) is the Fisher Information.

### Asymptotic Efficiency

MLE achieves the lowest possible variance among consistent estimators (the Cramér-Rao lower bound).

### Invariance

If θ̂ is the MLE of θ, then g(θ̂) is the MLE of g(θ).

---

## 6. Fisher Information

### Definition

```
I(θ) = E[(∂/∂θ log f(X; θ))²] = -E[∂²/∂θ² log f(X; θ)]
```

**Interpretation:** Fisher Information measures how much information the data provides about the parameter. Higher I(θ) = more precise estimation.

### For n Observations

```
I_n(θ) = n × I(θ)
```

Information grows linearly with sample size.

### Standard Error of MLE

```
SE(θ̂_MLE) ≈ 1/√(I_n(θ)) = 1/√(n × I(θ))
```

---

## 7. Numerical Optimization

When analytical solutions don't exist, we optimize numerically.

### Gradient Ascent

```
θ_{t+1} = θ_t + η × ∇ℓ(θ_t)
```

Where η is the learning rate.

### Using scipy.optimize

```python
from scipy.optimize import minimize
from scipy.stats import norm
import numpy as np

# Generate sample data
np.random.seed(42)
true_mu, true_sigma = 5, 2
data = np.random.normal(true_mu, true_sigma, 100)

# Negative log-likelihood (we minimize this)
def neg_log_likelihood(params, data):
    mu, log_sigma = params  # Use log_sigma to avoid negative sigma
    sigma = np.exp(log_sigma)
    return -np.sum(norm.logpdf(data, mu, sigma))

# Optimize
result = minimize(
    neg_log_likelihood,
    x0=[0, 0],  # Initial guess
    args=(data,),
    method='BFGS'
)

mu_hat = result.x[0]
sigma_hat = np.exp(result.x[1])
print(f"MLE: μ̂ = {mu_hat:.3f}, σ̂ = {sigma_hat:.3f}")
print(f"True: μ = {true_mu}, σ = {true_sigma}")
```

---

## 8. MLE for Regression

### Linear Regression

For Y = Xβ + ε where ε ~ N(0, σ²):

```
ℓ(β, σ²) = -n/2 log(2πσ²) - (1/(2σ²)) Σᵢ(yᵢ - xᵢᵀβ)²
```

Maximizing gives:
```
β̂ = (XᵀX)⁻¹Xᵀy   (same as least squares!)
```

**Key insight:** MLE with normal errors = ordinary least squares.

### Logistic Regression

For binary outcomes with P(Y=1) = σ(xᵀβ):

```
ℓ(β) = Σᵢ [yᵢ log p(xᵢ) + (1-yᵢ) log(1-p(xᵢ))]
```

No closed form—solved numerically.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Generate data
X, y = make_classification(n_samples=1000, n_features=5, random_state=42)

# Fit logistic regression (uses MLE internally)
model = LogisticRegression()
model.fit(X, y)
print(f"Estimated coefficients: {model.coef_}")
```

---

## 9. MLE vs Bayesian Estimation

| Aspect | MLE | Bayesian |
|--------|-----|----------|
| Philosophy | Frequentist | Bayesian |
| Output | Point estimate | Posterior distribution |
| Prior | Not used | Required |
| Interpretation | Most likely parameter | Distribution over parameters |

### Connection to MAP

Maximum a Posteriori (MAP) estimation:
```
θ̂_MAP = argmax_θ P(θ|data) = argmax_θ [P(data|θ) × P(θ)]
```

MLE is MAP with uniform prior.

---

## 10. Practical Considerations

### Likelihood Ratio Tests

Compare nested models:
```
LR = -2[ℓ(θ₀) - ℓ(θ̂)] ~ χ²_k
```

Where k is the difference in number of parameters.

### Information Criteria

Penalize model complexity:

**AIC (Akaike):**
```
AIC = -2ℓ(θ̂) + 2k
```

**BIC (Bayesian):**
```
BIC = -2ℓ(θ̂) + k log(n)
```

Lower is better. BIC penalizes complexity more heavily.

```python
# Model comparison with AIC
def aic(log_likelihood, k):
    return -2 * log_likelihood + 2 * k

# Compare two models
k1, ll1 = 3, -150  # Simpler model
k2, ll2 = 5, -145  # More complex model

print(f"Model 1: AIC = {aic(ll1, k1)}")
print(f"Model 2: AIC = {aic(ll2, k2)}")
# Choose model with lower AIC
```

---

## Key Takeaways

1. **Likelihood L(θ)** = probability of data given parameter

2. **MLE maximizes likelihood:** θ̂ = argmax L(θ)

3. **Use log-likelihood** to avoid numerical issues

4. **MLE is consistent and efficient** asymptotically

5. **Standard error from Fisher Information:** SE ≈ 1/√(nI(θ))

6. **Linear regression + normal errors = MLE = OLS**

7. **AIC/BIC** balance fit vs complexity for model selection

---

## Connections to Future Modules

- **Module 10:** Bayesian inference provides an alternative to MLE
- **Module 12:** Graphical model parameters often estimated via MLE
- *Course on ML:* Training neural networks maximizes (regularized) likelihood

---

## Practice Problems

1. Derive the MLE for λ in Poisson(λ) given observations x₁, ..., xₙ.

2. For data from Uniform(0, θ), show that the MLE is θ̂ = max(xᵢ). Is this unbiased?

3. Implement MLE for an exponential distribution using scipy.optimize.

4. Generate 100 samples from N(3, 4). Find MLEs for μ and σ². Compare to true values.

5. Compare two regression models using AIC. Which is preferred?

---

## Further Reading

- Casella & Berger, *Statistical Inference* - Chapter 7
- Murphy, K. *Machine Learning: A Probabilistic Perspective* - Chapter 5
- Bishop, *Pattern Recognition and ML* - Chapter 2
