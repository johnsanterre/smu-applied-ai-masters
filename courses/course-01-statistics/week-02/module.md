# Module 2: Random Variables and Distributions

## Overview

A random variable transforms outcomes into numbers, giving us the power of mathematics to analyze uncertainty. This module covers discrete and continuous random variables, their probability distributions, and the most important distributions you'll encounter in AI and machine learning.

By the end of this module, you'll understand how to describe random phenomena mathematically and recognize common distributions that appear repeatedly in data science.

---

## 1. What is a Random Variable?

### Definition

A **random variable** is a function that assigns a numerical value to each outcome in a sample space.

```
X: Ω → ℝ
```

Think of it as a measurement or quantity that depends on a random process.

### Examples

| Experiment | Sample Space | Random Variable X |
|------------|--------------|-------------------|
| Coin flip | {H, T} | X(H) = 1, X(T) = 0 |
| Die roll | {1,2,3,4,5,6} | X = face value |
| Customer visit | {purchase, browse, leave} | X = dollars spent |
| Email classification | {spam, not spam} | X = 1 if spam, 0 otherwise |

### Notation

- Capital letters (X, Y, Z) denote random variables
- Lowercase letters (x, y, z) denote specific values
- P(X = x) means "probability that X takes value x"

---

## 2. Discrete Random Variables

A **discrete random variable** can take on a finite or countably infinite set of values.

### Probability Mass Function (PMF)

The PMF gives the probability that X equals each possible value:

```
p(x) = P(X = x)
```

**Properties:**
1. p(x) ≥ 0 for all x
2. Σ p(x) = 1 (sum over all possible x)

### Example: Die Roll

```python
import numpy as np

# PMF for fair die
X_values = [1, 2, 3, 4, 5, 6]
pmf = {x: 1/6 for x in X_values}

# Verify it's a valid PMF
assert all(p >= 0 for p in pmf.values())
assert abs(sum(pmf.values()) - 1) < 1e-10
```

### Cumulative Distribution Function (CDF)

The CDF gives the probability that X is less than or equal to some value:

```
F(x) = P(X ≤ x) = Σ p(k) for all k ≤ x
```

**Properties:**
1. F is non-decreasing
2. F(-∞) = 0, F(∞) = 1
3. P(a < X ≤ b) = F(b) - F(a)

---

## 3. Important Discrete Distributions

### Bernoulli Distribution

A single trial with two outcomes (success/failure).

```
X ~ Bernoulli(p)

P(X = 1) = p
P(X = 0) = 1 - p
```

**Use cases:** 
- Coin flip
- Email is spam or not
- Customer converts or not

### Binomial Distribution

Number of successes in n independent Bernoulli trials.

```
X ~ Binomial(n, p)

P(X = k) = C(n,k) × p^k × (1-p)^(n-k)
```

**Parameters:**
- n: number of trials
- p: probability of success per trial

**Use cases:**
- Number of heads in n coin flips
- Number of defective items in a batch
- Number of clicks out of n impressions

```python
from scipy.stats import binom

# 10 coin flips, P(heads) = 0.5
n, p = 10, 0.5
X = binom(n, p)

print(f"P(X = 5) = {X.pmf(5):.4f}")      # 0.2461
print(f"P(X ≤ 3) = {X.cdf(3):.4f}")      # 0.1719
print(f"E[X] = {X.mean():.1f}")           # 5.0
print(f"Var(X) = {X.var():.2f}")          # 2.50
```

### Poisson Distribution

Number of events in a fixed interval when events occur at a constant average rate.

```
X ~ Poisson(λ)

P(X = k) = (λ^k × e^(-λ)) / k!
```

**Parameter:** λ = average number of events per interval

**Use cases:**
- Website visitors per hour
- Typos per page
- Server requests per second
- Rare disease occurrences

```python
from scipy.stats import poisson

# Average 5 emails per hour
λ = 5
X = poisson(λ)

print(f"P(X = 3) = {X.pmf(3):.4f}")      # 0.1404
print(f"P(X ≥ 10) = {1 - X.cdf(9):.4f}") # 0.0318
```

**Key property:** For Poisson, E[X] = Var(X) = λ

### Geometric Distribution

Number of trials until first success.

```
X ~ Geometric(p)

P(X = k) = (1-p)^(k-1) × p
```

**Use case:** 
- Number of attempts until a success
- Time until first failure

---

## 4. Continuous Random Variables

A **continuous random variable** can take any value in an interval (uncountably many values).

### Probability Density Function (PDF)

For continuous variables, we use a PDF f(x):

```
P(a ≤ X ≤ b) = ∫[a to b] f(x) dx
```

**Key insight:** P(X = x) = 0 for any specific value! Instead, we find probabilities over intervals.

**Properties:**
1. f(x) ≥ 0 for all x
2. ∫[-∞ to ∞] f(x) dx = 1

### CDF for Continuous Variables

```
F(x) = P(X ≤ x) = ∫[-∞ to x] f(t) dt
```

And the PDF is the derivative of the CDF:
```
f(x) = dF(x)/dx
```

---

## 5. Important Continuous Distributions

### Uniform Distribution

All values in an interval are equally likely.

```
X ~ Uniform(a, b)

f(x) = 1/(b-a)  for a ≤ x ≤ b
       0        otherwise
```

**Use cases:**
- Random number generators
- Modeling complete ignorance
- Simulation

```python
from scipy.stats import uniform

X = uniform(loc=0, scale=10)  # Uniform(0, 10)
print(f"P(2 < X < 5) = {X.cdf(5) - X.cdf(2):.2f}")  # 0.30
```

### Exponential Distribution

Time between events in a Poisson process.

```
X ~ Exponential(λ)

f(x) = λ × e^(-λx)  for x ≥ 0
```

**Key property:** Memoryless—P(X > s + t | X > s) = P(X > t)

**Use cases:**
- Time between customer arrivals
- Lifetime of electronic components
- Radioactive decay

```python
from scipy.stats import expon

# Average time between events = 2 minutes
λ = 0.5  # rate
X = expon(scale=1/λ)  # scale = 1/rate in scipy

print(f"P(X < 3) = {X.cdf(3):.4f}")      # 0.7769
print(f"Median = {X.median():.2f}")       # 1.39
```

### Normal (Gaussian) Distribution

The most important distribution in statistics.

```
X ~ Normal(μ, σ²)

f(x) = (1/√(2πσ²)) × exp(-(x-μ)²/(2σ²))
```

**Parameters:**
- μ: mean (center)
- σ²: variance (spread)
- σ: standard deviation

**Why so important?**
1. Central Limit Theorem (Module 6)
2. Many natural phenomena are approximately normal
3. Errors in measurements often normal
4. Maximum entropy distribution for given mean and variance

```python
from scipy.stats import norm

X = norm(loc=100, scale=15)  # IQ scores: μ=100, σ=15

print(f"P(X < 85) = {X.cdf(85):.4f}")           # 0.1587
print(f"P(85 < X < 115) = {X.cdf(115) - X.cdf(85):.4f}")  # 0.6827
print(f"95th percentile = {X.ppf(0.95):.1f}")   # 124.7
```

### The 68-95-99.7 Rule

For normal distributions:
- 68% of values within 1σ of mean
- 95% of values within 2σ of mean
- 99.7% of values within 3σ of mean

### Standard Normal Distribution

When μ = 0 and σ = 1:
```
Z ~ Normal(0, 1)
```

Any normal can be standardized:
```
Z = (X - μ) / σ
```

---

## 6. Visualizing Distributions

### PMF Visualization

```python
import matplotlib.pyplot as plt
from scipy.stats import binom

n, p = 20, 0.3
X = binom(n, p)
x_values = range(n + 1)

plt.bar(x_values, X.pmf(x_values))
plt.xlabel('Number of Successes')
plt.ylabel('Probability')
plt.title(f'Binomial(n={n}, p={p})')
plt.show()
```

### PDF Visualization

```python
import numpy as np
from scipy.stats import norm

x = np.linspace(-4, 4, 1000)
plt.plot(x, norm.pdf(x))
plt.fill_between(x, norm.pdf(x), alpha=0.3)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Standard Normal Distribution')
plt.show()
```

---

## 7. Quantiles and Percentiles

### Definitions

**p-th quantile (or percentile):** The value x such that P(X ≤ x) = p

```python
from scipy.stats import norm

X = norm(0, 1)

# Common quantiles
print(f"Median (50th percentile) = {X.ppf(0.5):.3f}")   # 0.000
print(f"25th percentile = {X.ppf(0.25):.3f}")           # -0.674
print(f"75th percentile = {X.ppf(0.75):.3f}")           # 0.674
```

### Important Quantiles

- **Median:** 50th percentile
- **Quartiles:** 25th, 50th, 75th percentiles
- **Interquartile Range (IQR):** Q3 - Q1

---

## 8. Summary: Choosing the Right Distribution

| Scenario | Distribution |
|----------|-------------|
| Yes/No outcome | Bernoulli |
| Count successes in n trials | Binomial |
| Count events in fixed time/space | Poisson |
| Trials until first success | Geometric |
| Equal likelihood over interval | Uniform |
| Time between Poisson events | Exponential |
| Sum of many small effects | Normal |

---

## Key Takeaways

1. **Random variables map outcomes to numbers**—essential for mathematical analysis

2. **Discrete variables have PMFs** (probability mass functions); **continuous have PDFs** (probability density functions)

3. **The CDF works for both** and gives P(X ≤ x)

4. **Binomial for counting successes**, Poisson for counting rare events

5. **Exponential for waiting times**, Normal for sums and natural phenomena

6. **Know the parameters:** Each distribution is characterized by specific parameters (p, n, λ, μ, σ)

---

## Connections to Future Modules

- **Module 3:** Expected value and variance give us summary statistics
- **Module 5:** Conditional distributions lead to Bayes' theorem
- **Module 6:** Central Limit Theorem explains why Normal is everywhere
- **Module 9:** Maximum likelihood finds the best-fitting distribution
- **Module 10:** Bayesian inference treats parameters as random variables

---

## Practice Problems

1. A website has a 2% conversion rate. Out of 500 visitors, what's the probability that at least 15 convert? (Use Binomial)

2. A call center receives an average of 3 calls per minute. What's the probability of receiving more than 5 calls in a given minute? (Use Poisson)

3. If customer wait times are exponentially distributed with mean 4 minutes, what's the probability a customer waits more than 6 minutes?

4. IQ scores are normally distributed with μ=100 and σ=15. What percentage of the population has an IQ above 130?

5. Generate 1000 samples from a Binomial(100, 0.5) distribution and plot the histogram. What shape do you notice?

---

## Further Reading

- Ross, S. *A First Course in Probability* - Chapters 3-4
- DeGroot & Schervish, *Probability and Statistics* - Chapters 3-5
- SciPy documentation: scipy.stats module
