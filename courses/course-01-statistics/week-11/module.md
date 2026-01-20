# Module 11: Markov Chains and Stochastic Processes

## Overview

Many real-world systems evolve over time with random dynamics: stock prices, customer behavior, weather, and web browsing patterns. Markov chains provide a mathematical framework for modeling such sequential random processes. This module covers the theory of Markov chains, their long-term behavior, and applications in AI and machine learning.

---

## 1. What is a Stochastic Process?

### Definition

A **stochastic process** is a collection of random variables indexed by time or space:

```
{X_t : t ∈ T}
```

Where T is the index set (often {0, 1, 2, ...} or [0, ∞)).

### Examples

| Process | State Space | Index |
|---------|-------------|-------|
| Stock price | ℝ⁺ | Continuous time |
| Number in queue | {0, 1, 2, ...} | Continuous time |
| Weather | {Sunny, Rainy, Cloudy} | Days |
| Random walk | ℤ | Discrete steps |

---

## 2. The Markov Property

### Definition

A process has the **Markov property** if:

```
P(X_{t+1} | X_t, X_{t-1}, ..., X_0) = P(X_{t+1} | X_t)
```

**In words:** The future depends only on the present, not the past.

"Given the present, the future is independent of the past."

### Memorylessness

The Markov property is also called memorylessness:
- Where you go next depends only on where you are now
- How you got here doesn't matter

---

## 3. Discrete-Time Markov Chains

### Definition

A **discrete-time Markov chain (DTMC)** is a sequence of random variables X₀, X₁, X₂, ... with:
1. Finite or countable state space S
2. Markov property
3. Time-homogeneous transitions: P(X_{t+1} = j | X_t = i) = p_{ij} (same for all t)

### Transition Matrix

```
P = [p_{ij}]  where p_{ij} = P(X_{t+1} = j | X_t = i)
```

Properties:
- Each row sums to 1 (probability distribution)
- All entries ≥ 0

### Example: Weather

States: S = {Sunny, Rainy}

| From \ To | Sunny | Rainy |
|-----------|-------|-------|
| Sunny | 0.8 | 0.2 |
| Rainy | 0.4 | 0.6 |

```python
import numpy as np

# Transition matrix
P = np.array([
    [0.8, 0.2],  # From Sunny
    [0.4, 0.6]   # From Rainy
])

# Verify it's a valid transition matrix
assert np.allclose(P.sum(axis=1), 1)
print("Transition matrix P:")
print(P)
```

---

## 4. Multi-Step Transitions

### n-Step Transition Probabilities

```
p_{ij}^{(n)} = P(X_{t+n} = j | X_t = i)
```

### Chapman-Kolmogorov Equation

```
P^{(n)} = P^n  (matrix power)
```

**The n-step transition matrix is the original matrix raised to the nth power!**

```python
# 2-step transitions
P2 = np.linalg.matrix_power(P, 2)
print("2-step transitions:")
print(P2)

# 10-step transitions
P10 = np.linalg.matrix_power(P, 10)
print("\n10-step transitions:")
print(P10)
```

---

## 5. Stationary Distribution

### Definition

A distribution π is **stationary** if:

```
π = π × P
```

If you start in the stationary distribution, you stay in it forever.

### Finding the Stationary Distribution

Solve: πP = π subject to Σπᵢ = 1

This is an eigenvalue problem: π is a left eigenvector with eigenvalue 1.

```python
def find_stationary(P):
    """Find stationary distribution of a transition matrix"""
    # Solve πP = π, which is π(P - I) = 0
    # Also need π.sum() = 1
    
    n = len(P)
    
    # Set up system: (P^T - I)π = 0 and sum(π) = 1
    A = np.vstack([P.T - np.eye(n), np.ones(n)])
    b = np.zeros(n + 1)
    b[-1] = 1
    
    # Least squares solution
    pi, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return pi

pi = find_stationary(P)
print(f"Stationary distribution: {pi}")
print(f"Verify πP = π: {np.allclose(pi @ P, pi)}")
```

### Interpretation

In the long run, the chain visits each state proportionally to π.

For weather: π ≈ [0.667, 0.333] means 2/3 of days are sunny.

---

## 6. Convergence to Stationary Distribution

### Conditions for Convergence

A Markov chain converges to its stationary distribution if it's:
1. **Irreducible:** Can reach any state from any state
2. **Aperiodic:** Doesn't cycle with a fixed period

### Ergodic Theorem

For irreducible, aperiodic chains:

```
lim_{n→∞} P^n = [π; π; ...; π]  (all rows become π)
```

Every row converges to the stationary distribution!

```python
# Watch convergence
initial = np.array([1, 0])  # Start sunny

print("Distribution evolution:")
dist = initial.copy()
for n in [0, 1, 5, 10, 50]:
    dist = initial @ np.linalg.matrix_power(P, n)
    print(f"n={n:2d}: {dist}")
```

---

## 7. Classification of States

### Recurrence

A state i is **recurrent** if the chain returns to i with probability 1.

A state i is **transient** if there's positive probability of never returning.

### Periodicity

The **period** of state i is the GCD of all n such that p_{ii}^{(n)} > 0.

- Period = 1: **aperiodic**
- Period > 1: **periodic**

### Communicating Classes

States i and j **communicate** if you can get from i to j and from j to i.

The chain is **irreducible** if all states communicate.

---

## 8. Random Walk Example

### Simple Random Walk

At each step, move +1 with probability p or -1 with probability 1-p.

```python
def random_walk(n_steps, p=0.5, start=0):
    """Simulate a simple random walk"""
    position = start
    path = [position]
    
    for _ in range(n_steps):
        step = 1 if np.random.random() < p else -1
        position += step
        path.append(position)
    
    return path

# Simulate multiple walks
plt.figure(figsize=(10, 6))
for _ in range(5):
    path = random_walk(100)
    plt.plot(path, alpha=0.7)
plt.xlabel('Step')
plt.ylabel('Position')
plt.title('Simple Random Walks')
```

### Gambler's Ruin

A gambler starts with $k, wins $1 with probability p, loses $1 with probability 1-p.

Game ends at $0 (ruin) or $N (win).

Probability of ruin:
```
P(ruin) = (q/p)^k - (q/p)^N) / (1 - (q/p)^N)  if p ≠ 0.5
        = 1 - k/N                              if p = 0.5
```

---

## 9. Applications

### PageRank

Google's original ranking algorithm models web browsing as a Markov chain:
- States = web pages
- Transitions = following links
- Stationary distribution = page importance

```python
def pagerank(adjacency, damping=0.85, n_iter=100):
    """Simple PageRank implementation"""
    n = len(adjacency)
    
    # Normalize adjacency to get transition matrix
    out_degree = adjacency.sum(axis=1, keepdims=True)
    out_degree[out_degree == 0] = 1  # Handle dangling nodes
    P = adjacency / out_degree
    
    # Add damping (random jump)
    P = damping * P + (1 - damping) / n * np.ones((n, n))
    
    # Power iteration
    pi = np.ones(n) / n
    for _ in range(n_iter):
        pi = pi @ P
    
    return pi

# Example: 4-page web
adj = np.array([
    [0, 1, 1, 0],
    [0, 0, 1, 1],
    [1, 0, 0, 1],
    [1, 1, 0, 0]
])

ranks = pagerank(adj)
print(f"PageRank scores: {ranks}")
```

### Hidden Markov Models (Preview)

States are hidden; we observe emissions from each state.
- Used in speech recognition, bioinformatics, NLP
- Covered more in later courses

### MCMC Algorithms

Markov chains designed to sample from target distributions.
- States = possible parameter values
- Construct chain whose stationary distribution = target
- Covered in Module 13

---

## 10. Continuous-Time Markov Chains

### Key Differences

- Time is continuous
- Rate matrix Q instead of transition matrix P
- Exponential holding times in each state

### Generator Matrix (Q)

```
Q = [q_{ij}]  where q_{ij} = rate of transition i → j (i ≠ j)
              q_{ii} = -Σ_{j≠i} q_{ij}
```

### Relationship to DTMC

```
P(t) = e^{Qt}  (matrix exponential)
```

---

## Key Takeaways

1. **Markov property:** Future depends only on present, not past

2. **Transition matrix P:** p_{ij} = P(move from i to j)

3. **Multi-step:** P^n gives n-step transition probabilities

4. **Stationary distribution π:** Satisfies πP = π; long-run behavior

5. **Convergence:** Irreducible + aperiodic → converges to π

6. **PageRank** is a Markov chain application

7. **MCMC** uses Markov chains for sampling (Module 13)

---

## Connections to Future Modules

- **Module 12:** Graphical models structure dependencies
- **Module 13:** MCMC uses Markov chains for Bayesian computation
- *Course on ML:* Hidden Markov Models, reinforcement learning

---

## Practice Problems

1. For the weather chain, what's the probability of Sunny-Sunny-Rainy?

2. Find the stationary distribution of:
   ```
   P = [[0.5, 0.5, 0],
        [0.25, 0.5, 0.25],
        [0, 0.5, 0.5]]
   ```

3. Is this chain irreducible?
   ```
   P = [[0.5, 0.5, 0],
        [0.5, 0.5, 0],
        [0, 0, 1]]
   ```

4. Simulate 1000 steps of the weather chain and compare empirical frequencies to π.

5. Implement PageRank for a 6-page web you design.

---

## Further Reading

- Ross, S. *Stochastic Processes*
- Norris, J.R. *Markov Chains*
- Grinstead & Snell, *Introduction to Probability* - Chapter 11
