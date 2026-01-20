# Module 14: Information Theory Basics

## Overview

Information theory provides a mathematical foundation for understanding information, uncertainty, and learning. Originally developed for communication systems, it now pervades machine learning—from decision trees to neural network training to generative models. This module covers entropy, mutual information, KL divergence, and their applications in AI.

---

## 1. What is Information?

### Intuition

Information is the resolution of uncertainty. A surprising event carries more information than an expected one.

**Example:**
- "The sun rose today" → low information (expected)
- "It snowed in Miami" → high information (surprising)

### Shannon's Key Insight

Information should be:
1. **Additive:** Independent events add information
2. **Non-negative:** Information can't be negative
3. **Related to probability:** Rare events = more information

### Self-Information

For an event with probability p:

```
I(p) = -log₂(p) bits
```

or equivalently:

```
I(p) = -ln(p) nats
```

**Examples:**
- Fair coin heads (p=0.5): I = -log₂(0.5) = 1 bit
- Die showing 6 (p=1/6): I = -log₂(1/6) ≈ 2.58 bits
- Certain event (p=1): I = -log₂(1) = 0 bits

---

## 2. Entropy

### Definition

**Entropy** is the expected information content—the average surprise:

```
H(X) = -Σ p(x) log p(x) = E[-log p(X)]
```

Units: bits (log₂) or nats (ln)

### Interpretation

Entropy measures:
- Average uncertainty in a random variable
- Minimum bits needed to encode outcomes
- How "spread out" a distribution is

### Examples

```python
import numpy as np

def entropy(probs, base=2):
    """Calculate entropy of a probability distribution"""
    probs = np.array(probs)
    probs = probs[probs > 0]  # Remove zeros
    return -np.sum(probs * np.log(probs)) / np.log(base)

# Fair coin: maximum uncertainty for binary
print(f"Fair coin: H = {entropy([0.5, 0.5]):.3f} bits")  # 1.0

# Biased coin: less uncertainty
print(f"p=0.9 coin: H = {entropy([0.9, 0.1]):.3f} bits")  # 0.469

# Fair die
print(f"Fair die: H = {entropy([1/6]*6):.3f} bits")  # 2.585
```

### Properties

1. **H(X) ≥ 0** (always non-negative)
2. **H(X) = 0** iff X is deterministic
3. **Maximum entropy:** For n outcomes, H ≤ log(n), achieved when uniform

---

## 3. Joint and Conditional Entropy

### Joint Entropy

For two random variables:

```
H(X, Y) = -Σₓ Σᵧ p(x, y) log p(x, y)
```

### Conditional Entropy

```
H(Y | X) = -Σₓ Σᵧ p(x, y) log p(y | x)
         = Σₓ p(x) H(Y | X = x)
```

**Interpretation:** Average uncertainty in Y after observing X.

### Chain Rule

```
H(X, Y) = H(X) + H(Y | X) = H(Y) + H(X | Y)
```

### Relationship

```
H(Y | X) ≤ H(Y)
```

Conditioning never increases entropy (on average)!

Equality holds iff X and Y are independent.

---

## 4. Mutual Information

### Definition

**Mutual information** measures the information shared between X and Y:

```
I(X; Y) = H(X) - H(X | Y) = H(Y) - H(Y | X)
```

Equivalently:

```
I(X; Y) = Σₓ Σᵧ p(x, y) log [p(x, y) / (p(x) × p(y))]
```

### Interpretation

- How much knowing X tells us about Y (and vice versa)
- Reduction in uncertainty about X from observing Y
- Zero iff X and Y are independent

### Properties

1. **I(X; Y) ≥ 0**
2. **I(X; Y) = I(Y; X)** (symmetric)
3. **I(X; X) = H(X)** (information about self = entropy)
4. **I(X; Y) = 0 ⟺ X ⊥ Y** (independence)

### Venn Diagram View

```
     ┌─────────────┐
     │   H(X|Y)    │
  ┌──┼─────┐       │
  │  │I(X;Y)│      │
  │  └─────┼──┐    │
  │        │  H(Y|X)
  └────────┴──┴────┘
      H(X)    H(Y)
```

---

## 5. Kullback-Leibler Divergence

### Definition

**KL divergence** measures how one distribution differs from another:

```
D_KL(P || Q) = Σ p(x) log [p(x) / q(x)] = E_P[log(P/Q)]
```

### Interpretation

- Extra bits needed to encode samples from P using code optimized for Q
- "Distance" from Q to P (but not a true distance!)

### Properties

1. **D_KL(P || Q) ≥ 0** (Gibbs' inequality)
2. **D_KL(P || Q) = 0 ⟺ P = Q**
3. **NOT symmetric:** D_KL(P || Q) ≠ D_KL(Q || P)
4. **NOT a distance:** Doesn't satisfy triangle inequality

```python
def kl_divergence(p, q, eps=1e-10):
    """KL divergence D_KL(P || Q)"""
    p = np.array(p) + eps
    q = np.array(q) + eps
    return np.sum(p * np.log(p / q))

# Two distributions
p = [0.5, 0.3, 0.2]
q = [0.4, 0.4, 0.2]

print(f"D_KL(P || Q) = {kl_divergence(p, q):.4f}")
print(f"D_KL(Q || P) = {kl_divergence(q, p):.4f}")  # Different!
```

---

## 6. Cross-Entropy

### Definition

```
H(P, Q) = -Σ p(x) log q(x) = H(P) + D_KL(P || Q)
```

### Connection to ML

Cross-entropy loss for classification:

```
L = -Σᵢ yᵢ log(ŷᵢ)
```

Where y is the true distribution (one-hot) and ŷ are predicted probabilities.

**Minimizing cross-entropy = minimizing KL divergence from true distribution**

```python
def cross_entropy_loss(y_true, y_pred, eps=1e-15):
    """Binary cross-entropy loss"""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + 
                    (1 - y_true) * np.log(1 - y_pred))

# Example predictions
y_true = np.array([1, 0, 1, 1, 0])
y_pred_good = np.array([0.9, 0.1, 0.8, 0.95, 0.2])
y_pred_bad = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

print(f"Good predictions: Loss = {cross_entropy_loss(y_true, y_pred_good):.4f}")
print(f"Bad predictions: Loss = {cross_entropy_loss(y_true, y_pred_bad):.4f}")
```

---

## 7. Information Gain

### Definition

In decision trees, **information gain** is the mutual information between a feature and the target:

```
IG(Y; X) = H(Y) - H(Y | X)
```

### Algorithm

For each feature X:
1. Compute H(Y) before split
2. Compute H(Y | X) after split (weighted average of child entropies)
3. Choose feature with highest IG

```python
def information_gain(y, x):
    """Information gain of splitting by x"""
    H_y = entropy(np.bincount(y) / len(y))
    
    # Conditional entropy
    H_y_given_x = 0
    for val in np.unique(x):
        mask = (x == val)
        weight = mask.sum() / len(x)
        if mask.sum() > 0:
            H_y_given_x += weight * entropy(np.bincount(y[mask]) / mask.sum())
    
    return H_y - H_y_given_x

# Example: Should I play tennis?
y = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0])  # Play
outlook = np.array([0, 0, 0, 1, 2, 2, 2, 0, 2, 1, 0, 1, 2, 1])  # Sunny, Overcast, Rain

print(f"Information Gain (Outlook) = {information_gain(y, outlook):.4f}")
```

---

## 8. Maximum Entropy Principle

### The Principle

Among all distributions consistent with known constraints, choose the one with **maximum entropy**.

**Rationale:** Maximum entropy is the most uniform, least assuming distribution.

### Example: Fair Die

Constraint: E[X] = 3.5

Maximum entropy distribution? The uniform distribution!

### Exponential Family

Maximum entropy with moment constraints gives exponential family distributions:
- Mean constraint → Normal
- Mean and non-negativity → Exponential
- Mean of log → Geometric

---

## 9. Connections to Machine Learning

### Neural Network Training

**Cross-entropy loss** is the standard for classification:
- Minimizes KL divergence between true labels and predictions
- Derived from maximum likelihood under categorical distribution

### Variational Autoencoders (VAEs)

ELBO (Evidence Lower Bound):
```
log p(x) ≥ E_q[log p(x|z)] - D_KL(q(z|x) || p(z))
```

### Information Bottleneck

Trade-off between:
- Compressing input X into representation T
- Preserving information about target Y

```
min I(X; T) - β × I(T; Y)
```

### Mutual Information Neural Estimation (MINE)

Estimate mutual information using neural networks for representation learning.

---

## 10. Data Compression

### Source Coding Theorem

You cannot compress data below its entropy:

```
Average code length ≥ H(X)
```

### Huffman Coding

Achieves approximately H(X) bits per symbol for known distributions.

```python
from collections import Counter
import heapq

def huffman_code(text):
    """Build Huffman code for text"""
    freq = Counter(text)
    
    # Build priority queue
    heap = [[count, [char, ""]] for char, count in freq.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    
    return dict(heap[0][1:])

text = "hello world"
codes = huffman_code(text)
print("Huffman codes:", codes)
```

---

## Key Takeaways

1. **Entropy H(X):** Average uncertainty / bits to encode

2. **Mutual Information I(X; Y):** Shared information between X and Y

3. **KL Divergence D_KL(P || Q):** Extra bits to encode P using Q-optimal code

4. **Cross-Entropy:** H(P) + D_KL(P || Q), used in ML loss functions

5. **Information Gain:** Reduction in entropy, used in decision trees

6. **Maximum Entropy:** Choose least assuming distribution given constraints

7. **Source Coding:** Can't compress below entropy

---

## Connections to Other Modules

- **Module 2:** Entropy characterizes distributions
- **Module 10:** KL divergence in Bayesian variational inference
- *Course on ML:* Cross-entropy loss, information-theoretic regularization

---

## Practice Problems

1. Compute the entropy of [0.25, 0.25, 0.25, 0.25] vs [0.1, 0.2, 0.3, 0.4].

2. Show that I(X; Y) = H(X) + H(Y) - H(X, Y).

3. Calculate information gain for the XOR problem.

4. Prove that KL divergence is non-negative (Gibbs' inequality).

5. Implement Huffman coding for a given text and compute average bits per character.

---

## Further Reading

- Cover & Thomas, *Elements of Information Theory*
- MacKay, *Information Theory, Inference, and Learning Algorithms*
- Shannon, "A Mathematical Theory of Communication" (1948)
