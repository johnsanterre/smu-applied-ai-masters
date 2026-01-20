# Module 12: Probabilistic Graphical Models

## Overview

Probabilistic Graphical Models (PGMs) combine probability theory with graph theory to represent complex relationships among random variables. They provide a visual language for modeling dependencies and efficient algorithms for inference. This module covers Bayesian networks and Markov random fields—foundational tools used throughout machine learning.

---

## 1. Why Graphical Models?

### The Challenge

Real-world systems involve many related variables. With n binary variables, a joint distribution requires 2ⁿ - 1 parameters—exponentially many.

### The Solution

Exploit structure! Most variables are only directly influenced by a few others.

**Example:** Weather affects whether I carry an umbrella, but not my coffee preference.

### Two Representations

1. **Directed graphs (Bayesian Networks):** Represent causal/generative relationships
2. **Undirected graphs (Markov Random Fields):** Represent symmetric associations

---

## 2. Bayesian Networks (Directed Models)

### Definition

A Bayesian network is:
1. A directed acyclic graph (DAG)
2. A conditional probability distribution for each node given its parents

### Factorization

The joint distribution factorizes:

```
P(X₁, X₂, ..., Xₙ) = ∏ᵢ P(Xᵢ | Parents(Xᵢ))
```

### Example: Student Network

```
         Difficulty ──┐
              │       │
              ▼       ▼
Intelligence ──► Grade ──► Letter
              │
              ▼
            SAT
```

Variables:
- D: Course difficulty
- I: Student intelligence
- G: Grade received
- S: SAT score
- L: Recommendation letter

Factorization:
```
P(D, I, G, S, L) = P(D) × P(I) × P(G | D, I) × P(S | I) × P(L | G)
```

```python
import numpy as np

# Define conditional probability tables (CPTs)

# P(D): Prior on difficulty
P_D = {'easy': 0.6, 'hard': 0.4}

# P(I): Prior on intelligence
P_I = {'low': 0.3, 'high': 0.7}

# P(G | D, I): Grade given difficulty and intelligence
P_G_given_DI = {
    ('easy', 'low'): {'A': 0.3, 'B': 0.4, 'C': 0.3},
    ('easy', 'high'): {'A': 0.9, 'B': 0.08, 'C': 0.02},
    ('hard', 'low'): {'A': 0.05, 'B': 0.25, 'C': 0.7},
    ('hard', 'high'): {'A': 0.5, 'B': 0.3, 'C': 0.2}
}

# Joint probability of specific assignment
def joint_prob(d, i, g, s, l):
    p = P_D[d] * P_I[i]
    p *= P_G_given_DI[(d, i)][g]
    # ... (complete for S and L)
    return p
```

---

## 3. Conditional Independence in Bayesian Networks

### D-Separation

Two node sets A and B are conditionally independent given C if:
- Every path from A to B is "blocked" by C

### Three Basic Patterns

**1. Chain:** A → B → C
- A and C independent given B
- A ⫫ C | B

**2. Fork:** A ← B → C
- A and C independent given B
- A ⫫ C | B

**3. V-Structure (Collider):** A → B ← C
- A and C independent marginally
- A and C become dependent given B!
- A ⫫ C but NOT A ⫫ C | B

### Explaining Away

In a v-structure, observing the child creates dependence between parents.

**Example:** Rain → Wet Grass ← Sprinkler

If grass is wet and we learn it rained, sprinkler becomes less likely (explained away).

---

## 4. Markov Random Fields (Undirected Models)

### Definition

A Markov Random Field (MRF) is:
1. An undirected graph
2. Potential functions over cliques

### Factorization

```
P(X₁, ..., Xₙ) = (1/Z) ∏_{c ∈ cliques} ψ_c(X_c)
```

Where Z is the normalizing constant (partition function).

### Markov Properties

A node is conditionally independent of all others given its neighbors (Markov blanket).

### Example: Ising Model

Binary variables on a grid. Neighboring nodes tend to have the same value.

```
ψ(xᵢ, xⱼ) = exp(θ × xᵢ × xⱼ)
```

Higher θ → stronger preference for neighbors to agree.

---

## 5. Inference in Graphical Models

### Types of Inference

1. **Marginalization:** P(X₁ | evidence)
2. **MAP inference:** argmax P(X | evidence)
3. **Partition function:** Z = ΣₓP̃(x)

### Exact Inference

**Variable Elimination:** Sum out variables one at a time

```python
def variable_elimination(factors, query, evidence, elimination_order):
    """
    Simplified variable elimination
    factors: list of factor tables
    query: variable to compute marginal for
    evidence: observed values
    elimination_order: order to eliminate variables
    """
    # Apply evidence
    restricted_factors = apply_evidence(factors, evidence)
    
    # Eliminate variables
    for var in elimination_order:
        if var == query:
            continue
        # Multiply factors containing var
        relevant = [f for f in restricted_factors if var in f.scope]
        product = multiply_factors(relevant)
        # Sum out var
        marginalized = sum_out(product, var)
        # Update factor list
        restricted_factors = [f for f in restricted_factors if var not in f.scope]
        restricted_factors.append(marginalized)
    
    # Multiply remaining factors
    result = multiply_factors(restricted_factors)
    return normalize(result)
```

### Approximate Inference

For large models, exact inference is intractable:

1. **Sampling methods:** MCMC, importance sampling
2. **Variational inference:** Approximate posterior with simpler distribution
3. **Loopy belief propagation:** Message passing (not guaranteed to converge)

---

## 6. Learning in Graphical Models

### Parameter Learning

Given structure and data, estimate CPT entries.

**Complete data:** Use MLE
```
P̂(xᵢ | parents) = count(xᵢ, parents) / count(parents)
```

**Missing data:** Use EM algorithm

### Structure Learning

Given data, learn the graph structure.

**Score-based:** Search over structures, evaluate with AIC/BIC
**Constraint-based:** Test conditional independencies

---

## 7. Common Models

### Naive Bayes (Revisited)

```
      Y
    / | \
   X₁ X₂ X₃
```

All features conditionally independent given class.

### Hidden Markov Model

```
Z₁ → Z₂ → Z₃ → ...  (hidden states)
↓    ↓    ↓
X₁   X₂   X₃        (observations)
```

### Latent Dirichlet Allocation (LDA)

```
Topics → Document → Words
```

Used for topic modeling in NLP.

---

## 8. Implementation with pgmpy

```python
# pip install pgmpy
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Define structure
model = BayesianNetwork([
    ('D', 'G'),
    ('I', 'G'),
    ('I', 'S'),
    ('G', 'L')
])

# Define CPDs
cpd_d = TabularCPD('D', 2, [[0.6], [0.4]])
cpd_i = TabularCPD('I', 2, [[0.3], [0.7]])

cpd_g = TabularCPD('G', 3,
    [[0.3, 0.05, 0.9, 0.5],   # P(G=0 | D, I)
     [0.4, 0.25, 0.08, 0.3],  # P(G=1 | D, I)
     [0.3, 0.7, 0.02, 0.2]],  # P(G=2 | D, I)
    evidence=['D', 'I'],
    evidence_card=[2, 2])

cpd_s = TabularCPD('S', 2,
    [[0.95, 0.2],
     [0.05, 0.8]],
    evidence=['I'],
    evidence_card=[2])

cpd_l = TabularCPD('L', 2,
    [[0.1, 0.4, 0.99],
     [0.9, 0.6, 0.01]],
    evidence=['G'],
    evidence_card=[3])

model.add_cpds(cpd_d, cpd_i, cpd_g, cpd_s, cpd_l)
model.check_model()

# Inference
infer = VariableElimination(model)

# P(I | G=0, S=1)  - Intelligence given good grade and high SAT
result = infer.query(['I'], evidence={'G': 0, 'S': 1})
print(result)
```

---

## 9. Factor Graphs

### Unified Representation

Factor graphs represent both directed and undirected models:
- Variable nodes (circles)
- Factor nodes (squares)
- Edges connect variables to factors they participate in

### Message Passing

Belief propagation passes messages between nodes:
```
μ_{f→x}(x) = Σ_{y} f(x, y) × μ_{y→f}(y)
μ_{x→f}(x) = ∏_{g ≠ f} μ_{g→x}(x)
```

---

## 10. Applications

| Domain | Application |
|--------|-------------|
| Medicine | Diagnosis networks, drug interactions |
| Vision | Image segmentation, object recognition |
| NLP | Parsing, named entity recognition |
| Genomics | Gene regulatory networks |
| Finance | Risk modeling, fraud detection |

---

## Key Takeaways

1. **PGMs represent joint distributions** using graph structure

2. **Bayesian Networks:** Directed, causal, P(X) = ∏ P(Xᵢ | Parents)

3. **Markov Random Fields:** Undirected, potential functions, P(X) ∝ ∏ ψ(clique)

4. **D-separation** determines conditional independence in directed graphs

5. **Variable elimination** is exact but exponential in treewidth

6. **MCMC and variational** methods approximate inference for large models

7. **Structure learning** discovers graph from data

---

## Connections to Future Modules

- **Module 13:** MCMC for approximate inference in PGMs
- *Course on ML:* HMMs, CRFs, deep generative models

---

## Practice Problems

1. Draw the Bayesian network for: A affects B and C; B and C both affect D.

2. Write the factorization for the network in problem 1.

3. Are A and C d-separated given D in problem 1?

4. Given the student network, compute P(D = hard | L = good).

5. Compare the number of parameters in a full joint distribution vs. a Bayesian network for 5 binary variables with sparse structure.

---

## Further Reading

- Koller & Friedman, *Probabilistic Graphical Models*
- Murphy, K. *Machine Learning: A Probabilistic Perspective* - Chapters 10, 19
- Bishop, *Pattern Recognition and ML* - Chapter 8
