# Module 5: Conditional Probability and Bayes' Theorem

## Overview

Conditional probability answers the question: "Given that event B occurred, what's the probability of event A?" Bayes' theorem inverts this question, allowing us to update our beliefs when we observe new evidence. This is the foundation of Bayesian reasoning, spam filters, medical diagnosis, and modern machine learning.

This module is arguably the most important in the course—mastering Bayes' theorem will change how you think about uncertainty.

---

## 1. Conditional Probability

### Definition

The probability of A given that B has occurred:

```
P(A | B) = P(A ∩ B) / P(B),  provided P(B) > 0
```

Read: "Probability of A given B"

### Intuition

We're restricting our attention to a smaller sample space—just the outcomes where B occurred—and asking what fraction of those have A.

### Example: Cards

What's the probability a card is a King given it's a face card?

```
P(King | Face card) = P(King ∩ Face) / P(Face)
                    = (4/52) / (12/52)
                    = 4/12 = 1/3
```

### The Conditioning Trap

**Common error:** Confusing P(A | B) with P(B | A)

Example:
- P(Wet Grass | Rain) ≈ 0.99 (rain almost always wets grass)
- P(Rain | Wet Grass) ≠ 0.99 (sprinklers also wet grass!)

---

## 2. The Multiplication Rule

Rearranging the conditional probability formula:

```
P(A ∩ B) = P(A | B) × P(B) = P(B | A) × P(A)
```

### Chain Rule (Multiple Events)

```
P(A ∩ B ∩ C) = P(A) × P(B | A) × P(C | A ∩ B)
```

Extends to any number of events.

**Example:** Drawing without replacement

P(first two cards are hearts) = P(♥₁) × P(♥₂ | ♥₁) = (13/52) × (12/51)

---

## 3. Law of Total Probability

If B₁, B₂, ..., Bₙ partition the sample space (mutually exclusive and exhaustive):

```
P(A) = Σᵢ P(A | Bᵢ) × P(Bᵢ)
```

### Intuition

To find P(A), consider all the different ways A could happen (through each Bᵢ) and add them up.

### Example: Medical Testing

A disease affects 1% of the population. A test has:
- 95% sensitivity: P(Positive | Disease) = 0.95
- 90% specificity: P(Negative | No Disease) = 0.90

What's P(Positive)?

```
P(Positive) = P(Positive | Disease) × P(Disease) 
            + P(Positive | No Disease) × P(No Disease)
            = 0.95 × 0.01 + 0.10 × 0.99
            = 0.0095 + 0.099
            = 0.1085
```

About 10.85% of tests are positive!

---

## 4. Bayes' Theorem

### The Formula

```
P(A | B) = P(B | A) × P(A) / P(B)
```

Or with the law of total probability:

```
P(A | B) = P(B | A) × P(A) / [P(B | A) × P(A) + P(B | A') × P(A')]
```

### The Components

- **P(A):** Prior probability (before seeing evidence)
- **P(B | A):** Likelihood (probability of evidence given hypothesis)
- **P(B):** Marginal likelihood (normalizing constant)
- **P(A | B):** Posterior probability (after seeing evidence)

### The Bayesian Mantra

```
Posterior ∝ Likelihood × Prior
```

---

## 5. Medical Diagnosis Example

Using the disease/test numbers from before:
- P(Disease) = 0.01 (prior)
- P(Positive | Disease) = 0.95 (sensitivity)
- P(Positive | No Disease) = 0.10 (false positive rate)

**Question:** If you test positive, what's the probability you have the disease?

```
P(Disease | Positive) = P(Positive | Disease) × P(Disease) / P(Positive)
                      = 0.95 × 0.01 / 0.1085
                      = 0.0095 / 0.1085
                      ≈ 0.088
```

**Only 8.8%!** Despite a 95% accurate test, most positives are false positives.

### Why So Low?

The **base rate** matters enormously. With a rare disease:
- True positives: 0.95 × 1% = 0.95% of population
- False positives: 0.10 × 99% = 9.9% of population
- Most positive tests are false positives!

```python
def bayes_disease(prior, sensitivity, specificity):
    """Calculate P(Disease | Positive)"""
    false_positive_rate = 1 - specificity
    p_positive = sensitivity * prior + false_positive_rate * (1 - prior)
    posterior = (sensitivity * prior) / p_positive
    return posterior

# Try different priors
for prior in [0.001, 0.01, 0.10, 0.50]:
    posterior = bayes_disease(prior, 0.95, 0.90)
    print(f"Prior: {prior:.1%} → Posterior: {posterior:.1%}")
```

Output:
```
Prior: 0.1% → Posterior: 0.9%
Prior: 1.0% → Posterior: 8.8%
Prior: 10.0% → Posterior: 51.4%
Prior: 50.0% → Posterior: 90.5%
```

---

## 6. Bayesian Updating

Bayes' theorem allows sequential updates as new evidence arrives.

### Process

1. Start with prior P(H)
2. Observe evidence E
3. Update: P(H | E) becomes new prior
4. Repeat with new evidence

### Example: Loaded Coin

A coin is either fair (p=0.5) or loaded (p=0.8). Prior: 50% each.

We flip it and get Heads. Update beliefs:

```
P(Loaded | H) = P(H | Loaded) × P(Loaded) / P(H)
              = 0.8 × 0.5 / [0.8 × 0.5 + 0.5 × 0.5]
              = 0.4 / 0.65
              ≈ 0.615
```

Now flip again, get Heads:
```
P(Loaded | HH) = 0.8 × 0.615 / [0.8 × 0.615 + 0.5 × 0.385]
               ≈ 0.72
```

```python
def update_belief(prior_loaded, observation):
    """Update P(Loaded) after observing heads (1) or tails (0)"""
    p_h_loaded = 0.8
    p_h_fair = 0.5
    
    if observation == 1:  # Heads
        likelihood_loaded = p_h_loaded
        likelihood_fair = p_h_fair
    else:  # Tails
        likelihood_loaded = 1 - p_h_loaded
        likelihood_fair = 1 - p_h_fair
    
    p_evidence = likelihood_loaded * prior_loaded + likelihood_fair * (1 - prior_loaded)
    posterior = (likelihood_loaded * prior_loaded) / p_evidence
    return posterior

# Observe: H, H, T, H, H
belief = 0.5
observations = [1, 1, 0, 1, 1]
for obs in observations:
    belief = update_belief(belief, obs)
    print(f"After {'H' if obs else 'T'}: P(Loaded) = {belief:.3f}")
```

---

## 7. Naive Bayes Classifier

### The Setup

Classify emails as spam or not-spam based on words.

### The Formula

```
P(Spam | words) ∝ P(words | Spam) × P(Spam)
```

### The "Naive" Assumption

Words are conditionally independent given the class:

```
P(w₁, w₂, ..., wₙ | Spam) = P(w₁ | Spam) × P(w₂ | Spam) × ... × P(wₙ | Spam)
```

This is often false but works surprisingly well!

### Implementation

```python
from collections import defaultdict
import numpy as np

class NaiveBayesSpam:
    def __init__(self):
        self.word_probs = {'spam': defaultdict(float), 'ham': defaultdict(float)}
        self.class_probs = {'spam': 0.5, 'ham': 0.5}
    
    def train(self, emails, labels):
        """Learn word probabilities from labeled emails"""
        spam_emails = [e for e, l in zip(emails, labels) if l == 'spam']
        ham_emails = [e for e, l in zip(emails, labels) if l == 'ham']
        
        self.class_probs['spam'] = len(spam_emails) / len(emails)
        self.class_probs['ham'] = len(ham_emails) / len(emails)
        
        # Count word frequencies (with Laplace smoothing)
        for cls, email_list in [('spam', spam_emails), ('ham', ham_emails)]:
            all_words = ' '.join(email_list).split()
            word_counts = defaultdict(int)
            for word in all_words:
                word_counts[word] += 1
            total = len(all_words)
            vocab_size = len(set(all_words))
            for word, count in word_counts.items():
                self.word_probs[cls][word] = (count + 1) / (total + vocab_size)
    
    def predict(self, email):
        """Predict spam probability for an email"""
        words = email.split()
        log_prob_spam = np.log(self.class_probs['spam'])
        log_prob_ham = np.log(self.class_probs['ham'])
        
        for word in words:
            log_prob_spam += np.log(self.word_probs['spam'].get(word, 1e-6))
            log_prob_ham += np.log(self.word_probs['ham'].get(word, 1e-6))
        
        # Convert back from log space
        prob_spam = np.exp(log_prob_spam)
        prob_ham = np.exp(log_prob_ham)
        return prob_spam / (prob_spam + prob_ham)
```

---

## 8. Common Bayes' Theorem Pitfalls

### 1. Base Rate Neglect

Ignoring the prior probability (like in the medical test example).

### 2. Prosecutor's Fallacy

Confusing P(Evidence | Innocent) with P(Innocent | Evidence).

Example: "The probability of this DNA match by chance is 1 in a million, so there's only a 1 in a million chance the defendant is innocent."

**Wrong!** If there are 8 million people in the city, about 8 would match by chance.

### 3. Confirmation Bias

Only updating on evidence that confirms your prior, ignoring disconfirming evidence.

---

## Key Takeaways

1. **Conditional probability:** P(A|B) = P(A∩B) / P(B)

2. **Bayes' theorem:** P(A|B) = P(B|A) × P(A) / P(B)

3. **Posterior ∝ Likelihood × Prior**

4. **Base rates matter:** A positive test for a rare disease is probably a false positive

5. **Sequential updating:** Posterior becomes the new prior when new evidence arrives

6. **Naive Bayes:** Despite "naive" conditional independence assumption, works well in practice

---

## Connections to Future Modules

- **Module 10:** Full Bayesian inference for parameter estimation
- **Module 12:** Graphical models encode conditional independence
- **Module 13:** MCMC samples from complex posterior distributions
- *Course on ML:* Naive Bayes, Bayesian neural networks

---

## Practice Problems

1. In a city, 5% of cars are Teslas. Teslas run red lights 1% of the time; other cars do so 5% of the time. If a car ran a red light, what's the probability it's a Tesla?

2. You have two coins: Fair (P(H) = 0.5) and biased (P(H) = 0.7). You pick one at random and flip it 5 times, getting HHTHT. What's P(Biased Coin)?

3. Implement a Naive Bayes classifier for sentiment analysis (positive/negative reviews).

4. A patient tests positive on two independent tests with 90% sensitivity and 95% specificity each. If the disease prevalence is 2%, what's P(Disease | Both Positive)?

5. Prove that P(A | B, C) = P(B | A, C) × P(A | C) / P(B | C).

---

## Further Reading

- McGrayne, S. *The Theory That Would Not Die* (History of Bayes)
- Kruschke, J. *Doing Bayesian Data Analysis*
- 3Blue1Brown, "Bayes theorem" (YouTube)
- Jaynes, E.T. *Probability Theory: The Logic of Science*
