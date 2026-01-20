# Module 1: Foundations of Probability

## Overview

Probability is the mathematical language of uncertainty. In AI and machine learning, nearly everything involves uncertainty—sensor readings are noisy, predictions are imperfect, and data is incomplete. This module establishes the foundational concepts of probability theory that underpin all statistical learning methods.

By the end of this module, you will understand sample spaces, events, probability axioms, and basic rules for combining probabilities. These concepts form the grammar of uncertainty that we'll use throughout this course and beyond.

---

## 1. Why Probability Matters for AI

### Uncertainty is Everywhere

Consider these AI scenarios:
- **Medical diagnosis:** A patient has symptoms that could indicate multiple diseases
- **Spam detection:** An email has features consistent with both spam and legitimate mail
- **Self-driving cars:** Sensor readings are noisy; is that shape a pedestrian or a mailbox?
- **Recommendation systems:** Will this user like this movie?

In each case, we can't be certain. Probability gives us a principled framework to:
- Quantify our uncertainty
- Update beliefs as we gather evidence
- Make optimal decisions despite uncertainty

### The Two Interpretations

**Frequentist view:** Probability is the long-run frequency of events. If we flip a fair coin infinitely many times, heads will appear 50% of the time.

**Bayesian view:** Probability represents degrees of belief. When we say there's a 70% chance of rain, we're expressing our confidence level, not predicting infinite weather trials.

Both views are useful. Frequentist methods dominate classical statistics; Bayesian methods are increasingly important in modern ML.

---

## 2. Sample Spaces and Events

### Sample Space (Ω)

The **sample space** is the set of all possible outcomes of an experiment.

**Examples:**
- Coin flip: Ω = {Heads, Tails}
- Die roll: Ω = {1, 2, 3, 4, 5, 6}
- Temperature tomorrow: Ω = ℝ (all real numbers)
- Customer behavior: Ω = {purchase, browse, leave}

Sample spaces can be:
- **Finite:** Ω = {1, 2, 3, 4, 5, 6}
- **Countably infinite:** Ω = {0, 1, 2, 3, ...} (number of website visits)
- **Uncountably infinite:** Ω = [0, ∞) (time until next event)

### Events

An **event** is a subset of the sample space—a collection of outcomes we're interested in.

**Examples for a die roll:**
- Event A = "roll an even number" = {2, 4, 6}
- Event B = "roll greater than 4" = {5, 6}
- Event C = "roll a 3" = {3}

### Special Events

- **Empty event (∅):** Contains no outcomes—a logical impossibility
- **Sure event (Ω):** Contains all outcomes—a certainty
- **Elementary event:** Contains exactly one outcome

---

## 3. Probability Axioms (Kolmogorov's Axioms)

All of probability theory is built from three simple axioms.

### Axiom 1: Non-negativity

For any event A:

```
P(A) ≥ 0
```

Probabilities are never negative.

### Axiom 2: Normalization

```
P(Ω) = 1
```

Something must happen; the probability of the entire sample space is 1.

### Axiom 3: Additivity

For mutually exclusive events A₁, A₂, A₃, ... (events that cannot occur together):

```
P(A₁ ∪ A₂ ∪ A₃ ∪ ...) = P(A₁) + P(A₂) + P(A₃) + ...
```

If events can't overlap, their probabilities add.

### Immediate Consequences

From these three axioms, we can derive everything else:

**Complement Rule:**
```
P(A') = 1 - P(A)
```
where A' (or Ā) is "not A"

**Probability bounds:**
```
0 ≤ P(A) ≤ 1
```

**Empty set:**
```
P(∅) = 0
```

---

## 4. Counting Methods for Discrete Probability

When outcomes are equally likely:

```
P(A) = |A| / |Ω| = (number of favorable outcomes) / (total number of outcomes)
```

This requires us to count outcomes systematically.

### The Multiplication Principle

If task 1 can be done in n₁ ways, and for each choice, task 2 can be done in n₂ ways, then both tasks can be done in n₁ × n₂ ways.

**Example:** A password has 2 letters followed by 3 digits.
- 26 × 26 × 10 × 10 × 10 = 676,000 possible passwords

### Permutations

**Permutation:** An ordered arrangement of objects.

Number of ways to arrange n distinct objects:
```
n! = n × (n-1) × (n-2) × ... × 2 × 1
```

Number of ways to arrange r objects from n distinct objects:
```
P(n,r) = n! / (n-r)!
```

**Example:** How many ways can 3 people finish a race with 8 runners?
```
P(8,3) = 8! / 5! = 8 × 7 × 6 = 336
```

### Combinations

**Combination:** An unordered selection of objects (order doesn't matter).

```
C(n,r) = (n choose r) = n! / [r!(n-r)!]
```

**Example:** How many ways can you choose 3 people from a group of 8 for a committee?
```
C(8,3) = 8! / (3! × 5!) = 56
```

### When to Use Which

- **Permutation:** Order matters (rankings, arrangements, passwords)
- **Combination:** Order doesn't matter (committees, hands of cards, samples)

---

## 5. Basic Probability Rules

### Addition Rule (General Form)

For any two events A and B:

```
P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
```

We subtract the intersection to avoid double-counting.

**Visual intuition:** Draw a Venn diagram. If we add the two circles, we've counted the overlap twice.

### Special Case: Mutually Exclusive Events

If A and B cannot both occur (A ∩ B = ∅):

```
P(A ∪ B) = P(A) + P(B)
```

**Example:** P(roll 2 OR roll 5) = 1/6 + 1/6 = 2/6

### Inclusion-Exclusion (Three Events)

```
P(A ∪ B ∪ C) = P(A) + P(B) + P(C) 
               - P(A ∩ B) - P(A ∩ C) - P(B ∩ C) 
               + P(A ∩ B ∩ C)
```

The pattern continues for more events: add singles, subtract pairs, add triples, etc.

---

## 6. Probability as a Python Dictionary

For finite sample spaces, we can represent probability distributions as dictionaries:

```python
# Die roll probabilities
die = {1: 1/6, 2: 1/6, 3: 1/6, 4: 1/6, 5: 1/6, 6: 1/6}

# Verify axioms
assert all(p >= 0 for p in die.values())  # Non-negativity
assert abs(sum(die.values()) - 1) < 1e-10  # Normalization

# P(even number)
P_even = sum(die[k] for k in [2, 4, 6])
print(f"P(even) = {P_even}")  # 0.5
```

For events:

```python
def P(event, distribution):
    """Calculate probability of an event given a distribution."""
    return sum(distribution[outcome] for outcome in event 
               if outcome in distribution)

# Events
even = {2, 4, 6}
greater_than_4 = {5, 6}

print(P(even, die))  # 0.5
print(P(even | greater_than_4, die))  # P(A ∪ B) = 0.667
print(P(even & greater_than_4, die))  # P(A ∩ B) = 0.167
```

---

## 7. Common Probability Distributions (Preview)

We'll explore these in depth in Module 2, but here's a preview:

### Discrete Distributions

**Uniform:** All outcomes equally likely
```
P(X = k) = 1/n  for k = 1, 2, ..., n
```

**Bernoulli:** Single yes/no trial
```
P(X = 1) = p,  P(X = 0) = 1-p
```

**Binomial:** Number of successes in n independent trials
```
P(X = k) = C(n,k) × p^k × (1-p)^(n-k)
```

### Continuous Distributions (teaser)

When the sample space is continuous (like time or temperature), we'll need probability density functions (PDFs) instead of probability mass functions (PMFs).

---

## 8. Worked Examples

### Example 1: The Birthday Problem

What's the probability that in a group of n people, at least two share a birthday?

**Strategy:** Calculate P(no shared birthdays) and subtract from 1.

```
P(no match with n people) = 365/365 × 364/365 × 363/365 × ... × (365-n+1)/365

P(at least one match) = 1 - P(no match)
```

**Results:**
- n = 23: P ≈ 0.507 (more likely than not!)
- n = 50: P ≈ 0.970
- n = 70: P ≈ 0.999

```python
def birthday_probability(n):
    p_no_match = 1.0
    for i in range(n):
        p_no_match *= (365 - i) / 365
    return 1 - p_no_match
```

### Example 2: Poker Hand Probabilities

What's the probability of being dealt a flush (5 cards of the same suit)?

**Count favorable outcomes:**
- Choose 1 of 4 suits: C(4,1) = 4
- Choose 5 of 13 cards in that suit: C(13,5) = 1,287
- Total flushes: 4 × 1,287 = 5,148

**Count total outcomes:**
- C(52,5) = 2,598,960

**Probability:**
```
P(flush) = 5,148 / 2,598,960 ≈ 0.00198 ≈ 0.2%
```

---

## Key Takeaways

1. **Probability quantifies uncertainty**—essential for AI systems that must act under incomplete information

2. **Three axioms define all of probability:** non-negativity, normalization, and additivity

3. **Sample spaces can be finite, countably infinite, or continuous**—the math differs for each

4. **Counting is crucial:** Permutations for ordered arrangements, combinations for unordered selections

5. **Addition rule:** P(A ∪ B) = P(A) + P(B) - P(A ∩ B)—don't double-count the overlap

6. **Complement rule:** P(not A) = 1 - P(A)—often easier to count what we don't want

---

## Connections to Future Modules

- **Module 2:** Random variables give us a numerical handle on events
- **Module 4:** Joint distributions extend probability to multiple variables
- **Module 5:** Conditional probability and Bayes' theorem let us update beliefs
- **Module 10:** Bayesian inference applies these foundations to parameter estimation
- **Module 12:** Probabilistic graphical models encode complex probability relationships

---

## Practice Problems

1. A bag contains 5 red balls and 3 blue balls. If you draw 2 balls without replacement, what's the probability that both are red?

2. How many 4-digit PINs are possible if no digit can repeat?

3. In a class of 30 students, what's the probability that at least two share a birthday?

4. A fair die is rolled twice. What's the probability that the sum is 7?

5. Prove from the axioms that P(A') = 1 - P(A).

---

## Further Reading

- Ross, S. *A First Course in Probability* - Chapters 1-2
- Blitzstein & Hwang, *Introduction to Probability* - Chapters 1-2
- 3Blue1Brown, "Visualizing Bayes' theorem" (YouTube)
