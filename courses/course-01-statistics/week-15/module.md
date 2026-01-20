# Module 15: Capstone Project – Statistical Modeling

## Overview

This final module brings together everything you've learned in a comprehensive statistical modeling project. You'll formulate a problem, collect and explore data, apply probabilistic models, perform inference, and communicate your findings. This is your opportunity to demonstrate mastery of statistical thinking.

---

## 1. Project Goals

### Learning Objectives

By completing this capstone, you will:

1. **Formulate statistical questions** from real-world problems
2. **Apply appropriate probability models** to data
3. **Perform both frequentist and Bayesian inference**
4. **Validate assumptions** and assess model fit
5. **Communicate findings** clearly with visualizations and uncertainty quantification

### Deliverables

1. **Written Report** (10-15 pages)
2. **Code Repository** (well-documented Python/R)
3. **Presentation** (15 minutes + Q&A)

---

## 2. Project Options

### Option A: Bayesian A/B Testing

**Context:** You work for an e-commerce company testing a new checkout flow.

**Data:** Conversion rates for control (A) and treatment (B) groups.

**Tasks:**
- Model conversions as Bernoulli/Binomial
- Apply Beta-Binomial Bayesian analysis
- Compute P(B > A | data)
- Compare to frequentist hypothesis test
- Determine required sample size for 80% power

### Option B: Time Series Modeling

**Context:** Forecast stock returns or energy demand.

**Data:** Historical time series data.

**Tasks:**
- Explore autocorrelation structure
- Fit ARIMA or state-space model
- Use Markov chain models for regime switching
- Evaluate forecast accuracy with proper cross-validation
- Quantify prediction uncertainty

### Option C: Probabilistic Classification

**Context:** Medical diagnosis or spam detection.

**Data:** Features and labels for classification task.

**Tasks:**
- Implement Naive Bayes from scratch
- Compare to logistic regression (MLE)
- Calibrate probabilities
- Analyze feature importance using mutual information
- Evaluate with proper metrics (ROC-AUC, calibration curves)

### Option D: Hierarchical Bayesian Model

**Context:** Comparing performance across groups (schools, hospitals, etc.).

**Data:** Outcomes nested within groups.

**Tasks:**
- Build hierarchical model allowing partial pooling
- Implement using PyMC or Stan
- Compare complete pooling, no pooling, and partial pooling
- Shrinkage analysis
- Posterior predictive checks

### Option E: Propose Your Own

Submit a 1-page proposal including:
- Problem description
- Data source
- Planned statistical methods
- Connections to course material

---

## 3. Project Structure

### Phase 1: Problem Definition (Week 1)

**Deliverable:** 1-2 page proposal

- Clear research question
- Description of data
- Initial hypotheses
- Planned statistical approach
- Success criteria

### Phase 2: Exploratory Analysis (Week 2)

**Deliverable:** Jupyter notebook with EDA

- Data cleaning and preprocessing
- Summary statistics
- Visualizations of distributions
- Initial patterns and hypotheses
- Identification of modeling challenges

### Phase 3: Model Development (Weeks 3-4)

**Deliverable:** Modeling notebook

- Model specification and justification
- Parameter estimation (MLE and/or Bayesian)
- Multiple models for comparison
- Sensitivity analysis

### Phase 4: Validation & Interpretation (Week 5)

**Deliverable:** Analysis notebook

- Model checking (residuals, posterior predictive)
- Cross-validation or held-out evaluation
- Uncertainty quantification
- Interpretation of results

### Phase 5: Final Report & Presentation (Week 6)

**Deliverable:** Report + slides + code

---

## 4. Technical Requirements

### Statistical Methods (use at least 4)

From the course:

| Module | Method |
|--------|--------|
| 2-3 | Probability distributions, expectation, variance |
| 5 | Bayes' theorem, conditional probability |
| 6 | Central Limit Theorem, sampling distributions |
| 7 | Confidence intervals |
| 8 | Hypothesis testing |
| 9 | Maximum likelihood estimation |
| 10 | Bayesian inference |
| 11 | Markov chains (if relevant) |
| 12 | Graphical models (if relevant) |
| 13 | Monte Carlo / MCMC |
| 14 | Information theory (if relevant) |

### Code Requirements

```python
# Required structure
project/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_validation.ipynb
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── models.py
│   └── utils.py
├── tests/
│   └── test_models.py
├── requirements.txt
└── README.md
```

### Visualization Standards

- Clear labels and titles
- Uncertainty bands (confidence/credible intervals)
- Proper scales
- Publication-quality figures

---

## 5. Evaluation Rubric

### Statistical Rigor (40%)

| Criterion | Excellent | Satisfactory | Needs Work |
|-----------|-----------|--------------|------------|
| Model choice | Justified, appropriate | Reasonable | Inappropriate |
| Assumptions | Checked, reasonable | Mentioned | Ignored |
| Inference | Correct, complete | Minor errors | Major errors |
| Uncertainty | Properly quantified | Partially | Missing |

### Technical Implementation (25%)

| Criterion | Excellent | Satisfactory | Needs Work |
|-----------|-----------|--------------|------------|
| Code quality | Clean, documented | Functional | Messy |
| Reproducibility | Fully reproducible | With effort | Not reproducible |
| Testing | Unit tests | Some checks | None |

### Communication (25%)

| Criterion | Excellent | Satisfactory | Needs Work |
|-----------|-----------|--------------|------------|
| Writing | Clear, precise | Understandable | Confusing |
| Visualizations | Publication-quality | Informative | Poor |
| Presentation | Engaging, clear | Adequate | Unclear |

### Creativity & Depth (10%)

| Criterion | Excellent | Satisfactory | Needs Work |
|-----------|-----------|--------------|------------|
| Insights | Novel, deep | Standard | Superficial |
| Extensions | Beyond requirements | Meets requirements | Falls short |

---

## 6. Report Template

### Title Page

- Descriptive title
- Your name
- Date

### Abstract (200 words)

- Problem context
- Methods used
- Key findings

### 1. Introduction (1-2 pages)

- Problem motivation
- Research questions
- Data description
- Preview of results

### 2. Methods (3-4 pages)

- Statistical model specification
- Mathematical formulation
- Estimation approach
- Computational details

### 3. Results (4-5 pages)

- Parameter estimates with uncertainty
- Model comparisons
- Key visualizations
- Sensitivity analyses

### 4. Discussion (2-3 pages)

- Interpretation of findings
- Limitations
- Comparison to prior work
- Future directions

### 5. Conclusion (0.5 page)

- Summary of contributions
- Practical implications

### References

- Properly formatted citations

### Appendix

- Additional figures
- Technical derivations
- Complete code listings

---

## 7. Example: Bayesian A/B Test

### Problem Setup

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Data
control_visitors = 1000
control_conversions = 50
treatment_visitors = 1000
treatment_conversions = 65

# Beta priors (weakly informative)
alpha_prior = 1
beta_prior = 1

# Posterior parameters
alpha_A = alpha_prior + control_conversions
beta_A = beta_prior + control_visitors - control_conversions
alpha_B = alpha_prior + treatment_conversions
beta_B = beta_prior + treatment_visitors - treatment_conversions

print(f"Posterior A: Beta({alpha_A}, {beta_A})")
print(f"Posterior B: Beta({alpha_B}, {beta_B})")
```

### Visualization

```python
x = np.linspace(0, 0.15, 1000)
plt.figure(figsize=(10, 6))
plt.plot(x, stats.beta.pdf(x, alpha_A, beta_A), label='Control (A)')
plt.plot(x, stats.beta.pdf(x, alpha_B, beta_B), label='Treatment (B)')
plt.xlabel('Conversion Rate')
plt.ylabel('Density')
plt.legend()
plt.title('Posterior Distributions')
```

### Key Computation

```python
# P(B > A | data) via Monte Carlo
n_samples = 100000
samples_A = np.random.beta(alpha_A, beta_A, n_samples)
samples_B = np.random.beta(alpha_B, beta_B, n_samples)

prob_B_better = np.mean(samples_B > samples_A)
print(f"P(B > A | data) = {prob_B_better:.3f}")

# Expected lift
lift_samples = (samples_B - samples_A) / samples_A
print(f"Expected lift: {np.mean(lift_samples):.1%} ({np.percentile(lift_samples, 2.5):.1%}, {np.percentile(lift_samples, 97.5):.1%})")
```

---

## 8. Tips for Success

### Do's

✓ Start early and iterate
✓ Seek feedback on your proposal
✓ Document as you go
✓ Test your code
✓ Show uncertainty everywhere
✓ Connect methods to course material
✓ Interpret results in context

### Don'ts

✗ Wait until the last week
✗ Ignore failed approaches (document them!)
✗ Report point estimates without uncertainty
✗ Use methods you don't understand
✗ Copy code without attribution
✗ Overfit to your data

---

## 9. Resources

### Datasets

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/)
- [Kaggle](https://www.kaggle.com/datasets)
- [data.gov](https://data.gov/)
- [Google Dataset Search](https://datasetsearch.research.google.com/)

### Tools

- **Python:** NumPy, SciPy, pandas, matplotlib, seaborn
- **Bayesian:** PyMC, Stan, ArviZ
- **Documentation:** Jupyter, Quarto

### Writing Help

- Strunk & White, *Elements of Style*
- Tufte, *The Visual Display of Quantitative Information*

---

## 10. Submission Checklist

- [ ] Code runs without errors
- [ ] All dependencies in requirements.txt
- [ ] README with setup instructions
- [ ] Report is 10-15 pages
- [ ] All figures have captions
- [ ] Uncertainty is quantified
- [ ] At least 4 statistical methods used
- [ ] Assumptions checked
- [ ] Results interpreted
- [ ] Slides prepared for presentation

---

## Key Takeaways

This capstone is your opportunity to:

1. **Integrate** concepts from the entire course
2. **Practice** end-to-end statistical modeling
3. **Develop** skills you'll use in your AI career
4. **Demonstrate** your statistical thinking

Good luck! 🎓
