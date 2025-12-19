# Bayesian Coin

## Overview

A Bayesian inference framework for modelling and updating beliefs about coin fairness. Demonstrates hierarchical Bayesian models with conjugate priors and mixture distributions.

## Mathematical Framework

### Core Idea: Composable Priors

The framework implements a **composable prior hierarchy** where any prior can contain sub-priors, which can themselves be mixtures. This recursive structure is not limited to the binomial/coin-flip case—it generalises to any likelihood function with compatible priors.

```
CompoundPrior (mixture)
├── DeltaPrior (p=0)          # Always tails
├── DeltaPrior (p=1)          # Always heads
├── DeltaPrior (p=0.5)        # Fair coin
├── BetaPrior (α, β)          # Unknown bias
└── CompoundPrior (nested)    # Sub-mixtures allowed!
    ├── ...
    └── ...
```

**Impossible priors are acceptable:** When a hypothesis becomes impossible given the evidence (e.g., DeltaPrior p=0 after observing heads), it is not an error. The mixture update simply sets its weight to zero ($\ln w = -\infty$), effectively excluding it from further inference while preserving the model structure.

---

## Formulas

### 1. Beta Prior (Conjugate to Bernoulli)

The Beta distribution is the conjugate prior for Bernoulli/Binomial likelihoods, enabling closed-form updates.

**Prior:** $p \sim \text{Beta}(\alpha_0, \beta_0)$

**Conjugate Update (after observing k heads, m tails):**
$$\alpha' = \alpha_0 + k, \quad \beta' = \beta_0 + m$$

**Mean:**
$$\mathbb{E}[p] = \frac{\alpha}{\alpha + \beta}$$

**Variance:**
$$\text{Var}(p) = \frac{\alpha \beta}{(\alpha + \beta)^2 (\alpha + \beta + 1)}$$

**Log-Normalisation (Evidence/Marginal Likelihood):**
$$\ln P(D | \text{Beta prior}) = \ln B(\alpha', \beta') - \ln B(\alpha_0, \beta_0)$$

where $B(\alpha, \beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}$ is the Beta function.

**Common Priors:**
| Name | Parameters | Interpretation |
|------|------------|----------------|
| Jeffreys | α=0.5, β=0.5 | Non-informative (default) |
| Uniform | α=1, β=1 | All biases equally likely |
| Haldane | α→0, β→0 | Improper, data-dominated |

---

### 2. Delta Prior (Point Mass / Improper)

Models fixed hypotheses where the coin bias is known exactly. These correspond to improper priors (point masses) but are valid within a mixture framework.

**Prior:** $p = p_0$ with probability 1

**Key insight:** Unlike Beta priors, Delta priors **do not update their parameters**. The value $p_0$ represents a firm belief that the probability IS exactly that value, regardless of evidence. What changes is only the **likelihood** (lognorm), which affects the prior's weight in a mixture.

**Log-Likelihood (after observing k heads, m tails):**
$$\ln P(D | p_0) = k \ln(p_0) + m \ln(1 - p_0)$$

**Impossible Hypotheses:**

| Hypothesis | p₀ | Becomes impossible when... | Resulting weight |
|------------|-----|----------------------------|------------------|
| Always tails | 0 | Any heads observed | $w \to 0$ ($\ln w = -\infty$) |
| Always heads | 1 | Any tails observed | $w \to 0$ ($\ln w = -\infty$) |
| Fair coin | 0.5 | Never | Remains valid |

**Why this is valid:** In a mixture, impossible hypotheses simply receive zero posterior weight. They don't break the inference—they get ruled out by the evidence, which is exactly what Bayesian updating should do. The mixture renormalises over the remaining valid hypotheses.

---

### 3. Compound Prior (Mixture Model)

A weighted mixture of sub-priors implementing Bayesian model averaging.

**Prior:**
$$P(p) = \sum_i w_i P_i(p)$$

where $w_i$ are mixture weights ($\sum_i w_i = 1$) and $P_i$ are sub-priors.

#### Weight Update (Bayes' Rule for Model Selection)

After observing data D:
$$w_i' = \frac{P(D | M_i) \cdot w_i^{(0)}}{\sum_j P(D | M_j) \cdot w_j^{(0)}}$$

In log-space (for numerical stability):
$$\ln w_i' = \ln P(D | M_i) + \ln w_i^{(0)} - \text{logsumexp}_j(\ln P(D | M_j) + \ln w_j^{(0)})$$

where $\ln P(D | M_i)$ is the `lognorm` from each sub-prior.

**Note:** If $P(D | M_i) = 0$ (impossible hypothesis), then $\ln P(D | M_i) = -\infty$, and after normalisation $w_i' = 0$. The hypothesis is excluded from the mixture automatically.

#### How Different Prior Types Update

| Prior Type | Parameters | On Evidence |
|------------|------------|-------------|
| **BetaPrior** | α, β | Parameters update: α += heads, β += tails. Distribution sharpens around observed frequency. |
| **DeltaPrior** | p₀ | **Parameters fixed.** p₀ never changes—it represents certainty. Only lognorm (likelihood) changes, affecting mixture weight. |
| **CompoundPrior** | weights, sub-priors | Recursively updates sub-priors, then recomputes weights from their lognorms. |

This distinction is crucial: Beta priors learn from evidence (posterior concentrates), while Delta priors maintain their fixed hypothesis and only get reweighted relative to other hypotheses in the mixture.

**The essence of Bayesian model comparison:** The Delta prior hypothesis itself doesn't "learn"—it either gets supported or ruled out by evidence. Beta priors adapt *within* a model; Delta priors compete *between* fixed models.

#### Mixture Mean
$$\mathbb{E}[p] = \sum_i w_i \mu_i$$

where $\mu_i = \mathbb{E}[p | M_i]$ is the mean of sub-prior $i$ (sum only over hypotheses with $w_i > 0$).

#### Mixture Variance (Law of Total Variance)
$$\text{Var}(p) = \underbrace{\sum_i w_i \sigma_i^2}_{\text{expected variance}} + \underbrace{\sum_i w_i \mu_i^2 - \left(\sum_i w_i \mu_i\right)^2}_{\text{variance of means}}$$

This decomposes into:
- **E[Var(p|M)]**: Average within-model uncertainty
- **Var(E[p|M])**: Between-model uncertainty

#### Compound Log-Normalisation (Marginal Likelihood)
$$\ln P(D | \text{mixture}) = \text{logsumexp}_i(\ln P(D | M_i) + \ln w_i^{(0)})$$

---

### 4. Sampling

**From Beta:**
$$p \sim \text{Beta}(\alpha, \beta)$$

**From Mixture (two-stage):**
1. Select component $i$ with probability $w_i$ (from valid hypotheses only)
2. Sample $p \sim P_i$

---

### 5. Hierarchical Composability

**Key Property:** Sub-priors can themselves be CompoundPriors, enabling arbitrary hierarchical structures.

Example: "One-sided" hypothesis as a sub-mixture:
```
Coin (CompoundPrior)
├── One-sided (CompoundPrior, w=0.001)
│   ├── Always tails (DeltaPrior p=0, w=0.5)
│   └── Always heads (DeltaPrior p=1, w=0.5)
├── Fair (DeltaPrior p=0.5, w=0.998)
└── Unknown (BetaPrior α=0.5, β=0.5, w=0.001)
```

When evidence arrives:
1. Each leaf prior updates its parameters and computes its `lognorm`
2. Parent mixtures recursively recompute weights using children's `lognorm` values
3. The root's `lognorm` gives the total marginal likelihood
4. Impossible leaf priors get weight 0; if all leaves in a sub-mixture become impossible, that sub-mixture also gets weight 0

**Generality:** This framework extends beyond coin flips:
- Replace BetaPrior with any conjugate prior (Normal-Normal, Gamma-Poisson, Dirichlet-Multinomial, etc.)
- Replace DeltaPrior with any fixed-parameter hypothesis
- The CompoundPrior mechanics (weight updates, mixture mean/variance) remain identical
- The recursive structure works for any depth of nesting

---

## Implementation Notes

- **Log-space arithmetic** throughout for numerical stability
- **Log-sum-exp trick** prevents overflow/underflow in weight normalisation
- **Impossible hypotheses** (`is_zero=True`) receive $\ln w = -\infty$; excluded from sums but retained in structure
- **Reweight operation** allows "checkpointing" the posterior as a new prior

## Structure

```
bayesian-coin/
├── app.py                 # Interactive Dash visualisation
├── src/
│   ├── bayesian_coin.py   # Prior, BetaPrior, DeltaPrior, CompoundPrior
│   └── utils.py           # mk_coin() factory
└── requirements.txt       # Dependencies
```

## Key Classes

| Class | Description |
|-------|-------------|
| `Prior` | Abstract base class defining the interface |
| `BetaPrior` | Conjugate prior for unknown bias |
| `DeltaPrior` | Point mass for fixed hypotheses (p=0, 0.5, 1) |
| `CompoundPrior` | Mixture model with recursive nesting support |
