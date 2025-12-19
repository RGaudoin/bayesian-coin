# Usage Guide

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies:**
- numpy, pandas, scipy
- dash, plotly
- matplotlib, seaborn (for notebooks)

## Running the App

```bash
python app.py
```

Open http://localhost:8050 in your browser.

## App Features

### Tabs

**Simulation** - Interactive visualisation of Bayesian updating

**Theory** - Mathematical formulas and concepts

### Prior Configuration (Left Panel)

Configure the initial mixture weights for each hypothesis:

| Parameter | Description |
|-----------|-------------|
| P(always tails) | Weight for p=0 hypothesis (DeltaPrior) |
| P(always heads) | Weight for p=1 hypothesis (DeltaPrior) |
| P(fair coin) | Weight for p=0.5 hypothesis (DeltaPrior) |
| P(unknown bias) | Weight for Beta prior (continuous) |
| Beta α, β | Shape parameters for the Beta prior |

**Combine one-sided priors**: Groups p=0 and p=1 into a single "one-sided" sub-mixture.

### Simulation Controls

| Control | Description |
|---------|-------------|
| True coin probability | The actual probability used to generate flips |
| Number of steps | How many coin flips to simulate |
| Reset / Initialize | Create a new prior with current settings |
| Run Steps | Simulate N flips and update the posterior |

### Plots (Right Panel)

**Hypothesis Weights Over Time**: Shows how mixture weights evolve as evidence accumulates. Watch impossible hypotheses get ruled out.

**Posterior Statistics**: Mean and standard deviation of the posterior. Confidence band shows uncertainty.

**Observed Flips**: Cumulative mean of simulated flips vs true probability.

### Display Options

| Option | Description |
|--------|-------------|
| Log scale for weights | Show log-weights (useful when weights span many orders of magnitude) |
| Show as probabilities | Convert log-weights to probabilities (sums to 1) |

## Typical Experiments

### 1. Fair Coin Detection

```
P(fair) = 1000000, P(unknown) = 0.001, True prob = 0.5
```
Run 100+ steps. Fair coin hypothesis should dominate.

### 2. Biased Coin Detection

```
P(fair) = 1000000, P(unknown) = 0.001, True prob = 0.7
```
Run 100+ steps. Watch "Unknown (Beta)" take over as evidence accumulates.

### 3. One-Sided Coin

```
P(always heads) = 1, P(fair) = 1000000, True prob = 1.0
```
Run 10+ steps. "Always heads" should quickly dominate (no tails observed).

### 4. Ruling Out Impossible Hypotheses

```
P(always tails) = 1, P(always heads) = 1, P(fair) = 1, True prob = 0.5
```
Run 2-3 steps. Watch one-sided hypotheses get ruled out (weight → 0) after seeing both heads and tails.

### 5. The "Lucky Streak" Story (see [Background](README.md#background))

This experiment visualises why hierarchical priors match intuition:

**Step 1:** Reset with default weights (P(fair) = 10⁶), run 20 steps with True prob = 1.0 (all heads)
- 20 heads is ~1 in a million (2²⁰ ≈ 10⁶)—same order as the fair prior weight!
- Watch the weights start to shift: "Fair" loses ground, "One-sided" gains
- This is the moment of doubt: maybe this isn't a normal coin?

**Step 2:** Continue with 100 steps at True prob = 0.5 (fair coin)
- Seeing tails immediately rules out "Always heads" (weight → 0)
- "Fair" recovers and dominates again
- The hierarchical prior handled both the lucky streak and the return to normal

A pure Beta prior would have been permanently biased by those first 20 heads. The mixture "forgets" the streak once contradicting evidence arrives.

## Programmatic Usage

```python
from src.bayesian_coin import BetaPrior, DeltaPrior, CompoundPrior
from src.utils import mk_coin
import numpy as np

# Create a compound prior
coin = mk_coin(
    p_0=1,           # Weight for p=0
    p_1=1,           # Weight for p=1
    p_fair=1000000,  # Weight for p=0.5
    p_noninf=0.001,  # Weight for Beta prior
    combine=True,    # Combine one-sided priors
    beta_alpha=0.5,  # Jeffreys prior
    beta_beta=0.5
)

# Simulate evidence
flips = np.random.binomial(1, 0.7, size=100)  # Biased coin

# Update with evidence
coin.update(flips)

# Get posterior statistics
mean, std = coin.stats(std=True)
print(f"Posterior mean: {mean:.4f} +/- {std:.4f}")

# Get mixture weights
weights = np.exp(coin._params['logweights'])
print(f"Weights: {weights}")
```
