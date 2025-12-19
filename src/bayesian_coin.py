
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import math
import seaborn as sns

from scipy.special import beta as beta_fn
from scipy.special import betaln
from scipy.stats import beta as beta_dist

import numpy as np


class Prior(ABC):
    """
    Abstract base class for prior distributions in Bayesian coin inference.

    Defines the interface for all prior types (Beta, Delta, Compound).
    Each prior tracks its parameters and a log-normalization factor (lognorm)
    used for computing posterior probabilities after observing evidence.
    """

    @abstractmethod
    def sample(self, size=1):
        """Draw random samples from the prior distribution."""
        pass

    @abstractmethod
    def mean(self):
        """Return the mean of the distribution."""
        pass

    @abstractmethod
    def stats(self, std=True):
        """
        Return distribution statistics.

        Args:
            std: If True, return (mean, std). If False, return (mean, variance).
        """
        pass

    @abstractmethod
    def update(self, evidence):
        """
        Update the prior with observed evidence (coin flips).

        Args:
            evidence: Array of observations (0 for tails, 1 for heads).
        """
        pass

    def reweight(self):
        """
        Reset the prior to current posterior and return accumulated log-normalization.

        This 'realizes' the current weights by setting the current parameters
        as the new starting point and resetting lognorm to 0.

        Returns:
            The accumulated lognorm before reset.
        """
        lognorm = self.lognorm
        self._params_start = {**self._params}
        self.lognorm = 0.0
        return lognorm

class BetaPrior(Prior):
    """
    Beta distribution prior for modeling unknown coin bias.

    Uses the Beta distribution as a conjugate prior to the Bernoulli likelihood,
    enabling closed-form Bayesian updates without MCMC sampling.

    Attributes:
        _params: Dict with 'alpha' and 'beta' shape parameters.
        lognorm: Log-normalization factor tracking evidence likelihood.
        is_zero: Always False for Beta priors (never impossible).
    """

    def __init__(self, params=None):
        """
        Initialize a Beta prior.

        Args:
            params: Optional dict with 'alpha' and 'beta' keys.
                    Defaults to Jeffreys prior (alpha=0.5, beta=0.5).
        """
        self._params = {
            'alpha': 0.5,
            'beta': 0.5,
        }

        if params is not None:
            self._params.update(params)
        self._params_start = {**self._params}

        self.is_zero = False
        self.lognorm = 0.0

    def sample(self, size=None):
        """Draw samples from the Beta distribution."""
        return beta_dist.rvs(self._params['alpha'], self._params['beta'], size=size)

    def mean(self):
        """Return the mean: alpha / (alpha + beta)."""
        mean = self._params['alpha'] / (self._params['alpha'] + self._params['beta'])
        return mean

    def stats(self, std=True):
        """Return (mean, std) or (mean, variance) of the Beta distribution."""
        mean = self.mean()
        var = mean * self._params['beta'] / (1.0 + self._params['alpha'] + self._params['beta'])
        var /= (self._params['alpha'] + self._params['beta'])

        if std:
            return mean, np.sqrt(var)
        else:
            return mean, var

    def _get_lognorm(self):
        """
        Compute log-normalization factor for evidence likelihood.

        Returns the log of the integral of (evidence * normalized_prior),
        which equals ln(B(alpha', beta') / B(alpha_0, beta_0)).
        """
        ln_diff = betaln(self._params['alpha'], self._params['beta'])
        ln_diff -= betaln(self._params_start['alpha'], self._params_start['beta'])
        return ln_diff

    def update(self, evidence):
        """
        Update Beta parameters with observed coin flips.

        Uses conjugate update: alpha += heads, beta += tails.

        Args:
            evidence: Array of observations (0 for tails, 1 for heads).
        """
        ones = sum(evidence == 1)
        zeros = sum(evidence == 0)
        n = ones + zeros
        assert n == evidence.shape[0]

        self._params['alpha'] += ones
        self._params['beta'] += zeros

        self.lognorm = self._get_lognorm()


class DeltaPrior(Prior):
    """
    Delta (point mass) prior for fixed coin bias hypotheses.

    Models deterministic coins with known bias (e.g., always heads p=1,
    always tails p=0, or fair p=0.5). The prior has zero variance.

    Attributes:
        _params: Dict with 'p0' (fixed probability), 'zeros', 'ones' counts.
        lognorm: Log-likelihood of observed data given this fixed p.
        is_zero: True if observed data is impossible under this prior.
    """

    def __init__(self, params=None):
        """
        Initialize a Delta prior.

        Args:
            params: Optional dict with 'p0' key (default 0.5).
        """
        self._params = {
            'p0': 0.5,
            'zeros': 0,
            'ones': 0
        }

        if params is not None:
            self._params.update(params)
        self._params_start = {**self._params}

        self.is_zero = False
        self._check_is_zero()
        self.lognorm = 0.0

    def sample(self, size=1):
        """Return the fixed probability p0 (deterministic)."""
        return np.array([self._params['p0']] * size)

    def mean(self):
        """Return the fixed probability p0."""
        return self._params['p0']

    def stats(self, std=True):
        """Return (p0, 0) since delta distributions have zero variance."""
        return self._params['p0'], 0.0

    def _check_is_zero(self):
        """Check if observed evidence makes this hypothesis impossible."""
        p0 = self._params['p0']
        if self._params['ones'] and p0 == 0.0:
            self.is_zero = True
        elif self._params['zeros'] and p0 == 1.0:
            self.is_zero = True

    def _get_lognorm(self):
        """
        Compute log-likelihood of evidence under fixed p0.

        Returns -inf if evidence is impossible (e.g., heads when p=0).
        """
        self._check_is_zero()

        if self.is_zero:
            return -np.inf

        diff_zeros = self._params['zeros'] - self._params_start['zeros']
        diff_ones = self._params['ones'] - self._params_start['ones']

        p0 = self._params['p0']
        if p0 != 0.0:
            ln_diff = diff_ones * np.log(p0)
        else:
            ln_diff = 0.0

        if p0 != 1.0:
            ln_diff += diff_zeros * np.log(1 - p0)

        return ln_diff

    def update(self, evidence):
        """
        Update observation counts with new evidence.

        Args:
            evidence: Array of observations (0 for tails, 1 for heads).
        """
        ones = sum(evidence == 1)
        zeros = sum(evidence == 0)
        n = ones + zeros
        assert n == evidence.shape[0]

        self._params['ones'] += ones
        self._params['zeros'] += zeros

        self.lognorm = self._get_lognorm()

# Vectorized helpers for operating on arrays of Prior objects
is_zero = np.vectorize(lambda t: t.is_zero)
sample = np.vectorize(lambda t: t.sample())


class CompoundPrior(Prior):
    """
    Mixture model combining multiple prior hypotheses with log-weights.

    Implements Bayesian model averaging over competing hypotheses about
    coin bias. Updates mixture weights based on evidence likelihood under
    each sub-prior.

    Attributes:
        _params: Dict with 'logweights' (log prior probs), 'priors' (sub-priors),
                 and 'is_zero' (mask for impossible hypotheses).
        lognorm: Log marginal likelihood of evidence under the mixture.
        is_zero: True if all sub-priors are impossible.
    """

    def __init__(self, params=None):
        """
        Initialize a compound prior mixture model.

        Args:
            params: Dict with 'logweights' (np.ndarray of log prior probs)
                    and 'priors' (np.ndarray of Prior objects).

        Raises:
            ValueError: If params is None.
        """
        if params is None:
            raise ValueError('CompoundPrior must have logweights and priors')
        assert isinstance(params['logweights'], np.ndarray)
        assert isinstance(params['priors'], np.ndarray)

        self._params = {
            'logweights': params['logweights'],
            'priors': params['priors'],
        }
        self._params['is_zero'] = is_zero(self._params['priors'])

        self._params_start = {**self._params}

        self._update_weights()  # normalise
        self._params_start['logweights'] = self._params['logweights']  # copy across
        self._update_weights()  # lognorm should be zero now
        assert self.lognorm == 0.0

        self.is_zero = all(self._params['is_zero'])

    def sample(self, size=1):
        """Sample from the mixture by first selecting a prior, then sampling from it."""
        probs = np.exp(self._params['logweights'][~self._params['is_zero']])
        priors = np.random.choice(self._params['priors'][~self._params['is_zero']], size, p=probs)
        return sample(priors)

    def mean(self):
        """Return the mixture mean (weighted average of sub-prior means)."""
        probs = np.exp(self._params['logweights'][~self._params['is_zero']])
        mean = sum([
            prior.mean() * prob for prior, prob in zip(
                self._params['priors'][~self._params['is_zero']], probs
            )]
        )
        return mean

    def stats(self, std=True):
        """
        Return mixture statistics using the law of total variance.

        Args:
            std: If True, return (mean, std). If False, return (mean, variance).
        """
        probs = np.exp(self._params['logweights'][~self._params['is_zero']])

        stats = [prior.stats(std=False) for prior in self._params['priors'][~self._params['is_zero']]]
        means, vars = zip(*stats)

        # Law of total variance: Var(X) = E[Var(X|Y)] + Var(E[X|Y])
        vars_sum = sum(var * prob for var, prob in zip(vars, probs))
        means_sum = sum(mean * prob for mean, prob in zip(means, probs))
        means_sq_sum = sum(mean * mean * prob for mean, prob in zip(means, probs))

        var = vars_sum + (means_sq_sum - means_sum * means_sum)

        if std:
            return means_sum, np.sqrt(var)
        else:
            return means_sum, var

    def _update_weights(self):
        """
        Recompute mixture weights based on sub-prior log-normalizations.

        Updates logweights to reflect posterior model probabilities and
        computes the mixture lognorm (log marginal likelihood).
        """
        lognorms = np.array([prior.lognorm for prior in self._params['priors'][~self._params['is_zero']]])
        logweights = self._params_start['logweights'][~self._params['is_zero']] + lognorms

        logweights_max = max(logweights)

        logweights -= logweights_max
        logsum = np.log(np.exp(logweights).sum())

        # Normalise (copy to avoid modifying _params_start reference)
        self._params['logweights'] = self._params['logweights'].copy()
        self._params['logweights'][~self._params['is_zero']] = logweights - logsum
        self._params['logweights'][self._params['is_zero']] = -np.inf

        self.lognorm = logsum + logweights_max

    def update(self, evidence):
        """
        Update all sub-priors with evidence and recompute mixture weights.

        Args:
            evidence: Array of observations (0 for tails, 1 for heads).
        """
        # Update sub-priors
        for prior in self._params['priors'][~self._params['is_zero']]:
            prior.update(evidence)

        # Check for newly impossible priors
        self._params['is_zero'] = is_zero(self._params['priors'])
        self.is_zero = all(self._params['is_zero'])

        if not self.is_zero:
            self._update_weights()

    def reweight(self):
        """Recursively reset all sub-priors and this compound prior."""
        for prior in self._params['priors']:
            _ = prior.reweight()

        super().reweight()
