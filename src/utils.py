

import matplotlib.pyplot as plt
import pandas as pd
import math
import seaborn as sns


from scipy.special import beta as beta_fn
from scipy.special import betaln
from scipy.stats import beta as beta_dist

import numpy as np

from bayesian_coin import DeltaPrior, BetaPrior, CompoundPrior


def mk_coin(p_0, p_1, p_fair, p_noninf, combine=True, beta_alpha=0.5, beta_beta=0.5):
    """
    Factory function to create a compound coin hypothesis model.

    Creates a mixture model over four coin hypotheses:
    - Always tails (p=0)
    - Always heads (p=1)
    - Fair coin (p=0.5)
    - Unknown bias (Beta prior)

    Args:
        p_0: Prior probability for always-tails hypothesis.
        p_1: Prior probability for always-heads hypothesis.
        p_fair: Prior probability for fair coin hypothesis.
        p_noninf: Prior probability for unknown bias (Beta prior).
        combine: If True, group p_0 and p_1 into a nested "one-sided" prior.
        beta_alpha: Alpha parameter for the Beta prior (default 0.5 = Jeffreys).
        beta_beta: Beta parameter for the Beta prior (default 0.5 = Jeffreys).

    Returns:
        CompoundPrior: A mixture model over the coin hypotheses.
    """
    prior_0 = DeltaPrior(params={'p0': 0.0})
    prior_1 = DeltaPrior(params={'p0': 1.0})
    prior_fair = DeltaPrior(params={'p0': 0.5})
    prior_noninf = BetaPrior(params={'alpha': beta_alpha, 'beta': beta_beta})

    if combine:
        priors = np.array([prior_0, prior_1])
        probs = [p_0, p_1]
        logprobs = np.log(probs)

        params_one_sided = {
            'logweights': logprobs,
            'priors': priors
        }

        prior_one_sided = CompoundPrior(params=params_one_sided)

        priors_coin = np.array([prior_one_sided, prior_fair, prior_noninf])
        probs_coin = [p_0 + p_1, p_fair, p_noninf]

    else:
        priors_coin = np.array([prior_0, prior_1, prior_fair, prior_noninf])
        probs_coin = [p_0, p_1, p_fair, p_noninf]

    logprobs_coin = np.log(probs_coin)

    params = {
        'logweights': logprobs_coin,
        'priors': priors_coin,
    }

    prior = CompoundPrior(params=params)

    return prior
