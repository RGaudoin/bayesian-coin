# Bayesian Coin

Hierarchical Bayes and everything you never knew you never knew about binomial Bayes.

*(Disclaimer: no rocket science here)*

## Background

Some time ago I had a bit too much time on my hands and wanted to see how Bayesian beliefs update as one flips a coin. Like most, I thought this was trivial—after all, what is difficult about beta priors and conjugate updates?

I thought to myself: let's assume I flip a coin I find on the road 10 times and see heads every time. What does a non-informative prior do? What happens when I flip it 20 times and see heads 20 times?

I watched the prior shift towards heads, and the posterior too—even after just 10 flips. Yet in reality, 10 heads in a row happens roughly 1 in 1000 times. Nobody actually suspects that a randomly found coin has a weird distribution other than 50/50.

So I tried a very informative prior like Beta(10⁶, 10⁶). But even then, as a Bayesian should I believe in an odd, lopsided distribution slightly shifted away from fair? Again, no way I believe that. What went wrong? Binomial Bayes is easy, no?

The answer is obvious once you see it: **nobody actually believes in beta priors for coins**. The real prior is something else:

- Very likely 50/50—several heads in a row just happens, like there will always be lottery winners
- After 20 heads, I might start to doubt, so there's a small probability I found a two-headed coin
- Maybe some odd unbalanced coins exist that aren't fair—I wouldn't rule it out, but surely this is very unlikely, even relative to a two-headed coin

That's a nice prior. Shame it's not conjugate... or is it?

Some quick algebra shows that **priors can be composed hierarchically**: sub-priors update as usual, and their weights change according to Bayes' formula. None of this is hard—I had just never thought about it because Bayesian coins are "easy" and "obvious."

A nice bonus: you can even include seemingly nonsensical priors like δ(p=p₀) that never change no matter what the evidence—only their relative weight changes. This allows for sub-priors like δ(p=1) ("always heads")—a prior that on its face makes no sense at all: a distribution that assigns probability 1 to a single value and ignores all evidence? Yet within a mixture it works perfectly: if I see tails, the prior doesn't update (it can't), but its weight goes to zero, effectively ruling it out.

It all makes sense. One should never think something is too simple or unworthy of deeper study. There is always something to learn, and maybe a second look at Bayesian coins offers a useful insight. It doesn't always have to be rocket science.

*Try the ["Lucky Streak" experiment](USAGE.md#5-the-lucky-streak-story-see-background) to see this in action.*

## Documentation

- **[THEORY.md](THEORY.md)** - Mathematical framework and formulas
- **[USAGE.md](USAGE.md)** - Running the app and features

## Quick Start

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:8050
```

## Structure

```
bayesian-coin/
├── README.md            # This file (overview)
├── THEORY.md            # Mathematical framework
├── USAGE.md             # Running and features
├── app.py               # Interactive Dash visualization
├── requirements.txt     # Dependencies
└── src/
    ├── bayesian_coin.py # Core classes: Prior, BetaPrior, DeltaPrior, CompoundPrior
    └── utils.py         # mk_coin() factory function
```

## Acknowledgments

I worked out the formulas and iterated in notebooks, but kept them private as the project was unfinished. Cleaning it all up and building the interactive app, I iterated productively with [Claude Code](https://claude.ai/code)—partly to finish the project, partly to familiarise myself with the tool and learn how to employ AI code assistants productively.
