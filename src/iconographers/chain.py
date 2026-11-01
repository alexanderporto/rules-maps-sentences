"""Computational model: iterated learning as a Markov chain over languages.

Following Griffiths & Kalish (2007), a transmission chain in which each generation is one learner
who learns from data produced by the previous generation is a Markov chain; its transition matrix
gives the expected course of evolution and its stationary distribution the long-run outcome:
Griffiths, T. L., & Kalish, M. L. (2007). Language evolution by iterated learning with Bayesian
agents. Cognitive Science, 31(3), 441-480. https://doi.org/10.1080/15326900701326576
Reali & Griffiths (2010) use the same analysis and relate it to the Wright-Fisher model:
Reali, F., & Griffiths, T. L. (2010). Words as alleles: connecting language evolution with Bayesian
learners to models of genetic drift. Proceedings of the Royal Society B, 277(1680), 429-436.
https://doi.org/10.1098/rspb.2009.1513

Where some languages are (practically) never left once reached, the chain has several absorbing
sets and no single stationary distribution; the long-run outcome is then reported as the
distribution reached from the starting distribution after many generations.

The rules are those of the agent-based model (prior, likelihood, production with gamma and
epsilon). A speaker's behaviour is fully described by its production language: for each meaning,
the form that maximises its production score (learning.production_scores). The chain's state is
therefore the production language of the current generation. The next learner starts from the
prior, observes `bottleneck` utterances (meaning uniform; the production-language form with
probability 1 - epsilon, otherwise one of the other three forms), and its own production language
is the next state. Ties are split equally, as random tie-breaking does in expectation.

Each row of the transition matrix is an expectation over data sets, estimated by Monte Carlo with
`datasets_per_row` data sets.
"""

import numpy as np
from numpy.typing import NDArray

from iconographers.languages import LANGUAGES, N_FORMS, N_LANGUAGES, N_MEANINGS
from iconographers.learning import LANGUAGES_WITH

type Matrix = NDArray[np.float64]


def sampled_counts(
    production_language: int, bottleneck: int, error_probability: float, datasets: int, rng: np.random.Generator
) -> NDArray[np.int64]:
    """Counts [dataset, meaning, form] of the utterances a learner observes."""
    meanings = rng.integers(N_MEANINGS, size=(datasets, bottleneck))
    intended = LANGUAGES[production_language][meanings]
    shift = rng.integers(1, N_FORMS, size=(datasets, bottleneck))  # a different form, uniformly
    erroneous = rng.random((datasets, bottleneck)) < error_probability
    forms = np.where(erroneous, (intended + shift) % N_FORMS, intended)
    cells = meanings * N_FORMS + forms
    counts = np.zeros((datasets, N_MEANINGS * N_FORMS), dtype=np.int64)
    np.add.at(counts, (np.repeat(np.arange(datasets), bottleneck), cells.ravel()), 1)
    return counts.reshape(datasets, N_MEANINGS, N_FORMS)


def learner_log_posteriors(counts: NDArray[np.int64], log_prior: Matrix, likelihoods: Matrix) -> Matrix:
    """Unnormalised log posteriors [dataset, language] after observing the counted utterances."""
    flat_likelihoods = likelihoods.reshape(N_LANGUAGES, N_MEANINGS * N_FORMS)
    return log_prior[None, :] + counts.reshape(len(counts), -1) @ flat_likelihoods.T


def production_language_probabilities(log_posteriors: Matrix, penalties: Matrix) -> Matrix:
    """Probability [dataset, language] that a learner with each posterior has each production language."""
    per_meaning = []
    for m in range(N_MEANINGS):
        weighted = log_posteriors + penalties[None, :, m]
        scores = np.stack([np.logaddexp.reduce(weighted[:, LANGUAGES_WITH[m][f]], axis=1) for f in range(N_FORMS)], axis=1)
        best = scores == scores.max(axis=1, keepdims=True)
        per_meaning.append(best / best.sum(axis=1, keepdims=True))  # [dataset, form]
    probabilities = np.ones((len(log_posteriors), N_LANGUAGES))
    for m in range(N_MEANINGS):
        probabilities *= per_meaning[m][:, LANGUAGES[:, m]]
    return probabilities


def transition_matrix(
    log_prior: Matrix,
    likelihoods: Matrix,
    penalties: Matrix,
    error_probability: float,
    bottleneck: int,
    datasets_per_row: int,
    rng: np.random.Generator,
) -> Matrix:
    """Q[l, l']: probability that a learner taught by production language l has production language l'."""
    rows = []
    for language in range(N_LANGUAGES):
        counts = sampled_counts(language, bottleneck, error_probability, datasets_per_row, rng)
        posteriors = learner_log_posteriors(counts, log_prior, likelihoods)
        rows.append(production_language_probabilities(posteriors, penalties).mean(axis=0))
    return np.array(rows)


def trajectory(initial: NDArray[np.float64], transitions: Matrix, generations: int) -> Matrix:
    """Distribution over production languages in generations 0..generations."""
    distributions = [initial]
    for _ in range(generations):
        distributions.append(distributions[-1] @ transitions)
    return np.array(distributions)


def unit_eigenvalue_count(transitions: Matrix) -> int:
    """Number of eigenvalues equal to 1 (to 1e-9). More than one means the chain has several
    absorbing (closed) sets of languages, so its long-run outcome depends on where it starts and
    there is no single stationary distribution."""
    return int(np.sum(np.abs(np.linalg.eigvals(transitions) - 1.0) < 1e-9))


def long_run_distribution(initial: NDArray[np.float64], transitions: Matrix, generations: int) -> NDArray[np.float64]:
    """Distribution over production languages after `generations` generations from `initial`."""
    return initial @ np.linalg.matrix_power(transitions, generations)
