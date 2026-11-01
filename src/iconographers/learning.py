"""Bayesian learning and communicative production, following Kirby & Tamariz (2022):
Kirby, S., & Tamariz, M. (2022). Cumulative cultural evolution, population structure and the
origin of combinatoriality in human language. Philosophical Transactions of the Royal Society B,
377(1843), 20200319. https://doi.org/10.1098/rstb.2020.0319
Reference code: https://github.com/smkirby/combinatoriality (MIT License, Simon Kirby).

Learning: P(l | d) is proportional to P(l) P(d | l), updated after every observed
<meaning, form> pair, with P(form | l, meaning) = 1 - epsilon if l pairs the form with the meaning
and epsilon / 3 otherwise.
Production: the speaker chooses the form f maximising
    sum over l of P(l | d) C(f | l, meaning),  C(f | l, meaning) = (1 / a)^gamma if l pairs f with
    the meaning and 0 otherwise,
where a is the number of meanings l pairs with f; ties are broken at random. The chosen form is
then replaced by a random other form with probability epsilon.
"""

import numpy as np
from numpy.typing import NDArray

from iconographers.languages import LANGUAGES, N_FORMS, N_MEANINGS, normalised

# LANGUAGES_WITH[m][f] lists the languages that pair meaning m with form f.
LANGUAGES_WITH: tuple[tuple[NDArray[np.int64], ...], ...] = tuple(
    tuple(np.flatnonzero(LANGUAGES[:, m] == f) for f in range(N_FORMS)) for m in range(N_MEANINGS)
)
# AMBIGUITY[l, m]: number of meanings that language l pairs with the form it gives meaning m.
AMBIGUITY: NDArray[np.int64] = np.array(
    [[int(np.sum(language == language[m])) for m in range(N_MEANINGS)] for language in LANGUAGES], dtype=np.int64
)


def log_likelihoods(error_probability: float) -> NDArray[np.float64]:
    """Table [l, m, f] of log P(form f | language l, meaning m)."""
    if not 0.0 < error_probability < 1.0:
        raise ValueError(f"error_probability must be in (0, 1); got {error_probability}.")
    paired = LANGUAGES[:, :, None] == np.arange(N_FORMS)[None, None, :]
    return np.where(paired, np.log(1.0 - error_probability), np.log(error_probability / (N_FORMS - 1)))


def updated(log_posterior: NDArray[np.float64], meaning: int, form: int, table: NDArray[np.float64]) -> NDArray[np.float64]:
    """Posterior after observing one <meaning, form> pair."""
    return normalised(log_posterior + table[:, meaning, form])


def ambiguity_penalties(communicative_rationality: float) -> NDArray[np.float64]:
    """Table [l, m] of log C for the form language l gives meaning m: gamma * log(1 / a)."""
    if communicative_rationality < 0.0:
        raise ValueError(f"communicative_rationality must be >= 0; got {communicative_rationality}.")
    return -communicative_rationality * np.log(AMBIGUITY.astype(np.float64))


def production_scores(log_posterior: NDArray[np.float64], meaning: int, penalties: NDArray[np.float64]) -> NDArray[np.float64]:
    """Log of sum over l of P(l | d) C(f | l, meaning), for every form f."""
    weighted = log_posterior + penalties[:, meaning]
    return np.array([np.logaddexp.reduce(weighted[LANGUAGES_WITH[meaning][f]]) for f in range(N_FORMS)])


def chosen_form(scores: NDArray[np.float64], rng: np.random.Generator) -> int:
    """The form with the highest score; ties broken uniformly at random."""
    return int(rng.choice(np.flatnonzero(scores == scores.max())))


def with_production_error(form: int, error_probability: float, rng: np.random.Generator) -> int:
    """With probability epsilon, a uniformly chosen different form replaces the intended one."""
    if rng.random() >= error_probability:
        return form
    others = [f for f in range(N_FORMS) if f != form]
    return int(rng.choice(others))


def produce(
    log_posterior: NDArray[np.float64],
    meaning: int,
    penalties: NDArray[np.float64],
    error_probability: float,
    rng: np.random.Generator,
) -> int:
    """One utterance (or one map) for a meaning."""
    return with_production_error(chosen_form(production_scores(log_posterior, meaning, penalties), rng), error_probability, rng)


def type_masses(log_posterior: NDArray[np.float64], language_types: NDArray[np.int64]) -> NDArray[np.float64]:
    """Posterior probability of each language type."""
    return np.bincount(language_types, weights=np.exp(log_posterior), minlength=4)
