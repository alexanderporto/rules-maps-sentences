"""The space of possible languages, the learners' prior, and iconicity.

The language space is that of Kirby et al. (2015), as implemented in the Simulating Language
course (Lab 6):
Kirby, S., Tamariz, M., Cornish, H., & Smith, K. (2015). Compression and communication in the
cultural evolution of linguistic structure. Cognition, 141, 87-102.
https://doi.org/10.1016/j.cognition.2015.03.016

Meanings are the four resources of the argument, each a combination of two features:
    feature 1: 0 = large and round, 1 = small and angular;
    feature 2: 2 = distant, 3 = near.
Forms are two symbols long, each symbol 'a' or 'b'. A language gives one form to each meaning,
so there are 4 ** 4 = 256 languages.

The same abstract forms are used for both expression systems:
    vocalisations: 'a' = a back rounded vowel, 'b' = a front unrounded vowel;
    maps: first symbol is a glyph ('a' = large round glyph, 'b' = small angular glyph),
          second symbol is a placement ('a' = far, 'b' = near).
In both systems 'a' is the symbol the argument associates with large, round and distant things.
"""

from itertools import product

import numpy as np
from numpy.typing import NDArray

from iconographers.kirby2015_priors import KIRBY_2015_LOG_PRIORS

MEANINGS: tuple[str, ...] = ("02", "03", "12", "13")
FORMS: tuple[str, ...] = ("aa", "ab", "ba", "bb")
N_MEANINGS = len(MEANINGS)
N_FORMS = len(FORMS)

DEGENERATE = 0
HOLISTIC = 1
OTHER = 2
COMPOSITIONAL = 3
TYPE_NAMES: tuple[str, ...] = ("degenerate", "holistic", "other", "compositional")

# Symbol the argument associates with each feature value: 'a' for large-and-round and for distant.
ICONIC_SYMBOL: dict[str, str] = {"0": "a", "1": "b", "2": "a", "3": "b"}

# LANGUAGES[l, m] is the index in FORMS of the form that language l gives to meaning m.
LANGUAGES: NDArray[np.int64] = np.array(list(product(range(N_FORMS), repeat=N_MEANINGS)), dtype=np.int64)
N_LANGUAGES = len(LANGUAGES)


def language_type(language: NDArray[np.int64]) -> int:
    """Kirby et al. (2015): degenerate (one form for all meanings), compositional (each symbol
    position consistently expresses one feature), holistic (all forms distinct but not
    compositional), or other."""
    forms = [FORMS[f] for f in language]
    if len(set(forms)) == 1:
        return DEGENERATE
    if len(set(forms)) < N_MEANINGS:
        return OTHER
    for first_position, second_position in ((0, 1), (1, 0)):
        expresses_both = all(
            (forms[i][first_position] == forms[j][first_position]) == (MEANINGS[i][0] == MEANINGS[j][0])
            and (forms[i][second_position] == forms[j][second_position]) == (MEANINGS[i][1] == MEANINGS[j][1])
            for i in range(N_MEANINGS)
            for j in range(N_MEANINGS)
        )
        if expresses_both:
            return COMPOSITIONAL
    return HOLISTIC


LANGUAGE_TYPES: NDArray[np.int64] = np.array([language_type(language) for language in LANGUAGES], dtype=np.int64)


def vocal_word_is_iconic(meaning: str, form: str) -> bool:
    """A word is iconic if its two vowels are the vowels the argument gives the resource's two
    features, in either order (the argument does not specify an order)."""
    return sorted(form) == sorted(ICONIC_SYMBOL[meaning[0]] + ICONIC_SYMBOL[meaning[1]])


def map_is_iconic(meaning: str, form: str) -> bool:
    """A map is iconic if its glyph depicts the resource's size and shape and its placement
    depicts the resource's distance."""
    return form[0] == ICONIC_SYMBOL[meaning[0]] and form[1] == ICONIC_SYMBOL[meaning[1]]


VOCAL_ICONIC_COUNTS: NDArray[np.int64] = np.array(
    [sum(vocal_word_is_iconic(MEANINGS[m], FORMS[f]) for m, f in enumerate(language)) for language in LANGUAGES],
    dtype=np.int64,
)
MAP_ICONIC_COUNTS: NDArray[np.int64] = np.array(
    [sum(map_is_iconic(MEANINGS[m], FORMS[f]) for m, f in enumerate(language)) for language in LANGUAGES],
    dtype=np.int64,
)


def normalised(log_values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalise log probabilities so that their exponentials sum to 1."""
    return log_values - np.logaddexp.reduce(log_values)


def log_prior(iconic_counts: NDArray[np.int64], iconic_bits: float) -> NDArray[np.float64]:
    """Kirby et al.'s (2015) simplicity prior, P(l) proportional to 2^-L(l), with each iconic
    expression in l saving iconic_bits bits of coding length: P(l) proportional to
    2^-(L(l) - iconic_bits * k(l)), where k(l) is the number of iconic expressions in l.
    iconic_bits = 0 gives Kirby et al.'s prior unchanged."""
    if iconic_bits < 0.0:
        raise ValueError(f"iconic_bits must be >= 0; got {iconic_bits}.")
    kirby = np.array(KIRBY_2015_LOG_PRIORS, dtype=np.float64)
    return normalised(kirby + iconic_bits * np.log(2.0) * iconic_counts)
