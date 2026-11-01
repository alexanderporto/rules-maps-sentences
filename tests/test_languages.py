import numpy as np
import pytest

from iconographers.kirby2015_priors import KIRBY_2015_LANGUAGE_TYPES, KIRBY_2015_LOG_PRIORS
from iconographers.languages import (
    COMPOSITIONAL,
    FORMS,
    LANGUAGE_TYPES,
    LANGUAGES,
    MAP_ICONIC_COUNTS,
    VOCAL_ICONIC_COUNTS,
    log_prior,
    map_is_iconic,
    vocal_word_is_iconic,
)


def language_index(forms: tuple[str, str, str, str]) -> int:
    """Index of the language giving these forms to meanings 02, 03, 12, 13."""
    target = [FORMS.index(form) for form in forms]
    return int(np.flatnonzero((LANGUAGES == target).all(axis=1))[0])


def test_language_space_and_types_match_kirby_et_al() -> None:
    assert LANGUAGES.shape == (256, 4)
    np.testing.assert_array_equal(LANGUAGE_TYPES, KIRBY_2015_LANGUAGE_TYPES)
    assert np.bincount(LANGUAGE_TYPES).tolist() == [4, 16, 228, 8]  # degenerate, holistic, other, compositional


def test_prior_reproduces_published_coding_lengths() -> None:
    """Kirby et al. (2015), section 2.1.3: L = 38.55 (degenerate aa), 55.20 (compositional), 67.29 (holistic) bits."""
    prior = log_prior(VOCAL_ICONIC_COUNTS, 0.0)
    np.testing.assert_allclose(prior, KIRBY_2015_LOG_PRIORS)
    bits = -prior / np.log(2.0)
    degenerate = language_index(("aa", "aa", "aa", "aa"))
    compositional = language_index(("aa", "ab", "ba", "bb"))
    holistic = language_index(("aa", "ab", "bb", "ba"))
    assert bits[compositional] - bits[degenerate] == pytest.approx(55.20 - 38.55, abs=0.01)
    assert bits[holistic] - bits[degenerate] == pytest.approx(67.29 - 38.55, abs=0.01)


def test_iconic_expressions_follow_the_argument() -> None:
    # 'a' = back rounded vowel; meaning 02 = large-and-round, distant; 13 = small-and-angular, near.
    assert vocal_word_is_iconic("02", "aa") and vocal_word_is_iconic("13", "bb")
    assert vocal_word_is_iconic("03", "ab") and vocal_word_is_iconic("03", "ba")  # vowel order is free
    assert not vocal_word_is_iconic("02", "bb")
    # Maps: first symbol is the glyph, second the placement; position matters.
    assert map_is_iconic("03", "ab") and not map_is_iconic("03", "ba")


def test_fully_iconic_unambiguous_languages_are_compositional() -> None:
    vocal = np.flatnonzero((VOCAL_ICONIC_COUNTS == 4) & (LANGUAGE_TYPES == COMPOSITIONAL))
    maps = np.flatnonzero((MAP_ICONIC_COUNTS == 4) & (LANGUAGE_TYPES == COMPOSITIONAL))
    assert len(vocal) == 2 and len(maps) == 1
    unambiguous = np.array([len(set(language)) == 4 for language in LANGUAGES])
    assert set(np.flatnonzero((VOCAL_ICONIC_COUNTS == 4) & unambiguous)) == set(vocal)


def test_each_iconic_expression_saves_the_given_bits() -> None:
    beta = 2.0
    shifted = log_prior(VOCAL_ICONIC_COUNTS, beta) - log_prior(VOCAL_ICONIC_COUNTS, 0.0)
    relative_bits = (shifted - shifted[0]) / np.log(2.0)
    np.testing.assert_allclose(relative_bits, beta * (VOCAL_ICONIC_COUNTS - VOCAL_ICONIC_COUNTS[0]), atol=1e-9)


def test_negative_iconic_bits_raise() -> None:
    with pytest.raises(ValueError, match="iconic_bits"):
        log_prior(VOCAL_ICONIC_COUNTS, -1.0)
