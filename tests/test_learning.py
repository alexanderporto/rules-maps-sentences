import numpy as np
import pytest

from iconographers.languages import FORMS, LANGUAGES, VOCAL_ICONIC_COUNTS, log_prior
from iconographers.learning import (
    ambiguity_penalties,
    chosen_form,
    log_likelihoods,
    production_scores,
    updated,
    with_production_error,
)

EPSILON = 0.05


def language_index(forms: tuple[str, str, str, str]) -> int:
    target = [FORMS.index(form) for form in forms]
    return int(np.flatnonzero((LANGUAGES == target).all(axis=1))[0])


def test_likelihood_follows_kirby_and_tamariz() -> None:
    table = log_likelihoods(EPSILON)
    language = language_index(("aa", "ab", "ba", "bb"))
    assert np.exp(table[language, 0, FORMS.index("aa")]) == pytest.approx(1 - EPSILON)
    assert np.exp(table[language, 0, FORMS.index("bb")]) == pytest.approx(EPSILON / 3)


def test_observing_a_pair_favours_languages_containing_it() -> None:
    prior = log_prior(VOCAL_ICONIC_COUNTS, 0.0)
    posterior = updated(prior, 0, FORMS.index("ab"), log_likelihoods(EPSILON))
    assert np.exp(posterior).sum() == pytest.approx(1.0)
    has_pair = LANGUAGES[:, 0] == FORMS.index("ab")
    assert np.all(posterior[has_pair] > prior[has_pair]) and np.all(posterior[~has_pair] < prior[~has_pair])


def test_communicative_rationality_avoids_ambiguous_forms() -> None:
    """90% belief in a degenerate language, 10% in a compositional one. Without communicative
    rationality the speaker uses the degenerate form; with gamma = 100 it avoids it."""
    log_posterior = np.full(len(LANGUAGES), -np.inf)
    log_posterior[language_index(("aa", "aa", "aa", "aa"))] = np.log(0.9)
    log_posterior[language_index(("ab", "aa", "bb", "ba"))] = np.log(0.1)
    rng = np.random.default_rng(0)
    assert FORMS[chosen_form(production_scores(log_posterior, 0, ambiguity_penalties(0.0)), rng)] == "aa"
    assert FORMS[chosen_form(production_scores(log_posterior, 0, ambiguity_penalties(100.0)), rng)] == "ab"


def test_production_error_replaces_form_with_a_different_one() -> None:
    rng = np.random.default_rng(1)
    outputs = np.array([with_production_error(0, EPSILON, rng) for _ in range(20000)])
    assert np.mean(outputs != 0) == pytest.approx(EPSILON, abs=0.005)
    assert set(outputs[outputs != 0]) == {1, 2, 3}
    assert all(with_production_error(2, 0.0, rng) == 2 for _ in range(100))


def test_invalid_learning_parameters_raise() -> None:
    with pytest.raises(ValueError, match="error_probability"):
        log_likelihoods(0.0)
    with pytest.raises(ValueError, match="communicative_rationality"):
        ambiguity_penalties(-1.0)
