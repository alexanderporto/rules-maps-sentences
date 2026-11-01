import numpy as np
import pytest

from iconographers.chain import (
    learner_log_posteriors,
    long_run_distribution,
    production_language_probabilities,
    sampled_counts,
    trajectory,
    transition_matrix,
    unit_eigenvalue_count,
)
from iconographers.languages import FORMS, LANGUAGES, VOCAL_ICONIC_COUNTS, log_prior
from iconographers.learning import ambiguity_penalties, log_likelihoods

EPSILON = 0.05


def test_sampled_counts_follow_the_production_language() -> None:
    rng = np.random.default_rng(0)
    language = 27  # compositional: aa ab ba bb
    counts = sampled_counts(language, 20, EPSILON, 5000, rng)
    assert np.all(counts.sum(axis=(1, 2)) == 20)
    correct = sum(counts[:, m, LANGUAGES[language, m]].sum() for m in range(4))
    assert correct / counts.sum() == pytest.approx(1 - EPSILON, abs=0.005)


def test_a_learner_with_much_data_reproduces_an_unambiguous_language() -> None:
    language = 27
    counts = np.zeros((1, 4, 4), dtype=np.int64)
    counts[0, np.arange(4), LANGUAGES[language]] = 20
    posteriors = learner_log_posteriors(counts, log_prior(VOCAL_ICONIC_COUNTS, 0.0), log_likelihoods(EPSILON))
    probabilities = production_language_probabilities(posteriors, ambiguity_penalties(100.0))
    assert probabilities[0, language] == pytest.approx(1.0)
    assert [FORMS[f] for f in LANGUAGES[language]] == ["aa", "ab", "ba", "bb"]


def test_transition_matrix_is_stochastic_and_long_run_matches_repeated_steps() -> None:
    rng = np.random.default_rng(1)
    q = transition_matrix(
        log_prior(VOCAL_ICONIC_COUNTS, 0.0), log_likelihoods(EPSILON), ambiguity_penalties(0.0), EPSILON, 20, 50, rng
    )
    assert np.all(q >= 0.0)
    np.testing.assert_allclose(q.sum(axis=1), 1.0)
    initial = np.full(len(q), 1.0 / len(q))
    np.testing.assert_allclose(long_run_distribution(initial, q, 30), trajectory(initial, q, 30)[-1], atol=1e-12)
    assert unit_eigenvalue_count(q) >= 1


def test_absorbing_languages_give_several_unit_eigenvalues() -> None:
    q = np.eye(3)
    q[2] = [0.5, 0.5, 0.0]  # states 0 and 1 absorb
    assert unit_eigenvalue_count(q) == 2
