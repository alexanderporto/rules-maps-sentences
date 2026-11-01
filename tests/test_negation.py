import numpy as np
import pytest

from iconographers.negation import (
    CONSTANT,
    IDENTITY,
    NEGATION,
    N_STATES,
    NegationModel,
    Parameters,
    interpreted,
    negated,
)


def test_negation_is_a_derangement_pairing_presence_and_absence() -> None:
    images = [negated(s) for s in range(N_STATES)]
    assert sorted(images) == list(range(N_STATES))  # a bijection
    assert all(negated(s) != s for s in range(N_STATES))  # no fixed points
    assert negated(0) == 4 and negated(4) == 0  # resource 0 present <-> absent


def test_interpretations_of_the_function_word() -> None:
    assert interpreted(IDENTITY, 2, 0) == 2
    assert interpreted(CONSTANT, 2, 0) == 0
    assert interpreted(NEGATION, 2, 0) == 6


def test_strategies_are_probability_distributions() -> None:
    model = NegationModel(Parameters(constant_target=0), rng=0)
    for _ in range(50):
        model.step()
    np.testing.assert_allclose(model.sender.signal_probabilities().sum(axis=1), 1.0)
    np.testing.assert_allclose(model.receiver.act_probabilities(0).sum(axis=1), 1.0)


def test_reproduces_published_results_for_four_resources() -> None:
    """Steinert-Threlkeld (2017), Table 2.2, n = 4, 100 trials of 1000 plays: expected payoff 0.978
    and probability of negation 0.995 when the constant interpretation points into X."""
    finals = []
    for seed in range(100):
        model = NegationModel(Parameters(constant_target=0), rng=seed)
        for _ in range(1000):
            model.step()
        finals.append((model.latest_record.expected_payoff, model.latest_record.negation_probability))
    payoff, negation = np.mean(finals, axis=0)
    assert payoff == pytest.approx(0.978, abs=0.015)
    assert negation == pytest.approx(0.995, abs=0.005)


def test_invalid_constant_target_raises() -> None:
    with pytest.raises(ValueError, match="constant_target"):
        Parameters(constant_target=8)
