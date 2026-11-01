from dataclasses import replace

import numpy as np
import pytest

from iconographers.quantifiers import (
    ALL_STRUCTURES,
    Parameters,
    QuantifierModel,
    forward,
    gradients,
    initial_network,
    is_degenerate,
    is_ultrafilter,
    judgments,
    monotonicity,
    quantity,
    structures_as_produced,
    structures_shuffled,
    trained,
)

DISTANT = ALL_STRUCTURES.sum(axis=1)


def test_measures_reproduce_published_values() -> None:
    """Carcassi et al. (2021), section 3.1.5: 'some' 1.0 and 'an even number of' 0.001.
    The paper also gives 0.7517 for 'between 3 and 5'; the formula as stated (and as written in the
    reference code) gives 0.729 for 10 objects, so that value is not reproduced (see ODD)."""
    assert monotonicity((DISTANT >= 1).astype(np.int64)) == pytest.approx(1.0)
    assert monotonicity((DISTANT % 2 == 0).astype(np.int64)) == pytest.approx(0.001, abs=5e-4)
    assert monotonicity(((DISTANT >= 3) & (DISTANT <= 5)).astype(np.int64)) == pytest.approx(0.729, abs=1e-3)


def test_quantity_and_ultrafilters() -> None:
    assert quantity((DISTANT >= 3).astype(np.int64)) == pytest.approx(1.0)
    first_resource = ALL_STRUCTURES[:, 0].copy()
    assert is_ultrafilter(first_resource) and is_ultrafilter(1 - first_resource)
    assert quantity(first_resource) < 0.2
    assert is_degenerate(np.ones(len(ALL_STRUCTURES), dtype=np.int64))


def test_backpropagation_matches_numerical_gradient() -> None:
    rng = np.random.default_rng(0)
    network = initial_network(rng)
    x = rng.integers(0, 2, (32, 10))
    y = rng.integers(0, 2, 32).astype(np.float64)

    def loss(candidate: object) -> float:
        p = forward(candidate, x).output
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    analytic = gradients(network, forward(network, x), y)
    for name in ("w1", "w2", "gamma", "beta", "w3"):
        weights = getattr(network, name)
        plus, minus = weights.copy(), weights.copy()
        plus.flat[0] += 1e-6
        minus.flat[0] -= 1e-6
        numerical = (loss(replace(network, **{name: plus})) - loss(replace(network, **{name: minus}))) / 2e-6
        assert getattr(analytic, name).flat[0] == pytest.approx(numerical, rel=1e-4, abs=1e-8)


def test_a_network_can_learn_a_monotone_quantifier() -> None:
    rng = np.random.default_rng(1)
    target = (DISTANT >= 3).astype(np.int64)
    network = trained(initial_network(rng), ALL_STRUCTURES, target, 8, 32, rng)
    assert np.mean(judgments(network, ALL_STRUCTURES) == target) > 0.95


def test_shuffling_keeps_the_number_of_distant_resources() -> None:
    rng = np.random.default_rng(2)
    structures = rng.integers(0, 2, (100, 10))
    np.testing.assert_array_equal(structures_shuffled(structures, rng).sum(axis=1), structures.sum(axis=1))
    assert structures_as_produced(structures, rng) is structures


def test_generations_are_replaced_and_reproducible() -> None:
    parameters = Parameters(agents_per_generation=4, bottleneck=64, epochs=1, batch_size=32)
    first = QuantifierModel(parameters, structures_shuffled, rng=3)
    second = QuantifierModel(parameters, structures_shuffled, rng=3)
    old = list(first.generation)
    first.step()
    second.step()
    assert not set(old) & set(first.generation) and set(first.agents) == set(first.generation)
    for a, b in zip(first.generation, second.generation, strict=True):
        np.testing.assert_array_equal(a.network.w1, b.network.w1)


def test_invalid_parameters_raise() -> None:
    with pytest.raises(ValueError, match="bottleneck"):
        Parameters(agents_per_generation=10, bottleneck=16, epochs=4, batch_size=32)
