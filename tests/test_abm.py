from dataclasses import replace

import numpy as np
import pytest

from iconographers.abm import IconographersModel, Parameters

BASE = Parameters(
    population_size=10,
    interactions_per_step=20,
    founder_training=50,
    record_size=80,
    error_probability=0.05,
    communicative_rationality=100.0,
    iconic_bits=2.0,
)


def test_founders_learn_the_shared_starting_language() -> None:
    model = IconographersModel(BASE, rng=0)
    for agent in model.population:
        assert np.exp(agent.vocal[model.starting_language]) > 0.9
        assert np.exp(agent.maps[model.starting_language]) > 0.9


def test_oldest_agent_is_replaced_by_a_newborn_each_step() -> None:
    model = IconographersModel(BASE, rng=1)
    oldest = model.population[-1]
    model.step()
    assert oldest not in model.population and oldest not in set(model.agents)
    np.testing.assert_allclose(model.population[0].vocal, model.vocal_prior)
    assert len(model.population) == BASE.population_size and set(model.agents) == set(model.population)


def test_record_keeps_only_the_most_recent_maps() -> None:
    model = IconographersModel(BASE, rng=2)
    for _ in range(3):
        model.step()
    assert len(model.record) == BASE.record_size


def test_same_seed_reproduces_run_exactly() -> None:
    first, second = IconographersModel(BASE, rng=3), IconographersModel(BASE, rng=3)
    for _ in range(10):
        first.step()
        second.step()
    for a, b in zip(first.population, second.population, strict=True):
        np.testing.assert_array_equal(a.vocal, b.vocal)
        np.testing.assert_array_equal(a.maps, b.maps)


def test_invalid_parameters_raise() -> None:
    with pytest.raises(ValueError, match="population_size"):
        replace(BASE, population_size=1)
