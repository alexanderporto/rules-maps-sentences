"""Agent-based model of the iconographers, using the population model of Kirby & Tamariz (2022).

Kirby, S., & Tamariz, M. (2022). Cumulative cultural evolution, population structure and the
origin of combinatoriality in human language. Philosophical Transactions of the Royal Society B,
377(1843), 20200319. https://doi.org/10.1098/rstb.2020.0319

Each step (one "iteration" in Kirby & Tamariz):
1. interactions_per_step times: a meaning (resource) is chosen at random; the oldest agent
   produces a vocalisation and a map for it; a different, randomly chosen agent hears the
   vocalisation and updates its vocal posterior; the map is added to the community's record
   (the most recent record_size maps), and the same agent sees one map drawn at random from
   the record and updates its map posterior.
2. The oldest agent's posteriors are measured.
3. The oldest agent is removed and a newborn, whose posteriors equal the prior, is added.

Vocalisations vanish once heard; maps stay in the record after their maker has died. Founders are
trained on a shared holistic language, as in Kirby & Tamariz (2022) and Kirby et al. (2015), drawn
at random from the 16 holistic languages so that no particular starting language is favoured.
"""

from collections import deque
from dataclasses import dataclass, fields
from functools import partial

import mesa
import numpy as np
from numpy.typing import NDArray

from iconographers.languages import (
    COMPOSITIONAL,
    HOLISTIC,
    LANGUAGE_TYPES,
    LANGUAGES,
    MAP_ICONIC_COUNTS,
    N_MEANINGS,
    VOCAL_ICONIC_COUNTS,
    log_prior,
)
from iconographers.learning import ambiguity_penalties, log_likelihoods, produce, type_masses, updated

type LogPosterior = NDArray[np.float64]


@dataclass(frozen=True)
class Parameters:
    """All parameters. No parameter has a default."""

    population_size: int  # Kirby & Tamariz (2022): 10
    interactions_per_step: int  # Kirby & Tamariz (2022): 20
    founder_training: int  # observations of the starting language per founder; Kirby & Tamariz (2022): 50
    record_size: int  # maps kept in the record; the memory size r = 80 of Kirby et al. (2015)
    error_probability: float  # epsilon; Kirby & Tamariz (2022): 0.05
    communicative_rationality: float  # gamma; Kirby & Tamariz (2022): 0 (none) and 100 (communicative)
    iconic_bits: float  # beta: coding-length bits saved per iconic expression; 0 = Kirby et al.'s prior

    def __post_init__(self) -> None:
        if self.population_size < 2:
            raise ValueError(f"population_size must be >= 2; got {self.population_size}.")
        if self.interactions_per_step < 1 or self.founder_training < 0 or self.record_size < 1:
            raise ValueError(
                "interactions_per_step and record_size must be >= 1 and founder_training >= 0; got "
                f"{self.interactions_per_step}, {self.record_size}, {self.founder_training}."
            )


@dataclass(frozen=True)
class Record:
    """Posterior of the oldest agent, one row per step; field names are the output columns."""

    vocal_degenerate: float
    vocal_holistic: float
    vocal_other: float
    vocal_compositional: float
    vocal_iconicity: float  # expected share of iconic words
    vocal_iconic_compositional: float  # probability of the fully iconic compositional languages
    map_degenerate: float
    map_holistic: float
    map_other: float
    map_compositional: float
    map_iconicity: float
    map_iconic_compositional: float


def channel_measures(log_posterior: LogPosterior, iconic_counts: NDArray[np.int64]) -> tuple[float, ...]:
    """Type masses, expected share of iconic expressions, and mass on fully iconic compositional languages."""
    probabilities = np.exp(log_posterior)
    iconic_compositional = (LANGUAGE_TYPES == COMPOSITIONAL) & (iconic_counts == N_MEANINGS)
    return (
        *(float(mass) for mass in type_masses(log_posterior, LANGUAGE_TYPES)),
        float(probabilities @ iconic_counts) / N_MEANINGS,
        float(probabilities[iconic_compositional].sum()),
    )


def latest(field_name: str, model: "IconographersModel") -> float:
    """DataCollector reporter. Mesa passes the model only to lambdas and functools.partial objects."""
    return float(getattr(model.latest_record, field_name))


class Iconographer(mesa.Agent):
    def __init__(self, model: "IconographersModel", vocal: LogPosterior, maps: LogPosterior) -> None:
        super().__init__(model)
        self.vocal: LogPosterior = vocal  # posterior over vocal languages
        self.maps: LogPosterior = maps  # posterior over map languages


class IconographersModel(mesa.Model):
    def __init__(self, parameters: Parameters, rng: int) -> None:
        super().__init__(rng=rng)
        self.parameters: Parameters = parameters
        self.likelihoods: NDArray[np.float64] = log_likelihoods(parameters.error_probability)
        self.penalties: NDArray[np.float64] = ambiguity_penalties(parameters.communicative_rationality)
        self.vocal_prior: LogPosterior = log_prior(VOCAL_ICONIC_COUNTS, parameters.iconic_bits)
        self.map_prior: LogPosterior = log_prior(MAP_ICONIC_COUNTS, parameters.iconic_bits)
        self.starting_language: int = int(self.rng.choice(np.flatnonzero(LANGUAGE_TYPES == HOLISTIC)))
        # Ordered from newest (index 0) to oldest (index -1).
        self.population: list[Iconographer] = [
            Iconographer(
                self,
                self.trained(self.vocal_prior, parameters.founder_training),
                self.trained(self.map_prior, parameters.founder_training),
            )
            for _ in range(parameters.population_size)
        ]
        self.record: deque[tuple[int, int]] = deque(
            (self.starting_pair() for _ in range(parameters.record_size)), maxlen=parameters.record_size
        )
        self.latest_record: Record = self.measured(self.population[-1])
        reporters = {field.name: partial(latest, field.name) for field in fields(Record)}
        self.datacollector: mesa.DataCollector = mesa.DataCollector(model_reporters={"step": "steps", **reporters})
        self.datacollector.collect(self)

    def starting_pair(self) -> tuple[int, int]:
        """A <meaning, form> pair from the starting language (no production error, as in Kirby & Tamariz)."""
        meaning = int(self.rng.integers(N_MEANINGS))
        return meaning, int(LANGUAGES[self.starting_language, meaning])

    def trained(self, prior: LogPosterior, observations: int) -> LogPosterior:
        posterior = prior
        for _ in range(observations):
            meaning, form = self.starting_pair()
            posterior = updated(posterior, meaning, form, self.likelihoods)
        return posterior

    def measured(self, agent: Iconographer) -> Record:
        return Record(*channel_measures(agent.vocal, VOCAL_ICONIC_COUNTS), *channel_measures(agent.maps, MAP_ICONIC_COUNTS))

    def interact(self, signaller: Iconographer, learner: Iconographer) -> None:
        p = self.parameters
        meaning = int(self.rng.integers(N_MEANINGS))
        word = produce(signaller.vocal, meaning, self.penalties, p.error_probability, self.rng)
        learner.vocal = updated(learner.vocal, meaning, word, self.likelihoods)
        drawn = produce(signaller.maps, meaning, self.penalties, p.error_probability, self.rng)
        self.record.append((meaning, drawn))
        seen_meaning, seen_map = self.record[int(self.rng.integers(len(self.record)))]
        learner.maps = updated(learner.maps, seen_meaning, seen_map, self.likelihoods)

    def step(self) -> None:
        signaller = self.population[-1]
        n_agents = len(self.population)
        for _ in range(self.parameters.interactions_per_step):
            learner = self.population[int(self.rng.integers(n_agents - 1))]  # any agent but the oldest
            self.interact(signaller, learner)
        self.latest_record = self.measured(signaller)
        self.datacollector.collect(self)
        signaller.remove()
        newborn = Iconographer(self, self.vocal_prior.copy(), self.map_prior.copy())
        self.population = [newborn, *self.population[:-1]]
