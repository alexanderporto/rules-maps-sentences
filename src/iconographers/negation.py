"""Negation submodel: the basic negation learning setup of Steinert-Threlkeld (2016, 2017).

Steinert-Threlkeld, S. (2016). Compositional signaling in a complex world. Journal of Logic,
Language and Information, 25(3-4), 379-397. https://doi.org/10.1007/s10849-016-9236-9
Steinert-Threlkeld, S. (2017). Communication and computation: New questions about
compositionality (Doctoral dissertation, Stanford University), chapter 2, sections 2.3 and 2.5.
ILLC Dissertation Series DS-2017-05. https://eprints.illc.uva.nl/id/eprint/2145/

The iconographers' world: for each of the 4 resources, the resource is present or absent
(2n = 8 states, n = 4). Minimal negation is the derangement f that maps "resource r present" to
"resource r absent" and back (the thesis's Example 2.1: leopard / no leopard). The present states
form the complementizer X.

Setup (thesis, section 2.5, "basic negation learning setup"):
- The sender and receiver already have a signaling system on X: in "resource r present" the
  sender sends the basic signal m_r, and the receiver responds to m_r with "resource r present".
- In the absent states, the sender's behaviour is learned: it draws a complex signal a m_r from
  an urn for that state. The thesis text leaves the sender's options open; the reference code
  (github.com/shanest/learning-evolution-compositionality, SemiFixedSender) restricts them to the
  complex signals, and with that restriction this implementation reproduces the thesis's
  Table 2.2 for n = 4 (tests/test_negation.py). Allowing basic signals too gives lower values.
- On hearing a m_r, the receiver draws an interpretation g of the function word a from an urn
  over F = {identity, constant c_i, negation f} and performs g(its act for m_r).
- Roth-Erev reinforcement: after a successful play (payoff 1), the ball drawn from each urn is
  doubled up (one ball of the same kind added). Urns start with one ball of every kind.
"""

from dataclasses import dataclass, fields
from functools import partial

import mesa
import numpy as np
from numpy.typing import NDArray

N_RESOURCES = 4
N_STATES = 2 * N_RESOURCES  # state r: resource r present; state r + 4: resource r absent
N_SIGNALS = 2 * N_RESOURCES  # signal r: basic m_r; signal r + 4: a m_r
PRESENT_STATES: tuple[int, ...] = tuple(range(N_RESOURCES))  # the complementizer X
IDENTITY, CONSTANT, NEGATION = 0, 1, 2
FUNCTION_NAMES: tuple[str, ...] = ("identity", "constant", "negation")


def negated(state: int) -> int:
    """Minimal negation f: 'resource r present' <-> 'resource r absent'."""
    return (state + N_RESOURCES) % N_STATES


def interpreted(function: int, act: int, constant_target: int) -> int:
    """Apply the receiver's chosen interpretation of the function word to an act."""
    if function == IDENTITY:
        return act
    if function == CONSTANT:
        return constant_target
    return negated(act)


@dataclass(frozen=True)
class Parameters:
    """All parameters. No parameter has a default."""

    constant_target: int  # the act c_i maps to; the thesis runs i in X and i in f[X]

    def __post_init__(self) -> None:
        if not 0 <= self.constant_target < N_STATES:
            raise ValueError(f"constant_target must be a state 0..{N_STATES - 1}; got {self.constant_target}.")


@dataclass(frozen=True)
class Record:
    """One row per play; field names are the output columns."""

    expected_payoff: float  # probability of a successful play under the current strategies
    negation_probability: float  # probability that the receiver interprets a as negation
    sender_uses_negation: float  # mean probability of sending a m_r in 'resource r absent'


class Sender(mesa.Agent):
    def __init__(self, model: "NegationModel") -> None:
        super().__init__(model)
        # Urns for the absent states only (rows: resource r absent; columns: a m_0 .. a m_3).
        self.urns: NDArray[np.float64] = np.ones((N_RESOURCES, N_RESOURCES))

    def signal_probabilities(self) -> NDArray[np.float64]:
        """Row per state, column per signal."""
        probabilities = np.zeros((N_STATES, N_SIGNALS))
        probabilities[np.arange(N_RESOURCES), np.arange(N_RESOURCES)] = 1.0  # signaling system on X
        probabilities[N_RESOURCES:, N_RESOURCES:] = self.urns / self.urns.sum(axis=1, keepdims=True)
        return probabilities

    def send(self, state: int) -> int:
        if state in PRESENT_STATES:
            return state
        urn = self.urns[state - N_RESOURCES]
        return N_RESOURCES + int(self.model.rng.choice(N_RESOURCES, p=urn / urn.sum()))

    def reinforce(self, state: int, signal: int, payoff: float) -> None:
        if state not in PRESENT_STATES:
            self.urns[state - N_RESOURCES, signal - N_RESOURCES] += payoff


class Receiver(mesa.Agent):
    def __init__(self, model: "NegationModel") -> None:
        super().__init__(model)
        self.function_urn: NDArray[np.float64] = np.ones(len(FUNCTION_NAMES))

    def function_probabilities(self) -> NDArray[np.float64]:
        return self.function_urn / self.function_urn.sum()

    def act_probabilities(self, constant_target: int) -> NDArray[np.float64]:
        """Row per signal, column per act."""
        probabilities = np.zeros((N_SIGNALS, N_STATES))
        probabilities[np.arange(N_RESOURCES), np.arange(N_RESOURCES)] = 1.0  # m_r -> resource r present
        for r in range(N_RESOURCES):
            for function, weight in enumerate(self.function_probabilities()):
                probabilities[N_RESOURCES + r, interpreted(function, r, constant_target)] += weight
        return probabilities

    def respond(self, signal: int, constant_target: int) -> tuple[int, int | None]:
        """The act, and the interpretation of a that was drawn (None for a basic signal)."""
        if signal < N_RESOURCES:
            return signal, None
        function = int(self.model.rng.choice(len(FUNCTION_NAMES), p=self.function_probabilities()))
        return interpreted(function, signal - N_RESOURCES, constant_target), function

    def reinforce(self, function: int | None, payoff: float) -> None:
        if function is not None:
            self.function_urn[function] += payoff


def latest(field_name: str, model: "NegationModel") -> float:
    """DataCollector reporter. Mesa passes the model only to lambdas and functools.partial objects."""
    return float(getattr(model.latest_record, field_name))


class NegationModel(mesa.Model):
    """One step = one play of the game."""

    def __init__(self, parameters: Parameters, rng: int) -> None:
        super().__init__(rng=rng)
        self.parameters: Parameters = parameters
        self.sender: Sender = Sender(self)
        self.receiver: Receiver = Receiver(self)
        self.latest_record: Record = self.measured()
        reporters = {field.name: partial(latest, field.name) for field in fields(Record)}
        self.datacollector: mesa.DataCollector = mesa.DataCollector(model_reporters={"play": "steps", **reporters})
        self.datacollector.collect(self)

    def measured(self) -> Record:
        sending = self.sender.signal_probabilities()
        acting = self.receiver.act_probabilities(self.parameters.constant_target)
        success = float(np.mean([sending[s] @ acting[:, s] for s in range(N_STATES)]))
        uses_negation = float(np.mean([sending[N_RESOURCES + r, N_RESOURCES + r] for r in range(N_RESOURCES)]))
        return Record(success, float(self.receiver.function_probabilities()[NEGATION]), uses_negation)

    def step(self) -> None:
        state = int(self.rng.integers(N_STATES))
        signal = self.sender.send(state)
        act, function = self.receiver.respond(signal, self.parameters.constant_target)
        payoff = 1.0 if act == state else 0.0
        self.sender.reinforce(state, signal, payoff)
        self.receiver.reinforce(function, payoff)
        self.latest_record = self.measured()
        self.datacollector.collect(self)
