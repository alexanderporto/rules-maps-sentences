"""Quantifier submodel: iterated learning of quantifiers by neural-network agents.

Carcassi, F., Steinert-Threlkeld, S., & Szymanik, J. (2021). Monotone quantifiers emerge via
iterated learning. Cognitive Science, 45(8), e13027. https://doi.org/10.1111/cogs.13027
Reference code: https://github.com/thelogicalgrammar/NeuralNetIteratedQuantifiers (PyTorch).

The iconographers' world: A is the 10 resources in view, B the distant ones. A structure is a
vector of 10 bits, bit i = 1 if resource i is distant. A quantifier ("some / all / at least two of
the resources are distant") is a function from structures to truth values.

As in the paper and its code:
- each agent is a network with 10 inputs (bits mapped to -1 / 1), two hidden layers of 16 ReLU
  units, batch normalisation after the second hidden layer, and a sigmoid output. Batch
  normalisation always uses the statistics of the current batch (the reference code never
  switches the networks to evaluation mode);
- weights and biases start uniform in (-1/sqrt(fan_in), 1/sqrt(fan_in)), PyTorch's default;
- learning: binary cross-entropy, Adam (learning rate 0.001, betas 0.9 and 0.999, eps 1e-8),
  mini-batches of 32 (an incomplete last batch is dropped), data reshuffled every epoch;
- production: the network's output for a structure, rounded to 0 or 1;
- each generation has 10 agents; each child learns from one parent drawn at random from the
  previous generation, on `bottleneck` random structures (drawn with replacement) and the
  parent's judgments of them. In Experiment 2 the child sees each structure with its bits shuffled,
  so that a quantifier cannot track particular resources.
This is a NumPy reimplementation, not the authors' PyTorch code; random streams differ.
"""

from collections.abc import Callable
from dataclasses import dataclass, fields
from functools import partial
from itertools import product

import mesa
import numpy as np
from numpy.typing import NDArray

type Array = NDArray[np.float64]
type Bits = NDArray[np.int64]
type Transmission = Callable[[Bits, np.random.Generator], Bits]

STRUCTURE_SIZE = 10  # resources in view
HIDDEN = 16
BATCH_NORM_EPSILON = 1e-5
ALL_STRUCTURES: Bits = np.array(list(product((0, 1), repeat=STRUCTURE_SIZE)), dtype=np.int64)


@dataclass(frozen=True)
class Network:
    """Weights of one agent's network (also used for gradients and Adam moments)."""

    w1: Array
    b1: Array
    w2: Array
    b2: Array
    gamma: Array
    beta: Array
    w3: Array
    b3: Array


def network_map(function: Callable[..., Array], *networks: Network) -> Network:
    """Apply a function to corresponding weight arrays of several networks."""
    return Network(*(function(*(getattr(n, f.name) for n in networks)) for f in fields(Network)))


def initial_network(rng: np.random.Generator) -> Network:
    def linear(fan_in: int, fan_out: int) -> tuple[Array, Array]:
        bound = 1.0 / np.sqrt(fan_in)
        return rng.uniform(-bound, bound, (fan_in, fan_out)), rng.uniform(-bound, bound, fan_out)

    w1, b1 = linear(STRUCTURE_SIZE, HIDDEN)
    w2, b2 = linear(HIDDEN, HIDDEN)
    w3, b3 = linear(HIDDEN, 1)
    return Network(w1, b1, w2, b2, np.ones(HIDDEN), np.zeros(HIDDEN), w3, b3)


@dataclass(frozen=True)
class ForwardPass:
    """Intermediate values needed for backpropagation."""

    inputs: Array
    z1: Array
    h1: Array
    z2: Array
    normalised: Array
    inverse_std: Array
    normed_output: Array
    output: Array


def forward(network: Network, structures: Bits) -> ForwardPass:
    inputs = 2.0 * structures - 1.0
    z1 = inputs @ network.w1 + network.b1
    h1 = np.maximum(z1, 0.0)
    z2 = h1 @ network.w2 + network.b2
    h2 = np.maximum(z2, 0.0)
    mean = h2.mean(axis=0)
    inverse_std = 1.0 / np.sqrt(h2.var(axis=0) + BATCH_NORM_EPSILON)  # biased variance, as in training mode
    normalised = (h2 - mean) * inverse_std
    normed_output = network.gamma * normalised + network.beta
    output = 1.0 / (1.0 + np.exp(-(normed_output @ network.w3 + network.b3)))
    return ForwardPass(inputs, z1, h1, z2, normalised, inverse_std, normed_output, output[:, 0])


def gradients(network: Network, passed: ForwardPass, targets: Array) -> Network:
    """Gradients of the mean binary cross-entropy with respect to every weight."""
    batch = len(targets)
    d_logit = ((passed.output - targets) / batch)[:, None]
    d_w3 = passed.normed_output.T @ d_logit
    d_b3 = d_logit.sum(axis=0)
    d_normed = d_logit @ network.w3.T
    d_gamma = (d_normed * passed.normalised).sum(axis=0)
    d_beta = d_normed.sum(axis=0)
    d_hat = d_normed * network.gamma
    d_h2 = passed.inverse_std / batch * (
        batch * d_hat - d_hat.sum(axis=0) - passed.normalised * (d_hat * passed.normalised).sum(axis=0)
    )
    d_z2 = d_h2 * (passed.z2 > 0)
    d_w2 = passed.h1.T @ d_z2
    d_b2 = d_z2.sum(axis=0)
    d_z1 = (d_z2 @ network.w2.T) * (passed.z1 > 0)
    d_w1 = passed.inputs.T @ d_z1
    d_b1 = d_z1.sum(axis=0)
    return Network(d_w1, d_b1, d_w2, d_b2, d_gamma, d_beta, d_w3, d_b3)


@dataclass(frozen=True)
class Adam:
    """Adam optimiser state (Kingma & Ba, 2015), with PyTorch's default settings."""

    first: Network
    second: Network
    steps: int


ADAM_LEARNING_RATE = 0.001
ADAM_BETAS = (0.9, 0.999)
ADAM_EPSILON = 1e-8


def adam_start(network: Network) -> Adam:
    zeros = network_map(np.zeros_like, network)
    return Adam(zeros, zeros, 0)


def adam_step(network: Network, gradient: Network, state: Adam) -> tuple[Network, Adam]:
    beta1, beta2 = ADAM_BETAS
    steps = state.steps + 1
    first = network_map(lambda m, g: beta1 * m + (1 - beta1) * g, state.first, gradient)
    second = network_map(lambda v, g: beta2 * v + (1 - beta2) * g * g, state.second, gradient)
    correction1, correction2 = 1 - beta1**steps, 1 - beta2**steps
    updated = network_map(
        lambda w, m, v: w - ADAM_LEARNING_RATE * (m / correction1) / (np.sqrt(v / correction2) + ADAM_EPSILON),
        network,
        first,
        second,
    )
    return updated, Adam(first, second, steps)


def trained(network: Network, structures: Bits, judgments: Bits, epochs: int, batch_size: int, rng: np.random.Generator) -> Network:
    """Train on <structure, judgment> pairs: `epochs` passes, reshuffled each time."""
    state = adam_start(network)
    for _ in range(epochs):
        order = rng.permutation(len(structures))
        for batch in range(len(structures) // batch_size):
            rows = order[batch * batch_size : (batch + 1) * batch_size]
            passed = forward(network, structures[rows])
            network, state = adam_step(network, gradients(network, passed, judgments[rows].astype(np.float64)), state)
    return network


def judgments(network: Network, structures: Bits) -> Bits:
    """Production: the rounded output for each structure (one batch)."""
    return np.round(forward(network, structures).output).astype(np.int64)


def structures_as_produced(structures: Bits, rng: np.random.Generator) -> Bits:
    """Experiment 1: the child sees the structures the parent judged."""
    return structures


def structures_shuffled(structures: Bits, rng: np.random.Generator) -> Bits:
    """Experiment 2: the child sees each structure with its bits shuffled."""
    return rng.permuted(structures, axis=1)


# ---------- measures (Carcassi et al., 2021, sections 3.1.5, 4.1.2 and Appendix A) ----------

# SUBSTRUCTURE[i, j]: structure j is a substructure of structure i (j's distant resources are among i's).
SUBSTRUCTURE: NDArray[np.float64] = ((ALL_STRUCTURES[None, :, :] & (1 - ALL_STRUCTURES[:, None, :])).sum(axis=2) == 0).astype(
    np.float64
)


def entropy(probabilities: Array) -> float:
    positive = probabilities[probabilities > 0]
    return float(-(positive * np.log2(positive)).sum())


def upward_monotonicity(quantifier: Bits, substructure: NDArray[np.float64]) -> float:
    """1 - H(1_Q | 1_Q<) / H(1_Q), where 1_Q< says whether a structure has a true substructure.
    A degenerate quantifier gets 1, as in the reference code."""
    if quantifier.min() == quantifier.max():
        return 1.0
    has_true_sub = (substructure @ quantifier) > 0
    joint = np.array(
        [[np.mean((quantifier == q) & (has_true_sub == t)) for q in (0, 1)] for t in (False, True)]
    )  # rows: has true substructure; columns: truth value
    conditional = sum(entropy(row / row.sum()) * row.sum() for row in joint if row.sum() > 0)
    return 1.0 - conditional / entropy(np.array([1 - quantifier.mean(), quantifier.mean()]))


def monotonicity(quantifier: Bits) -> float:
    """The highest upward monotonicity among the four readings of the bits (Appendix A): truth and
    falsity of the output swapped or not, and of the structures swapped or not (the latter reverses
    the substructure order, which covers downward monotonicity)."""
    return max(
        upward_monotonicity(quantifier, SUBSTRUCTURE),
        upward_monotonicity(1 - quantifier, SUBSTRUCTURE),
        upward_monotonicity(1 - quantifier, SUBSTRUCTURE.T),
        upward_monotonicity(quantifier, SUBSTRUCTURE.T),
    )


def quantity(quantifier: Bits) -> float:
    """1 - H(1_Q | number of distant resources) / H(1_Q); degenerate quantifiers get 1."""
    if quantifier.min() == quantifier.max():
        return 1.0
    counts = ALL_STRUCTURES.sum(axis=1)
    conditional = 0.0
    for k in range(STRUCTURE_SIZE + 1):
        share_true = quantifier[counts == k].mean()
        conditional += np.mean(counts == k) * entropy(np.array([share_true, 1 - share_true]))
    return 1.0 - conditional / entropy(np.array([1 - quantifier.mean(), quantifier.mean()]))


def is_degenerate(quantifier: Bits) -> bool:
    return bool(quantifier.min() == quantifier.max())


def is_ultrafilter(quantifier: Bits) -> bool:
    """True iff the truth value depends only on whether one particular resource is distant (or not)."""
    return bool(np.any(np.all(ALL_STRUCTURES == quantifier[:, None], axis=0) | np.all(ALL_STRUCTURES != quantifier[:, None], axis=0)))


# ---------- agent-based model ----------


@dataclass(frozen=True)
class Parameters:
    """All parameters. No parameter has a default."""

    agents_per_generation: int  # 10 in Carcassi et al. (2021)
    bottleneck: int  # structures per child: 200, 512, 715 or 1024
    epochs: int  # 4 or 8
    batch_size: int  # 32 in the reference code

    def __post_init__(self) -> None:
        if self.agents_per_generation < 1 or self.bottleneck < self.batch_size or self.epochs < 1 or self.batch_size < 2:
            raise ValueError(
                "agents_per_generation >= 1, epochs >= 1, batch_size >= 2 and bottleneck >= batch_size are required; got "
                f"{self.agents_per_generation}, {self.epochs}, {self.batch_size}, {self.bottleneck}."
            )


@dataclass(frozen=True)
class Record:
    """One row per generation; field names are the output columns."""

    monotonicity: float  # mean over agents
    quantity: float  # mean over agents
    degenerate: float  # share of agents
    ultrafilter: float  # share of agents
    perfectly_monotone: float  # share of agents with monotonicity 1 that are not degenerate


def generation_record(quantifiers: list[Bits]) -> Record:
    mono = [monotonicity(q) for q in quantifiers]
    return Record(
        float(np.mean(mono)),
        float(np.mean([quantity(q) for q in quantifiers])),
        float(np.mean([is_degenerate(q) for q in quantifiers])),
        float(np.mean([is_ultrafilter(q) for q in quantifiers])),
        float(np.mean([m > 1 - 1e-9 and not is_degenerate(q) for m, q in zip(mono, quantifiers, strict=True)])),
    )


def latest(field_name: str, model: "QuantifierModel") -> float:
    """DataCollector reporter. Mesa passes the model only to lambdas and functools.partial objects."""
    return float(getattr(model.latest_record, field_name))


class Speaker(mesa.Agent):
    def __init__(self, model: "QuantifierModel", network: Network) -> None:
        super().__init__(model)
        self.network: Network = network


class QuantifierModel(mesa.Model):
    """One step = one generation of iterated learning."""

    def __init__(self, parameters: Parameters, transmission: Transmission, rng: int) -> None:
        super().__init__(rng=rng)
        self.parameters: Parameters = parameters
        self.transmission: Transmission = transmission
        self.generation: list[Speaker] = [Speaker(self, initial_network(self.rng)) for _ in range(parameters.agents_per_generation)]
        self.latest_record: Record = self.measured()
        reporters = {field.name: partial(latest, field.name) for field in fields(Record)}
        self.datacollector: mesa.DataCollector = mesa.DataCollector(model_reporters={"generation": "steps", **reporters})
        self.datacollector.collect(self)

    def quantifiers(self) -> list[Bits]:
        return [judgments(agent.network, ALL_STRUCTURES) for agent in self.generation]

    def measured(self) -> Record:
        return generation_record(self.quantifiers())

    def child_of(self, parent: Speaker) -> Speaker:
        p = self.parameters
        structures = self.rng.integers(0, 2, size=(p.bottleneck, STRUCTURE_SIZE))
        parent_judgments = judgments(parent.network, structures)
        seen = self.transmission(structures, self.rng)
        return Speaker(self, trained(initial_network(self.rng), seen, parent_judgments, p.epochs, p.batch_size, self.rng))

    def step(self) -> None:
        parents = [self.generation[int(i)] for i in self.rng.integers(len(self.generation), size=len(self.generation))]
        children = [self.child_of(parent) for parent in parents]
        for agent in self.generation:
            agent.remove()
        self.generation = children
        self.latest_record = self.measured()
        self.datacollector.collect(self)
