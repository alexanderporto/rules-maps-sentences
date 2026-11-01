"""Computational model: Markov-chain analysis of the same rules (Griffiths & Kalish, 2007).

Same rules and values as the agent-based model: epsilon = 0.05, gamma in {0, 100}, beta in {0, 2}.
Bottleneck: 20 utterances per learner, the expected number an agent in the agent-based model hears
before it becomes the oldest (20 interactions per step, 9 possible learners, 9 steps).
Each row of the transition matrix is estimated from 4000 data sets (Monte Carlo standard error
of each entry at most 0.008). Initial distribution: uniform over the 16 holistic languages.
The vocal prior and the map prior differ only in which expressions count as iconic; the record
of maps is a population-level feature and is in the agent-based model only.

Outputs in results/: chain_trajectories.csv, chain_long_run.csv
"""

import os
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from iconographers.chain import long_run_distribution, trajectory, transition_matrix, unit_eigenvalue_count
from iconographers.languages import (
    COMPOSITIONAL,
    HOLISTIC,
    LANGUAGE_TYPES,
    MAP_ICONIC_COUNTS,
    N_MEANINGS,
    TYPE_NAMES,
    VOCAL_ICONIC_COUNTS,
    log_prior,
)
from iconographers.learning import ambiguity_penalties, log_likelihoods

RESULTS = Path(__file__).resolve().parent.parent / "results"
GENERATIONS = 2000
LONG_RUN = 100_000  # generations for the long-run distribution
EPSILON = 0.05
BOTTLENECK = 20
DATASETS_PER_ROW = 4000
CHANNELS = {"vocal": VOCAL_ICONIC_COUNTS, "map": MAP_ICONIC_COUNTS}


def summarised(distribution: np.ndarray, iconic_counts: np.ndarray) -> dict[str, float]:
    types = np.bincount(LANGUAGE_TYPES, weights=distribution, minlength=4)
    iconic_compositional = (LANGUAGE_TYPES == COMPOSITIONAL) & (iconic_counts == N_MEANINGS)
    return {
        **{name: float(share) for name, share in zip(TYPE_NAMES, types, strict=True)},
        "iconicity": float(distribution @ iconic_counts) / N_MEANINGS,
        "iconic_compositional": float(distribution[iconic_compositional].sum()),
    }


def analyse(setting: tuple[float, float, str, int]) -> tuple[pd.DataFrame, dict[str, float | str]]:
    gamma, beta, channel, seed = setting
    counts = CHANNELS[channel]
    rng = np.random.default_rng(seed)
    q = transition_matrix(
        log_prior(counts, beta), log_likelihoods(EPSILON), ambiguity_penalties(gamma), EPSILON, BOTTLENECK, DATASETS_PER_ROW, rng
    )
    holistic = (LANGUAGE_TYPES == HOLISTIC).astype(np.float64)
    initial = holistic / holistic.sum()
    path = trajectory(initial, q, GENERATIONS)
    rows = pd.DataFrame([summarised(p, counts) for p in path]).assign(generation=range(GENERATIONS + 1))
    labels = {"gamma": gamma, "beta": beta, "channel": channel}
    long_run = summarised(long_run_distribution(initial, q, LONG_RUN), counts)
    return rows.assign(**labels), {**labels, "unit_eigenvalues": unit_eigenvalue_count(q), **long_run}


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    count = os.cpu_count()
    if count is None:
        raise RuntimeError("os.cpu_count() could not determine the number of CPUs.")
    settings = [(*setting, seed) for seed, setting in enumerate(product((0.0, 100.0), (0.0, 2.0), tuple(CHANNELS)))]
    with ProcessPoolExecutor(max_workers=max(1, count - 1)) as executor:
        results = list(executor.map(analyse, settings))
    pd.concat([r[0] for r in results], ignore_index=True).to_csv(RESULTS / "chain_trajectories.csv", index=False)
    long_run = pd.DataFrame([r[1] for r in results]).round(3)
    long_run.to_csv(RESULTS / "chain_long_run.csv", index=False)
    print(f"Long-run distributions ({LONG_RUN} generations from the holistic start), with the number of")
    print("unit eigenvalues (> 1: several absorbing sets, so no single stationary distribution):")
    print(long_run.to_string(index=False))


if __name__ == "__main__":
    main()
