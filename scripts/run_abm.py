"""Agent-based design: communicative rationality gamma x iconic bits beta.

Settings from Kirby & Tamariz (2022): 10 agents, 20 interactions per step, founders trained on 50
observations of a holistic language, epsilon = 0.05, gamma in {0, 100}, the oldest agent teaches,
2000 steps, 100 runs per condition. Record of maps: the last 80 (Kirby et al., 2015, memory r = 80).
beta in {0, 2}: Kirby et al.'s prior unchanged, and each iconic expression saving 2 bits
(an illustrative value).

Predictions (descriptive; no inferential test):
    beta = 0 replicates the pipeline: degenerate languages without communicative rationality
        (gamma = 0); compositional languages emerging, slowly, with it (gamma = 100).
    beta = 2 raises iconicity. Because the only unambiguous fully iconic languages are
        compositional, it also favours compositional languages directly (this follows from the
        assumption). Among compositional languages, the iconic ones should dominate:
        abstract structure that is not arbitrary.
    Maps are learned from a record of maps made by earlier, older agents; Kirby & Tamariz (2022)
        found slower change when learners learn from older producers, so maps are expected to
        change more slowly than vocalisations. The direction is not fixed in advance.

Outputs in results/: abm_mean_trajectories.csv, abm_summary.csv
"""

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from itertools import product
from pathlib import Path

import pandas as pd

from iconographers.abm import IconographersModel, Parameters

RESULTS = Path(__file__).resolve().parent.parent / "results"
STEPS = 2000
RUNS = tuple(range(100))
FINAL_WINDOW = 200
BASE = Parameters(
    population_size=10,
    interactions_per_step=20,
    founder_training=50,
    record_size=80,
    error_probability=0.05,
    communicative_rationality=0.0,
    iconic_bits=0.0,
)


def run(setting: tuple[float, float, int]) -> pd.DataFrame:
    gamma, beta, seed = setting
    model = IconographersModel(replace(BASE, communicative_rationality=gamma, iconic_bits=beta), rng=seed)
    for _ in range(STEPS):
        model.step()
    frame = model.datacollector.get_model_vars_dataframe()
    return frame.assign(gamma=gamma, beta=beta, run=seed)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    count = os.cpu_count()
    if count is None:
        raise RuntimeError("os.cpu_count() could not determine the number of CPUs.")
    settings = list(product((0.0, 100.0), (0.0, 2.0), RUNS))
    with ProcessPoolExecutor(max_workers=max(1, count - 1)) as executor:
        series = pd.concat(list(executor.map(run, settings)), ignore_index=True)
    keys = ["gamma", "beta"]
    measures = [c for c in series.columns if c.startswith(("vocal_", "map_"))]
    series.groupby(keys + ["step"])[measures].mean().to_csv(RESULTS / "abm_mean_trajectories.csv")
    final = series[series["step"] > STEPS - FINAL_WINDOW].groupby(keys + ["run"])[measures].mean()
    summary = final.groupby(keys).agg(["mean", "std"]).round(3)
    summary.to_csv(RESULTS / "abm_summary.csv")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(f"Oldest agent's posterior, mean (sd) over {len(RUNS)} runs of the last {FINAL_WINDOW} steps")
        print(summary.to_string())


if __name__ == "__main__":
    main()
