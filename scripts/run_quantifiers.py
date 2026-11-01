"""Quantifier design: Carcassi, Steinert-Threlkeld & Szymanik (2021), Experiments 1 and 2.

A = the 10 resources in view, B = the distant ones. 10 agents per generation, 300 generations,
20 trials for each bottleneck size (200, 512, 715, 1024) and number of epochs (4, 8), as in the
paper. Experiment 1: children see the structures the parent judged. Experiment 2: children see each
structure with its bits shuffled, so that quantifiers cannot track particular resources.

Published results: monotone quantifiers evolve for intermediate amounts of training (Experiment 1,
mostly ultrafilters, which name one resource); with shuffling, quantifiers are mostly both monotone
and quantitative, or degenerate (Experiment 2).

Outputs in results/: quantifier_mean_trajectories.csv, quantifier_summary.csv
"""

import os
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import pandas as pd

from iconographers.quantifiers import Parameters, QuantifierModel, structures_as_produced, structures_shuffled

RESULTS = Path(__file__).resolve().parent.parent / "results"
GENERATIONS = 300
TRIALS = tuple(range(20))
FINAL_WINDOW = 50
EXPERIMENTS = {"1 (as produced)": structures_as_produced, "2 (shuffled)": structures_shuffled}


def run(setting: tuple[str, int, int, int]) -> pd.DataFrame:
    experiment, bottleneck, epochs, seed = setting
    parameters = Parameters(agents_per_generation=10, bottleneck=bottleneck, epochs=epochs, batch_size=32)
    model = QuantifierModel(parameters, EXPERIMENTS[experiment], rng=seed)
    for _ in range(GENERATIONS):
        model.step()
    frame = model.datacollector.get_model_vars_dataframe()
    return frame.assign(experiment=experiment, bottleneck=bottleneck, epochs=epochs, trial=seed)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    count = os.cpu_count()
    if count is None:
        raise RuntimeError("os.cpu_count() could not determine the number of CPUs.")
    settings = list(product(tuple(EXPERIMENTS), (200, 512, 715, 1024), (4, 8), TRIALS))
    with ProcessPoolExecutor(max_workers=max(1, count - 1)) as executor:
        series = pd.concat(list(executor.map(run, settings)), ignore_index=True)
    keys = ["experiment", "bottleneck", "epochs"]
    measures = ["monotonicity", "quantity", "degenerate", "ultrafilter", "perfectly_monotone"]
    series.groupby(keys + ["generation"])[measures].mean().to_csv(RESULTS / "quantifier_mean_trajectories.csv")
    final = series[series["generation"] > GENERATIONS - FINAL_WINDOW].groupby(keys + ["trial"])[measures].mean()
    summary = final.groupby(keys).agg(["mean", "std"]).round(3)
    summary.to_csv(RESULTS / "quantifier_summary.csv")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(f"Generation 0 (random networks), mean over all runs:")
        print(series[series["generation"] == 0][measures].mean().round(3).to_string())
        print(f"\nLast {FINAL_WINDOW} generations, mean (sd) over {len(TRIALS)} trials:")
        print(summary.to_string())


if __name__ == "__main__":
    main()
