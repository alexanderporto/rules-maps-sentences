"""Negation design: the basic negation learning setup (Steinert-Threlkeld, 2017, section 2.5.2).

The 4 resources, each present or absent (n = 4, 8 states). 100 trials of 1000 plays for each of the
thesis's two conditions: the receiver's 'new word' interpretation of the function word (the
constant function c_i) points to a present state (i in X; here 'resource 0 present') or to an
absent state (i in f[X]; here 'resource 0 absent').

Published values for n = 4 (Table 2.2): expected payoff 0.978 and probability of interpreting the
function word as negation 0.995 (i in X); 0.870 and 0.807 (i in f[X]).

Outputs in results/: negation_mean_trajectories.csv, negation_summary.csv
"""

from pathlib import Path

import pandas as pd

from iconographers.negation import NegationModel, Parameters

RESULTS = Path(__file__).resolve().parent.parent / "results"
PLAYS = 1000
TRIALS = tuple(range(100))
CONDITIONS = {"constant in X (resource 0 present)": 0, "constant in f[X] (resource 0 absent)": 4}


def run(condition: str, seed: int) -> pd.DataFrame:
    model = NegationModel(Parameters(constant_target=CONDITIONS[condition]), rng=seed)
    for _ in range(PLAYS):
        model.step()
    return model.datacollector.get_model_vars_dataframe().assign(condition=condition, trial=seed)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    series = pd.concat([run(condition, seed) for condition in CONDITIONS for seed in TRIALS], ignore_index=True)
    measures = ["expected_payoff", "negation_probability", "sender_uses_negation"]
    series.groupby(["condition", "play"])[measures].mean().to_csv(RESULTS / "negation_mean_trajectories.csv")
    final = series[series["play"] == PLAYS].groupby("condition")[measures].agg(["mean", "std"]).round(3)
    final.to_csv(RESULTS / "negation_summary.csv")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(f"After {PLAYS} plays, mean (sd) over {len(TRIALS)} trials:")
        print(final.to_string())


if __name__ == "__main__":
    main()
