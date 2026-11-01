"""Summaries of the NetLogo designs, computed exactly as for the Python designs.

Reads results/netlogo_<model>_table.csv.gz (written by scripts/run_netlogo.sh) and writes, with the
column names of the Python outputs:
    netlogo_abm_mean_trajectories.csv, netlogo_abm_summary.csv               (run_abm.py: last 200 steps)
    netlogo_negation_mean_trajectories.csv, netlogo_negation_summary.csv     (run_negation.py: play 1000)
    netlogo_quantifier_mean_trajectories.csv, netlogo_quantifier_summary.csv (run_quantifiers.py: last 50 generations)
NetLogo records submodel A every 10 steps and every step after step 1800, and submodel B every 10
plays, so its trajectories have fewer points than the Python ones; the summary windows are complete.
"""

from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"
TABLE_HEADER_ROWS = 6  # BehaviorSpace table format: six lines of metadata before the column names

ABM_COLUMNS = {
    "communicative-rationality": "gamma",
    "iconic-bits": "beta",
    "[step]": "step",
    "[run number]": "run",
    **{
        f"type-mass teacher-{channel} {index}": f"{prefix}_{name}"
        for channel, prefix in (("vocal", "vocal"), ("maps", "map"))
        for index, name in enumerate(("degenerate", "holistic", "other", "compositional"))
    },
    "iconicity teacher-vocal vocal-iconic-counts": "vocal_iconicity",
    "iconic-compositional teacher-vocal vocal-iconic-counts": "vocal_iconic_compositional",
    "iconicity teacher-maps map-iconic-counts": "map_iconicity",
    "iconic-compositional teacher-maps map-iconic-counts": "map_iconic_compositional",
}
NEGATION_COLUMNS = {
    "constant-points-to": "condition",
    "[step]": "play",
    "[run number]": "trial",
    "expected-payoff": "expected_payoff",
    "item 2 function-probabilities": "negation_probability",
    "sender-uses-negation": "sender_uses_negation",
}
NEGATION_CONDITIONS = {
    "resource 0 present": "constant in X (resource 0 present)",
    "resource 0 absent": "constant in f[X] (resource 0 absent)",
}
QUANTIFIER_MEASURES = ("monotonicity", "quantity", "degenerate", "ultrafilter", "perfectly_monotone")
QUANTIFIER_COLUMNS = {
    "experiment": "experiment",
    "bottleneck": "bottleneck",
    "epochs": "epochs",
    "[step]": "generation",
    "[run number]": "trial",
    **{f"item {index} measures": name for index, name in enumerate(QUANTIFIER_MEASURES)},
}
QUANTIFIER_EXPERIMENTS = {"1: as produced": "1 (as produced)", "2: shuffled": "2 (shuffled)"}


def table(model: str, columns: dict[str, str]) -> pd.DataFrame:
    path = RESULTS / f"netlogo_{model}_table.csv.gz"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run scripts/run_netlogo.sh first.")
    frame = pd.read_csv(path, skiprows=TABLE_HEADER_ROWS)
    missing = set(columns) - set(frame.columns)
    if missing:
        raise KeyError(f"{path} lacks the columns {sorted(missing)}.")
    return frame[list(columns)].rename(columns=columns)


def summarised(series: pd.DataFrame, keys: list[str], unit: str, measures: list[str]) -> pd.DataFrame:
    return series.groupby(keys + [unit])[measures].mean().groupby(keys).agg(["mean", "std"]).round(3)


def abm() -> None:
    series = table("iconographers", ABM_COLUMNS)
    measures = [c for c in series.columns if c.startswith(("vocal_", "map_"))]
    keys = ["gamma", "beta"]
    series.groupby(keys + ["step"])[measures].mean().to_csv(RESULTS / "netlogo_abm_mean_trajectories.csv")
    summary = summarised(series[series["step"] > 1800], keys, "run", measures)
    summary.to_csv(RESULTS / "netlogo_abm_summary.csv")
    print("Submodel A, oldest agent's posterior over the last 200 steps, mean (sd) over runs:")
    print(summary.to_string())


def negation() -> None:
    series = table("negation", NEGATION_COLUMNS).assign(condition=lambda f: f["condition"].map(NEGATION_CONDITIONS))
    measures = ["expected_payoff", "negation_probability", "sender_uses_negation"]
    series.groupby(["condition", "play"])[measures].mean().to_csv(RESULTS / "netlogo_negation_mean_trajectories.csv")
    summary = series[series["play"] == 1000].groupby("condition")[measures].agg(["mean", "std"]).round(3)
    summary.to_csv(RESULTS / "netlogo_negation_summary.csv")
    print("Submodel B, after 1000 plays, mean (sd) over trials:")
    print(summary.to_string())


def quantifiers() -> None:
    series = table("quantifiers", QUANTIFIER_COLUMNS).assign(
        experiment=lambda f: f["experiment"].map(QUANTIFIER_EXPERIMENTS)
    )
    measures = list(QUANTIFIER_MEASURES)
    keys = ["experiment", "bottleneck", "epochs"]
    series.groupby(keys + ["generation"])[measures].mean().to_csv(RESULTS / "netlogo_quantifier_mean_trajectories.csv")
    summary = summarised(series[series["generation"] > 250], keys, "trial", measures)
    summary.to_csv(RESULTS / "netlogo_quantifier_summary.csv")
    print("Submodel C, last 50 generations, mean (sd) over trials:")
    print("Generation 0:", series[series["generation"] == 0][measures].mean().round(3).to_dict())
    print(summary.to_string())


def main() -> None:
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        abm()
        negation()
        quantifiers()


if __name__ == "__main__":
    main()
