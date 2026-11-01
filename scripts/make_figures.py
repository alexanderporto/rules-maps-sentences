"""Figures comparing the Python (Mesa) and NetLogo implementations of the three submodels.

Reads the Python outputs (run_abm.py, run_negation.py, run_quantifiers.py) and the NetLogo summaries
(summarise_netlogo.py) from results/, and writes PNG files to results/figures/. In every figure the
colour identifies the measure, solid lines are Mesa and dashed lines NetLogo.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

RESULTS = Path(__file__).resolve().parent.parent / "results"
FIGURES = RESULTS / "figures"

# Pastel and dark earth-tone blue, dark green, black and grey only (the author's specification).
# Each set below passes the colour-vision-deficiency and normal-vision separation checks of the
# dataviz palette validator for adjacent colours; line colours reach 3:1 contrast on the surface.
LINE_BLUE, DARK_GREEN, BLACK = "#4a7fb5", "#2d5a3d", "#1f1f1f"
DARK_BLUE, PASTEL_BLUE, GREY, LIGHT_GREY = "#2b4c6f", "#a9c4e0", "#8c8c8a", "#d6d5cf"
SURFACE, INK, INK_SECONDARY, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
IMPLEMENTATIONS = {"Mesa (Python)": "-", "NetLogo": "--"}

# Published values: Steinert-Threlkeld (2017), Table 2.2, n = 4.
PUBLISHED_NEGATION = {
    "constant in X (resource 0 present)": {"expected_payoff": 0.978, "negation_probability": 0.995},
    "constant in f[X] (resource 0 absent)": {"expected_payoff": 0.870, "negation_probability": 0.807},
}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 9,
        "text.color": INK,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": INK_SECONDARY,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "lines.linewidth": 1.8,
    }
)


def read_trajectories(name: str) -> dict[str, pd.DataFrame]:
    return {
        "Mesa (Python)": pd.read_csv(RESULTS / f"{name}_mean_trajectories.csv"),
        "NetLogo": pd.read_csv(RESULTS / f"netlogo_{name}_mean_trajectories.csv"),
    }


def read_summaries(name: str, index_columns: int) -> dict[str, pd.DataFrame]:
    def means(path: Path) -> pd.DataFrame:
        frame = pd.read_csv(path, header=[0, 1], index_col=list(range(index_columns)))
        return frame.xs("mean", axis=1, level=1)

    return {
        "Mesa (Python)": means(RESULTS / f"{name}_summary.csv"),
        "NetLogo": means(RESULTS / f"netlogo_{name}_summary.csv"),
    }


def line_legend(figure: Figure, measures: dict[str, str], extra: list[Line2D]) -> None:
    handles = [Line2D([], [], color=colour, label=label) for label, colour in measures.items()]
    handles += [Line2D([], [], color=GREY, linestyle=style, label=name) for name, style in IMPLEMENTATIONS.items()]
    entries = handles + extra
    figure.legend(handles=entries, loc="outside lower center", ncol=len(entries) if len(entries) <= 4 else 3)


def plot_measures(axes: Axes, frames: dict[str, pd.DataFrame], x: str, measures: dict[str, str]) -> None:
    """Draw each measure (column -> colour) for each implementation."""
    for implementation, frame in frames.items():
        for column, colour in measures.items():
            axes.plot(frame[x], frame[column], color=colour, linestyle=IMPLEMENTATIONS[implementation])


def compositional_trajectories() -> Figure:
    frames = read_trajectories("abm")
    figure, grid = plt.subplots(2, 2, figsize=(8, 5.4), sharex=True, sharey=True, layout="constrained")
    for row, (channel, channel_label) in enumerate((("vocal", "Vocalisations"), ("map", "Maps"))):
        for col, beta in enumerate((0.0, 2.0)):
            axes = grid[row][col]
            selected = {k: f[(f["gamma"] == 100.0) & (f["beta"] == beta)] for k, f in frames.items()}
            plot_measures(axes, selected, "step", {f"{channel}_compositional": LINE_BLUE, f"{channel}_iconic_compositional": DARK_GREEN})
            axes.set_title(f"{channel_label}, {'no iconic bias (β = 0)' if beta == 0 else 'iconic bias (β = 2)'}", loc="left")
            axes.set_ylim(0, 1)
            axes.set_xlim(0, 2000)
        grid[row][0].set_ylabel("Probability (oldest agent)")
    for axes in grid[1]:
        axes.set_xlabel("Step (one agent replaced per step)")
    figure.suptitle("A. Compositional languages emerge, and some keep the iconic mapping (γ = 100)", x=0.02, ha="left", fontweight="bold")
    line_legend(figure, {"Compositional": LINE_BLUE, "Fully iconic compositional": DARK_GREEN}, [])
    return figure


def final_languages() -> Figure:
    summaries = read_summaries("abm", 2)
    segments = {"Degenerate": BLACK, "Holistic": GREY, "Compositional, not fully iconic": PASTEL_BLUE, "Fully iconic compositional": DARK_GREEN, "Other": LIGHT_GREY}
    figure, panels = plt.subplots(1, 2, figsize=(8.5, 4.4), sharey=True, layout="constrained")
    conditions = [(0.0, 0.0), (0.0, 2.0), (100.0, 0.0), (100.0, 2.0)]
    labels = [f"γ = {g:g}, β = {b:g}  {name}" for g, b in conditions for name in IMPLEMENTATIONS]
    for axes, (channel, title) in zip(panels, (("vocal", "Vocalisations"), ("map", "Maps")), strict=True):
        for position, ((gamma, beta), implementation) in enumerate((c, i) for c in conditions for i in IMPLEMENTATIONS):
            row = summaries[implementation].loc[(gamma, beta)]
            values = [
                row[f"{channel}_degenerate"],
                row[f"{channel}_holistic"],
                row[f"{channel}_compositional"] - row[f"{channel}_iconic_compositional"],
                row[f"{channel}_iconic_compositional"],
                row[f"{channel}_other"],
            ]
            y = position + (position // 2) * 0.6  # gap between conditions
            left = 0.0
            for value, colour in zip(values, segments.values(), strict=True):
                axes.barh(y, value, left=left, height=0.8, color=colour, edgecolor=SURFACE, linewidth=1)
                left += value
        axes.set_title(title, loc="left")
        axes.set_xlim(0, 1)
        axes.set_xlabel("Probability (oldest agent, mean of last 200 steps)")
        axes.grid(axis="x")
        axes.grid(axis="y", visible=False)
    positions = [p + (p // 2) * 0.6 for p in range(len(labels))]
    panels[0].set_yticks(positions, labels)
    panels[0].invert_yaxis()
    figure.suptitle("A. Where the languages end up: Mesa and NetLogo, condition by condition", x=0.02, ha="left", fontweight="bold")
    figure.legend(handles=[Patch(color=c, label=l) for l, c in segments.items()], loc="outside lower center", ncol=3)
    return figure


def negation_learning() -> Figure:
    frames = {k: f.rename(columns={"play": "step"}) for k, f in read_trajectories("negation").items()}
    measures = {"expected_payoff": LINE_BLUE, "negation_probability": DARK_GREEN, "sender_uses_negation": BLACK}
    figure, panels = plt.subplots(1, 2, figsize=(8, 3.6), sharey=True, layout="constrained")
    titles = {
        "constant in X (resource 0 present)": "Alternative reading points to a present state",
        "constant in f[X] (resource 0 absent)": "Alternative reading points to an absent state",
    }
    for axes, (condition, title) in zip(panels, titles.items(), strict=True):
        plot_measures(axes, {k: f[f["condition"] == condition] for k, f in frames.items()}, "step", measures)
        for column, value in PUBLISHED_NEGATION[condition].items():
            axes.plot([1000], [value], marker="o", markersize=8, color=measures[column], markeredgecolor=INK, markeredgewidth=1.2, clip_on=False, zorder=5)
        axes.set_title(title, loc="left")
        axes.set_xlim(0, 1000)
        axes.set_ylim(0, 1)
        axes.set_xlabel("Play")
    panels[0].set_ylabel("Probability (mean of 100 trials)")
    figure.suptitle("B. Learning to use a function word as negation", x=0.02, ha="left", fontweight="bold")
    published = Line2D([], [], linestyle="none", marker="o", markersize=7, color=SURFACE, markeredgecolor=INK, label="Published (play 1000)")
    line_legend(figure, {"Communication success": LINE_BLUE, "Listener reads a as 'not'": DARK_GREEN, "Speaker uses a + word": BLACK}, [published])
    return figure


def quantifier_trajectories() -> Figure:
    frames = read_trajectories("quantifier")
    figure, grid = plt.subplots(2, 4, figsize=(10, 5.2), sharex=True, sharey=True, layout="constrained")
    for row, experiment in enumerate(("1 (as produced)", "2 (shuffled)")):
        for col, bottleneck in enumerate((200, 512, 715, 1024)):
            axes = grid[row][col]
            selected = {k: f[(f["experiment"] == experiment) & (f["bottleneck"] == bottleneck) & (f["epochs"] == 8)] for k, f in frames.items()}
            plot_measures(axes, selected, "generation", {"monotonicity": LINE_BLUE, "quantity": DARK_GREEN})
            axes.set_title(f"{bottleneck} examples", loc="left")
            axes.set_ylim(0, 1)
            axes.set_xlim(0, 300)
        grid[row][0].set_ylabel(f"Experiment {experiment}\nmean of 20 trials")
    for axes in grid[1]:
        axes.set_xlabel("Generation")
    figure.suptitle("C. Quantifiers become monotone; with shuffled input they come to count (8 epochs)", x=0.02, ha="left", fontweight="bold")
    line_legend(figure, {"Monotonicity": LINE_BLUE, "Quantity": DARK_GREEN}, [])
    return figure


def quantifier_types() -> Figure:
    summaries = read_summaries("quantifier", 3)
    segments = {"Ultrafilter ('resource k is distant')": DARK_BLUE, "Other perfectly monotone": PASTEL_BLUE, "Degenerate (always true / false)": BLACK, "Not perfectly monotone": LIGHT_GREY}
    settings = [(b, e) for b in (200, 512, 715, 1024) for e in (4, 8)]
    figure, panels = plt.subplots(1, 2, figsize=(9, 5.6), sharey=True, layout="constrained")
    labels = [f"{b} ex., {e} ep.  {name.split(' ')[0]}" for b, e in settings for name in IMPLEMENTATIONS]
    positions = [p + (p // 2) * 0.6 for p in range(len(labels))]
    for axes, experiment in zip(panels, ("1 (as produced)", "2 (shuffled)"), strict=True):
        for y, ((bottleneck, epochs), implementation) in zip(positions, ((s, i) for s in settings for i in IMPLEMENTATIONS), strict=True):
            row = summaries[implementation].loc[(experiment, bottleneck, epochs)]
            values = [
                row["ultrafilter"],
                row["perfectly_monotone"] - row["ultrafilter"],
                row["degenerate"],
                1.0 - row["perfectly_monotone"] - row["degenerate"],
            ]
            left = 0.0
            for value, colour in zip(values, segments.values(), strict=True):
                axes.barh(y, max(value, 0.0), left=left, height=0.8, color=colour, edgecolor=SURFACE, linewidth=1)
                left += max(value, 0.0)
        axes.set_title(f"Experiment {experiment}", loc="left")
        axes.set_xlim(0, 1)
        axes.set_xlabel("Share of agents (mean of last 50 generations)")
        axes.grid(axis="x")
        axes.grid(axis="y", visible=False)
    panels[0].set_yticks(positions, labels)
    panels[0].invert_yaxis()
    figure.suptitle("C. What kinds of quantifier the agents end up with", x=0.02, ha="left", fontweight="bold")
    figure.legend(handles=[Patch(color=c, label=l) for l, c in segments.items()], loc="outside lower center", ncol=2)
    return figure


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    for name, make in (
        ("a1_compositional_trajectories", compositional_trajectories),
        ("a2_final_languages", final_languages),
        ("b_negation_learning", negation_learning),
        ("c1_quantifier_trajectories", quantifier_trajectories),
        ("c2_quantifier_types", quantifier_types),
    ):
        figure = make()
        figure.savefig(FIGURES / f"{name}.png", dpi=200)
        plt.close(figure)
        print(f"wrote results/figures/{name}.png")


if __name__ == "__main__":
    main()
