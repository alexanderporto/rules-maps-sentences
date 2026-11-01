# Results (Python 2026-10-31, NetLogo 2026-11-01)

Comparisons are descriptive: no inferential statistic has been chosen. β = 2 is an illustrative value. Each agent-based design was run twice with the same settings, in Python (Mesa) and in NetLogo 7.0.4; in each section the NetLogo results follow the Python ones. Figures comparing the two are in `figures/`.

## Files

| File | Produced by | Content |
|---|---|---|
| `abm_mean_trajectories.csv` | `scripts/run_abm.py` | Per condition and step: mean over 100 runs of the oldest agent's posterior measures |
| `abm_summary.csv` | `scripts/run_abm.py` | Per condition: mean and SD over runs of the last-200-step averages |
| `abm_console_output.txt` | `scripts/run_abm.py` | Console output |
| `chain_trajectories.csv` | `scripts/run_chain.py` | Per condition and generation: distribution over production-language types |
| `chain_long_run.csv` | `scripts/run_chain.py` | Per condition: distribution after 100,000 generations, and the number of unit eigenvalues |
| `chain_console_output.txt` | `scripts/run_chain.py` | Console output |
| `negation_mean_trajectories.csv` | `scripts/run_negation.py` | Submodel B: per condition and play, mean over 100 trials of the three measures |
| `negation_summary.csv` | `scripts/run_negation.py` | Submodel B: per condition, mean and SD over trials after 1000 plays |
| `negation_console_output.txt` | `scripts/run_negation.py` | Console output |
| `quantifier_mean_trajectories.csv` | `scripts/run_quantifiers.py` | Submodel C: per experiment, bottleneck, epochs and generation, mean over 20 trials of the five measures |
| `quantifier_summary.csv` | `scripts/run_quantifiers.py` | Submodel C: per setting, mean and SD over trials of the last-50-generation averages |
| `quantifier_console_output.txt` | `scripts/run_quantifiers.py` | Console output |
| `netlogo_<model>_table.csv.gz` | `scripts/run_netlogo.sh` | Raw BehaviorSpace output for `iconographers` (submodel A), `negation` and `quantifiers`: one row per run and recorded step; gzip-compressed CSV with six lines of metadata before the column names |
| `netlogo_console_output.txt` | `scripts/run_netlogo.sh` | Start and end time of each NetLogo design |
| `netlogo_abm_*`, `netlogo_negation_*`, `netlogo_quantifier_*` | `scripts/summarise_netlogo.py` | The summaries of the Python files of the same name, computed in the same way from the NetLogo tables |
| `figures/*.png` | `scripts/make_figures.py` | Python and NetLogo compared: submodel A (`a1`, `a2`), B (`b`), C (`c1`, `c2`) |

Sections "Design" to "Findings" cover submodel A (compositional vocalisations and maps). Submodels B and C follow.

## Design

- **Agent-based model:** γ ∈ {0, 100} × β ∈ {0, 2}, 100 runs each, 2000 steps (200 complete replacements of the population). Other settings from Kirby & Tamariz (2022): 10 agents, 20 interactions per step, the oldest agent teaches, founders trained on 50 observations of a holistic language, ε = 0.05. Maps kept in a record of the last 80.
- **Computational model:** the same rules as a transmission chain, with 20 utterances per learner, for both priors (vocal, map). The transition matrices are estimated from 4,000 data sets per row.

**Chance levels.** If every language were equally likely, the share of iconic expressions would be 0.375 (vocalisations) and 0.25 (maps). Among compositional languages, 2 of 8 vocal and 1 of 8 map languages are fully iconic (25% and 12.5%).

## Agent-based model

Posterior of the oldest agent, averaged over the last 200 steps, then mean (SD) over 100 runs:

| γ | β | | Degenerate | Holistic | Compositional | Iconic compositional | Iconicity |
|---|---|---|---|---|---|---|---|
| 0 | 0 | vocal | 0.96 (0.20) | 0.00 | 0.04 (0.20) | 0.01 | 0.34 |
| | | map | 0.91 (0.29) | 0.00 | 0.09 (0.29) | 0.00 | 0.24 |
| 0 | 2 | vocal | 0.88 (0.33) | 0.00 | 0.08 (0.27) | 0.07 | 0.45 |
| | | map | 0.93 (0.26) | 0.00 | 0.07 (0.26) | 0.04 | 0.29 |
| 100 | 0 | vocal | 0.00 | 0.55 (0.37) | 0.31 (0.46) | 0.06 (0.24) | 0.34 |
| | | map | 0.00 | 0.63 (0.30) | 0.18 (0.38) | 0.02 (0.14) | 0.21 |
| 100 | 2 | vocal | 0.00 | 0.42 (0.39) | 0.46 (0.49) | 0.20 (0.40) | 0.53 |
| | | map | 0.00 | 0.53 (0.34) | 0.29 (0.45) | 0.05 (0.22) | 0.31 |

The large SDs reflect runs that end either in a compositional or in a holistic language.

Compositional languages over time (γ = 100, mean over runs):

| β | | Step 100 | 500 | 1000 | 2000 |
|---|---|---|---|---|---|
| 0 | vocal | 0.01 | 0.07 | 0.15 | 0.33 |
| | map | 0.00 | 0.07 | 0.09 | 0.21 |
| 2 | vocal | 0.04 | 0.11 | 0.31 | 0.47 |
| | map | 0.01 | 0.14 | 0.19 | 0.31 |

**NetLogo.** Same design and measure, mean (SD) over 100 runs:

| γ | β | | Degenerate | Holistic | Compositional | Iconic compositional | Iconicity |
|---|---|---|---|---|---|---|---|
| 0 | 0 | vocal | 0.94 (0.24) | 0.00 | 0.06 (0.24) | 0.02 (0.14) | 0.36 (0.16) |
| | | map | 0.94 (0.24) | 0.00 | 0.06 (0.24) | 0.00 | 0.24 (0.06) |
| 0 | 2 | vocal | 0.88 (0.33) | 0.00 | 0.12 (0.33) | 0.11 (0.31) | 0.49 (0.21) |
| | | map | 0.92 (0.27) | 0.00 | 0.08 (0.27) | 0.08 (0.27) | 0.31 (0.20) |
| 100 | 0 | vocal | 0.00 | 0.56 (0.36) | 0.29 (0.44) | 0.08 (0.26) | 0.39 (0.25) |
| | | map | 0.00 | 0.54 (0.35) | 0.30 (0.44) | 0.06 (0.24) | 0.31 (0.25) |
| 100 | 2 | vocal | 0.00 | 0.36 (0.39) | 0.53 (0.49) | 0.27 (0.44) | 0.58 (0.28) |
| | | map | 0.00 | 0.45 (0.37) | 0.41 (0.48) | 0.16 (0.37) | 0.40 (0.32) |

Compositional languages over time (γ = 100, mean over runs):

| β | | Step 100 | 500 | 1000 | 2000 |
|---|---|---|---|---|---|
| 0 | vocal | 0.00 | 0.06 | 0.11 | 0.31 |
| | map | 0.00 | 0.08 | 0.14 | 0.33 |
| 2 | vocal | 0.01 | 0.16 | 0.32 | 0.54 |
| | map | 0.04 | 0.14 | 0.27 | 0.42 |

**Agreement.** Most differences between the implementations are a few hundredths; the largest are for maps with γ = 100. Python reuses the seeds 0–99 in every condition, so all conditions start from the same 100 holistic languages, whose mean map iconicity is 0.21 (expected 0.25). NetLogo seeds each run separately; its starting languages for γ = 100, β = 0 have a mean map iconicity of 0.28. About half of the outcome is still holistic after 2000 steps, so the starting sample carries through. Differences in the compositional share of up to about 0.13 lie within seed-to-seed variation (twice the standard error of a difference between two means of 100 runs with SD ≈ 0.45).

## Computational model

Distribution over production-language types (γ = 100):

| β | | Gen. 200 compositional | Gen. 2000 compositional | Long run: compositional | Long run: iconic compositional | Long run: iconicity |
|---|---|---|---|---|---|---|
| 0 | vocal | 0.29 | 0.95 | 1.00 | 0.26 | 0.40 |
| | map | 0.28 | 0.95 | 1.00 | 0.13 | 0.26 |
| 2 | vocal | 0.25 | 0.90 | 1.00 | 0.56 | 0.74 |
| | map | 0.25 | 0.90 | 1.00 | 0.47 | 0.58 |

With γ = 0, degenerate languages take over (long run 1.00, except 0.93 for maps with β = 2). Every transition matrix has between 3 and 8 unit eigenvalues: degenerate or compositional languages, once reached, are in practice never left, so there is no single stationary distribution and the long run depends on the start (here, the holistic languages).

## Findings

**P1, replication: confirmed.** With β = 0, degenerate languages dominate without communicative rationality (agent-based model 0.96 and 0.91 in Python, 0.94 and 0.94 in NetLogo; computational model 1.00). With it, compositional languages emerge: slowly in the agent-based model, where the oldest agent teaches (0.33 vocal after 2000 steps in Python, 0.31 in NetLogo), as Kirby & Tamariz (2022) report for learning from the oldest; and fully in the long run of the computational model.

**P2, iconic and compositional: confirmed, partly by construction.** With β = 2 and γ = 100:
- *Iconicity* rises above chance: 0.53 against 0.34 (vocal) and 0.31 against 0.21 (maps) in the Python agent-based model, 0.58 against 0.39 and 0.40 against 0.31 in NetLogo; 0.74 and 0.58 in the long run of the computational model.
- *Iconic compositional languages* are over-represented among compositional ones. Agent-based model, Python: 0.20 of 0.46 (43%, against 25% by chance) for vocalisations, 0.05 of 0.29 (18%, against 12.5%) for maps; NetLogo: 0.27 of 0.53 (51%) and 0.16 of 0.41 (40%). Computational model: 56% and 47%, against 26% and 13% with β = 0. Compositional systems that emerge keep the iconic mapping more often than chance: abstract structure that is not arbitrary. This follows partly from the prior, which favours the fully iconic languages, and all of these are compositional.
- *Speed of compositionality: the two models disagree.* In the agent-based model, β = 2 brings compositional languages sooner, in both implementations (Python: 0.46 against 0.31 vocal, 0.29 against 0.18 maps; NetLogo: 0.53 against 0.29 and 0.41 against 0.30). In the computational model it makes them slightly slower (0.90 against 0.95 at generation 2000). The two differ in population (one teacher per learner vs a population in which learners hear nine successive teachers) and in measure (production language vs posterior). Which difference is responsible has not been tested.

**P3, maps as records: not settled.** Without the iconic bias (β = 0, γ = 100), where vocalisations and maps differ only in the record, maps lag behind vocalisations in Python (compositional 0.18 against 0.31) but not in NetLogo (0.30 against 0.29); both differences lie within seed-to-seed variation (about 0.13). With β = 2, maps lag in both implementations (Python 0.29 against 0.46; NetLogo 0.41 against 0.53), but there the two channels also differ in their prior (one fully iconic compositional map language against two vocal ones), so the record cannot be singled out as the cause. The slowing expected from Kirby & Tamariz's (2022) finding that learning from older producers slows change is therefore not established here. Iconicity is lower for maps throughout, partly because their chance level is lower (0.25 against 0.375). The computational model has no record, so it cannot test P3; its vocal and map chains differ only through the prior.

## Submodel B: Negation

**Design.** Steinert-Threlkeld's (2017, §2.5) basic negation learning setup with the four resources: 100 trials of 1000 plays in each of the thesis's two conditions, the receiver's "constant" interpretation of *a* pointing to "resource 0 present" (in X) or "resource 0 absent" (in f[X]).

After 1000 plays, mean (SD) over 100 trials, with the published values (Steinert-Threlkeld, 2017, Table 2.2, n = 4):

| Constant points to | Expected payoff | Published | Receiver reads *a* as negation | Published | Sender uses *a* m_r for "r absent" |
|---|---|---|---|---|---|
| resource 0 present (in X) | 0.983 (0.001) | 0.978 | 0.995 (0.000) | 0.995 | 0.972 (0.002) |
| resource 0 absent (in f[X]) | 0.910 (0.104) | 0.870 | 0.836 (0.261) | 0.807 | 0.881 (0.171) |
| resource 0 present (in X), NetLogo | 0.983 (0.001) | 0.978 | 0.995 (0.000) | 0.995 | 0.971 (0.002) |
| resource 0 absent (in f[X]), NetLogo | 0.913 (0.091) | 0.870 | 0.853 (0.223) | 0.807 | 0.890 (0.143) |

**Findings.** The published result is reproduced. When the alternative reading of *a* points to a present state, the pair learns to use *a* as negation in practically every trial. When it points to an absent state, where reading *a* m_r as one fixed word is right for one of the four resources, negation is still learned on average but less reliably (larger SD), as in the thesis. Our values in this condition are a little above the published ones and within one SD of ours. The two implementations agree to within 0.02. The setting differs from the thesis text in one respect, documented in ODD §8: the sender in an absent state chooses only among complex signals, as in the reference code.

## Submodel C: Quantifiers

**Design.** Carcassi, Steinert-Threlkeld & Szymanik's (2021) Experiments 1 and 2. A = the 10 resources in view, B = the distant ones. Bottleneck 200, 512, 715 and 1024 structures; 4 and 8 epochs; 20 trials of 300 generations each; 10 agents per generation. Measures averaged over the last 50 generations, then mean (SD) over trials. Random networks at generation 0 have monotonicity 0.134 and quantity 0.036 (mean over all runs); none is degenerate, an ultrafilter or perfectly monotone.

| Experiment | Bottleneck | Epochs | Monotonicity | Quantity | Degenerate | Ultrafilter | Perfectly monotone |
|---|---|---|---|---|---|---|---|
| 1 (as produced) | 200 | 4 | 0.14 (0.01) | 0.04 | 0.00 | 0.00 | 0.00 |
| | 200 | 8 | 0.16 (0.01) | 0.04 | 0.00 | 0.00 | 0.00 |
| | 512 | 4 | 0.19 (0.01) | 0.04 | 0.00 | 0.00 | 0.00 |
| | 512 | 8 | 0.92 (0.01) | 0.08 | 0.00 | 0.36 (0.02) | 0.36 (0.02) |
| | 715 | 4 | 0.43 (0.08) | 0.06 | 0.00 | 0.01 | 0.01 |
| | 715 | 8 | 0.95 (0.16) | 0.08 | 0.00 | 0.79 (0.27) | 0.80 (0.26) |
| | 1024 | 4 | 0.94 (0.01) | 0.08 | 0.00 | 0.43 (0.02) | 0.43 (0.02) |
| | 1024 | 8 | 0.77 (0.33) | 0.07 | 0.00 | 0.45 (0.51) | 0.61 (0.47) |
| 2 (shuffled) | 200 | 4 | 0.15 (0.01) | 0.05 | 0.00 | 0.00 | 0.00 |
| | 200 | 8 | 0.31 (0.02) | 0.14 (0.01) | 0.00 | 0.00 | 0.00 |
| | 512 | 4 | 0.58 (0.02) | 0.33 (0.01) | 0.00 | 0.00 | 0.00 |
| | 512 | 8 | 0.91 (0.01) | 0.62 (0.01) | 0.00 | 0.00 | 0.04 (0.01) |
| | 715 | 4 | 0.77 (0.01) | 0.48 (0.01) | 0.00 | 0.00 | 0.00 |
| | 715 | 8 | 0.92 (0.11) | 0.70 (0.09) | 0.09 (0.18) | 0.00 | 0.23 (0.12) |
| | 1024 | 4 | 0.93 (0.00) | 0.64 (0.01) | 0.00 | 0.00 | 0.06 (0.01) |
| | 1024 | 8 | 0.94 (0.05) | 0.86 (0.05) | 0.43 (0.40) | 0.00 | 0.37 (0.42) |

SDs below 0.005 are omitted. Large SDs reflect trials that end in different kinds of quantifier.

**NetLogo.** Same design and measures. Random networks at generation 0 have monotonicity 0.138 and quantity 0.038.

| Experiment | Bottleneck | Epochs | Monotonicity | Quantity | Degenerate | Ultrafilter | Perfectly monotone |
|---|---|---|---|---|---|---|---|
| 1 (as produced) | 200 | 4 | 0.14 (0.01) | 0.04 | 0.00 | 0.00 | 0.00 |
|  | 200 | 8 | 0.16 (0.01) | 0.04 (0.01) | 0.00 | 0.00 | 0.00 |
|  | 512 | 4 | 0.19 (0.02) | 0.05 (0.01) | 0.00 | 0.00 | 0.00 |
|  | 512 | 8 | 0.92 (0.01) | 0.07 | 0.00 | 0.36 (0.02) | 0.36 (0.02) |
|  | 715 | 4 | 0.44 (0.08) | 0.06 (0.01) | 0.00 | 0.01 (0.01) | 0.01 (0.01) |
|  | 715 | 8 | 0.96 (0.16) | 0.07 (0.01) | 0.00 | 0.83 (0.20) | 0.83 (0.20) |
|  | 1024 | 4 | 0.94 (0.01) | 0.08 | 0.00 | 0.44 (0.02) | 0.44 (0.02) |
|  | 1024 | 8 | 0.84 (0.30) | 0.07 (0.03) | 0.00 | 0.70 (0.47) | 0.74 (0.44) |
| 2 (shuffled) | 200 | 4 | 0.15 (0.01) | 0.05 | 0.00 | 0.00 | 0.00 |
|  | 200 | 8 | 0.32 (0.02) | 0.15 (0.01) | 0.00 | 0.00 | 0.00 |
|  | 512 | 4 | 0.57 (0.02) | 0.33 (0.01) | 0.00 | 0.00 | 0.00 |
|  | 512 | 8 | 0.91 | 0.62 | 0.00 | 0.00 | 0.04 (0.01) |
|  | 715 | 4 | 0.77 (0.01) | 0.48 (0.01) | 0.00 | 0.00 | 0.00 |
|  | 715 | 8 | 0.95 (0.07) | 0.73 (0.06) | 0.04 (0.11) | 0.00 | 0.27 (0.07) |
|  | 1024 | 4 | 0.93 | 0.64 | 0.00 | 0.00 | 0.06 (0.01) |
|  | 1024 | 8 | 0.94 (0.05) | 0.87 (0.04) | 0.44 (0.41) | 0.00 | 0.38 (0.43) |

**Findings.** The paper's two conclusions are reproduced qualitatively; the paper's figures were not compared number by number.
- *Monotone quantifiers emerge* once each child gets enough training (Experiment 1, from bottleneck 512 with 8 epochs, or 1024 with 4). With too little training, quantifiers stay as unstructured as the random starting networks. In Experiment 1 the perfectly monotone quantifiers are almost all ultrafilters: "true if resource k is distant", for one particular resource. They are monotone but say nothing about number (quantity 0.04–0.08).
- *With shuffling, quantifiers come to depend on number* (Experiment 2): quantity rises with training up to 0.86, while monotonicity rises as in Experiment 1. Ultrafilters cannot survive shuffling and do not appear. With the most training, trials end either in degenerate quantifiers ("always true" or "always false") or in monotone quantifiers that largely depend on number, as in the paper.
- *The two implementations agree.* In 15 of the 16 settings they differ by at most 0.06 on every measure. The exception is Experiment 1 with bottleneck 1024 and 8 epochs. There, trials end either in perfectly monotone quantifiers or in a quantifier that is neither monotone nor numerical, fixed within the first 20–40 generations and then transmitted unchanged (5 of 20 NetLogo trials; Python's SD of 0.33 for monotonicity points to the same split). The differences in the ultrafilter (0.25) and perfectly monotone (0.13) shares lie within twice the standard error of a difference between two means of 20 trials (about 0.3). Why trials freeze in this setting was not examined.
