# ODD description: The Iconographers

The model is described following the ODD protocol, second update:

> Grimm, V., Railsback, S. F., Vincenot, C. E., Berger, U., Gallagher, C., DeAngelis, D. L., Edmonds, B., Ge, J., Giske, J., Groeneveld, J., Johnston, A. S. A., Milles, A., Nabe-Nielsen, J., Polhill, J. G., Radchuk, V., Rohwäder, M.-S., Stillman, R. A., Thiele, J. C., & Ayllón, D. (2020). The ODD protocol for describing agent-based and other simulation models: A second update to improve clarity, replication, and structural realism. *Journal of Artificial Societies and Social Simulation*, 23(2), 7. https://doi.org/10.18564/jasss.4259

The iconographers' world is modelled by three submodels, each a published model applied to the same resources. Each has a Python (Mesa) and a NetLogo version.

| Submodel | Python (Mesa) | NetLogo | Published model |
|---|---|---|---|
| A. Compositional vocalisations and maps | `src/iconographers/abm.py` | `netlogo/iconographers.nlogo` | Kirby et al. (2015); Kirby & Tamariz (2022) |
| B. Negation | `src/iconographers/negation.py` | `netlogo/negation.nlogo` | Steinert-Threlkeld (2016, 2017) |
| C. Quantifiers | `src/iconographers/quantifiers.py` | `netlogo/quantifiers.nlogo` | Carcassi, Steinert-Threlkeld & Szymanik (2021) |

The submodels are not linked to one another; each answers its own question about the same world. Submodel A also has a computational (Markov-chain) version, `src/iconographers/chain.py`; B and C are agent-based only.

Shared parts of A: `src/iconographers/languages.py` (language space, prior, iconicity) and `src/iconographers/learning.py` (learning and production). Design scripts: `scripts/run_abm.py`, `scripts/run_chain.py` (A), `scripts/run_negation.py` (B), `scripts/run_quantifiers.py` (C). The NetLogo versions run the same designs as BehaviorSpace experiments (`netlogo/experiments/*.xml`, via `scripts/run_netlogo.sh`; run with NetLogo 7.0.4).

Sections 1–7 describe submodel A. Submodels B and C are described in sections 8 and 9. Results: [results/README.md](results/README.md). The scenario, and the passages quoted below as "the argument", come from an unpublished manuscript by the author.

---

## 1. Purpose and patterns

**Purpose.** The model asks whether a community whose vocalisations and maps start out iconic can arrive at compositional expressions (signals whose parts combine, as in sentences) that remain iconic (i.e. *abstract* but not arbitrary).

**The established pipeline.** The model combines two published models from the same group, which share one learner:

- *Language space and prior:* Kirby, Tamariz, Cornish & Smith (2015), as implemented in the Simulating Language course, Lab 6 (Kirby & Smith; MIT License). Four meanings, each made of two features; forms of two symbols; 256 possible languages; a prior favouring languages with short grammars. In that model, compositional languages emerge when there is both a pressure to be learnable (new learners) and a pressure to be expressive (communication).
- *Learning, production and population:* Kirby & Tamariz (2022). Bayesian learners, speakers who avoid ambiguous signals, and a population of 10 agents with turnover in which the oldest agent teaches. Kirby & Tamariz describe their model as "a relatively small modification" of Kirby et al. (2015); this model uses their population and production with Kirby et al.'s language space.
- *Computational model:* the Markov-chain analysis of iterated learning (Griffiths & Kalish, 2007), which Reali & Griffiths (2010) relate to the Wright–Fisher model of population genetics.

**Argument structure and model elements.**

| The argument | The model |
|---|---|
| "a small set of valuable resources" that are "large, round, and/or distant" or "small, angular, and/or nearby" | Four resources, each a combination of two features: large-and-round vs small-and-angular, and distant vs near |
| "back rounded vowels to refer to large, round, and/or distant objects and front unrounded vowels to refer to small, angular, and/or nearby objects" | A vocalisation has two vowels, each back rounded or front unrounded. A word is *iconic* if its vowels are those the argument gives the resource's two features |
| Maps as "cartographic ways of navigating an environment", "expressions that keep a record" (the author's specification) | A map has two marks: a glyph (large round or small angular) and a placement (far or near). A map is iconic if its glyph depicts the resource's size and shape and its placement its distance. Maps are kept in a record that outlives their makers |
| "simple maps are cheap but minimally informative whereas complex maps are expensive but maximally informative" | Built into the shared prior: simple expression systems (one expression for everything) have short grammars and are cheap to learn but carry no information; systems with distinct expressions for every resource have longer grammars and are costly to learn but fully informative |
| The iconic mapping as a bias of the iconographers | The prior: each iconic expression makes a language β bits shorter. Maps "use the same prior as iconic vocalisations" (the author's specification) |
| "equal amounts of competition and cooperation"; "We will see how this community establishes rules" | Dyadic communication in a community with generational turnover; speakers try to be understood (communicative rationality γ) |
| "from rules to maps to sentences" | Compositional languages: one part of the expression for size and shape, one for distance. The course implementation describes such a form as "a two-word sentence" |

**Pressures for sentential structure.** The argument introduces three linguistic pressures: to "negate more complex information", to "quantify more complex information (and use those quantifications reliably)", and "abstract predication of complex information (for example, the epistemic states of others)". Negation is submodel B and quantification submodel C. No published agent-based model of the emergence of language for epistemic states (ascribing knowledge and error to others) was found, so it is not modelled. The normative pressures (coordination, sanctioning) are not modelled either. Pointing gestures are not modelled separately.

### 1.1 Predictions

| ID | Prediction | Measure | Basis |
|---|---|---|---|
| P1 | **Replication.** With β = 0: without communicative rationality (γ = 0), degenerate languages dominate; with it (γ = 100), compositional languages emerge, slowly when the oldest agent teaches | Language-type shares | Kirby et al. (2015); Kirby & Tamariz (2022) |
| P2 | **Iconic and compositional.** With β > 0, expressions are more iconic, and among compositional languages the iconic ones are more frequent than chance (2 of 8 for vocalisations, 1 of 8 for maps) | Iconicity; probability of the fully iconic compositional languages | Partly built in: with two-symbol expressions, the only unambiguous, fully iconic languages are compositional (2 of the 8 compositional vocal languages, 1 of the 8 map languages), so the prior favours them over the other compositional languages. Whether compositional languages as a whole emerge faster is *not* implied, because β also favours partly iconic holistic and other languages |
| P3 | **Maps as records.** Maps are learned from a record of maps made by earlier, older agents. Kirby & Tamariz (2022) found slower change when learners learn from older producers, so maps are expected to change more slowly than vocalisations | Type shares and iconicity, maps vs vocalisations | Expected direction from Kirby & Tamariz (2022); not fixed in advance |

**Chance levels.** If every language were equally likely, the expected share of iconic expressions would be 0.375 for vocalisations (order of vowels free) and 0.25 for maps.

### 1.2 Measures

All measures are taken from the oldest agent (the current teacher) at the end of each step, as in Kirby & Tamariz (2022), separately for vocalisations and maps.

| Measure | Definition |
|---|---|
| `degenerate`, `holistic`, `other`, `compositional` | Posterior probability of each language type (Kirby et al., 2015) |
| `iconicity` | Expected share of iconic expressions under the posterior |
| `iconic_compositional` | Posterior probability of the fully iconic compositional languages |

The computational model reports the same quantities for the distribution over production languages. Comparisons are descriptive: no inferential statistic has been chosen.

## 2. Entities, state variables, and scales

| Entity | State | Meaning |
|---|---|---|
| `Iconographer` | `vocal` | Log posterior over the 256 vocal languages |
| | `maps` | Log posterior over the 256 map languages |
| `IconographersModel` | `population` | Agents ordered from newest to oldest |
| | `record` | The most recent `record_size` maps, as ⟨resource, map⟩ pairs |
| | `starting_language` | The founders' shared holistic language |

**Meanings (resources).** Two features with two values each: feature 1, 0 = large-and-round, 1 = small-and-angular; feature 2, 2 = distant, 3 = near. The four resources are 02, 03, 12 and 13.

**Forms.** Two symbols, each `a` or `b`: `aa`, `ab`, `ba`, `bb`. In a vocalisation, `a` is a back rounded vowel and `b` a front unrounded vowel. In a map, the first symbol is the glyph (`a` large round, `b` small angular) and the second the placement (`a` far, `b` near). In both, `a` is the symbol the argument associates with large, round and distant things.

**Languages.** A language gives one form to each resource: 4⁴ = 256 languages. Types (Kirby et al., 2015): 4 degenerate (one form for all), 8 compositional (each symbol position consistently expresses one feature), 16 holistic (all forms distinct, not compositional), 228 other.

**Scales.** No space. One step is one replacement of the oldest agent; ten steps replace the whole population (one generation in Kirby & Tamariz's sense).

## 3. Process overview and scheduling

Each step (`IconographersModel.step`), as in Kirby & Tamariz (2022):

1. `interactions_per_step` times:
   1. A resource is chosen at random.
   2. The oldest agent produces a vocalisation for it (§7.3). A different agent, chosen at random, hears it and updates its vocal posterior (§7.2).
   3. The oldest agent draws a map for the same resource. The map is added to the record (only the most recent `record_size` are kept). The same listener sees one map drawn at random from the record and updates its map posterior.
2. The oldest agent's posteriors are measured.
3. The oldest agent is removed; a newborn whose posteriors equal the prior is added.

## 4. Design concepts

- **Basic principles.** Iterated learning: languages change because each new learner infers a language from limited data produced by others, with a prior bias (Kirby et al., 2015; Griffiths & Kalish, 2007). The trade-off between a pressure for simplicity (learning) and for expressivity (communication) produces compositional structure (Kirby et al., 2015).
- **Emergence.** The type of language the community converges on and how iconic it is.
- **Adaptation and learning.** Bayesian updating after every observed expression (§7.2). There is no biological evolution; the prior is the same for every agent.
- **Objectives.** Speakers choose the expression most likely to be understood (§7.3).
- **Interaction.** The oldest agent teaches; listeners are chosen at random. Maps are also passed on through the record.
- **Stochasticity.** Resources, listeners, production errors, ties between expressions, record draws and the starting language all come from one seeded generator. A run is reproducible from its seed (tested).
- **Observation.** §1.2.

## 5. Initialization

- A starting language is drawn uniformly from the 16 holistic languages and shared by both channels, so that no particular holistic language (and its iconicity) is favoured. Kirby et al. (2015) and Kirby & Tamariz (2022) also start from a shared holistic language.
- Each founder is trained on `founder_training` ⟨resource, form⟩ pairs from the starting language, separately for vocalisations and maps, without production errors (as in Kirby & Tamariz's code).
- The record is filled with `record_size` maps from the starting language.

| Parameter | Value | Source |
|---|---|---|
| `population_size` | 10 | Kirby & Tamariz (2022) |
| `interactions_per_step` | 20 | Kirby & Tamariz (2022) |
| `founder_training` | 50 | Kirby & Tamariz (2022), learning-from-the-oldest runs |
| `error_probability` (ε) | 0.05 | Kirby et al. (2015); Kirby & Tamariz (2022) |
| `communicative_rationality` (γ) | 0, 100 | Kirby & Tamariz (2022) |
| `record_size` | 80 | The memory size r = 80 of Kirby et al. (2015) |
| `iconic_bits` (β) | 0, 2 | 0 = Kirby et al.'s prior; 2 is illustrative (no published value) |

No parameter has a default in the code.

## 6. Input data

None. The prior table (coding lengths of the 256 grammars) is copied from the Simulating Language course, Lab 6, and reproduces the coding lengths published in Kirby et al. (2015): 38.55, 55.20 and 67.29 bits for their example degenerate, compositional and holistic languages (tested).

## 7. Submodels

### 7.1 Prior
P(l) ∝ 2^−(L(l) − β·k(l)), where L(l) is the coding length in bits of the grammar of language l (Kirby et al., 2015, §2.1.3) and k(l) the number of iconic expressions in l (0–4). With β = 0 this is Kirby et al.'s prior. The vocal and map priors differ only in what counts as iconic:
- a word is iconic if its two vowels are the argument's vowels for the resource's two features, in either order (the argument does not specify an order);
- a map is iconic if its glyph depicts the resource's size and shape and its placement its distance.

### 7.2 Learning
After observing ⟨resource m, form f⟩: P(l | d) ∝ P(l | d_previous) · P(f | l, m), with P(f | l, m) = 1 − ε if l pairs f with m and ε/3 otherwise (Kirby & Tamariz, 2022).

### 7.3 Production
The speaker chooses the form f that maximises Σ_l P(l | d) · C(f | l, m), where C(f | l, m) = (1/a)^γ if l pairs f with m and 0 otherwise, and a is the number of resources l pairs with f. Ties are broken at random. With probability ε, a different form, chosen at random, is produced instead (Kirby & Tamariz, 2022). With γ = 0 the speaker only cares that the form is right; with γ = 100 it avoids forms that are ambiguous.

### 7.4 Computational model
A transmission chain in which each generation is one learner (Griffiths & Kalish, 2007). A speaker's behaviour is fully described by its *production language*, the form §7.3 chooses for each resource, so the chain's state is the current production language. The next learner starts from the prior (§7.1), observes 20 utterances (resource uniform; the production-language form with probability 1 − ε, otherwise one of the other three forms), and its own production language is the next state. Twenty is the expected number of utterances an agent hears in the agent-based model before it becomes the oldest (20 interactions per step, 9 possible listeners, 9 steps). Each row of the 256 × 256 transition matrix is estimated from 4,000 simulated data sets (Monte Carlo standard error ≤ 0.008 per entry). The model reports the expected trajectory from a uniform distribution over holistic languages and the long-run distribution reached from it (after 100,000 generations). Compositional and degenerate languages are, in practice, never left once reached, so the chain has several absorbing languages and no single stationary distribution; the long-run outcome therefore depends on the starting distribution, and the number of unit eigenvalues of the transition matrix is reported as a check. The record of maps is a population-level feature and is in the agent-based model only.

### 7.5 Departures from the reference code
- Kirby & Tamariz's paper says the listener is "a different, randomly chosen agent"; their code can choose the speaker itself. This model follows the paper.
- The Simulating Language Lab 6 pragmatic speaker passes the list of signals where the list of meanings is expected; this model uses Kirby & Tamariz's production rule (§7.3) instead, so the issue does not arise.

---

---

## 8. Submodel B: Negation

**Purpose.** Can the iconographers learn to use a function word as negation, "*a* + word for resource r" meaning "resource r is absent"? This is the argument's need "to negate".

**Published model.** The basic negation learning setup of Steinert-Threlkeld (2017, chapter 2, section 2.5; also 2016), a Lewis–Skyrms signaling game with Roth–Erev reinforcement learning.

**Entities and state.** A sender and a receiver (Mesa agents).
- *States:* each of the 4 resources present or absent (8 states). Minimal negation f maps "resource r present" to "resource r absent" and back. This pairing is the thesis's Example 2.1 (leopard / no leopard).
- *Signals:* the basic signals m_r (the word for resource r) and the complex signals *a* m_r.
- *Sender:* in "resource r present" it sends m_r (a signaling system on the present states is given). In "resource r absent" it draws a complex signal from an urn for that state.
- *Receiver:* it responds to m_r with "resource r present" (given). On hearing *a* m_r it draws an interpretation g of *a* from an urn over three functions and performs g("resource r present"). The three functions are identity (ignore *a*), a constant (treat *a* m_r as one new word, always meaning one state c), and negation f.

**Process.** Each play: a state is drawn uniformly; the sender signals; the receiver acts; if the act matches the state (payoff 1), each player adds one ball of the kind it drew to the urn it drew from. Urns start with one ball of each kind.

**Design** (`scripts/run_negation.py`). 100 trials of 1000 plays for each of the thesis's two conditions: c is a present state ("resource 0 present") or an absent state ("resource 0 absent").

**Measures.** Expected payoff (probability of a successful play under the current strategies); probability that the receiver interprets *a* as negation; mean probability that the sender sends *a* m_r in "resource r absent".

**Departure from the thesis text.** The thesis leaves open which signals the sender may use in the absent states. The reference code restricts it to complex signals; with that restriction, this implementation reproduces the thesis's Table 2.2 for n = 4 (expected payoff 0.978 and probability of negation 0.995 when c is a present state; tested). Allowing basic signals as well gives lower values (0.955 and 0.994; 0.839 and 0.720).

**Iconicity.** None. The published model has none, and none is added. The basic signals are the resources' words, but their form plays no role.

## 9. Submodel C: Quantifiers

**Purpose.** Do the iconographers' quantifiers, ways of judging situations like "some / all / at least two of the resources are distant", come to have the properties of natural-language quantifiers? 

**Published model.** Carcassi, Steinert-Threlkeld & Szymanik (2021): iterated learning with neural-network agents. Their result is that monotone quantifiers evolve, and with shuffled input (Experiment 2) quantifiers that depend only on number.

**Entities and state.** Agents (Mesa agents) in generations of 10; each agent is a neural network.
- *World:* A is the 10 resources in view and B the distant ones. A structure is a vector of 10 bits (1 = distant); there are 1,024 structures.
- *Network:* 10 inputs (bits mapped to −1/1), two hidden layers of 16 ReLU units, batch normalisation after the second, and a sigmoid output. An agent's judgment of a structure is the rounded output. Batch normalisation always uses the current batch's statistics, as in the reference code, which never switches to evaluation mode.

**Process.** Each generation, each of 10 children draws a parent at random from the previous generation, with replacement. It sees `bottleneck` random structures with the parent's judgments and trains on them: binary cross-entropy, Adam (learning rate 0.001), batches of 32, `epochs` passes, reshuffled each pass. Weights start uniform in (−1/√fan-in, 1/√fan-in), PyTorch's default. Experiment 1: the child sees the structures as the parent judged them. Experiment 2: the child sees each structure with its bits shuffled, so a quantifier cannot track particular resources.

**Design** (`scripts/run_quantifiers.py`). As published: both experiments; bottleneck 200, 512, 715 and 1024; 4 and 8 epochs; 20 trials; 300 generations.

**Measures** (on the 1,024 judgments of each agent).
- *Monotonicity:* 1 − H(Q | has a true substructure) / H(Q), maximised over the four readings of the bits (Appendix A of the paper).
- *Quantity:* 1 − H(Q | number of distant resources) / H(Q).
- *Shares:* degenerate quantifiers (always true or always false); ultrafilters (truth depends on one particular resource); perfectly monotone, non-degenerate quantifiers.

Degenerate quantifiers get monotonicity and quantity 1, as in the reference code.

**Implementation.** A NumPy reimplementation of the authors' PyTorch code; random streams differ. Backpropagation is checked against numerical gradients (tested). The measure reproduces two of the three values given in the paper ("some" 1.0; "an even number of" 0.001; tested). For "between 3 and 5" the paper gives 0.7517. The formula as stated, and as written in the reference code, gives 0.729 for 10 objects, so that value is not reproduced. The NetLogo version uses the bundled matrix extension; its backpropagation agrees with numerical gradients to eight significant digits. Because `matrix:times-scalar` overwrites its argument in NetLogo 7.0.4, the model scales matrices with its own `scaled` reporter.

**Iconicity.** None; the published model has none and none is added.

---

## Assumptions and their sources

| Assumption | Source |
|---|---|
| Vowel mapping (back rounded ↔ large, round, distant) | Argument assumption/specification. Empirical support: rounder drawings for back rounded vowels (Porto et al., under review); distal deictics with back vowels (Tanz, 1971; Woodworth, 1991).  |
| Large and round grouped as one feature | Argument assumption/specification ("large, round, and/or distant") |
| Maps keep a record and share the vocal prior | Argument assumption/specification |
| Iconicity as a bias in the prior (β bits per iconic expression) | In iterated learning, learner biases are expressed in the prior (Griffiths & Kalish, 2007; Reali & Griffiths, 2010). The form "bits saved per iconic expression" is this model's; β = 0 is the published prior |
| Vowel order free; map marks positional | Argument does not order vowels; a map's glyph and placement are different marks |
| Everything else in submodel A | Kirby et al. (2015); Kirby & Tamariz (2022) |
| Negation states: each resource present or absent; negation pairs them | Steinert-Threlkeld (2017), Example 2.1, applied to the four resources |
| Quantifier world: A = 10 resources in view, B = the distant ones | Model specification; structure size 10 as in Carcassi et al. (2021) |
| Epistemic language not modelled | No published agent-based model |

## References

Carcassi, F., Steinert-Threlkeld, S., & Szymanik, J. (2021). Monotone quantifiers emerge via iterated learning. *Cognitive Science*, 45(8), e13027. https://doi.org/10.1111/cogs.13027

Epstein, J. M., & Axtell, R. L. (1996). *Growing artificial societies: Social science from the bottom up*. MIT Press. https://doi.org/10.7551/mitpress/3374.001.0001

Griffiths, T. L., & Kalish, M. L. (2007). Language evolution by iterated learning with Bayesian agents. *Cognitive Science*, 31(3), 441–480. https://doi.org/10.1080/15326900701326576

Kirby, S., & Tamariz, M. (2022). Cumulative cultural evolution, population structure and the origin of combinatoriality in human language. *Philosophical Transactions of the Royal Society B*, 377(1843), 20200319. https://doi.org/10.1098/rstb.2020.0319

Kirby, S., Tamariz, M., Cornish, H., & Smith, K. (2015). Compression and communication in the cultural evolution of linguistic structure. *Cognition*, 141, 87–102. https://doi.org/10.1016/j.cognition.2015.03.016

Perfors, A. (2002). Simulated evolution of language: A review of the field. *Journal of Artificial Societies and Social Simulation*, 5(2), 4. https://www.jasss.org/5/2/4.html

Porto, A., Huckle, N., Basalyga, A., Perlman, M., Cuskley, C., & Kranjec, A. (under review). The shape and size of speech sounds. *Open Mind*.

Steinert-Threlkeld, S. (2016). Compositional signaling in a complex world. *Journal of Logic, Language and Information*, 25(3–4), 379–397. https://doi.org/10.1007/s10849-016-9236-9

Steinert-Threlkeld, S. (2017). *Communication and computation: New questions about compositionality* (Doctoral dissertation, Stanford University). ILLC Dissertation Series DS-2017-05. https://eprints.illc.uva.nl/id/eprint/2145/

Reali, F., & Griffiths, T. L. (2010). Words as alleles: Connecting language evolution with Bayesian learners to models of genetic drift. *Proceedings of the Royal Society B*, 277(1680), 429–436. https://doi.org/10.1098/rspb.2009.1513

Tanz, C. (1971). Sound symbolism in words relating to proximity and distance. *Language and Speech*, 14(3), 266–276. https://doi.org/10.1177/002383097101400307

Woodworth, N. L. (1991). Sound symbolism in proximal and distal forms. *Linguistics*, 29(2), 273–300. https://doi.org/10.1515/ling.1991.29.2.273

Software: Kirby, S., & Smith, K. Simulating Language course materials, https://github.com/smkirby/simlang2024-25 (MIT License); Kirby, S. combinatoriality model, https://github.com/smkirby/combinatoriality (MIT License); Steinert-Threlkeld, S. negation game code, https://github.com/shanest/learning-evolution-compositionality (consulted, not copied; no licence); Carcassi, F., et al. quantifier code, https://github.com/thelogicalgrammar/NeuralNetIteratedQuantifiers (consulted, not copied; no licence).
