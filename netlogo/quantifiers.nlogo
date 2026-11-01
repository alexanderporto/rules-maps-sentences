;; The Iconographers: quantifier submodel (NetLogo version)
;; Same rules as src/iconographers/quantifiers.py. See the Info tab.
;;
;; Carcassi, F., Steinert-Threlkeld, S., & Szymanik, J. (2021). Monotone quantifiers emerge via iterated
;; learning. Cognitive Science, 45(8), e13027. https://doi.org/10.1111/cogs.13027
;;
;; A = the 10 resources in view; B = the distant ones. A structure is a list of 10 bits (1 = distant).
;; Each agent's quantifier is a small neural network (10 inputs, two hidden layers of 16 ReLU units,
;; batch normalisation, sigmoid output) trained with Adam on its cultural parent's judgments.

extensions [ matrix ]

breed [ speakers speaker ]
speakers-own [ net ]   ;; list of 8 matrices: w1 b1 w2 b2 gamma beta w3 b3

globals [
  all-structures      ;; the 1024 structures, in binary order (first resource = most significant bit)
  all-inputs          ;; the same as a 1024 x 10 matrix with bits mapped to -1 / 1
  substructure        ;; 1024 x 1024 matrix: entry [i j] = 1 if structure j is a substructure of structure i
  superstructure      ;; its transpose
  distant-counts      ;; number of distant resources in each structure
  measures            ;; current generation's means: monotonicity, quantity, degenerate, ultrafilter, perfectly monotone
]

to setup
  clear-all
  set all-structures structures-of-length 10
  set all-inputs inputs-of all-structures
  set substructure matrix:from-row-list substructure-rows 10
  set superstructure matrix:transpose substructure
  set distant-counts map [ s -> sum s ] all-structures
  create-speakers agents-per-generation [ set net initial-net hide-turtle ]
  reset-ticks
  plot-measures
end

to go
  let parents n-values agents-per-generation [ one-of speakers ]
  let child-nets map [ p -> child-net-of p ] parents
  ask speakers [ die ]
  foreach child-nets [ n -> create-speakers 1 [ set net n hide-turtle ] ]
  tick
  plot-measures
end

to-report child-net-of [ parent ]
  let structures n-values bottleneck [ n-values 10 [ random 2 ] ]
  let parent-judgments judgments-of ([ net ] of parent) structures
  let seen ifelse-value (experiment = "2: shuffled") [ map [ s -> shuffle s ] structures ] [ structures ]
  report trained initial-net (inputs-of seen) parent-judgments
end

;; ---------- structures ----------

to-report structures-of-length [ n ]
  if n = 0 [ report [ [] ] ]
  let shorter structures-of-length (n - 1)
  report sentence (map [ s -> fput 0 s ] shorter) (map [ s -> fput 1 s ] shorter)
end

to-report substructure-rows [ n ]   ;; built recursively: [[S, 0], [S, S]]
  if n = 0 [ report [ [ 1 ] ] ]
  let smaller substructure-rows (n - 1)
  let zeros n-values (length smaller) [ 0 ]
  report sentence (map [ row -> sentence row zeros ] smaller) (map [ row -> sentence row row ] smaller)
end

to-report inputs-of [ structures ]
  report matrix:from-row-list map [ s -> map [ b -> 2 * b - 1 ] s ] structures
end

;; ---------- the network ----------

to-report uniform-matrix [ rows cols bound ]
  report matrix:from-row-list n-values rows [ n-values cols [ (random-float (2 * bound)) - bound ] ]
end

to-report initial-net   ;; PyTorch's default: uniform in (-1/sqrt(fan_in), 1/sqrt(fan_in))
  report (list
    (uniform-matrix 10 16 (1 / sqrt 10)) (uniform-matrix 1 16 (1 / sqrt 10))
    (uniform-matrix 16 16 (1 / sqrt 16)) (uniform-matrix 1 16 (1 / sqrt 16))
    (matrix:make-constant 1 16 1) (matrix:make-constant 1 16 0)
    (uniform-matrix 16 1 (1 / sqrt 16)) (uniform-matrix 1 1 (1 / sqrt 16)))
end

to-report scaled [ m k ]   ;; k times m as a new matrix; matrix:times-scalar overwrites its argument in NetLogo 7.0.4
  report matrix:map [ v -> v * k ] m
end

to-report broadcast [ row-matrix rows ]   ;; repeat a 1 x k matrix as rows x k
  report matrix:times (matrix:make-constant rows 1 1) row-matrix
end

to-report column-sums [ m ]
  report matrix:times (matrix:make-constant 1 (first matrix:dimensions m) 1) m
end

to-report sigmoid [ x ]
  if x >= 0 [ report 1 / (1 + exp (- x)) ]
  let exponential exp x
  report exponential / (1 + exponential)
end

to-report forward-pass [ weights x ]   ;; reports [x z1 h1 z2 normalised inverse-std normed-output output]
  let rows first matrix:dimensions x
  let z1 matrix:plus (matrix:times x item 0 weights) (broadcast item 1 weights rows)
  let h1 matrix:map [ v -> max list 0 v ] z1
  let z2 matrix:plus (matrix:times h1 item 2 weights) (broadcast item 3 weights rows)
  let h2 matrix:map [ v -> max list 0 v ] z2
  let mean-row scaled (column-sums h2) (1 / rows)
  let centered matrix:minus h2 (broadcast mean-row rows)
  let variances scaled (column-sums matrix:times-element-wise centered centered) (1 / rows)
  let inverse-std matrix:map [ v -> 1 / sqrt (v + 0.00001) ] variances
  let normalised matrix:times-element-wise centered (broadcast inverse-std rows)
  let normed matrix:plus (matrix:times-element-wise normalised (broadcast item 4 weights rows)) (broadcast item 5 weights rows)
  let logits matrix:plus (matrix:times normed item 6 weights) (broadcast item 7 weights rows)
  report (list x z1 h1 z2 normalised inverse-std normed (matrix:map [ v -> sigmoid v ] logits))
end

to-report gradients [ weights passed targets ]   ;; gradients of the mean binary cross-entropy
  let x item 0 passed
  let z1 item 1 passed
  let h1 item 2 passed
  let z2 item 3 passed
  let normalised item 4 passed
  let inverse-std item 5 passed
  let normed item 6 passed
  let output item 7 passed
  let rows first matrix:dimensions x
  let d-logit scaled (matrix:minus output targets) (1 / rows)
  let d-w3 matrix:times (matrix:transpose normed) d-logit
  let d-b3 column-sums d-logit
  let d-normed matrix:times d-logit (matrix:transpose item 6 weights)
  let d-gamma column-sums matrix:times-element-wise d-normed normalised
  let d-beta column-sums d-normed
  let d-hat matrix:times-element-wise d-normed (broadcast item 4 weights rows)
  let sum-hat column-sums d-hat
  let sum-hat-normalised column-sums matrix:times-element-wise d-hat normalised
  let inner matrix:minus (matrix:minus (scaled d-hat rows) (broadcast sum-hat rows))
                         (matrix:times-element-wise normalised (broadcast sum-hat-normalised rows))
  let d-h2 matrix:times-element-wise (broadcast (scaled inverse-std (1 / rows)) rows) inner
  let d-z2 matrix:times-element-wise d-h2 (matrix:map [ v -> ifelse-value (v > 0) [ 1 ] [ 0 ] ] z2)
  let d-w2 matrix:times (matrix:transpose h1) d-z2
  let d-b2 column-sums d-z2
  let d-z1 matrix:times-element-wise (matrix:times d-z2 (matrix:transpose item 2 weights)) (matrix:map [ v -> ifelse-value (v > 0) [ 1 ] [ 0 ] ] z1)
  let d-w1 matrix:times (matrix:transpose x) d-z1
  let d-b1 column-sums d-z1
  report (list d-w1 d-b1 d-w2 d-b2 d-gamma d-beta d-w3 d-b3)
end

to-report trained [ weights inputs targets ]   ;; Adam (lr 0.001, betas 0.9 / 0.999, eps 1e-8), batches of 32
  let target-matrix matrix:from-column-list (list targets)
  let first-moments map [ w -> scaled w 0 ] weights
  let second-moments map [ w -> scaled w 0 ] weights
  let steps 0
  let n first matrix:dimensions inputs
  repeat epochs [
    let order shuffle range n
    foreach range (floor (n / batch-size)) [ b ->
      let rows sublist order (b * batch-size) ((b + 1) * batch-size)
      let batch-x matrix:from-row-list map [ i -> matrix:get-row inputs i ] rows
      let batch-y matrix:from-row-list map [ i -> matrix:get-row target-matrix i ] rows
      let grads gradients weights (forward-pass weights batch-x) batch-y
      set steps steps + 1
      set first-moments (map [ [ m g ] -> matrix:plus (scaled m 0.9) (scaled g 0.1) ] first-moments grads)
      set second-moments (map [ [ v g ] -> matrix:plus (scaled v 0.999) (scaled (matrix:times-element-wise g g) 0.001) ] second-moments grads)
      let correction1 1 - 0.9 ^ steps
      let correction2 1 - 0.999 ^ steps
      set weights (map [ [ w m v ] ->
        matrix:minus w (scaled
          (matrix:times-element-wise m (matrix:map [ s -> 1 / (sqrt (s / correction2) + 0.00000001) ] v))
          (0.001 / correction1)) ] weights first-moments second-moments)
    ]
  ]
  report weights
end

to-report judgments-of [ weights structures ]   ;; production: rounded outputs, one batch
  report map [ v -> round v ] matrix:get-column (item 7 forward-pass weights (inputs-of structures)) 0
end

;; ---------- measures (Carcassi et al., 2021) ----------

to-report entropy [ probabilities ]
  report (- sum map [ p -> ifelse-value (p > 0) [ p * log p 2 ] [ 0 ] ] probabilities)
end

to-report upward-monotonicity [ q order ]   ;; 1 - H(Q | has true substructure) / H(Q)
  if min q = max q [ report 1 ]
  let has-true matrix:get-column (matrix:times order (matrix:from-column-list (list q))) 0
  let n length q
  let conditional 0
  foreach [ true false ] [ t ->
    let group filter [ i -> (item i has-true > 0) = t ] range n
    if not empty? group [
      let share-true mean map [ i -> item i q ] group
      set conditional conditional + (length group / n) * entropy (list share-true (1 - share-true))
    ]
  ]
  let p mean q
  report 1 - conditional / entropy (list p (1 - p))
end

to-report monotonicity [ q ]   ;; highest of the four readings of the bits
  let flipped map [ v -> 1 - v ] q
  report max (list (upward-monotonicity q substructure) (upward-monotonicity flipped substructure)
                   (upward-monotonicity flipped superstructure) (upward-monotonicity q superstructure))
end

to-report quantity [ q ]   ;; 1 - H(Q | number of distant resources) / H(Q)
  if min q = max q [ report 1 ]
  let n length q
  let conditional 0
  foreach range 11 [ k ->
    let group filter [ i -> item i distant-counts = k ] range n
    let share-true mean map [ i -> item i q ] group
    set conditional conditional + (length group / n) * entropy (list share-true (1 - share-true))
  ]
  let p mean q
  report 1 - conditional / entropy (list p (1 - p))
end

to-report ultrafilter? [ q ]   ;; truth depends only on whether one particular resource is distant
  report reduce [ [ a b ] -> a or b ] map [ k ->
    (map [ s -> item k s ] all-structures) = q or (map [ s -> 1 - item k s ] all-structures) = q ] range 10
end

to plot-measures
  let quantifiers [ judgments-of net all-structures ] of speakers
  let monotonicities map [ q -> monotonicity q ] quantifiers
  set measures (list
    (mean monotonicities)
    (mean map [ q -> quantity q ] quantifiers)
    (mean map [ q -> ifelse-value (min q = max q) [ 1 ] [ 0 ] ] quantifiers)
    (mean map [ q -> ifelse-value (ultrafilter? q) [ 1 ] [ 0 ] ] quantifiers)
    (mean (map [ [ m q ] -> ifelse-value (m > 1 - 0.000000001 and min q != max q) [ 1 ] [ 0 ] ] monotonicities quantifiers)))
  set-current-plot "Quantifiers"
  set-current-plot-pen "monotonicity" plot item 0 measures
  set-current-plot-pen "quantity" plot item 1 measures
  set-current-plot-pen "degenerate" plot item 2 measures
  set-current-plot-pen "ultrafilter" plot item 3 measures
end
@#$#@#$#@
GRAPHICS-WINDOW
800
10
973
184
-1
-1
5.0
1
10
1
1
1
0
1
1
1
-16
16
-16
16
0
0
1
generations
30.0

BUTTON
15
10
95
43
NIL
setup
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
105
10
185
43
NIL
go
T
1
T
OBSERVER
NIL
NIL
NIL
NIL
0

BUTTON
105
50
185
83
go once
go
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
0

CHOOSER
15
95
200
140
experiment
experiment
"1: as produced" "2: shuffled"
1

SLIDER
15
150
200
183
agents-per-generation
agents-per-generation
1
20
10.0
1
1
NIL
HORIZONTAL

SLIDER
15
190
200
223
bottleneck
bottleneck
32
1024
512.0
1
1
NIL
HORIZONTAL

SLIDER
15
230
200
263
epochs
epochs
1
8
4.0
1
1
NIL
HORIZONTAL

SLIDER
15
270
200
303
batch-size
batch-size
2
64
32.0
1
1
NIL
HORIZONTAL

MONITOR
15
315
110
360
generation
ticks
0
1
11

PLOT
215
10
785
340
Quantifiers
generation
mean over agents
0.0
10.0
0.0
1.0
true
true
"" ""
PENS
"monotonicity" 1.0 0 -16777216 true "" ""
"quantity" 1.0 0 -13345367 true "" ""
"degenerate" 1.0 0 -955883 true "" ""
"ultrafilter" 1.0 0 -10899396 true "" ""

@#$#@#$#@
## WHAT IS IT?

The quantifier submodel of the iconographers. Ten resources are in view; some of them are distant. Each agent has a quantifier, a way of judging situations like "some / all / at least two of the resources are distant" as true or false. Quantifiers are passed from generation to generation: each child learns from the judgments of one parent.

This is the model of Carcassi, Steinert-Threlkeld & Szymanik (2021).

## HOW IT WORKS

- A situation (structure) is a list of 10 bits: bit i is 1 if resource i is distant.
- Each agent is a small neural network: 10 inputs, two hidden layers of 16 units, batch normalisation, and one output (the probability that the quantifier is true). Its judgment is that output rounded to 0 or 1.
- Each generation, every child picks a parent at random from the previous generation, sees `bottleneck` random situations with the parent's judgments, and trains on them for `epochs` passes (Adam, batches of `batch-size`).
- Experiment 1: the child sees the same situations the parent judged. Experiment 2: the child sees each situation with its bits shuffled, so a quantifier cannot track particular resources, only how many are distant.

## HOW TO USE IT

Press **setup**, then **go**. The plot shows, averaged over the generation's agents:
- **monotonicity**: how monotone the quantifiers are (1 = perfectly, like "some" or "at least two");
- **quantity**: how far the truth value depends only on how many resources are distant;
- **degenerate**: share of quantifiers that are always true or always false;
- **ultrafilter**: share of quantifiers that just name one resource.

NetLogo is much slower than the Python version; use smaller bottlenecks or fewer generations to explore.

## THINGS TO NOTICE

Carcassi et al. found that monotone quantifiers evolve with intermediate amounts of training; in Experiment 1 they mostly name one resource, in Experiment 2 they are mostly monotone and quantitative, or degenerate.

## CREDITS AND REFERENCES

Carcassi, F., Steinert-Threlkeld, S., & Szymanik, J. (2021). Monotone quantifiers emerge via iterated learning. *Cognitive Science*, 45(8), e13027. https://doi.org/10.1111/cogs.13027
@#$#@#$#@
default
true
0
Polygon -7500403 true true 150 5 40 250 150 205 260 250

@#$#@#$#@
NetLogo 6.4.0
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
default
0.0
-0.2 0 0.0 1.0
0.0 1 1.0 0.0
0.2 0 0.0 1.0
link direction
true
0
Line -7500403 true 150 150 90 180
Line -7500403 true 150 150 210 180
@#$#@#$#@
0
@#$#@#$#@
