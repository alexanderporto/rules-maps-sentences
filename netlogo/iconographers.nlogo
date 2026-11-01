;; The Iconographers (NetLogo version)
;; Same rules as the Python/Mesa model in src/iconographers/abm.py. See the Info tab.
;;
;; Language space and prior: Kirby, S., Tamariz, M., Cornish, H., & Smith, K. (2015). Compression and
;; communication in the cultural evolution of linguistic structure. Cognition, 141, 87-102.
;; https://doi.org/10.1016/j.cognition.2015.03.016
;; Learning, production and population: Kirby, S., & Tamariz, M. (2022). Cumulative cultural evolution,
;; population structure and the origin of combinatoriality in human language. Philosophical
;; Transactions of the Royal Society B, 377(1843), 20200319. https://doi.org/10.1098/rstb.2020.0319
;; The prior table (kirby-log-priors, kirby-language-types) is copied from the Simulating Language
;; course, Lab 6 (https://github.com/smkirby/simlang2024-25), MIT License, Copyright (c) 2019 Simon Kirby.

globals [
  meanings            ;; the four resources: feature 1 (0 large-and-round, 1 small-and-angular) x feature 2 (2 distant, 3 near)
  forms               ;; "aa" "ab" "ba" "bb"; vocal: a = back rounded vowel, b = front unrounded vowel;
                      ;; map: first symbol a glyph (a large round, b small angular), second a placement (a far, b near)
  languages           ;; 256 lists of 4 form indices, one per meaning
  language-types      ;; 0 degenerate, 1 holistic, 2 other, 3 compositional
  vocal-iconic-counts ;; number of iconic words in each language
  map-iconic-counts   ;; number of iconic maps in each language
  vocal-prior
  map-prior
  log-likelihoods     ;; 16 lists (index meaning * 4 + form) of 256 values: log P(form | language, meaning)
  penalties           ;; 256 lists of 4 values: -gamma * ln(ambiguity of the form the language gives each meaning)
  languages-with      ;; 16 lists (index meaning * 4 + form) of the languages pairing that meaning with that form
  starting-language
  map-record          ;; the most recent maps, as [meaning form] pairs
  teacher-vocal       ;; posteriors of the oldest agent at the end of the last step, before it dies (the measured agent)
  teacher-maps
]

turtles-own [
  vocal               ;; log posterior over vocal languages
  maps                ;; log posterior over map languages
]

to setup
  clear-all
  set meanings ["02" "03" "12" "13"]
  set forms ["aa" "ab" "ba" "bb"]
  set languages all-languages
  set language-types kirby-language-types
  set vocal-iconic-counts map [ l -> vocal-iconic-count l ] languages
  set map-iconic-counts map [ l -> map-iconic-count l ] languages
  set vocal-prior prior-with vocal-iconic-counts
  set map-prior prior-with map-iconic-counts
  set log-likelihoods n-values 16 [ i -> map [ l -> pair-log-likelihood l (floor (i / 4)) (i mod 4) ] languages ]
  set penalties map [ l -> map [ m -> (- communicative-rationality) * ln (ambiguity l m) ] range 4 ] languages
  set languages-with n-values 16 [ i -> filter [ j -> item (floor (i / 4)) (item j languages) = i mod 4 ] range 256 ]
  set starting-language one-of filter [ j -> item j language-types = 1 ] range 256
  create-turtles population-size [
    set vocal trained vocal-prior founder-training
    set maps trained map-prior founder-training
    hide-turtle
  ]
  set map-record n-values record-size [ starting-pair ]
  set teacher-vocal [ vocal ] of min-one-of turtles [ who ]
  set teacher-maps [ maps ] of min-one-of turtles [ who ]
  reset-ticks
end

to go
  let signaller min-one-of turtles [ who ]   ;; the oldest agent teaches
  repeat interactions-per-step [
    let learner one-of turtles with [ self != signaller ]
    let meaning random 4
    let utterance produce [ vocal ] of signaller meaning
    ask learner [ set vocal updated vocal meaning utterance ]
    let drawn produce [ maps ] of signaller meaning
    set map-record lput (list meaning drawn) map-record
    if length map-record > record-size [ set map-record but-first map-record ]
    let seen one-of map-record
    ask learner [ set maps updated maps (first seen) (last seen) ]
  ]
  set teacher-vocal [ vocal ] of signaller
  set teacher-maps [ maps ] of signaller
  plot-oldest signaller
  ask signaller [ die ]
  create-turtles 1 [
    set vocal vocal-prior
    set maps map-prior
    hide-turtle
  ]
  tick
end

;; ---------- language space ----------

to-report all-languages
  let result []
  foreach range 4 [ a -> foreach range 4 [ b -> foreach range 4 [ c -> foreach range 4 [ d ->
    set result lput (list a b c d) result
  ] ] ] ]
  report result
end

to-report iconic-symbol [ value ]   ;; the argument's symbol for a feature value: a for large-and-round and for distant
  report ifelse-value (value = "0" or value = "2") [ "a" ] [ "b" ]
end

to-report vocal-word-iconic? [ m f ]   ;; vowels match the resource's features, in either order
  let meaning item m meanings
  let form item f forms
  report sort (list (iconic-symbol first meaning) (iconic-symbol last meaning)) = sort (list (first form) (last form))
end

to-report map-iconic? [ m f ]   ;; glyph depicts size and shape, placement depicts distance
  let meaning item m meanings
  let form item f forms
  report (first form = iconic-symbol first meaning) and (last form = iconic-symbol last meaning)
end

to-report vocal-iconic-count [ l ]
  report length filter [ m -> vocal-word-iconic? m (item m l) ] range 4
end

to-report map-iconic-count [ l ]
  report length filter [ m -> map-iconic? m (item m l) ] range 4
end

to-report ambiguity [ l m ]   ;; number of meanings the language pairs with the form it gives meaning m
  report length filter [ x -> x = item m l ] l
end

;; ---------- prior, likelihood, learning ----------

to-report logsumexp [ xs ]
  let biggest max xs
  report biggest + ln sum map [ x -> exp (x - biggest) ] xs
end

to-report normalised [ xs ]
  let total logsumexp xs
  report map [ x -> x - total ] xs
end

to-report prior-with [ counts ]   ;; Kirby et al.'s prior, each iconic expression saving iconic-bits bits
  report normalised (map [ [ p k ] -> p + iconic-bits * ln 2 * k ] kirby-log-priors counts)
end

to-report pair-log-likelihood [ l m f ]
  report ifelse-value (item m l = f) [ ln (1 - error-probability) ] [ ln (error-probability / 3) ]
end

to-report updated [ posterior m f ]
  report normalised (map + posterior (item (m * 4 + f) log-likelihoods))
end

to-report starting-pair
  let m random 4
  report list m (item m (item starting-language languages))
end

to-report trained [ prior n ]
  let posterior prior
  repeat n [
    let pair starting-pair
    set posterior updated posterior (first pair) (last pair)
  ]
  report posterior
end

;; ---------- production ----------

to-report produce [ posterior m ]
  let scores map [ f -> logsumexp map [ l -> item l posterior + item m (item l penalties) ] (item (m * 4 + f) languages-with) ] range 4
  let best max scores
  let form one-of filter [ f -> item f scores = best ] range 4
  if random-float 1 < error-probability [ set form one-of remove form range 4 ]
  report form
end

;; ---------- measures ----------

to-report type-mass [ posterior t ]
  report sum (map [ [ p ty ] -> ifelse-value (ty = t) [ exp p ] [ 0 ] ] posterior language-types)
end

to-report iconicity [ posterior counts ]
  report sum (map [ [ p k ] -> exp p * k ] posterior counts) / 4
end

to-report iconic-compositional [ posterior counts ]   ;; probability of the fully iconic compositional languages
  report sum (map [ [ p ty k ] -> ifelse-value (ty = 3 and k = 4) [ exp p ] [ 0 ] ] posterior language-types counts)
end

to plot-oldest [ agent ]
  let v [ vocal ] of agent
  let mp [ maps ] of agent
  set-current-plot "Vocalisations (oldest agent)"
  set-current-plot-pen "degenerate" plot type-mass v 0
  set-current-plot-pen "holistic" plot type-mass v 1
  set-current-plot-pen "compositional" plot type-mass v 3
  set-current-plot "Maps (oldest agent)"
  set-current-plot-pen "degenerate" plot type-mass mp 0
  set-current-plot-pen "holistic" plot type-mass mp 1
  set-current-plot-pen "compositional" plot type-mass mp 3
  set-current-plot "Iconicity (oldest agent)"
  set-current-plot-pen "vocal" plot iconicity v vocal-iconic-counts
  set-current-plot-pen "maps" plot iconicity mp map-iconic-counts
end

to-report starting-language-forms
  report map [ f -> item f forms ] (item starting-language languages)
end

;; ---------- Kirby et al. (2015) prior table (Simulating Language Lab 6, MIT License) ----------

to-report kirby-log-priors
  report [
    -0.9178860550328204 -10.749415928290118 -10.749415928290118 -11.272664072079987
    -10.749415928290118 -10.749415928290118 -16.95425710594061 -17.294055179550075
    -10.749415928290118 -16.95425710594061 -10.749415928290118 -17.294055179550075
    -11.272664072079987 -17.294055179550075 -17.294055179550075 -11.272664072079987
    -10.749415928290118 -10.749415928290118 -16.95425710594061 -17.294055179550075
    -10.749415928290118 -10.749415928290118 -16.95425710594061 -17.294055179550075
    -16.95425710594061 -16.95425710594061 -16.95425710594061 -12.460704095246543
    -17.294055179550075 -17.294055179550075 -20.83821243446749 -17.294055179550075
    -10.749415928290118 -16.95425710594061 -10.749415928290118 -17.294055179550075
    -16.95425710594061 -16.95425710594061 -16.95425710594061 -12.460704095246543
    -10.749415928290118 -16.95425710594061 -10.749415928290118 -17.294055179550075
    -17.294055179550075 -20.83821243446749 -17.294055179550075 -17.294055179550075
    -11.272664072079987 -17.294055179550075 -17.294055179550075 -11.272664072079987
    -17.294055179550075 -17.294055179550075 -20.83821243446749 -17.294055179550075
    -17.294055179550075 -20.83821243446749 -17.294055179550075 -17.294055179550075
    -11.272664072079987 -17.294055179550075 -17.294055179550075 -11.272664072079987
    -10.749415928290118 -10.749415928290118 -16.95425710594061 -17.294055179550075
    -10.749415928290118 -10.749415928290118 -16.95425710594061 -17.294055179550075
    -16.95425710594061 -16.95425710594061 -16.95425710594061 -20.83821243446749
    -17.294055179550075 -17.294055179550075 -12.460704095246543 -17.294055179550075
    -10.749415928290118 -10.749415928290118 -16.95425710594061 -17.294055179550075
    -10.749415928290118 -2.304180416152711 -11.272664072079987 -10.749415928290118
    -16.95425710594061 -11.272664072079987 -11.272664072079987 -16.95425710594061
    -17.294055179550075 -10.749415928290118 -16.95425710594061 -10.749415928290118
    -16.95425710594061 -16.95425710594061 -16.95425710594061 -20.83821243446749
    -16.95425710594061 -11.272664072079987 -11.272664072079987 -16.95425710594061
    -16.95425710594061 -11.272664072079987 -11.272664072079987 -16.95425710594061
    -20.83821243446749 -16.95425710594061 -16.95425710594061 -16.95425710594061
    -17.294055179550075 -17.294055179550075 -12.460704095246543 -17.294055179550075
    -17.294055179550075 -10.749415928290118 -16.95425710594061 -10.749415928290118
    -20.83821243446749 -16.95425710594061 -16.95425710594061 -16.95425710594061
    -17.294055179550075 -10.749415928290118 -16.95425710594061 -10.749415928290118
    -10.749415928290118 -16.95425710594061 -10.749415928290118 -17.294055179550075
    -16.95425710594061 -16.95425710594061 -16.95425710594061 -20.83821243446749
    -10.749415928290118 -16.95425710594061 -10.749415928290118 -17.294055179550075
    -17.294055179550075 -12.460704095246543 -17.294055179550075 -17.294055179550075
    -16.95425710594061 -16.95425710594061 -16.95425710594061 -20.83821243446749
    -16.95425710594061 -11.272664072079987 -11.272664072079987 -16.95425710594061
    -16.95425710594061 -11.272664072079987 -11.272664072079987 -16.95425710594061
    -20.83821243446749 -16.95425710594061 -16.95425710594061 -16.95425710594061
    -10.749415928290118 -16.95425710594061 -10.749415928290118 -17.294055179550075
    -16.95425710594061 -11.272664072079987 -11.272664072079987 -16.95425710594061
    -10.749415928290118 -11.272664072079987 -2.304180416152711 -10.749415928290118
    -17.294055179550075 -16.95425710594061 -10.749415928290118 -10.749415928290118
    -17.294055179550075 -12.460704095246543 -17.294055179550075 -17.294055179550075
    -20.83821243446749 -16.95425710594061 -16.95425710594061 -16.95425710594061
    -17.294055179550075 -16.95425710594061 -10.749415928290118 -10.749415928290118
    -17.294055179550075 -16.95425710594061 -10.749415928290118 -10.749415928290118
    -11.272664072079987 -17.294055179550075 -17.294055179550075 -11.272664072079987
    -17.294055179550075 -17.294055179550075 -20.83821243446749 -17.294055179550075
    -17.294055179550075 -20.83821243446749 -17.294055179550075 -17.294055179550075
    -11.272664072079987 -17.294055179550075 -17.294055179550075 -11.272664072079987
    -17.294055179550075 -17.294055179550075 -20.83821243446749 -17.294055179550075
    -17.294055179550075 -10.749415928290118 -16.95425710594061 -10.749415928290118
    -12.460704095246543 -16.95425710594061 -16.95425710594061 -16.95425710594061
    -17.294055179550075 -10.749415928290118 -16.95425710594061 -10.749415928290118
    -17.294055179550075 -20.83821243446749 -17.294055179550075 -17.294055179550075
    -12.460704095246543 -16.95425710594061 -16.95425710594061 -16.95425710594061
    -17.294055179550075 -16.95425710594061 -10.749415928290118 -10.749415928290118
    -17.294055179550075 -16.95425710594061 -10.749415928290118 -10.749415928290118
    -11.272664072079987 -17.294055179550075 -17.294055179550075 -11.272664072079987
    -17.294055179550075 -10.749415928290118 -16.95425710594061 -10.749415928290118
    -17.294055179550075 -16.95425710594061 -10.749415928290118 -10.749415928290118
    -11.272664072079987 -10.749415928290118 -10.749415928290118 -0.9178860550328204
  ]
end

to-report kirby-language-types
  report [
    0 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3 2 2 1 2
    2 2 2 2 2 2 2 3 2 2 2 2 2 1 2 2 2 2 2 2 2 2 1 2 2 1 2 2 2 2 2 2
    2 2 2 2 2 2 2 2 2 2 2 1 2 2 3 2 2 2 2 2 2 0 2 2 2 2 2 2 2 2 2 2
    2 2 2 1 2 2 2 2 2 2 2 2 1 2 2 2 2 2 3 2 2 2 2 2 1 2 2 2 2 2 2 2
    2 2 2 2 2 2 2 1 2 2 2 2 2 3 2 2 2 2 2 1 2 2 2 2 2 2 2 2 1 2 2 2
    2 2 2 2 2 2 2 2 2 2 0 2 2 2 2 2 2 3 2 2 1 2 2 2 2 2 2 2 2 2 2 2
    2 2 2 2 2 2 1 2 2 1 2 2 2 2 2 2 2 2 1 2 2 2 2 2 3 2 2 2 2 2 2 2
    2 1 2 2 3 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 0
  ]
end
@#$#@#$#@
GRAPHICS-WINDOW
1010
10
1183
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
steps
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

SLIDER
15
95
200
128
population-size
population-size
2
50
10.0
1
1
NIL
HORIZONTAL

SLIDER
15
135
200
168
interactions-per-step
interactions-per-step
1
100
20.0
1
1
NIL
HORIZONTAL

SLIDER
15
175
200
208
founder-training
founder-training
0
200
50.0
1
1
NIL
HORIZONTAL

SLIDER
15
215
200
248
record-size
record-size
1
400
80.0
1
1
NIL
HORIZONTAL

SLIDER
15
255
200
288
error-probability
error-probability
0.01
0.5
0.05
0.01
1
NIL
HORIZONTAL

SLIDER
15
295
200
328
communicative-rationality
communicative-rationality
0
100
100.0
1
1
NIL
HORIZONTAL

SLIDER
15
335
200
368
iconic-bits
iconic-bits
0
5
2.0
0.5
1
NIL
HORIZONTAL

MONITOR
15
380
100
425
step
ticks
0
1
11

MONITOR
105
380
200
425
start language
starting-language-forms
0
1
11

PLOT
215
10
600
250
Vocalisations (oldest agent)
step
posterior
0.0
10.0
0.0
1.0
true
true
"" ""
PENS
"degenerate" 1.0 0 -955883 true "" ""
"holistic" 1.0 0 -10899396 true "" ""
"compositional" 1.0 0 -8630108 true "" ""

PLOT
610
10
995
250
Maps (oldest agent)
step
posterior
0.0
10.0
0.0
1.0
true
true
"" ""
PENS
"degenerate" 1.0 0 -955883 true "" ""
"holistic" 1.0 0 -10899396 true "" ""
"compositional" 1.0 0 -8630108 true "" ""

PLOT
215
260
600
460
Iconicity (oldest agent)
step
share iconic
0.0
10.0
0.0
1.0
true
true
"" ""
PENS
"vocal" 1.0 0 -13345367 true "" ""
"maps" 1.0 0 -2674135 true "" ""

@#$#@#$#@
## WHAT IS IT?

A community of *iconographers* asks one another for four resources. Each resource is large-and-round or small-and-angular, and distant or near. They can ask with a two-vowel vocalisation (each vowel back rounded, "a", or front unrounded, "b") or with a map (a glyph showing size and shape, and a placement showing distance). Vocalisations vanish once heard; maps are kept in a record.

The model asks whether the community's expressions become compositional (one part for size and shape, one part for distance), as in sentences, and whether those compositional systems keep the iconic mapping of the argument: back rounded vowels, round glyphs and far placements for large, round, distant things.

## HOW IT WORKS

This is the population model of Kirby & Tamariz (2022) with the language space of Kirby et al. (2015).

- Every agent has a belief (a posterior probability) over all 256 possible vocal languages and over all 256 possible map languages.
- Learners prefer simple languages: the prior favours languages whose grammar is short (Kirby et al., 2015). Each iconic expression makes a language `iconic-bits` bits shorter; with `iconic-bits` = 0 the prior is exactly Kirby et al.'s.
- Each step, the oldest agent speaks `interactions-per-step` times. Each time, a resource is chosen at random; the oldest agent vocalises and draws a map for it. Another agent hears the vocalisation and updates its belief. The map goes into the record (the last `record-size` maps), and the same agent sees one map from the record and updates its belief about maps.
- Speakers choose the expression that is most likely to be right and, with `communicative-rationality` > 0, avoid expressions that are ambiguous. With probability `error-probability` they make a mistake.
- After each step, the oldest agent dies and a newborn, knowing only the prior, joins.

## HOW TO USE IT

Press **setup**, then **go**. The plots show the belief of the oldest agent (the current teacher): how much it believes in degenerate (one expression for everything), holistic (distinct but unstructured) and compositional languages, and the expected share of iconic expressions.

## THINGS TO TRY

- `communicative-rationality` = 0: degenerate languages take over (Kirby & Tamariz, 2022).
- `communicative-rationality` = 100, `iconic-bits` = 0: compositional languages emerge, slowly.
- `iconic-bits` = 2: compare iconicity and the speed at which compositional languages emerge.

## RELATED MODELS

The Python/Mesa version (src/iconographers/abm.py) follows the same rules. The computational model (src/iconographers/chain.py) analyses the same rules as a Markov chain.

## CREDITS AND REFERENCES

Kirby, S., Tamariz, M., Cornish, H., & Smith, K. (2015). Compression and communication in the cultural evolution of linguistic structure. *Cognition*, 141, 87-102. https://doi.org/10.1016/j.cognition.2015.03.016

Kirby, S., & Tamariz, M. (2022). Cumulative cultural evolution, population structure and the origin of combinatoriality in human language. *Philosophical Transactions of the Royal Society B*, 377(1843), 20200319. https://doi.org/10.1098/rstb.2020.0319

The prior table is copied from the Simulating Language course, Lab 6 (https://github.com/smkirby/simlang2024-25). MIT License, Copyright (c) 2019 Simon Kirby.
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
