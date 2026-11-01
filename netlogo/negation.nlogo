;; The Iconographers: negation submodel (NetLogo version)
;; Same rules as src/iconographers/negation.py. See the Info tab.
;;
;; Steinert-Threlkeld, S. (2016). Compositional signaling in a complex world. Journal of Logic, Language
;; and Information, 25(3-4), 379-397. https://doi.org/10.1007/s10849-016-9236-9
;; Steinert-Threlkeld, S. (2017). Communication and computation: New questions about compositionality
;; (Doctoral dissertation, Stanford University), chapter 2, sections 2.3 and 2.5.
;; https://eprints.illc.uva.nl/id/eprint/2145/
;;
;; States 0-3: resource r present; states 4-7: resource r absent. Signals 0-3: basic m_r; 4-7: a m_r.

breed [ senders sender ]
breed [ receivers receiver ]

senders-own [ urns ]             ;; for each absent state, an urn over the complex signals a m_0 .. a m_3
receivers-own [ function-urn ]   ;; urn over interpretations of a: identity, constant, negation

globals [ constant-target ]

to setup
  clear-all
  set constant-target ifelse-value (constant-points-to = "resource 0 present") [ 0 ] [ 4 ]
  create-senders 1 [ set urns n-values 4 [ n-values 4 [ 1 ] ] hide-turtle ]
  create-receivers 1 [ set function-urn [ 1 1 1 ] hide-turtle ]
  reset-ticks
  plot-measures
end

to go
  let state random 8
  let signal send state
  let response respond signal         ;; [act function], function = -1 for a basic signal
  let act first response
  let payoff ifelse-value (act = state) [ 1 ] [ 0 ]
  if state >= 4 [
    ask one-of senders [ set urns replace-item (state - 4) urns (add-to (item (state - 4) urns) (signal - 4) payoff) ]
  ]
  if last response >= 0 [
    ask one-of receivers [ set function-urn add-to function-urn (last response) payoff ]
  ]
  tick
  plot-measures
end

to-report negated [ state ]   ;; minimal negation: resource r present <-> resource r absent
  report (state + 4) mod 8
end

to-report interpreted [ function act ]   ;; 0 identity, 1 constant, 2 negation
  if function = 0 [ report act ]
  if function = 1 [ report constant-target ]
  report negated act
end

to-report draw [ weights ]   ;; index drawn with probability proportional to its weight
  let r random-float sum weights
  let i 0
  let total item 0 weights
  while [ r >= total ] [
    set i i + 1
    set total total + item i weights
  ]
  report i
end

to-report add-to [ weights index amount ]
  report replace-item index weights (item index weights + amount)
end

to-report send [ state ]
  if state < 4 [ report state ]   ;; signaling system on the present states
  report 4 + draw item (state - 4) [ urns ] of one-of senders
end

to-report respond [ signal ]
  if signal < 4 [ report list signal (- 1) ]
  let function draw [ function-urn ] of one-of receivers
  report list (interpreted function (signal - 4)) function
end

;; ---------- measures ----------

to-report function-probabilities
  let urn [ function-urn ] of one-of receivers
  report map [ w -> w / sum urn ] urn
end

to-report expected-payoff   ;; probability of success under the current strategies
  let f function-probabilities
  let absent-success 0
  foreach range 4 [ r ->
    let urn item r [ urns ] of one-of senders
    foreach range 4 [ i ->
      let p-signal item i urn / sum urn
      let p-act (ifelse-value (i = r) [ item 2 f ] [ 0 ]) + (ifelse-value (constant-target = 4 + r) [ item 1 f ] [ 0 ])
      set absent-success absent-success + p-signal * p-act
    ]
  ]
  report (4 + absent-success) / 8
end

to-report sender-uses-negation   ;; mean probability of sending a m_r for 'resource r absent'
  let urns-now [ urns ] of one-of senders
  report mean map [ r -> item r (item r urns-now) / sum item r urns-now ] range 4
end

to plot-measures
  set-current-plot "Learning negation"
  set-current-plot-pen "expected payoff" plot expected-payoff
  set-current-plot-pen "receiver: a = negation" plot item 2 function-probabilities
  set-current-plot-pen "sender: a m_r for absent r" plot sender-uses-negation
end
@#$#@#$#@
GRAPHICS-WINDOW
790
10
963
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
plays
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
constant-points-to
constant-points-to
"resource 0 present" "resource 0 absent"
0

MONITOR
15
150
100
195
plays
ticks
0
1
11

MONITOR
105
150
200
195
expected payoff
expected-payoff
3
1
11

PLOT
215
10
775
330
Learning negation
plays
probability
0.0
10.0
0.0
1.0
true
true
"" ""
PENS
"expected payoff" 1.0 0 -16777216 true "" ""
"receiver: a = negation" 1.0 0 -2674135 true "" ""
"sender: a m_r for absent r" 1.0 0 -13345367 true "" ""

@#$#@#$#@
## WHAT IS IT?

The negation submodel of the iconographers. Each of the four resources can be present or absent. A sender and a receiver already have words for the four resources being present. The question is whether they learn to use a function word, *a*, as negation: "*a* + word for resource r" meaning "resource r is absent".

This is the basic negation learning setup of Steinert-Threlkeld (2016; 2017, chapter 2, section 2.5).

## HOW IT WORKS

- Each play, a state is chosen at random (one of the 4 resources present or absent).
- If a resource is present, the sender says its word, and the receiver understands it.
- If a resource is absent, the sender draws a signal "*a* + word" from an urn for that state.
- On hearing "*a* + word for resource i", the receiver draws an interpretation of *a* from its urn: ignore it (identity), treat "*a* + word" as one fixed new word (constant), or as negation. It then acts accordingly.
- If the act matches the state, both add a ball of the kind they drew to the urn they drew from (Roth-Erev reinforcement). All urns start with one ball of each kind.

## HOW TO USE IT

Choose where the receiver's "fixed new word" interpretation points, press **setup**, then **go**. The thesis runs 1000 plays.

## THINGS TO NOTICE

Published values after 1000 plays (100 trials): expected payoff 0.978 and probability of reading *a* as negation 0.995 when the constant points to a present state; 0.870 and 0.807 when it points to an absent state.

## CREDITS AND REFERENCES

Steinert-Threlkeld, S. (2016). Compositional signaling in a complex world. *Journal of Logic, Language and Information*, 25(3-4), 379-397. https://doi.org/10.1007/s10849-016-9236-9

Steinert-Threlkeld, S. (2017). *Communication and computation: New questions about compositionality*. Doctoral dissertation, Stanford University. https://eprints.illc.uva.nl/id/eprint/2145/
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
