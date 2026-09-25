# RB21 Self-Learning Driver

A neural driver that teaches itself to drive a 2025-spec Red Bull RB21 around
scale-accurate, elevation-aware reconstructions of Monza, Spa-Francorchamps and
the Circuit of the Americas — in the browser, with the car visible on track
while it learns.

```bash
python3 tools/serve.py 8777    # then open http://localhost:8777
```

Use `tools/serve.py` rather than `python3 -m http.server`: it sends `no-store`,
and a cached `app.js` against fresh weights silently rejects the checkpoint and
leaves an *untrained* policy driving, which looks like a trained one that
crashes instantly.

## The driver is Jev-shaped, not a Jev checkpoint

[Jev](https://beam.ai/agentic-insights/jev-typesafe-ai-agents) is TypeSafe AI's
non-autoregressive model for structured decisions. Rather than emitting text it
returns a **typed decision** plus a **calibrated confidence**, so the calling
code can act automatically or escalate to a human. It is trained with RLCD —
Reinforcement Learning for Calibrated Decisions — which scores a decision on
whether the outcome was actually correct, not on whether a human approved of it.

Jev is a decision model, so it cannot be dropped into a racing sim as a
continuous controller. What this project does is build the driver *in that
architecture*:

| Jev concept | Here |
| --- | --- |
| Typed decision | One of seven longitudinal decisions: `HARD_BRAKE`, `BRAKE`, `TRAIL_BRAKE`, `COAST`, `FEATHER`, `MODULATE`, `FULL_POWER` |
| Probabilistic output | Softmax over those seven; the argmax is emitted, the distribution is displayed |
| Calibrated confidence | A sigmoid head predicting P(this decision turns out well) |
| Escalate when unsure | Below 35 % confidence, control hands off to a reference fallback controller |
| Outcome-graded training (RLCD) | Every confidence is graded ~0.75 s later against what actually happened — did the car stay on track and inside the grip limit? — and the Brier score of those predictions is part of fitness |

Steering stays continuous, because it genuinely is; only the longitudinal
decision is typed.

**Escalation is off during training, deliberately.** A policy allowed to
escalate would converge on being permanently unsure and letting the fallback
drive — the confidence head has to be a prediction, not an exit.

## How it learns

Evolution strategies (Salimans et al. 2017): each generation samples antithetic
parameter perturbations around the current weights, rank-shapes the returns so
no single lucky run dominates, and steps with Adam. 2,649 parameters across a
25 → 40 → 32 → 9 MLP. This beat a genetic algorithm clearly at this parameter
count — the GA plateaued while ES kept climbing with the population mean tracking
the best, which is what convergence should look like.

Fitness is distance covered in a fixed time (so lap time is optimised directly)
minus 200 × the Brier score of the confidence head, and a crash **forfeits the
distance the car could still have made in the time remaining**.

That forfeit is not cosmetic. With a flat crash penalty, braking costs distance
immediately while surviving only pays off later, so carrying straight on into a
braking zone scored about as well as getting through it slowly. Every
configuration I tried stalled at the same metre of the same corner — 939 m at
Monza, the Variante del Rettifilo — for hundreds of generations:

```
flat penalty      gen  50   Q938x  r711x  r1077x  r851x
flat penalty      gen 200   Q939x  r712x  r1085x  r889x     <- 150 gens, no movement
time forfeit      gen 200   Q1975  r1452x r1986   r2333     <- braking zone solved
```

Making any survivor beat any crasher supplied the gradient that was missing.
Larger mutations (σ 0.09) were needed *with* it — σ alone changed nothing.

Everything is a **qualifying lap**. The timed start crosses the start/finish
line already at racing speed — 329 km/h at Monza, 246 at Spa, 226 at COTA,
taken from the limit lap's own speed at the line — carrying the momentum a
flying lap arrives with rather than launching from a standstill. The lap clock
starts at t=0, so the first crossing is a real timed lap.

The curriculum matters more than the optimiser:

- Start points are **fixed for the entire run**. Only episode length grows, set
  relative to each circuit's own limit lap (0.25× → 0.5× → 0.9× → 1.5×), so the
  final stage always has room to finish a lap whether the circuit takes 79 s or
  99 s.
- Rolling starts sit **on the racing line**, launched at a fraction of the
  reference speed — that speed belongs to the racing line's radius, not the
  centreline's.
- Rolling starts begin near racing pace. Starting slow produced a safe-and-slow
  local optimum that nothing pushed it out of.

**The expensive lesson:** an earlier curriculum swapped in *different* start
points for the final stage. A policy managing 3,482 m in 55 s collapsed to
~370 fitness and sat there for 300 generations. Rank-shaped ES has no gradient
when every probe fails in the same place, so it random-walked instead of
recovering. The first diagnosis — that the spawn was physically impossible —
was wrong; testing the reference driver from those same spawns showed it
survived all of them. The actual fault was changing the evaluation
distribution under the optimiser.

### What the driver can see

Speed, slip, yaw rate, position across the track, heading error, ten curvature
lookaheads from 6 m to 280 m, grade, vertical curvature, its own last inputs,
and how much of each axle's grip it is currently using.

Plus four **braking-demand** features: for points 40, 80, 150 and 240 m ahead,
how hard the car would have to decelerate — as a fraction of what the tyres can
actually deliver — to arrive at a speed that corner's radius can hold. This is
derivable from the curvature and speed the network already sees, but it needs a
square root and a comparison per lookahead, which is a lot to ask of a small
MLP and is the one thing it must get right. Adding it lifted fitness from 1910
to 2250 at generation 200 and took the policy from crashing on one of its four
start points to surviving all four.

## The circuits are real

- **Geometry and width**: TUM racetrack-database centrelines, in metres.
- **Georeferencing**: rigid ICP fit (rotation + translation, no scaling) onto
  OpenStreetMap `highway=raceway` ways. Median residual 1.3–1.5 m.
- **Elevation**: EU-DEM 25 m (Europe) and USGS NED 10 m (US) sampled every 10 m
  along the georeferenced centreline, smoothed over 25 m to remove tree-canopy
  and bridge artefacts.

Elevation is not decoration — grade changes the gravity component along the
track, and vertical curvature adds or removes tyre load over crests and through
compressions. Raidillon's climb lands at 1,290 m into the Spa lap, where it
belongs.

| Circuit | Length | Elevation range | Limit lap | Real 2025 pole |
| --- | --- | --- | --- | --- |
| Monza | 5.789 km | 18 m | 1:18.778 | 1:18.792 (Verstappen) |
| Spa | 6.999 km | 106 m | 1:39.364 | 1:40.562 (Norris) |
| COTA | 5.506 km | 30 m | 1:31.316 | 1:32.510 (Verstappen) |

## Results

**One set of weights drives every circuit.** The policy is trained across five
at once and never sees the two marked `*`, which exist only to measure whether
it transfers.

| Circuit | Unified policy | Physics limit | vs limit |
| --- | --- | --- | --- |
| Monza | 1:26.275 | 1:18.778 | +9.5 % |
| Zandvoort | 1:21.450 | 1:08.852 | +18.3 % |
| Spa | 1:57.575 | 1:39.364 | +18.3 % |
| Hungaroring | 1:28.900 | 1:13.623 | +20.8 % |
| COTA | 1:51.346 | 1:31.330 | +21.9 % |
| **Suzuka\*** | **1:42.325** | 1:23.241 | +22.9 % |
| **Shanghai\*** | dnf at 4,806 m of 5,445 | 1:27.107 | — |

`*` never trained on.

**Suzuka is the headline: a circuit the policy has never seen, driven start to
finish.** Shanghai gets 88 % of the way round and then fails, consistently, at
the exit of the Turn 1-3 spiral.

### What generalising costs

Against per-circuit specialists trained on nothing else:

| Circuit | Specialist | Unified | Cost |
| --- | --- | --- | --- |
| Monza | 1:25.075 | 1:26.275 | +1.2 s |
| Spa | 1:53.367 | 1:57.575 | +4.2 s |
| COTA | 1:34.500 | 1:51.346 | +16.8 s |

Monza and Spa give up little. COTA gives up a lot, and that is the honest
headline number for this trade: one policy that drives seven circuits is
meaningfully slower on at least one of them than a policy that drives only
that one.

### Transfer needs circuit diversity, not more generations

Trained on three circuits, the policy improved steadily on those three while
transfer went nowhere -- over 300 generations Suzuka stayed at ~2,400 m and
Shanghai at ~850 m. Adding Zandvoort and Hungaroring (tight and slow, closest
to Shanghai's character) moved Shanghai from 850 m to 4,806 m. That came from
diversity, not training time.

The policy has no position or track identifier as input, only local curvature,
grade and speed, so it cannot memorise a layout in the usual sense. What it
overfits to is the *distribution of corner shapes* three circuits happen to
present.

### Transfer is volatile, so the shipped weights are selected on it

Held-out performance is not monotonic: a checkpoint that gets round an unseen
circuit can lose that ability within a few generations while training fitness
barely moves. The two held-out circuits also trade off against each other --
configurations that suit Suzuka often collapse on Shanghai. `tools/select_by_heldout.js`
therefore evaluates checkpoints on the held-out circuits and keeps the best,
because selecting on training fitness is blind to exactly the property that
matters when you drop the model on a new track.

### Throttle and steering

An earlier build never used full throttle on two of the three circuits:

| | Before | After |
| --- | --- | --- |
| Monza, full throttle on straights | 92 % | 91 % |
| Spa | **0 %** | 83 % |
| COTA | **0 %** | 83 % |
| Spa top speed | 310 km/h | 345 km/h |
| COTA top speed | 295 km/h | 331 km/h |

The policy had been using `MODULATE` (0.70 pedal) as an implicit speed
governor. Forcing full throttle on demonstrably clear straights proved the time
was real — COTA dropped 1.8 s — but at Spa the car then *crashed*, because its
braking was tuned for the lower arrival speeds. It was co-adaptation, not
simply a stuck decision head.

Steering had a related problem: the command flipped sign on **56–76 % of
control steps**, and the car visibly weaved. The only constraint was a 50 ms
lag, which permits ~13 rad/s at the road wheel. Adding a 4 rad/s slew limit —
lock-to-lock in 0.17 s, which is what a rack and a pair of arms can actually do
— brought sign changes *at the wheel* down to 4–6 % and lateral wander to
4–7 cm per control step. Fixing that also unlocked the throttle: with the
degenerate control mode gone, the search found full power on its own.

## The car

Mass (815 kg), dimensions and power (735 kW) follow the 2025 FIA technical
regulations and published figures. Aero and tyre coefficients are not public, so
they were fitted (`tools/cal2.js`) until the quasi-steady-state limit lap lands
~1 % under real pole at all three circuits simultaneously. Top speeds fall out
at 362 / 352 / 336 km/h, which match reality without being targeted.

The vehicle model is a dynamic bicycle with load-sensitive Pacejka tyres,
longitudinal load transfer, a friction ellipse coupling grip between braking,
traction and cornering, altitude-corrected air density, and downforce. Three
findings from debugging it against the reference driver are baked in:

- A tyre at full longitudinal slip still points part of its force vector
  sideways. Capping `Fx` *on* the friction circle leaves a traction-limited axle
  with literally zero lateral stiffness, and any yaw disturbance diverges.
- Brake force is distributed in proportion to each axle's actual grip, plus a
  forward offset. A fixed split over-saturates the rear once load transfers
  forward; with a rearward CG that makes the car yaw-unstable in a straight
  line, which is precisely why real cars run forward bias.
- F1 rear tyres are wider than fronts (405 vs 305 mm), so the rear axle carries
  more lateral grip per newton of load. Without that the model spins under
  braking.
- Steering needs a slew-rate limit, not just a lag. A first-order lag alone let
  the policy bang the wheel lock-to-lock at controller rate and exploit the
  averaging, which is both unphysical and fragile to the timestep.

## What was tried and measured worse

Kept here because the negative results are as informative as the fixes, and all
of them looked reasonable beforehand.

- **Conjugate gradient with an active set for the racing line.** Solving the
  quadratic exactly instead of descending it reached objective 91 against the
  old 76: the first unconstrained solve pins far too many variables at the
  track edges and never releases them.
- **An offset-smoothness term on the racing line.** Lap times got monotonically
  worse as lambda rose. Related trap: a *lower* objective is not a faster lap —
  40k descent iterations reach a lower objective but a 0.85 s slower lap than
  the 6k actually used.
- **Sampling typed decisions from the softmax during training.** The reasoning
  was sound — argmax makes fitness piecewise-constant in the weights, so ES
  gets no gradient — but measured 2132 against 2402 for plain greedy, and still
  never found full throttle.
- **A control-effort penalty on steering.** At weight 300 the optimiser simply
  stopped steering and could not get round a corner; at 60 it scored 2190
  against 2402 without it. The rate limit had already solved what it was for.

Both switches survive in `driver.js` (`stochastic`, `chatterWeight`) because
the result was measured rather than assumed.

## What you see

The car on track, coloured by elevation, with its trail coloured by the typed
decision that was active. The Jev panel shows the live decision distribution,
the confidence, and a reliability diagram — each dot is a confidence bucket
plotted against how often those decisions actually worked out. On the diagonal
means honest: 70 % confidence really does work out 70 % of the time.

Trained weights ship in `weights/`, so the car drives properly on load. Pressing
**Start training** restarts learning from scratch in a worker and switches the
live car to the policy being trained, so you watch it improve generation by
generation.

## Layout

```
physics.js   track geometry, RB21 vehicle dynamics, observation vector
lapsim.js    minimum-curvature racing line + quasi-steady-state limit lap
driver.js    Jev-style policy, reference fallback, ES trainer
worker.js    training off the main thread
app.js       rendering, telemetry, live car
tools/       track builder, calibration, headless trainer
```

`tools/build_tracks.py` regenerates `tracks.js` from source data; it needs
`numpy` and network access to Overpass and OpenTopoData.
