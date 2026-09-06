# Two-turn recruiting pilot

This curriculum carries the actual shop, hand, board, frozen state, promised
Busker income, hero damage, counters and Spellcraft effects through the first
combat into a second recruit turn. It trains action values at every own decision,
then evaluates the selected neural policy choosing every action across both
turns. It is an engineering pilot with a restricted horizon and no full-game
strength or MMR claim.

The objective is the mean of the first and second sampled combat scores, with
win = 1, tie = 0.5 and loss = 0. Frozen shops and deferred gold can therefore
change the reward through their actual effects on the second turn. Remaining
gold and frozen state are not assigned invented terminal prices. Investment
beyond the second combat remains outside this objective.

All 22 current Tier-1 minions and eight current Tier-1 Tavern spells can occur,
subject to an explicitly sampled five-tribe lobby. The current snapshot is Solo
36.4.2.251332. All eight players are Patchwerk. Opponents use the existing
practical heuristic; pairings are declared synthetic fixtures, with player zero
facing player one on turn one and player two on turn two. This is not matchmaking
or full-lobby placement evaluation.

## Finite decisions and supported trajectories

The same mandatory timing wrapper applies to collection, candidate rollouts and
neural evaluation. It charges one complete command per second, the configured
extra animation/input delays, and a five-second reserve. Player-zero windows
are synthetic 15, 20 or 30 seconds; other players receive 60 seconds. These
inputs are not claimed to be a verified live turn-duration table. Clocks and
spent actions survive forks; combat starts a new clock for the next turn.

The declared shortlist is the existing maximum-16 candidate policy, including
legal upgrades, Freeze, both practical/raw-stat baselines, and canonical play
position zero; moves are omitted. A single unsupported shortlisted branch
rejects the entire training trajectory. Legal upgrades are not hidden to evade
the Tier-2 next-shop frontier. Failed paired evaluations are counted and both
policy results for that seed are excluded. Neither conditional scores nor
successful simulation counts are treated as a promotion gate.

Current engine frontiers include a Tier-2 refresh, ordinary triples,
pending-choice timeout, some contextual off-board effects and temporary
Spellcraft buffs merged by Magnetize. Native combat is disabled; one actual
seeded Firestone sample per pair produces a receipt bound to the full request.
Monte Carlo mean damage never mutates persistent hero state.

## Learning and reuse

The first iteration collects every player-zero decision along complete practical
heuristic trajectories. Each candidate is followed by that same policy through
the second combat. Every decision and candidate sibling stays in its original
trajectory split. Equivalent visible decision states across train/validation
splits reject the later entire trajectory. Validation averages decisions within
each episode before averaging episodes. The test plan and selected checkpoint
are frozen before new full-policy evaluation seeds are simulated.

The new schema has 1,129 semantic features. It retains the exact 1,118-feature v3
prefix and appends ten lobby tribe bits plus the exact visible promised gold.
Frozen state, action affordability and activation counters remain in the prefix.
Card IDs only locate visible entities; IDs, names, hidden hands, future shops,
combat receipts and RNG state do not enter features. Missing lobby/deferred-gold
fields fail closed rather than being interpreted as zero.

One known information gap remains: this schema does not encode the visible
duration of temporary enchantments or hand Spellcraft expiry. Equal current
stats can therefore encode identically while their second-turn persistence
differs. The simulator applies the actual duration rules; the policy cannot
always distinguish those states. A later explicitly versioned schema should
append duration features and migrate shared weights. The completed pilot's
schema and evidence remain unchanged.

`warm_start_two_turn_ranker` can expand exact frozen v2/v3 feature schemas while
preserving predictions, normalization, shared weights, Adam moments, optimizer
step and RNG state. Newly introduced inputs start at zero weight. The original
checkpoint is unchanged. This migration is tested; no faster learning or higher
strength from it has been established. Future checkpoints with the exact new
schema can resume directly across rotations once the current engine and card
pool have passed conformance checks.

The optional `--policy-checkpoint` uses a frozen, hash-recorded checkpoint to
collect every own decision and initializes a separate warm-fit copy. The source
model's parameters, normalization, optimizer and RNG remain unchanged during
collection and fitting. `--continuation-policy practical` is the default: each
candidate still receives the practical policy's complete buy/play continuation.
This collects learner-visited states without losing multiaction purchase value;
it is dataset aggregation, not a claim of on-policy action-value convergence.
Explicit `--continuation-policy checkpoint` instead uses the frozen checkpoint
for candidate continuation too. Opponents remain the same practical heuristic.
Both modes are recorded before execution. Current ruleset, exact feature schema
and timing profile must match the source checkpoint. Previously reserved source
test seeds cannot become new training, validation or test episodes. Checkpoints
retain the union of inherited and newly reserved test seeds.

After the original pilot, exception handling was tightened: only explicit
unsupported engine transitions and the explicit duplicate-state condition are
eligible for scenario rejection. Unexpected code, worker and file failures stop
the run. The exact source files executed by the original pilot are preserved in
`execution_sources.json.gz`, with SHA256 values in
`execution_source_manifest.json`.

## Completed current-patch pilot

The live source guards passed before execution at 14:11 UTC and afterward at
14:17 UTC on September 6, 2026. The preregistered run used 40 trajectory attempts,
two samples per candidate, a 16-unit hidden layer, and validation-selected
checkpoints at five and fifteen additional epochs.

| Measure | Result |
| --- | ---: |
| Supported complete trajectories | 33 of 40 |
| Train / validation trajectories | 27 / 6 |
| Rankable train / validation decisions | 199 / 46 |
| Minions / spells observed in accepted own states | 22 / 8 |
| Complete paired evaluation episodes | 14 of 16 |
| Samples per evaluated policy and episode | 4 |
| Neural mean two-combat score | 36.6071% |
| Practical heuristic mean two-combat score | 44.6429% |
| Neural minus practical | −8.0357 percentage points |
| Paired 95% bootstrap interval | [−21.4286, +6.2500] percentage points |
| Actual sampled combats, including rejected branches | 16,236 |
| Recorded time/action budget violations | 0 |

The model did not demonstrate an advantage and was not promoted. The interval
uses whole paired episodes, averaging combat samples within each episode first.
Seven training attempts were rejected: six for unresolved-choice timeout and
one for an unverified Tavern Fugitive trigger. Of two rejected evaluation seeds,
one failed under the neural policy on that Fugitive frontier, and one under the
practical policy on pending-choice timeout. Conditional scores do not measure
behavior on these omitted cases; failure rates must accompany any comparison.

Across 56 retained neural evaluation rollouts, it executed 100 buys, 84 plays,
40 spell casts, 40 choices, eight activations, 20 explicit turn endings and 1,164
Freeze commands. The clock bounded those loops, but the policy still wastes
time. Its actual buy/play behavior gives the next learner-state collection a
useful starting point. In the accepted turn-one decision labels, Freeze was
better than the practical first action in 27 cases, worse in 45 and equal in 27;
these low-sample diagnostic counts show the two-turn objective distinguishes
future shop value, not that Freeze has any universal value or penalty.

The selected checkpoint SHA256 is
`7028c293277a528932c18729f294b0510ab9d7bd70e606cb32332279947301a4`.
The measured pilot took about 6.4 minutes through evaluation, with a subsequent
successful live-source guard. All models and raw compressed evidence are under
`runs/20260906-two-turn-recruit-v1`.

## Reproduction and evidence

Run a fresh pilot with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/train_two_turn_recruit.py \
  --engine-root /path/to/pinned/HSBRSIM --out runs/new-two-turn-pilot \
  --trajectories 40 --label-samples 2 --evaluation-episodes 16 \
  --evaluation-samples 4 --seed 202609061 --hidden 16 --epochs 5 15
```

The command checks live source consistency before and after execution and
refuses to overwrite an existing preregistration. Explicit `--allow-historical`
is reserved for reproducibility/smoke work and marks the results accordingly.
The run records compressed actual requests, receipts, commands, timing traces,
visible decision observations, candidate features and scores, rejected seeds,
all candidate checkpoints, validation selection and independent evaluations.

The smoke under `runs/20260906-two-turn-recruit-v1/pipeline-smoke` is a plumbing
check with only two evaluated episodes. Its learned policy repeatedly selected
Freeze until the finite clock stopped it and performed worse than the practical
baseline. That is preserved as a failure of this tiny learned policy, not
suppressed or repaired by declaring Freeze globally illegal. The two-turn
objective does distinguish valuable and harmful freezes; learning repeated
decisions reliably still requires more data and subsequent on-policy iterations.

The subsequent `checkpoint-behavior-smoke` exercised the declared checkpoint
collection, practical continuation and separate warm-fit path in about 94
seconds. Four of ten complete trajectories survived, with 50 train and 69
validation decisions; six were rejected for pending-choice timeout. One of two
paired evaluation episodes survived. It ran 6,300 actual sampled combats with
zero timing violations. This is a plumbing check, not useful statistical
evidence of strength: its one-episode bootstrap interval is degenerate. The
original selected checkpoint remained unchanged. The source files used for
this smoke are archived alongside its preregistration and traces.
