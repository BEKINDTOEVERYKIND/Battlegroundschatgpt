# Finite-budget early recruitment curriculum

This adds a trained recruitment component to the positioning learner. It chooses
one legal first action from the player's current visible recruit state, then a
fixed practical heuristic performs the rest of that turn. It is **not a full-game
agent** and does not assign lobby-placement or MMR value to its decisions.

The environment executes actual commands in the pinned external HSBRSIM/hsrl2
engine, using a freshly constructed database from the current client snapshot.
The existing Firestone adapter evaluates each resulting board. No simulator
source is copied into this repository and no unsupported effect becomes a no-op.

## Exact scope

- Two recruit turns, stopping before turn-three Dark Gifts.
- Patchwerk and a fixed five-tribe lobby: Elemental, Naga, Pirate, Quilboar, Undead.
- The adapter's seven current minions and three current Tavern spells, plus
  explicit generated Blood Gems. The raw traces show which cards were actually
  encountered; this is an intentionally restricted curriculum pool.
- Initial boards and hands are empty. Turn-two states are reached by real
  turn-one purchases, plays, spells, choices, and combat transitions.
- Shop draws are sampled by the actual engine. Each candidate's first action is
  followed by the same fixed visible-state heuristic, including newly revealed
  shops. A candidate cannot use its future refreshed shop as a policy input.
- The opponent is one actual, same-turn heuristic board generated separately
  from a fork of the scenario. It is fixed across that scenario's candidates and
  is used only by the evaluator. Opposing private information is absent from
  policy features.

`TimedRecruitEngine.for_fixture` supplies the same finite resource accounting as
full recruit training will require. The root player's supplied remaining windows
are 10, 14, 20, 30, 45, or 60 seconds, with the configured five-second reserve and
one complete action per second plus per-command delays. These windows are
explicit synthetic input assumptions, **not a claim about Blizzard's timer table**.
Other fixture players receive a supplied 60-second window. Buy, play, sell,
refresh, spell cast, target choice, activate, upgrade, and moves are separate
commands. Forking copies the spent budget instead of refilling it.

Pending choices that reach an unverified timeout abort the scenario. So do
unvalidated triple transitions or other adapter scope failures. Failures are
preserved and excluded from training; their rate limits the experiment's scope.
The benchmark is conditional on **every shortlisted counterfactual branch**
remaining supported. It is not an estimate of performance on every sampled
early-turn state. The retained test set is fixed before neural choices and final
combat simulation.

## Learning and validation

The input encoder reuses semantic minion statistics, tribes, keywords, and effect
text hashing. It adds visible board/hand/shop summaries, seven ordered board slots, the candidate
action's card and target zone, scalar choice semantics, public spell-scaling
counters, activation-used flags, economic context, and remaining time/actions.
The v3 inputs also expose known per-command costs and whether an immediate
buy/play or buy/cast/target chain fits the remaining time. This is a visible
public-information estimate; the actual engine legal mask remains authoritative.
Legacy observations recover costs only when their profile hash matches the
known configuration; otherwise the input explicitly marks costs as unknown. Card IDs and entity
IDs do not index weights. IDs only identify which visible card an action refers
to. Unknown and unaffordable actions fail before encoding.

The first-action candidate list is fixed before evaluation. It includes the two
frozen heuristic baselines, legal buys, sells, spell casts, choices, activations,
refresh, freeze, and upgrade where available, with one canonical minion insertion
position. The list has at most 16 actions. The benchmark measures choosing within
that list, not exhaustive optimal planning.

Each scenario has its own game seed and belongs wholly to train, validation, or
test. Equivalent visible candidate-feature states are deduplicated globally
before training. Test labels never enter normalization, gradients, or model
selection. Hidden sizes and epoch checkpoints are declared before each run and compared
using validation combat score only. The completed v2 run used 16/32 hidden units;
the larger v3 run declares 32/64 units and checkpoints at 10/20/40 epochs. The selected test choices are saved
before a separate Firestone combat RNG stream produces final results.

The objective is `P(win) + 0.5 P(tie)` in the combat immediately following this
recruit turn. That deliberately limited target cannot properly value tavern
leveling, next-turn income, hand value, or lobby survival. The ranker must not
replace a future full-game value function on those decisions.

The saved candidate terminal boards and command sequences make the supervised
labels inspectable. The saved timing trace shows every charge. Final comparisons
use paired scenario bootstrap intervals; these do not include engine-model error
or transfer to a complete current minion pool.

## Reproduce

Use the exact clean external checkout declared in `config/recruit-engine.lock.json`.
The command performs current-source checks before and after the run by default:

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=python python scripts/train_recruit_curriculum.py \
  --engine-root /absolute/path/to/HSBRSIM \
  --out runs/20260905-recruit-v3 \
  --train-validation-attempts 8000 --holdout-attempts 2000 \
  --exclude-test-from runs/20260905-recruit-v2 --seed 202609053 \
  --hidden 32 64 --epochs 10 20 40 --trials 128 --evaluation-trials 2048
```

`--reuse` reuses the saved complete scenario archive and labels with an unchanged
preregistration. `--allow-historical` is an explicit offline reproduction option;
it cannot claim that the saved snapshot is still current.

Outputs include `preregistration.json`, `scenarios.jsonl.gz` with visible inputs
and full candidate traces, `generation_failures.json`, raw simulator labels, all
six candidate checkpoints, validation selection, frozen test choices, fresh test
outcomes, and `results.json`. `reports/recruit_curriculum_results.json` is the
compact result for the completed run.


## Completed v2 result

The first run preserved 1,341 supported unique scenarios and 592,384 combat
labels, then stopped before its first training epoch on an integer/float array
conversion error. Its failure record is retained. The correction explicitly
constructs floating-point score arrays.

V2 reencoded the original 820 training and 260 validation scenarios with reviewed
choice, target-zone, ordered-board, and counter features. It generated independent
new holdout seeds instead of reusing the first run's test states. Of 600 new
attempts, 327 were retained; 155 duplicated an existing visible state, 71 reached
an unverified pending-choice timeout, and 47 had fewer than two legal candidates.
No retained Firestone label or fresh-evaluation call failed.

The selected model scored 48.860% versus the practical heuristic's 48.397% on 327
retained test scenarios. The paired difference was **+0.463 percentage points**,
with a 95% interval of **−0.290 to +1.318**: there is no demonstrated improvement
over that baseline. It beat the simpler raw-stat first-action heuristic by
10.043 points (95% interval 7.674–12.483), which establishes only that narrower
comparison. The practical heuristic remains the baseline to beat.

This run used 620,288 training/validation/test-label combats, including reused
base labels, and 1,468,416 independently seeded final combats. The live client,
minion, hero, Tavern spell, Dark Gift, and known-issue source checks matched both
before and after the run. Ten commands was the largest observed per-turn count;
no recorded timing trace exceeded its budget. These early turns do not exercise
late-game APM compositions; the clock's repeating-loop tests cover termination.

Only five of the adapter's seven minions appeared in this distribution: Southsea
Busker, Razorfen Geomancer, Crackling Cyclone, Suspicious Prisonguard, and Risen
Rider. Shell Collector and Intrepid Botanist were absent, so this run does not
establish learned Choose One behavior. Many selected freeze commands leave the
same immediate combat board as a direct useful action or ending; this is a
short-horizon objective limitation and can waste time without improving combat.

`inspection_examples.json` contains the first five test scenarios and clearly
labeled diagnostic extremes, with initial state, selected commands, final boards,
timing charges, and independent combat outcomes. The examples are not a strength
estimate. `encode_legal_actions_v2` and `RECRUIT_V2_FEATURE_NAMES` retain the exact
frozen v2 encoder projection for separate transfer tests.

## Portable archives

Original per-row provenance made the traces unnecessarily repetitive. Each
completed run saves `scenarios.compact.jsonl.gz`, `scenario_provenance.json`, and
`scenario_archive.json`. The archive replaces only identical provenance objects
with content-addressed references. Restoring them reproduces the **exact original
uncompressed JSONL hash**. Gzip uses deterministic headers and compression. The
trainer can read this compact form directly when the original local archive is
absent; it likewise reads compressed simulator labels.

```bash
python scripts/archive_recruit_runs.py runs/20260905-recruit-v2 \
  --restore /absolute/output/scenarios.jsonl.gz
```

Raw labels, frozen choices, and fresh outcomes also have verified deterministic
gzip files and `artifact_archives.json` with original/compressed hashes. The raw
local copies are ignored only after their verified archives exist. The original
v1 failure record and every completed candidate checkpoint are retained.


## Inspect a trained recommendation

The advisor accepts an explicit checkpoint and a saved **visible** timed-fixture
observation. It checks the ruleset, hero, permitted cards, timing profile, budget,
and exact feature version. It does not execute the action or reset the clock.

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=python python scripts/recommend_recruit.py \
  --observation examples/recruit-observation-v2.json \
  --checkpoint runs/20260905-recruit-v2/selected_model.json
```

The saved example recommends buying Crackling Cyclone, while the practical
heuristic chooses Suspicious Prisonguard. Its supplied 45-second remaining window
has a five-second reserve: 40 usable seconds and at most 40 commands remain. The
buy costs 1.25 seconds under the configured profile. Candidate values are
uncalibrated ranking scores, not combat probabilities. The captured output is in
`examples/recruit-recommendation-v2.json`; it passed a live-source check.


## Completed v3 result

The larger preregistered run attempted 8,000 new training/validation states and
2,000 subsequent new holdout states, excluding previously evaluated v2 visible
states. It retained **2,659 train, 669 validation, and 469 test** scenarios. The
remaining attempts comprised 3,812 duplicates, 634 old-test states, 977
unsupported pending-choice timeouts, and 780 states with fewer than two choices.

The v3 features add public command costs and immediate follow-up affordability.
Six checkpoints compared 32/64 hidden units at 10/20/40 epochs using validation
only. Validation selected 64 hidden units at epoch 10. Training and initial labels
used 1,826,176 actual combat simulations; the frozen new holdout used another
2,371,584 independent simulations. The complete run took 1,672 seconds and was
not interrupted. Current-source checks matched before and after it.

The selected model scored **48.565%**, versus **48.338%** for the practical
heuristic: **+0.227 percentage points**, with a 95% interval of **−0.340 to +0.886**.
This again shows **no demonstrated improvement over the practical heuristic**,
so the model is not promoted. The comparison against raw-stat first actions was
+11.561 points (95% interval 9.467–13.685), a weaker baseline. This experiment does
not establish stronger full-turn or full-game play. The next useful milestone is
training against the expanded current opening pool and a separately registered
objective addressing redundant first-action sequences.

Large compact traces can be stored as `.part-000`, `.part-001`, and further
chunks. `scenario_archive.json` records every part's size/hash and the exact
reassembled gzip hash. Both the direct reader and restore command work without
the unsplit local file. All six v3 candidate checkpoints and the selected model
remain directly loadable JSON files. Every recorded v3 command trace respected
its finite budget; the largest observed per-turn count was 13 commands.
