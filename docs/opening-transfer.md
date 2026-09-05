# Frozen-model transfer to the current opening pool

This experiment evaluates the existing restricted-pool recruit-v2 model on
previously unseen **current** cards. It performs no training. It does not measure
adaptation to a future rotation, faster retraining, whole-turn neural play, or
full-game strength.

The fixed checkpoint is
`runs/20260905-recruit-v2/selected_model.json`, SHA-256
`b0ae096c38f3c2c04a4c8f70157b305b54f701adcf5bc15d3eccb96adacf787a`.
The benchmark requires its exact v2 feature names, ordering, and metadata.
`encode_legal_actions_v2` retains that representation when later encoders add
features. An incompatible model or schema aborts evaluation.

## Scope and fixed design

- The opening database offers all **22 current Tier-1 minions and 8 Tier-1
  Tavern spells**, filtered into ten predetermined five-tribe lobbies. Each tribe
  appears in five of those lobbies; a dual-tribe card is eligible when either
  tribe is present.
- Cards with unresolved play transitions remain in the offering pool, but
  playing them aborts a trajectory. The exact blocked IDs and tested playable
  count are taken from the engine at preregistration and included in the result.
  Pool membership is not presented as complete rules validation.
- Every player starts as Patchwerk in an actual first-turn recruitment state.
  A predetermined random-length legal heuristic prefix creates decisions with
  purchased cards, generated spells, and pending choices. Initial hand and board
  contents are not fabricated to make the policy look stronger.
- One model-selected first action is followed by the same frozen practical
  heuristic used for both baseline continuations. The alternatives are the
  practical heuristic's first action and a raw-stat first action. This evaluates
  the decision boundary that the source model was trained to handle.
- The shortlist includes both baselines, legal economic/choice actions,
  canonical board insertion at position zero, and explicit magnetic plays.
  Board movement is outside this first-action shortlist. Every executed command,
  including target/discovery choices, consumes the shared finite turn budget.
- All eight recruit turns finish before combat snapshots are taken. End-of-turn
  effects and expired Spellcraft are resolved. Firestone receives both the
  complete board and **actual hand**, including necessary enchantment context.
  Opponent hand/board snapshots are evaluator data and never policy features.
- A fixed opponent from the practical baseline branch is shared by every
  candidate. The primary score is first-combat `P(win) + 0.5 P(tie)`; it cannot
  value saved economy or Tavern upgrades fairly over later turns.

## Holdout and reporting protections

The predeclared target is 600 supported unique decision states, with at most
2,400 attempted states. The scenario seed is `202609057`. States matching any
saved source-dataset v2 decision fingerprint are excluded, as are duplicates
within this holdout. Source **training** observations determine which card IDs
are classified as unseen; configured pool membership alone does not imply the
model saw that card during training.

Inclusion is conditional on every shortlisted recruitment branch being
supported. Unsupported plays, triples, refreshes beyond the validated tier,
unresolved choices at timeout, and other failures are recorded with their
original scenario index and reason. They do not receive invented rewards.
Because counterfactual failures can depend on future draws, confidence intervals
apply to this conditional survivor population, not every sampled opening state.

All model choices are frozen before final combat simulation. The independent
final combat seed is `982453021`, mixed with the original scenario index. Every
selected policy receives 1,024 actual combats per distinct snapshot. Missing or
failed final evaluations abort the result rather than quietly removing those
test states. Result validation checks scenario order, frozen candidate indices,
trial counts, RNG metadata, and unchanged model/choice checksums.

Paired differences are bootstrapped over whole scenarios with 10,000 resamples.
A positive lower 95% bound against both baselines would support only this
conditional opening benchmark. Statistical intervals do not include simulator
model error or unvalidated interactions.

## Reproduce and inspect

```bash
PYTHONPATH=python python scripts/evaluate_opening_transfer.py \
  --engine-root ../research/HSBRSIM \
  --out runs/20260905-opening-transfer-v1
```

The script refuses to overwrite an existing preregistration. Use a fresh output
directory for a new independent evaluation. It checks current-source status
before and after the run; historical evaluation requires an explicit flag and
is labeled accordingly.

Artifacts include the preregistration, exact candidate features and complete
recruit/timing traces, actual combat snapshots, rejection log, frozen choices,
raw combat outcomes, paired intervals, and observed/terminal card coverage.
`inspection_examples.json` contains post-evaluation examples with the largest
positive and negative differences; those examples never tune this frozen model.

`tests/test_opening_transfer.py` verifies checkpoint/schema compatibility, paired
interval behavior, balanced tribe coverage, and rejection of missing, reordered,
wrong-seed, wrong-trial, or changed-selection final results.

## Completed result: September 5, 2026

The frozen model **did not show a transfer advantage**. The run accepted 600
unique states from 623 attempts and completed 686,080 independent final combats.
Current-source checks passed before and after evaluation. No training, model
selection, or tuning used these results.

| First-action policy | Mean first-combat score | Model minus baseline, 95% paired interval |
| --- | ---: | ---: |
| Frozen recruit-v2 | 0.511413 | — |
| Practical heuristic | 0.514340 | −0.002927 [−0.012066, +0.005771] |
| Raw stats | 0.510596 | +0.000817 [−0.010154, +0.011523] |

The model selected Freeze in **496 of 600** decisions. Every one of those
continuations produced the exact same complete combat snapshot as the practical
baseline; 563 of all 600 model snapshots matched that baseline exactly. The
fixed continuation performs most subsequent recruitment decisions. Similar
scores therefore do not establish effective autonomous play on unseen cards.
First-combat scoring does not measure Freeze's effect on the next shop.

All 22 current Tier-1 minions appeared in decision observations and reached
candidate player combat boards. All eight Tier-1 Tavern spells appeared and were
actually bought and cast in player-zero candidate continuations. These are
coverage of the evaluated alternatives, not evidence that the model learned to
use each card: its selected continuations contained 20 of the 22 minions, and
it never selected a spell cast as its first action. The model's other first
actions were 49 plays, 49 buys, and six choices. Of the accepted decisions, 597
contained card IDs absent from the source training observations, and 147 had a
nonempty actual owned hand.

The 23 rejected attempts comprised two unsupported off-board Scarlet threshold
interactions, nine unresolved choices at timeout, eight Tavern Fugitive spell
trigger interactions, and four duplicate visible states. No accepted trajectory
exceeded its timing budget; the maximum recorded usage was eight charged actions
in a turn. The source file snapshots, checkpoint, frozen choices, combat results,
and rejection log remain under `runs/20260905-opening-transfer-v1/`.
`descriptive_coverage.json` records the post-evaluation coverage and identical
snapshot counts; it does not change any inclusion, policy choice, or score.
