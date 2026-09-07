# Complete-trajectory recruitment policy training

`scripts/train_recruit_policy.py` trains a separate actor by imitating the
frozen practical heuristic. It addresses the previous ranker's tendency to
repeat Freeze by directly teaching action choice on complete trajectories.
These are binary teacher-preference labels, **not combat win probabilities**.
Improved combat performance is tested separately and is never assumed from
classification accuracy or lower training loss.

## Data and acting

Each expert episode executes both recruit turns and both sampled Firestone
combats for eight Patchwerk players. No counterfactual combat labels are
generated for imitation. Every own decision uses the existing bounded shortlist
(`candidates_for`, at most 16, practical action always first). Every action tied
for the highest frozen heuristic value is in the positive teacher set.
`policy_learning.fit_policy` minimizes minus log probability assigned to that
set, with equal decision weights. It fits training episodes only.

The actor receives the 1,311-feature v2 visible observation, including audited
temporary enchantment lifetimes. It controls every own decision in evaluation.
Opponents retain the practical policy. RNG, sampled receipts, future shops and
opponent private information never enter the actor. Training archives retain
actual observations, actions, budgets and combat requests/receipts for audit.
Archives may therefore contain evaluator information that is not an input.

`ChoiceTimedTwoTurnOpeningEngine` reserves time for known mandatory choice
commands and charges each command explicitly. Existing one-command-per-second,
extra-action-delay and five-second-reserve settings remain active. Timer windows
are declared synthetic 15/20/30 seconds for player zero and 60 seconds for the
other players. An unsupported transition rejects the complete expert trajectory;
partial evidence and the failure reason are retained separately. Unexpected
errors fail the run.

## Selection and independent test

All expert episodes belong to training. Fresh validation seeds are the base seed
plus 20,000,000, then increments of 1,009; fresh test seeds use +30,000,000. Seed
families are checked for overlap. The initial run trains widths 32/64 through
epochs 10/30 unless explicitly preregistered otherwise.

Every checkpoint plays complete validation episodes on the same seed/sample
plan as the practical control. An episode must complete every requested sample;
otherwise that policy receives zero for that episode. Every attempted seed stays
in the mean's denominator. Candidate-specific failures and estimates restricted
to the supported intersection are both reported, so an actor cannot gain a
selection advantage merely by failing on difficult games.

Validation freezes one experimental learner: prefer zero-failure candidates
whose paired lower 95% bound is at least −2.5 percentage points, then choose the
highest full-policy mean. If no candidate passes that screen, retain the best
failure-penalized learner as experimental. Practical remains recommended during
validation. Small validation sets and model comparisons do not establish a
population guarantee.

The selected checkpoint and exact test plan are saved before any test games.
The test compares only that frozen learner with practical; it cannot select a
different checkpoint. One preregistered gate recommends the learner **within
this fixture only** if its independent paired 95% lower bound exceeds zero,
both policies have zero failed episodes and zero timing violations, and the
live ruleset passed preflight and postflight. Otherwise practical remains the
deployment choice. The experimental network is preserved regardless, for later
training. There is no full-game or MMR promotion.

## Running and reusing data

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=python \
python scripts/train_recruit_policy.py \
  --engine-root ../research/HSBRSIM \
  --out runs/unique-policy-run \
  --trajectories 512 --hidden 32 64 --epochs 10 30 \
  --validation-episodes 32 --validation-samples 4 \
  --test-episodes 128 --test-samples 8 --seed 202609071
```

Use `--data runs/prior-matching-expert-run` to reuse expert data with the same
base seed and attempted training seed plan. The archive checksum, ruleset,
reference cards, timing, choice wrapper and feature schema must match. Reused
observations are re-encoded and teacher labels recomputed for equality.
Validation and test episodes cannot be imported as training decisions. This
initial version does not warm-start from older checkpoints.

Each output directory is immutable once populated. Preregistration and exact
execution-source text/hashes are saved before generation or fitting, alongside
compressed expert/validation/test traces, model checkpoints, manifests,
failures, selection and result summaries. `--allow-historical` supports local
engineering smoke tests and explicitly prevents promotion by the live-snapshot
gate. It is not current-patch validation.
