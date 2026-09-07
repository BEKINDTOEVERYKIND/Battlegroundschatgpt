# Recruitment policy results, 7 September 2026

The new actor no longer repeats Freeze in either fresh test set. It completes
its two recruitment turns reliably, but neither experiment establishes that a
learned policy is stronger than the practical heuristic. Practical remains the
default recommendation. This is a two-turn opening result, not full-game strength.

Both main runs checked the live sources before and after execution. They used
the unchanged current Solo snapshot, patch 36.4.2.251332. The training pool
contains all 22 current Tier-1 minions and eight Tier-1 Tavern spells, filtered
to each lobby's five tribes. All 30 identities occur in verified accepted expert
observations. The fixture uses eight Patchwerk players; higher-tier shops,
seasonal choices and unverified contextual effects remain explicit frontiers.

## Completed experiments

| Experiment | Frozen test policy | Policy score | Practical score | Paired difference, percentage points | 95% interval |
| --- | --- | ---: | ---: | ---: | --- |
| Expert imitation | 32-unit actor, epoch 30 | 49.63% | 49.17% | +0.46 | [−1.15, +2.05] |
| Conservative follow-up | Unchanged cloned reference retained by validation | 50.07% | 50.81% | −0.73 | [−2.29, +0.68] |

Each test uses 128 independently seeded fixtures and eight complete samples per
policy and fixture. A sample's score is the mean of its first two actual combat
outcomes: win 1, tie 0.5, loss 0. Confidence intervals resample fixtures after
averaging their samples. Both tests completed all 1,024 samples per policy,
with zero failed samples, zero learned-policy Freeze commands and zero recorded
timing violations across all players. The test seeds differ between experiments.

The imitation run generated **512 complete expert trajectories and 3,999 training
decisions**, then compared widths 32/64 at epochs 10/30 using complete-policy
validation. All equally preferred expert actions are positive training targets;
these labels are not combat-value estimates. The selected checkpoint was frozen
before its test. This run performed **25,600 sampled combats** and finished in
1,647.8 seconds.

The follow-up began from that exact cloned checkpoint. It completed **16 batches
of eight paired training episodes**, made **32 optimizer updates**, and rejected
no training batch. Four gradient snapshots all scored 50.59% on validation,
equal to the unchanged cloned reference; the frozen tie-breaking order retained
the reference. Therefore the follow-up's independent test evaluates that reference,
not a newly selected gradient actor. There is no independent-test claim about
any rejected gradient snapshot. This run performed **24,576 sampled combats**
and finished in 501.7 seconds.

The two main runs total **50,176 sampled combats**, excluding plumbing smokes and
deterministic evidence reconstruction. The result supports reliable imitation
within this fixture. It does not support promoting a learned recruitment policy
or claiming faster adaptation after a future rotation.

## Evidence integrity and recovery

The original expert archive was truncated after 493 complete trajectories and
3,851 decisions, despite the run reporting 512/3,999. It lacked both the last
19 trajectories and a valid gzip footer. Its original bytes, manifest and
[integrity report](../reports/bc-expert-archive-integrity.json) are preserved.
The exact cause of the apparent file replacement during writing is unresolved;
ordinary gzip flushing does not explain the observed file timestamps.

Recovery regenerated only the 512 preregistered **training** seeds. All 493
available original records matched exactly as JSON, and refitting the recovered
3,999 decisions reproduced **all four original model files byte for byte**.
The selected model SHA256 remains
`d4dc1102cc10b6c3f7803559ee9a5241940ebe28e23e0d769bd645fa63b7807d`.
The complete recovered archive SHA256 is
`1da5de0ab814652b873e1c44f21f696a479bd81370c2ab8a247bbedc2884e72e`.
Recovery neither read holdout trajectories nor changed model selection.

The independent audits recomputed both test/validation score reports and their
10,000-bootstrap intervals from raw sampled receipts. They separately checked
the complete policy/seed/sample grids, source contents and hashes, checkpoint
identities, action counts, timing traces and split boundaries. The imitation
audit explicitly uses the verified recovered training data; its ordinary audit
mode continues to reject the damaged original. The gradient run's original
archives passed the full audit without recovery.

Run directories:

- [Imitation checkpoints, original evidence and verified recovery](../runs/20260907-policy-imitation-v1/)
- [Conservative learning checkpoints and complete evidence](../runs/20260907-policy-gradient-v1/)

Large archives are stored as exact verified parts. Restore them before auditing
or reusing their data; this does not change their original bytes:

```bash
python scripts/package_training_evidence.py unpack runs/20260907-policy-imitation-v1
python scripts/package_training_evidence.py unpack runs/20260907-policy-gradient-v1
python scripts/audit_recruit_policy_run.py --run runs/20260907-policy-imitation-v1 --verified-recovery
python scripts/audit_recruit_policy_run.py --run runs/20260907-policy-gradient-v1
```

For a future compatible imitation run, the reusable data directory is
`runs/20260907-policy-imitation-v1/recovered_expert_data/`, with the original
seed plan and source/schema requirements. The damaged original is diagnostic
evidence, not a reusable dataset. See the [independent audit procedure](recruit-policy-evidence-audit.md)
and [archive publication checks](trajectory-archive-integrity.md).

## What changes next

The original single-action continuation objective rewarded behavior that failed
when the learner controlled complete turns. The new objective and validation
correct that problem, and required target choices now reserve their own time.
Every buy, play, cast, target choice and other command remains separately charged:
one command per second, additional delays and a five-second reserve. The fixture
deliberately supplies synthetic 15/20/30-second own clocks and 60-second opponent
clocks; a live adapter must supply the player's actual remaining time.

The next useful expansion is longer faithful games. A separate tested prototype
now preserves Tarecgosa gains and Eternal Knight deaths across sampled combat,
including defeated players, and a separate Forest Rover repair limits its buffs
to Beetles. Completing the remaining Tier-2 effects and generated-card pools is
necessary before these components can support a broader training curriculum.
