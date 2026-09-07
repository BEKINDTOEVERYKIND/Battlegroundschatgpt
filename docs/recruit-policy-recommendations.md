# Offline recruitment policy advice

`scripts/recommend_recruit_policy.py` scores one saved visible observation using
a completed two-turn actor run. It honors `deployment_policy.json`: when that
file retains the practical baseline, the default recommendation uses practical.
The unpromoted actor is available only with an explicit `--experimental` flag.
If a run records a passing independent-test decision, its frozen selected actor
becomes the default within this fixture.

A later gradient run can retain an already approved BC actor as `bc_reference`.
The adviser checks its separate frozen checkpoint against the preregistered hash
and the archived prior passing selection. It does not replace that retained
baseline with an unsuccessful gradient candidate unless `--experimental` is
explicit.

```bash
PYTHONPATH=python python scripts/recommend_recruit_policy.py \
  --run-dir runs/20260907-policy-imitation-v1 \
  --observation examples/recruit-policy-observation.json
```

The adviser validates the frozen checkpoint checksum, exact 1,311-feature
schema, current ruleset, timing profile, choice-wrapper version, observed card
pool and explicit budget before prediction. It uses the existing legal shortlist
of at most 16 actions. Policy logits are relative action preferences, not combat
win probabilities or full-game expected values.

Use `--remaining-ms 12000` to supply a new actual timer reading. This can only
tighten the archived clock, and actions that can no longer complete their
mandatory choice sequence are removed from consideration. If an already pending
choice no longer fits, advice fails instead of inventing a timeout resolution.
The actual timer and legal game state must be checked again immediately before
acting; an archived snapshot is not a live client connection.

Output describes exactly one command, its cost, the remaining budget after that
command, and any separately reserved mandatory choices. For example, Alliance
Flag costs 1,500 ms to initiate and reserves two explicit 2,000 ms choices. Only
the initiation command is charged by that recommendation. Subsequent choices
require new observations and decisions. Ending a turn is marked explicitly;
any unspent clock shown afterwards belongs to the closed turn.

The committed example comes from a **training** decision in the historical
10-episode plumbing smoke, with provenance in
`examples/recruit-policy-observation-source.json`. Its two output examples show
the practical default and the deliberately undertrained experimental actor:

```bash
PYTHONPATH=python python scripts/recommend_recruit_policy.py \
  --run-dir runs/20260907-policy-imitation-smoke \
  --observation examples/recruit-policy-observation.json \
  --allow-historical --out examples/recruit-policy-practical-recommendation.json

PYTHONPATH=python python scripts/recommend_recruit_policy.py \
  --run-dir runs/20260907-policy-imitation-smoke \
  --observation examples/recruit-policy-observation.json \
  --experimental --allow-historical \
  --out examples/recruit-policy-experimental-recommendation.json
```

Historical mode skips live patch checks and is labeled in the output. These
examples establish the interface and fallback behavior, not playing strength.
The tool executes no game action. Its scope remains the current Tier1 two-turn,
eight-Patchwerk fixture; it provides no full-game, seasonal or MMR claim.
