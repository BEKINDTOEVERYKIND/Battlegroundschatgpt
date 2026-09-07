# Conservative complete-episode recruitment learning

This experiment initializes from the new lifetime-aware behavior-cloning actor
and learns from the outcomes of its own complete two-turn trajectories. It does
not use the earlier first-action Q ranker. This remains an audited Tier-1 fixture,
not a full Battlegrounds player or a claim of improved strength.

The actor samples a softmax over the unchanged canonical legal action shortlist
(at most 16 candidates). It controls every own recruitment decision. Opponents
use the fixed practical policy. Every command, including an explicit Discover
choice, goes through `ChoiceTimedTwoTurnOpeningEngine`: one command per second,
extra action delays and a five-second reserve. Own observed fixture timers are
explicitly synthetic 15, 20 or 30 seconds; opponent timers are 60 seconds.

Each sampled decision stores its visible 1,311-feature candidate matrix, selected
index, old action probabilities and probabilities from the immutable BC
reference. No opponent private state or future combat outcome enters features.
Training archives also retain actual commands, timing receipts and both combats.
The return is the mean of the first two combat outcomes: win 1, tie 0.5, loss 0.
A separate practical-policy trajectory uses the same fixture and combat seeds.
Its score is an action-independent baseline; the actor's action RNG is separate.

The minimized loss is the negative PPO-style clipped surrogate, plus
`0.1 × KL(frozen BC || current actor)` at every recorded decision. Ratios compare
the current policy with the actor that actually sampled the action. The complete
episode return minus the practical baseline is the advantage at every decision.
Decisions are summed within an episode, then episodes are averaged; there is no
action-count normalization that could change the objective as trajectory lengths
change. This is a Monte Carlo actor without a learned critic or bootstrapping.
The clipping construction and reuse of sampled data for bounded update epochs
follow [Schulman et al., Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).
The fixed-reference KL term and practical baseline are explicit choices for this
experiment, not claims that the original paper establishes these choices here.

Weights and feature normalization start from BC. Adam moments and its step are
explicitly reset because the objective changes; the BC checkpoint is copied and
hash-checked. A total gradient norm no larger than `1e-12` leaves weights, Adam
moments and its step unchanged. This prevents fresh Adam from amplifying numerical
KL roundoff when the actor equals its reference and the sampled advantage is
zero. Logs distinguish accepted collection batches, rejected batches, actual
optimizer updates and skipped zero-signal updates. Default preregistration is temperature 0.5, ratio clipping 0.15,
learning rate 0.0001, two update epochs, gradient norm limit 1, and 16 iterations
of eight paired episodes. Each iteration samples fresh fixture seeds; no prior
on-policy batch can be reused after its prescribed update epochs.

Unsupported transitions are not assigned invented game rewards. Every attempt is
archived, and any actor or control failure rejects that entire preregistered
update batch. No replacement fixtures are drawn. This avoids learning from only
the successful members of a batch, but accepted-batch learning is still
conditional on the supported engine scope. Counts of attempted/failed pairs and
rejected batches are mandatory results. Zero accepted batches means zero policy
gradient learning, regardless of whether a checkpoint was saved.

Snapshots after every fourth iteration and the unchanged BC reference are
evaluated as complete deterministic actors on the same fresh validation fixtures.
The practical policy remains the deployment fallback during validation. Existing
validation eligibility requires zero failures and a paired lower confidence
bound no worse than the declared 2.5-percentage-point noninferiority margin; the
highest mean eligible learner is frozen for test. If none is eligible, the highest
mean learner remains explicitly experimental. The selected checkpoint is frozen
before independent test episodes.
Every attempted validation/test fixture stays in the denominator; a policy gets
score zero for a fixture if any required sample fails. Both stochastic training
and deterministic evaluation are reported separately. A single preregistered
test gate can promote only that frozen gradient learner within the fixture: lower
95% paired confidence bound above zero versus both practical and frozen BC, zero
learner/both-control failures or timing violations, and a verified current
snapshot. Selecting unchanged BC cannot count as gradient improvement. A prior
BC promotion is preserved only when its saved decision references the exact
initial checkpoint hash; otherwise practical remains the fallback. Test results never tune temperature, clipping,
KL, epoch count or candidate selection.

Run the preregistered default experiment with a current verified BC checkpoint:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=python python scripts/train_recruit_policy_gradient.py \
  --engine-root /path/to/pinned/HSBRSIM \
  --initial runs/bc-experiment/experimental_model.json \
  --out runs/unique-policy-gradient-experiment --seed 2026090717
```

For a plumbing smoke, use `--iterations 2 --episodes-per-iteration 1
--checkpoint-every 1 --validation-episodes 2 --validation-samples 1
--test-episodes 2 --test-samples 1`. Such a smoke cannot establish playing strength.
The runner checks the live ruleset before and after execution by default.
`--allow-historical` explicitly disables those checks and marks results historical.

Numerical tests cover exact finite-difference gradients, both clipped directions,
unclipped gradients in the opposite directions, the frozen-reference KL's
restoring gradient, immutable stored features/probabilities, reproducible
sampling, optimizer/checkpoint continuation, and rejection of leaked or partial
training episodes. Actual complete-episode smoke results are saved separately;
they are not inferred from these synthetic tests.

The runner uses `CachedPolicyFactory` with at most 16 retained tribe database
views. It caches only a verified immutable export and reconstructs independent
definitions for each view. Separate tests compare complete seeded trajectories
against the original factory, including eviction and rebuild; the cache changes
setup cost while preserving commands, combat receipts, timing and provenance.

Three historical plumbing artifacts are preserved from September 7. The initial
`runs/20260907-policy-gradient-smoke` exposed Adam's amplification of a negligible
KL gradient with zero return advantage; its actors are diagnostic only. Repeating
the exact fixture seeds after the fix in
`runs/20260907-policy-gradient-zero-signal-regression` produced 116 sampled combats,
four explicit optimizer skips, and checkpoints whose every parameter still
exactly equals BC. The fresh eight-episode-batch run in
`runs/20260907-policy-gradient-cached-smoke` completed in 13.9 seconds with 348
sampled combats. Its first eight complete pairs produced two real optimizer
updates. An unverified Tavern Fugitive trigger rejected the entire second batch;
the next checkpoint preserved the exact preceding weights and optimizer. All
attempts and failures remain in the archives. These use the deliberately
undertrained BC plumbing checkpoint, retain practical deployment, and provide no
playing-strength evidence.
