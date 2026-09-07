# Why the two-turn learner repeatedly freezes

The two-turn engine did not fix the mismatch between the learner's labels and its deployed behavior. A candidate receives a label for **one action followed by the practical heuristic**. The deployed policy instead makes every subsequent decision itself. The old validation selector also measures the one-action deviation, so it can prefer a policy that performs worse when actually controlling both turns.

The completed second run's large loss motivated this investigation. The diagnostic below reads **only the earlier pilot's training/validation archive**: 27 training and six validation trajectories. It never opens either run's held-out evaluation archive. Existing sources and checkpoints are hashed before and after and remain unchanged.

## Direct evidence from training labels

| Diagnostic | Train | Validation |
| --- | ---: | ---: |
| Decisions with at least two candidates | 199 | 46 |
| Exactly tied candidate pairs, receiving zero pairwise gradient | 1,013 / 2,741 | 255 / 621 |
| Freeze has exactly the practical action's label | 129 / 192 | 34 / 43 |
| Freeze has exactly the same complete remaining combat requests | 111 / 192 | 25 / 43 |
| Freeze positively / negatively ranked against other actions by labels | 447 / 98 | 99 / 10 |

For every matched complete-combat request above, Freeze spends an extra **1,000 ms** before the heuristic takes over. This comparison checks all sampled remaining combat requests, including hands and opponents, rather than only comparing equal Monte Carlo scores. It does not claim that frozen shops or recruitment states beyond the two-turn objective are identical.

The existing loss ignores exact ties even with `tie_tolerance=0`. Freeze often inherits a strong continuation and beats weak alternatives, but its ordering against a tied progress action is unconstrained. For example, the first audited training state matching these conditions, `two-turn-202609061-decision-7`, labels both end turn and Freeze at 0.75. Freeze has a predicted ranking margin of 1.4477 over ending the turn, despite costing another second. This is a strict learned preference, not an `argmax` tie artifact.

True exact policy improvement does not inherently fail merely because labels use a fixed continuation. Here the finite, approximate ranker, missing supervision between tied actions, shifted visited states, and one-deviation validation leave a practical failure mode unchecked. The measurements establish that mismatch in this implementation; they do not isolate the contribution of each approximation.

## Small executable control

A fixed diagnostic fits a fresh 16-unit ranker for 15 epochs on the same 199 training decisions, with target 1 for the practical expert's action and 0 for other legal candidates. Candidate zero is verified against the actual heuristic for every record. The heuristic is a training target only: learned inference still uses the normal legal candidate list and neural argmax. Freeze remains legal.

All three policies then control both turns on the six **development validation** episodes, using four independently specified combat sample indices per episode. No checkpoint is selected or promoted by this diagnostic.

| Policy | Old one-deviation validation score, episode weighted | Actual two-turn validation score | Freeze commands in 24 rollouts |
| --- | ---: | ---: | ---: |
| Frozen pilot ranker | 0.48967 | 0.30208 | 688 |
| Fresh expert imitation | 0.47396 | 0.43750 | 0 |
| Practical heuristic | — | 0.45833 | 0 |

The old selection metric favors the frozen ranker over imitation, while actual repeated-policy execution reverses that ordering. Imitation agrees with the expert on 95.65% of validation decisions and removes the Freeze loop in this small sample. Every policy respects the time/action budget; there are zero violations. These six episodes are too small for a playing-strength claim. Initialization and training history differ between the frozen ranker and fresh imitation, so this is a diagnostic rather than a matched causal ablation.

## Implemented next experiment

Use supervised imitation as a functioning initial controller and preserve the semantic representation for later improvement. Select every candidate by **complete repeated-policy validation episodes**, then freeze the selection before a fresh test. Keep the practical policy as an explicit control. A subsequent improvement stage should produce training-only better-action targets from verified counterfactual rollouts; equivalent actions may form a positive set, avoiding arbitrary ordering among duplicated teacher choices. The new `policy_learning.py` objective assigns probability to that whole positive set.

Imitating this expert cannot teach strategic Freeze usage that the expert never demonstrates. That is why imitation is an initialization/control and not the endpoint, and why neither Freeze nor any other economic action is removed from the legal mask. The two-turn horizon, current Tier-1 fixture, synthetic timer assumptions, and unsupported transition exclusions still limit the scope.

Reproduce the diagnostic with:

```bash
OPENBLAS_NUM_THREADS=1 python scripts/audit_two_turn_policy_collapse.py \
  --engine-root /absolute/path/to/pinned/HSBRSIM
```

The machine-readable evidence is `reports/policy-collapse-audit-sept7.json`.
