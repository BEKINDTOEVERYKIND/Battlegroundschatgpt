# Recruit objective audit

The priority is **a verified transition into the next recruit turn and an explicitly two-turn opening objective**. More labels for the current one-combat target cannot teach the future value of a frozen shop, saved cards, or deferred income. A small command-efficiency experiment is useful as a separate, limited ablation; it cannot establish that Freeze is a bad strategic action.

This review inspected individual records only from the completed recruit-v2 **820 training and 260 validation scenarios**. The completed opening-transfer report's 496 Freeze selections out of 600 tests are descriptive context, not data for selecting a new policy. No model, labels, legal masks, frozen source files, or running v3 experiment were changed.

## What the current learner is rewarded for

`scripts/train_recruit_curriculum.py` executes the candidate's first command, then `finish_player` repeatedly follows the same practical heuristic until this recruit turn ends. Firestone labels the resulting combat against one fixed opponent with `P(win) + 0.5 P(tie)`. The policy inputs are the original visible observation and candidate action; the realized continuation is evaluator information.

`Ranker._loss_gradient` in `python/bg_ai/learning.py` uses a pairwise logistic loss, weighted by absolute combat-score gaps and divided by the number of all candidate pairs. Pairs with an absolute gap **≤0.02** contribute zero. Entire scenarios whose labels fit inside that tolerance contribute no pairwise training signal. Neither milliseconds nor commands spent during continuation enter this target.

This rewards a broadly available Freeze command whenever the heuristic continuation after it beats a weak alternative such as selling a useful minion. It supplies no ordering preference between Freeze-plus-heuristic and the direct heuristic action when their combat labels are equal. Adding visible time-cost features, as v3 does, helps describe timing feasibility; it does not create a preference missing from these labels.

## Measured train/validation evidence

| Diagnostic | Training | Validation |
| --- | ---: | ---: |
| Scenarios | 820 | 260 |
| Freeze available | 819 | 259 |
| Freeze has the practical action's exact recorded combat projection and label | 819 | 259 |
| Extra paid time for those Freeze continuations | 1,000 ms each | 1,000 ms each |
| Extra charged commands for those continuations | 1 each | 1 each |
| Frozen model selects Freeze | 461 | 146 |
| Selected Freeze has an exactly tied top prediction | 0 | 0 |
| Lowest selected-Freeze winning rank margin | 0.00186 | 0.02954 |
| Scenarios with every pair discarded at tolerance 0.02 | 48 | 30 |
| Candidate pairs discarded at tolerance 0.02 | 4,961 / 12,205 | 1,646 / 3,756 |
| Candidate pairs with identical combat-board projection | 2,176 | 719 |
| Selected action has a cheaper candidate with the same combat projection | 489 | 164 |

The margins are **uncalibrated ranking units**, not probabilities. `np.argmax` chooses the earliest candidate on an exact prediction tie, but that rule does not explain these Freeze selections: every selected Freeze has a strictly positive winning margin. The issue is unconstrained ranking of equal-label actions, reinforced by their comparisons against weaker alternatives.

For example, the first qualifying training record, `recruit-202609051-2`, chooses `freeze → end_turn` over `end_turn`. Both labels are 0.5. The former costs 1,000 ms; the latter costs zero in the configured wrapper. Their predicted values are 7.154 and 1.362. The first qualifying validation example shows the same pattern. These are fixed first examples, not examples selected for extreme outcomes.

In training, Freeze participates in 2,053 positive and 98 negative label comparisons outside the combat tolerance. Its summed positive gap weight is 94.733, versus 0.966 negative. These are label weights before the margin-dependent gradient, not measured parameter updates. A common action that usually inherits a strong heuristic continuation can therefore learn a high general score without learning better recruitment.

All Freeze continuations also match the practical action's **recorded** board, hand, gold, and tier fields after removing entity IDs. This does **not** imply equivalent complete game states: the saved terminal record does not include the frozen shop, and Freeze changes what can be offered next turn. The audit deliberately calls the equality a recorded combat projection/label equality.

The current shortlist itself is limited: it omits 282 move and 137 additional insertion commands in training, and 106 moves and 32 insertions in validation. No economic action was removed by this audit. These existing omissions must remain disclosed; a new complete-action benchmark should enumerate every affordable legal command and placement.

## The next useful experiment

The smallest meaningful economic horizon is to carry a real opening state through **two recruit turns and their combats**. The current broad opening engine cannot yet supply that transition: `OpeningRecruitEngine.combat_snapshot` explicitly exports `next_turn_state_available: false` and `combat_changes_applied_to_recruitment: false`. Reusing that snapshot as a next-turn state would invent persistence.

Before generating the new labels:

1. Implement and verify combat-to-recruitment persistence, actual health/armor damage, frozen-shop retention, hand and enchantment lifetime, scheduled next-turn income, upgrade cost, and the next shop draw. Keep engine RNG and hidden pool inventory internal. A full-turn execution test must show that Freeze changes the following shop through the actual engine.
2. Version the public feature schema to expose active lobby tribes and exact deferred income. The opening observation currently exposes both a lobby list and a deferred field, but the encoder does not consume them. The deferred field currently counts scheduled actions; it must not be assumed to equal arbitrary amounts of future gold.
3. Predeclare a two-combat target, such as the equally weighted sum of the two combat scores, and report damage, survival, and paid milliseconds separately. Sample multiple hidden futures consistent with the visible state, use paired randomization across candidate arms where valid, and reserve independent evaluation RNG. This is still a finite opening objective: an upgrade whose benefit arrives after the second combat remains undervalued.
4. Collect labels at each actual visible decision, including choices and legal placements, with the fixed heuristic as the first continuation policy. Split entire trajectories before training; related decisions must never cross splits. First compare a model's first-action value against the same fixed continuation. Then evaluate a separately trained iteration that actually selects every command, with new on-policy states and a newly frozen holdout. A first-action policy cannot simply be called repeatedly and described as a validated full-turn learner.

Neither future shops, hidden opponent hands, RNG states, realized terminal boards, nor remaining-continuation costs belong in inference features. They may be training targets or evaluator state. Keep every legal economic action, including Freeze and upgrade; learn their consequences instead of suppressing them.

## Optional same-combat efficiency ablation

While the transition is being built, one matched supervised ablation can test whether the model stops paying for redundant **first-combat** continuations. Keep the architecture, visible features, existing candidates, splits, continuation, primary combat loss, optimizer budget, and seeds fixed. Add only an auxiliary pairwise preference for lower paid continuation time between candidates whose **complete combat inputs** are verified identical. Do not treat merely equal Monte Carlo scores as equivalent inputs, and do not reuse the legacy board-only cache key for hand-sensitive opening combat.

A concrete proposed preregistration is auxiliary weight 0.01, with each cost comparison weighted by the time difference divided by the root's remaining usable time, capped at one. The auxiliary loss bypasses the combat loss's 0.02 tolerance. These settings have **not** been trained or selected in this audit. A scalar penalty such as 0.001 combat-score units per second would disappear under that existing tolerance for the one-second Freeze difference; increasing it enough to survive the filter could exchange material combat value for time.

Use a matched control and ablation, and declare the validation rule before running: for example, a combat noninferiority margin of 0.25 percentage points plus a decrease in paid time. Report both quantities separately. The shared network can still trade combat quality against the auxiliary objective, so improvement must be measured, not assumed. Use ordinary visible-input argmax at inference; a terminal-board oracle must never break ties during deployment. A stronger-combat claim still requires a positive paired confidence bound against the practical heuristic on a fresh holdout. Reduced Freeze frequency alone is not success.

This optional ablation values efficient execution under a truncated target. It does not value Freeze's future shop and is not a replacement for the two-turn experiment above. The previously examined 327-state recruit test and 600-state transfer test must remain diagnostic, not become model-selection sets.

## Reproducible evidence

`scripts/audit_recruit_objective.py` restores and hashes the archived v2 data, discards individual test rows before diagnostics, loads the frozen selected checkpoint read-only, reconstructs the exact label-cache projection, verifies equal labels, totals actual timing-trace charges, and checks all audited source hashes again before writing. `reports/recruit_objective_audit.json` contains split-level counts, prediction margins, action-label signal, deterministic examples, source hashes, limitations, and the proposed experiment. It does not alter or restart training.

```bash
OPENBLAS_NUM_THREADS=1 python scripts/audit_recruit_objective.py
```
