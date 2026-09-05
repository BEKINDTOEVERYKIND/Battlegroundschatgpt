# First positioning experiments — 5 September 2026

**The combined learned-proposal and simulation-search policy beats all four
declared heuristics on 1,000 new synthetic combat scenarios. The neural model
alone does not clearly beat attack sorting. A full-game player is not ready.**

The current source snapshot is Solo patch 36.4.2.251332. Firestone 1.1.750
resolves the combats using frozen reference data. The curriculum covers 138
screened current minions in five-tribe lobbies, with explicit exclusions for
unaudited history, hand, seasonal, and generation effects.

## Fresh final benchmark

Combat score is P(win) + 0.5 P(tie). These are percentages of combat score, not
game win rates, placement improvements, or MMR estimates.

| Policy | Mean combat score | Difference from attack sorting |
|---|---:|---:|
| Random incoming order | 50.563% | −2.785 pp |
| Sort by health | 52.597% | −0.751 pp |
| Sort by attack | 53.348% | — |
| Attack order with Taunts last | 53.469% | +0.122 pp |
| Revised neural ranker alone | 53.222% | −0.125 pp |
| **Neural proposals + simulation search** | **54.560%** | **+1.213 pp** |

The combined policy gains **1.091 percentage points** over the strongest tested
heuristic (attack order with Taunts last), with a paired scenario bootstrap 95%
interval of **0.815 to 1.397 points**. Its interval against attack sorting is
**0.921 to 1.533 points**. It passes the declared requirement for a positive lower
95% interval against every baseline.

All choices were frozen before final evaluation, which used 1,024 new combat
simulations per distinct selected order, a separate RNG stream, and the same
1,000 scenarios for every policy. The paired bootstrap resamples scenarios,
not individual combat trials. The search configuration was fixed before this
test: top three neural proposals plus attack/taunt-last proposals, with 128
additional simulations per included order. The search never reads the stored
candidate outcome labels. Search and final-evaluation seeds are different.

This gain belongs to the **combined policy**. We have not established that neural
proposals outperform equally budgeted search proposals without a learned model.

## Training and failed first attempt

The original dataset has 6,000 scenarios and 10 candidate orders each, labelled
with 128 simulations: 7.68 million combats total. The split is 4,800 training,
600 validation, and 600 initial test scenarios. Training uses 1,251 stable
stat/tribe/keyword/effect-text/order features, no card-ID embedding lookup, and
a small NumPy MLP with exact optimizer/RNG checkpointing.

The first ranker beat random order but lost to attack sorting by 0.477 percentage
points on its fresh 600-scenario test. It was retained, not promoted.

The loss normalized absolute score differences within each scenario, which gave
tiny noisy differences the same total weight as material ordering differences.
The revised objective weights by absolute gap over all candidate pairs. A
predeclared 16-candidate sweep used only training and validation: hidden size
16/32, ignored gaps 0/0.02, and epochs 5/10/20/40. The selected model has 32 hidden
units, a 0.02 gap threshold, and 40 epochs. Validation score increased from
0.53135 to 0.53868. Its subsequent new test did not establish an advantage over
attack sorting, so the standalone network is still experimental.

The new 1,000-scenario test used a different generator seed, and generated
1.28 million label combats that were never used to train the revised model.
Search then used 430,592 additional combats to freeze the combined policy's
choices before independent testing. Raw results and unsuccessful candidates are
preserved; the initial test was not recycled as the revised model's final test.

## Reproduction and limits

See `reports/training_summary.json` for compact numeric results and each run's
`evaluation.json`, `fresh_evaluation.jsonl`, and `frozen_selections.jsonl` for
the evidence. Datasets are saved as deterministic `.jsonl.gz`; their archive
manifests bind both compressed and uncompressed hashes. The 16 candidate
checkpoints are archived in four repository-sized parts, with bundle and member
hashes in `runs/20260905-positioning-v2/candidate_archives.json`. Restore them with
`python scripts/restore_candidates.py`. The selected checkpoint is also
available directly as JSON.

For exact initial-loss reproduction, use `--objective legacy-normalized
--tie-tolerance 0 --pure-model`. New training defaults to gap weighting and the
combined search policy. Use `--allow-historical` explicitly to reproduce the
frozen snapshot offline; ordinary training verifies watched live sources before
and after a run.

The benchmark contains synthetic boards with known opponents, no combat hero
powers or seasonal effects, ordinary stat buffs, and a restricted starting-card
subset. It does not establish real-game positioning strength, rotation-transfer
speed, recruiting ability, or placement performance. The recruitment audit
documents the missing full-game engine work.
