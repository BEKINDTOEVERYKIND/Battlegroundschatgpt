# Corrected adapter audit of the published positioning result

The hybrid's restricted positioning advantage survives the canonical-Mech correction: **+1.066 percentage points** over the strongest heuristic, with a **95% paired interval of +0.787 to +1.369 points**. The original legacy estimate was +1.091 points. The neural model alone still does not beat every baseline. This is an audit of frozen synthetic positioning decisions, not a new model promotion or evidence of full-game strength.

The coverage claim needs a separate correction. Both the original 6,000-scenario dataset and the frozen 1,000-scenario test contain **123 observed minion definitions out of 138 nominally eligible definitions**. Fifteen pure Mechs never appeared. Correcting combat semantics does not add their missing boards to these datasets.

## Frozen protocol

Before simulating, `runs/20260905-positioning-corrected-audit/audit_plan.json` bound the exact input hashes, evaluator source, corrected adapter source, seeds, trial count, policies, baselines, and confidence-interval procedure. The audit used:

- All 1,000 original holdout boards and opponents from `20260905-holdout-v2`, unchanged.
- Exact hybrid choices from `20260905-positioning-search-v1/frozen_selections.jsonl` and exact pure-model choices from `20260905-positioning-v2/frozen_selections.jsonl`. The hybrid's frozen original search results and model-proposal provenance were checked against its choices.
- The original random, attack, health, and taunt-last baseline permutations, unchanged.
- `firestone-combat-v2-tribes`, dataset schema 2, Firestone combat engine 1.1.750, and reference package 3.0.196. Original dataset provenance remains explicitly legacy-v1/schema 1. Every output row is marked as an audit-only reinterpretation.
- Only the lobby alias `MECHANICAL` → `MECH` changed in the input interpretation. Original entities already used `MECH`; no own or opponent entity, candidate permutation, card snapshot, or ruleset snapshot was edited. Source scenario and board hashes were checked.
- 1,024 fresh combats per distinct selected order, with identical per-scenario seeds for all compared orders. Seed `(32452843 + imul(inputIndex + 1, 2654435761)) >>> 0` differs from each scenario's label, hybrid-search, and original-evaluation seed.
- 10,000 paired percentile bootstrap resamples over scenarios, seed 20260909, with the same sampled rows across all policy comparisons. All four predeclared baselines are reported. A restricted advantage requires a positive lower interval against every baseline.

There were **3,651,584 actual combats**, covering 6,000 policy/order evaluations; duplicate selected orders shared their identical outcomes. No training, candidate search, or new order selection occurred. Each of the 1,000 scenarios completed and passed the independent report validator.

## Corrected results

The score is combat win probability plus half the tie probability. Differences below are percentage points in this score, not changes in full-game win rate.

| Policy | Mean score |
| --- | ---: |
| Frozen hybrid, neural proposals plus original simulation search | 54.5300% |
| Frozen neural model alone | 53.1829% |
| Random order | 50.5560% |
| Attack order | 53.3460% |
| Health order | 52.5936% |
| Taunt-last order | 53.4641% |

| Paired comparison | Difference | 95% paired interval |
| --- | ---: | ---: |
| Hybrid minus random | +3.974 pp | [+3.537, +4.422] |
| Hybrid minus attack | +1.184 pp | [+0.883, +1.505] |
| Hybrid minus health | +1.936 pp | [+1.614, +2.278] |
| Hybrid minus taunt-last | +1.066 pp | [+0.787, +1.369] |
| Model minus random | +2.627 pp | [+2.181, +3.090] |
| Model minus attack | −0.163 pp | [−0.449, +0.125] |
| Model minus health | +0.589 pp | [+0.287, +0.901] |
| Model minus taunt-last | −0.281 pp | [−0.558, −0.004] |
| Hybrid minus model | +1.347 pp | [+1.069, +1.647] |

The strongest baseline by mean remains taunt-last. The model's interval against it narrowly excludes zero in the negative direction, and its interval against attack includes zero. The hybrid's advantage cannot be attributed to the neural model alone.

## Actual legacy card coverage

The legacy generation path compared reference race `MECH` against lobby alias `MECHANICAL`, so pure Mechs did not enter the sampled pools. Dual-tribe Mechs could still enter through their other tribe. The original manifest's 138 count described nominal eligibility, not the number sampled.

| Dataset | Scenarios | Original splits | Lobbies with legacy `MECHANICAL` alias | Observed / nominal minions |
| --- | ---: | --- | ---: | ---: |
| Original positioning dataset | 6,000 | 4,800 train, 600 validation, 600 original test | 3,008 | 123 / 138 |
| Frozen final holdout, rescored here | 1,000 | 1,000 test | 492 | 123 / 138 |

Both datasets are missing the same 15 nominally eligible minions:

| Card ID | Minion |
| --- | --- |
| BG26_146 | Lullabot |
| BG26_147 | Accord-o-Tron |
| BG26_148 | Scrap Scraper |
| BG26_152 | Utility Drone |
| BG28_741 | Charging Czarina |
| BG29_503 | Clunker Junker |
| BG31_177 | Mechagnome Interpreter |
| BG32_170 | Metallic Hunter |
| BG32_172 | Auto Assembler |
| BG35_341 | Enchanted Sentinel |
| BG36_506 | Drone Duplicator |
| BG36_853 | Glambot |
| BG36_854 | Rescue Bot |
| BGS_071 | Deflect-o-Bot |
| BG_BOT_911 | Annoy-o-Module |

The report also separates the 492 corrected-alias lobbies from the other 508 as a diagnostic. These subsets were not used to select or promote a policy. The corrected adapter can support generation involving the absent Mechs; this fixed-board audit does not measure such a new distribution.

## Evidence and reproducibility

`scripts/reevaluate_positioning_v2.mjs` performs the frozen resimulation, writes exclusive worker shards, and refuses to overwrite a completed audit. `scripts/report_positioning_corrected_audit.py` independently checks all input hashes, exact choices, unchanged boards, simulation counts, seeds, identical-order outcomes, and the original published report. It also reproduces the original confidence intervals from the original raw counts and counts observed card IDs directly in both datasets.

```bash
# Already completed; the evaluator deliberately refuses a second overwrite.
node scripts/reevaluate_positioning_v2.mjs --prepare true
node scripts/reevaluate_positioning_v2.mjs --workers 6

# Rebuild the audit report from the saved evidence.
python scripts/report_positioning_corrected_audit.py
```

The machine-readable summary is `reports/positioning_corrected_audit.json`; the run directory preserves its protocol, all raw outcomes, worker shards, and detailed summary. Corrected raw outcomes have SHA-256 `e4dc9a6f38170a9e8e51c218506a669458a2806b87eda005b1cba193e8073f6f`. The adapter source used has SHA-256 `b9a1ec0c56bbcc2f957a654ee41ae5cd5acc2342558d5a254702deba34dc580d`.

The fresh RNG and corrected interpretation both differ from the original evaluation, so the small numerical score change is not a causal estimate of the adapter change alone. Synthetic boards are not a ranked-player distribution, candidate orders are a restricted action set, and the missing Mech boards remain missing. These now-audited test results must not guide later model selection; a future promotion needs fresh, separately frozen evaluation data.
