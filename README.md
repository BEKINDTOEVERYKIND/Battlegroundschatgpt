# Battlegrounds AI

Goal: a strong current-patch **Solo Battlegrounds** player whose learned
representations remain useful after rotations.

**Current implementation: trained offline positioning and early-recruitment
components.** Firestone supplies combat simulation; a pinned HSBRSIM/hsrl2 bridge
executes finite-budget recruit commands against a rebuilt current-card database.
The new recruitment pilot acts at every decision through two recruit turns and
two sampled combats; it has not beaten the practical heuristic. Earlier
first-action models are also preserved. This is not yet a complete eight-player
self-play agent. Unsupported transitions
stop the affected rollout instead of becoming silent no-ops.

**New corrected-pool positioning result:** on 800 new test scenarios, the
learned-proposal + simulation-search policy gains **1.43 percentage points of
combat score** over the strongest tested heuristic (95% interval: 1.11–1.76).
That is the warm-start arm; the separately trained scratch arm gains 1.33 points.
Their direct comparison does not establish an advantage from warm-starting or
faster adaptation. All 138 eligible minions occur in the new dataset. See
[the completed experiment and both checkpoints](docs/positioning-transfer-results.md).

The earlier audit found a 1.07-point hybrid gain on 1,000 frozen scenarios.
The neural model alone did not establish an advantage. A Mech alias bug meant
the old data covered 123 distinct minions, not all 138 nominally eligible minions.
Corrected adapter semantics are versioned separately; 3,651,584 fresh combats
confirmed the restricted hybrid result without retraining or changing choices.
See the [corrected audit](docs/positioning-corrected-audit.md),
[original results](reports/results.md), and [coverage correction](docs/tribe-normalization.md).

The completed recruitment v3 model beat a raw-stat baseline but **did not
demonstrate an improvement over the practical heuristic**: +0.227 percentage
points, 95% interval −0.340 to +0.886, on 469 retained first-action test states.
Its 10,000 generation attempts yielded 2,659 train, 669 validation, and 469 test
states, with 4,197,760 label and independent evaluation combats.
The broader opening-pool transfer test of the frozen v2 model also found no
clear advantage on 600 retained states. All 22 minions reached candidate combat
boards. See
[recruitment results](docs/recruit-curriculum.md) and
[transfer results and action-selection limitations](docs/opening-transfer.md).

The new [two-turn recruitment pilot](docs/two-turn-recruit-pilot.md) retained
33 of 40 complete training trajectories and observed all 22 Tier-1 minions and
eight Tier-1 spells. Its learned policy scored 36.61% against 44.64% for the
practical heuristic on 14 supported paired test episodes. The small pilot
does **not** establish a recruitment advantage. It exposed repeated Freeze
decisions while confirming zero action-budget violations. The next declared
experiment collects states reached by that policy and retains practical
counterfactual continuations to teach complete buy-and-play sequences.

## Current ruleset

Snapshot: **5 September 2026, patch 36.4.2.251332, Season 14**. Dark Gifts,
Trinkets, Activate, Lockbox, and Fishbait are current. Ordinary Quests, Buddies,
Anomalies, and the Timewarped Tavern are not active seasonal systems; specific
hero-generated exceptions still require separate handlers.

- Official Solo pool: 246 minions, of which 234 are ordinary shop candidates and
  12 require special Tier 7 acquisition; 116 heroes; 71 Tavern spells; 43 Dark Gifts.
- Corrected positioning curriculum: **138 eligible minions**, with explicit
  exclusions for unaudited history, hand, seasonal, and random-generation effects.
  It uses Patchwerk and no combat-affecting seasonal effects.
- The separate opening bridge offers all **22 current Tier-1 minions and eight
  Tier-1 spells**, filtered to the lobby's five tribes. All have tested play paths;
  explicit contextual and later-turn limits still apply. The completed v2
  recruitment model was trained on a smaller pool, with five minions observed.
- The [two-turn bridge](docs/opening-transition.md) now preserves actual frozen
  shops, hands, buffs, deferred income and remaining action budgets through
  sampled combat. Tier-2 next-turn shops and other recorded frontiers still
  prevent full-game training.
- Source data and engine correctness have separate gates. An unresolved Trinket
  source discrepancy and incomplete recruitment mechanics block full-game training.
- Tokens, goldens, and other reference definitions are retained for correct
  scripted resolution. Being present in reference data never makes a card
  eligible for the shop or training sampler.

See [current rules and sources](docs/current_rules.md),
[simulator comparison](docs/simulator-selection.md), and
[recruit-engine audit](docs/recruit-engine-audit.md).

## Setup and verification

Requires Node 22+ and Python 3.12 with NumPy. No GPU is required for this stage.

```bash
npm ci --ignore-scripts
python -m pip install -r requirements.txt
python scripts/prepare_recruit_engine.py
python scripts/sync_ruleset.py --check
python scripts/check_live_ruleset.py
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
npm test
```

`--check` is offline and checks the frozen snapshot's integrity. It does **not**
claim that today's live game is still on this patch. `--refresh` retrieves into a
separate candidate directory; new patches cannot silently overwrite the approved
snapshot. Read the source reconciliation instructions before promoting a rotation.

## Train and evaluate

```bash
python scripts/train_positioning.py \
  --out runs/new-positioning-run \
  --count 6000 --candidates 10 --trials 128 \
  --eval-trials 1024 --epochs 40 --workers 6
```

The command checks watched live sources, generates seeded combat labels, trains
a shared feature ranker, refines its proposals with simulation search, freezes
its test-set choices, and independently simulates those choices against
four heuristic baselines. It saves model/optimizer/RNG state, source hashes,
candidate boards, coverage exclusions, raw evaluation results, and paired
confidence intervals. Existing runs are not overwritten. Logs appear in the run
directory throughout execution.

Use `--pure-model` to audit the network without search. Use
`--allow-historical` explicitly for offline reproduction; the run then records
that it is historical. Default training stops if the watched live sources change
or cannot be checked. Datasets are also archived as `.jsonl.gz`.

Try the saved supported example with the selected model:

```bash
OPENBLAS_NUM_THREADS=1 python scripts/recommend_position.py \
  --input examples/position.json \
  --checkpoint runs/20260905-positioning-v2/selected_positioning.json
```

This optimizes the supplied combat board only. It does not make recruitment or
seasonal choices. The model, all 16 candidate checkpoints, frozen decisions,
datasets, and test outcomes are preserved under `runs/`.

The input must include observed `recruitTimeRemainingMs`; alternatively supply
`--remaining-seconds`. Recommendations include a minimal sequence of minion
drags and exclude orders that exceed the remaining action budget. The saved
example has seven seconds remaining, allowing at most two drags after the
five-second reserve. This is offline advice; computation time is not included,
so a live integration must recheck the timer before executing the plan.

Train/validation/test are split by entire scenario (80/10/10), before learning.
Different permutations of one board never cross splits. The learning objective
is combat win probability plus half the tie probability; it is not final lobby
placement. Evaluation has fresh random streams and uses the scenario as its
statistical unit. The benchmark uses synthetic current-card snapshots, not a
sample of human games.

Inspect a trained recruitment recommendation on a saved visible state:

```bash
OPENBLAS_NUM_THREADS=1 python scripts/recommend_recruit.py \
  --input examples/recruit-observation-v2.json \
  --checkpoint runs/20260905-recruit-v2/selected_model.json
```

It reports one legal first action, its execution cost, and the remaining budget.
It checks the exact feature schema and current fixture scope. It does not execute
actions or provide a complete turn plan. See [recruitment training](docs/recruit-curriculum.md)
for the training command and archived comparisons.

## Transfer after a rotation

The 1,251-feature representation shares stats, tribes, keywords, effect text,
and order interactions. It has no card-ID embedding table. JSON checkpoints keep
normalization, Adam state, RNG state, and a feature-schema fingerprint. Compatible
checkpoints can warm-start a new run:

```bash
python scripts/train_positioning.py \
  --out runs/new-rotation \
  --warm-start runs/previous-run/positioning.json
```

First reconcile and validate the new ruleset and engine. Regenerate current-rule
labels. A compatible representation makes transfer possible; **faster adaptation
has not yet been measured on a subsequent rotation**.

## What is needed for the full player

Recruit actions have a finite execution budget: the default is **one complete
action per second**, with a five-second reserve and added delays for slower
actions. The tested wrapper masks unaffordable actions and expires the turn;
infinite economy loops cannot create unlimited decisions. The adapter must
supply actual player-specific remaining time and validated timeout behavior.
See [turn budgets and current implementation scope](docs/turn-budgets.md).

HSBRSIM's `hsrl2` is pinned externally in `config/recruit-engine.lock.json`.
The bridge now replaces its stale database with current definitions and adds
tested current effects, explicit triple rewards, corrected Battlecry dispatch,
Galewing transitions, and 39 Trinket IDs. These modules cover different scopes
and are not collectively a full-game certification. The original
[engine audit](reports/recruit_engine_coverage.json) records the unmodified
upstream baseline; the extension documents describe the subsequent repairs:

- [Current database](docs/current-recruit-data.md) and [opening pool](docs/opening-pool.md)
- [Triple rules](docs/core-recruit-rules.md) and [Battlecries](docs/battlecry-migration.md)
- [Current effects](docs/current-effect-extensions.md), [hero powers](docs/hero-power-extensions.md), and [Trinkets](docs/trinket-integration.md)

The [completed GitHub positioning workflow](docs/training-job.md) trained both
arms on 8,000 corrected-pool scenarios. All results are now committed. The
[next bounded two-turn experiment](docs/two-turn-training-job.md) uses the pilot's
policy to collect new states, checks live sources, preserves partial and complete
outputs, and stops within a fixed runtime. Neither workflow automatically
promotes a model or schedules an endless training loop.

The next milestone is a validated current-ruleset recruit bridge with complete
legal actions and persistent combat effects, followed by masked actor-critic
self-play and independent placement evaluation. See the
[training design](docs/training-design.md) and the
[recruitment objective audit](docs/recruit-objective-audit.md). The next learning
objective must value across-turn economy: the same-combat target cannot teach
the value of a frozen shop or a Tavern upgrade. No human MMR or full-game strength is
claimed from a positioning benchmark.

Third-party packages and game-data provenance are described in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
