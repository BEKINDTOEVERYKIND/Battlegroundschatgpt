# Battlegrounds AI

Goal: a strong current-patch **Solo Battlegrounds** player whose learned
representations remain useful after rotations.

**Current implementation: an offline combat-positioning trainer.** It reuses
Firestone's maintained combat engine and learns how to rank minion orders.
Recruiting, hero choice, seasonal choices, and full eight-player self-play are
not trained yet. The recruit-engine audit identifies the concrete missing work;
missing effects are never silently treated as no-ops.

**Measured first result:** the learned-proposal + simulation-search policy gains
**1.09 percentage points of combat score** over the strongest tested heuristic
on 1,000 new synthetic scenarios (95% interval: 0.81–1.40 points). The neural
model alone did not clearly beat attack sorting. See the
[full results, failed attempts, and limitations](reports/results.md).

## Current ruleset

Snapshot: **5 September 2026, patch 36.4.2.251332, Season 14**. Dark Gifts,
Trinkets, Activate, Lockbox, and Fishbait are current. Ordinary Quests, Buddies,
Anomalies, and the Timewarped Tavern are not active seasonal systems; specific
hero-generated exceptions still require separate handlers.

- Official Solo pool: 246 minions, of which 234 are ordinary shop candidates and
  12 require special Tier 7 acquisition; 116 heroes; 71 Tavern spells; 43 Dark Gifts.
- Current positioning curriculum: **138 eligible minions**, with explicit
  exclusions for unaudited history, hand, seasonal, and random-generation effects.
  It uses Patchwerk and no combat-affecting seasonal effects.
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

The best recruitment scaffold found is HSBRSIM's `hsrl2`, pinned externally in
`config/recruit-engine.lock.json`. Its existing tests pass, but it does not yet
implement the current Trinket system and has missing/deferred active effects and
outdated balance values. The machine-readable
[coverage report](reports/recruit_engine_coverage.json) enumerates the gaps.

The next milestone is a validated current-ruleset recruit bridge with complete
legal actions and persistent combat effects, followed by masked actor-critic
self-play and independent placement evaluation. See the
[training design](docs/training-design.md). No human MMR or full-game strength is
claimed from a positioning benchmark.

Third-party packages and game-data provenance are described in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
