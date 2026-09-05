# Recruitment engine audit — 5 September 2026

**Decision:** HSBRSIM's `hsrl2` is the strongest reusable candidate found for recruitment and eight-player Solo play. It is not yet a correct current-patch environment. Reuse its machinery only after the gaps below are resolved. Training a whole-game policy against its unchanged rules would teach the wrong game. The working combat/positioning trainer is independent of this decision.

## Candidates inspected

| Candidate | Inspected revision | Reusable part | Current full-game blocker |
|---|---|---|---|
| [Gallo13th/HSBRSIM](https://github.com/Gallo13th/HSBRSIM) | `4e0525a198352557ef816bd3a53566d6ffd4c164`, 1 September 2026 | Python `hsrl2` recruitment, shared pools, triples, spells, hero powers, Dark Gifts, combat and lobby flow | Snapshot is patch 36.2.2; all 228 candidate trinkets have no scripts or offering system; other concrete gaps below |
| [utilForever/RosettaStone](https://github.com/utilForever/RosettaStone) | `e10749b5f0c08d3a6135bce317cb11d1738846ad`, 4 August 2026 | C++ recruitment/lobby structure, Python bindings | Only 26 Battlegrounds `cards.emplace` definitions; `Player::PlayCard` spell branch is `TODO: Cast spell`; old tribe model |
| [Stone Ground Hearth Battles](https://github.com/JDBumgardner/stone_ground_hearth_battles) | `05dba5d0c2c58b1c2f6a6011f5896e8d1b8f3617`, 14 November 2021 | Complete historical Python loop, bots and PPO/A2C examples | Historical content and mechanics; no current seasonal system |
| [yossielimelech/BGSimulator](https://github.com/yossielimelech/BGSimulator) | Public README inspected | Historical deep-learning simulator | README excludes heroes and states all rights reserved; not a current reusable full-game engine |

RosettaStone has actual recruitment and combat code; describing its entire Battlegrounds tree as stubs would be inaccurate. Its missing contemporary card and spell support is sufficient to reject it for this project. The combat-only Firestone engine is audited separately; a combat simulator does not supply shopping, discover, lobby economy or seasonal selection.

## HSBRSIM: read the new engine, not the stale README

The repository contains both `hsrl/` and `hsrl2/`. Its README chiefly describes the older engine and is not a reliable statement of the new engine's coverage. The new engine's `data/bg_summary.json` reports **36.2.2.249896**, with Season 14 Dark Gifts and Activate implemented in code.

Executed, rather than assumed, at the pinned commit:

```text
python -m unittest discover -s hsrl2/tests -q
Ran 1426 tests in 4.370s
OK (skipped=4)
```

These tests establish that upstream's own assertions pass, not that its game matches Blizzard. Some explicitly approve deferred/no-effect behavior, and several derive their expected values from the same card data used by the implementation.

The package metadata declares MIT in `pyproject.toml`; the inspected checkout does not include a standalone LICENSE file. This project references the external dependency and vendors none of its source. RosettaStone is AGPL-3.0; Stone Ground includes Apache-2.0 terms.

## Concrete current-patch gaps

The current official Solo gallery was joined to canonical card IDs from HearthSim's current CardDefs snapshot, rather than matching localized names. All gallery IDs exist in the old HSBRSIM database, which demonstrates why ID membership alone is insufficient. The gallery contains a removed trinket and disagrees with the client about another trinket's eligibility; the ruleset therefore keeps trinkets as candidates and blocks that component. The audit does not promote those candidates into a training pool.

| Current category | Required gallery entries | Missing effect implementation |
|---|---:|---|
| Minions | 246 | Rimescale Priestess (`BG33_319`); keywords and explicitly implemented core effects are accounted for |
| Hero base cards | 116 | 21 linked hero powers have no registered script |
| Tavern spells | 71 | Unmasked Identity (`EBG_Spell_037`) |
| Dark Gifts | 43 | Polarization (`BG36_MidGameEffect_000t65`) is explicitly deferred |
| Trinkets | 228 candidates; zero verified active IDs | All 228 candidates lack `hsrl2` scripts; no trinket offer/purchase system |

**Numeric drift:** 22 current minions have changed base attack, health or tavern tier relative to HSBRSIM's snapshot. This includes Goldrinn moving from tier 6, 8/8 to tier 5, 7/7; Cagey Conjurer moving from tier 3 to 4; and Private Investigator changing from 2/4 to 5/6. The audit also detects 27 hero armor differences, Natural Blessing's cost change, and 42 candidate-trinket cost differences. These differences exclude meaningless comparisons between minion mana cost and the 3-gold purchase price. The generated coverage JSON supplies exact per-card differences; card-text and effect-parameter conformance remains a separate required gate.

The 21 missing hero-power implementations belong to Buttons, Cariel Roame, Galewing, Genn, Guff, Jim Raynor, Kerrigan, Loh, Marin, Master Nguyen, Morchie, Murozond, Professor Putricide, Silas, Sir Finley, Sire Denathrius, Tavish, The Great Akazamzarak, Tickatus, Varden and Zephrys. Simply dropping these heroes changes the training population; replacing their powers with no-ops is worse.

Further issues found in source:

- `hsrl2/game.py::apply_dark_gift` accepts Polarization, writes a `DEFERRED` audit entry and returns without applying the effect. A warning in a log does not make the resulting trajectory valid.
- `hsrl2/darkgifts.py::RARE_GIFT_WEIGHT = 0.5` is explicitly an assumed offering weight because the precise probability was not published. This uncertainty matters to economy/choice learning and should be represented and sensitivity-tested.
- `hero.trinkets` has start/end/combat hook loops but nothing implements current selection, pricing, eligibility, guarantee rules or purchase. Hook plumbing is not seasonal support.
- Generated cards, golden forms, hero-power replacements and cross-card dependencies must be validated beyond the gallery's offerable roots. A selected trinket can generate a minion absent from the ordinary tavern pool.
- `CardDB.register` merges duplicate definitions' pool flags/keywords but keeps the first loaded numeric values. Overlaying new subset JSON on old `bg_cards.json` does not reliably update stats. Rebuild a clean authoritative database.
- Full-game observations/actions are not supplied by the audit adapter. Before connecting a learner, target selection and every pending discovery must be explicit, with opponents' hidden shops/hands and future RNG excluded from observations.

## Reproduce the external audit

The lock is `config/recruit-engine.lock.json`. Clone outside this repository; no source is copied or rewritten:

```bash
git clone https://github.com/Gallo13th/HSBRSIM.git ../HSBRSIM
git -C ../HSBRSIM checkout --detach 4e0525a198352557ef816bd3a53566d6ffd4c164
PYTHONPATH=python python -m bg_ai.recruiting \
  --engine-root ../HSBRSIM \
  --ruleset data/ruleset.json \
  --cards data/reference_cards.json \
  --out reports/recruit_engine_coverage.json
```

Exit code **2 is the expected blocked result**, not a successful training gate. The adapter loads upstream in a child Python process, checks the revision, compares the current explicit pools, treats deferred gifts as unsupported even if a dispatcher entry exists, and records blockers. It does not install dependencies or mutate the checkout. Its protocol defines the future recruitment interface without pretending that a usable current full-game environment already exists.

## Integration order

1. Keep the combat implementation and pretrained semantic/position features; build a current authoritative `hsrl2` data adapter and traceable dependency closure.
2. Add current trinket selection/purchase/eligibility and scripts, resolve the listed card/hero gaps, and expose exact legal action targets and discoveries.
3. Differential-test combat, then validate full recruitment traces against actual current-client games. Preserve permanent effects that occur during combat when returning to recruitment.
4. Validate shared pool conservation, five-tribe lobby restrictions, hero armor tiers, ties/placements, damage caps, ghosts, action ordering and RNG reproducibility across whole games.
5. Only after these gates pass, train eight-player policies and benchmark held-out seeds/lineups/opponents. Neither passing unit tests nor complete registry membership alone opens this gate.

Rotation portability belongs in separate semantic card features, reusable action pointers, patch-keyed replay/checkpoints and explicit seasonal modules. Reuse old representations as initialization; do not continue optimizing the previous rotation's objective or mix obsolete legal pools into current self-play.
