# Simulator selection

Inspected on **5 September 2026**. Target: current Solo Battlegrounds, with a
transferable learned policy. A complete recruiting engine and a combat engine are
different requirements. The trainer must not treat a combat-only simulator as an
eight-player game simulator.

## Chosen combat engine

Use **Firestone's `@firestone-hs/simulate-bgs-battle` 1.1.750**, with
`@firestone-hs/reference-data` fixed to **3.0.196**. The exact transitive package
versions and integrity hashes are in `package-lock.json`.

- The npm registry reports publication on **2026-09-04 at 07:57:27 UTC**, MIT
  licensing metadata, and upstream git head
  `c53c77b127fa6d64479c03ad70f3a8e5511b3d7c`.
- The published JavaScript package exposes `simulateSingleCombat` and
  `simulateBattle`, along with `createSeededRng` and `withRng`.
- Firestone's own application imports this package. Its public application
  repository is not the source repository for the latest simulator package.
- Published package code is the actual dependency. Older public forks must not be
  represented as the current implementation.

Primary sources: [pinned npm metadata](https://registry.npmjs.org/@firestone-hs%2fsimulate-bgs-battle/1.1.750),
[npm package](https://www.npmjs.com/package/@firestone-hs/simulate-bgs-battle),
[Firestone dependency declaration](https://github.com/Zero-to-Heroes/firestone/blob/master/package.json).

This selection saves most of the difficult combat implementation work. It does
**not** establish that every card, state transition, or seasonal interaction is
correct. In particular, the inspected package's generic
`CardsData.getRandomTavernSpell()` returns `null`. Our adapter replaces that
silent gap with an explicit error. The inactive Timewarped generator also raises
an error, and the implemented two-cost Tavern spell pool is filtered against the
verified current active list.

## Other candidates inspected

| Candidate | Inspected revision / date | Reuse decision |
|---|---|---|
| [Gallo13th/HSBRSIM](https://github.com/Gallo13th/HSBRSIM/tree/4e0525a198352557ef816bd3a53566d6ffd4c164) | `4e0525a198352557ef816bd3a53566d6ffd4c164`; pushed 2026-09-01 | Best modern full recruiting-engine candidate found. Separate audit and execution probes are required; see the repository's recruit audit artifacts. |
| [RosettaStone](https://github.com/utilForever/RosettaStone/tree/e10749b5f0c08d3a6135bce317cb11d1738846ad) | `e10749b5f0c08d3a6135bce317cb11d1738846ad`; pushed 2026-08-19 | Maintained C++ framework, but current Battlegrounds coverage is far from a complete trainer. |
| [Stone Ground Hearth Battles](https://github.com/JDBumgardner/stone_ground_hearth_battles/tree/05dba5d0c2c58b1c2f6a6011f5896e8d1b8f3617) | `05dba5d0c2c58b1c2f6a6011f5896e8d1b8f3617`; pushed 2023-03-05 | Useful historical simulator and distributed PyTorch training patterns; its old pool/rules cannot be used as today's game. |
| [svgop Firestone simulator fork](https://github.com/svgop/api-simulate-battlegrounds-battle/tree/33f025fd70bc75a7a0a82034967a26e21b2b31f1) | `33f025fd70bc75a7a0a82034967a26e21b2b31f1`; package 1.1.668; pushed 2025-12-09 | Useful for reading historical architecture; use the current npm release instead. |
| [twanvl battle simulator](https://github.com/twanvl/hearthstone-battlegrounds-simulator) | Historical combat-only project | An existing fast C++ combat simulator, but not a maintained current full-game replacement. |

### Licensing evidence

- Firestone simulator and the inspected svgop package declare MIT in package
  metadata. Dependency licensing notices remain upstream; we do not relicense
  their code or vendor their source.
- HSBRSIM declares MIT in its README and project metadata, but the inspected tree
  lacks a standalone license file. It is inspected as an external checkout.
- RosettaStone includes an AGPL-3.0 license and identifies its SabberStone ancestry.
- Stone Ground Hearth Battles includes `LICENSE.txt` under Apache-2.0.
- The twanvl simulator includes an MIT license.

Primary license files and declarations:
[HSBRSIM README](https://github.com/Gallo13th/HSBRSIM/blob/4e0525a198352557ef816bd3a53566d6ffd4c164/README.md),
[RosettaStone license](https://github.com/utilForever/RosettaStone/blob/e10749b5f0c08d3a6135bce317cb11d1738846ad/LICENSE),
[Stone Ground license](https://github.com/JDBumgardner/stone_ground_hearth_battles/blob/05dba5d0c2c58b1c2f6a6011f5896e8d1b8f3617/LICENSE.txt),
[svgop package](https://github.com/svgop/api-simulate-battlegrounds-battle/blob/33f025fd70bc75a7a0a82034967a26e21b2b31f1/package.json),
[twanvl license](https://github.com/twanvl/hearthstone-battlegrounds-simulator/blob/master/LICENSE).

### Why repository activity is not enough

RosettaStone's recent maintenance does not imply current Battlegrounds card
coverage. In the inspected
[Battlegrounds card implementation](https://github.com/utilForever/RosettaStone/blob/e10749b5f0c08d3a6135bce317cb11d1738846ad/Sources/Rosetta/Battlegrounds/CardSets/BattlegroundsCardsGen.cpp),
`AddHeroes`, `AddHeroPowers`, and `AddTier2Minions` through `AddTier6Minions`
are empty. Only a small historical Tier 1 group and related tokens/enchantments
are implemented there.

Likewise, HSBRSIM's top-level claims need comparison with executable code. Its
[historical audit](https://github.com/Gallo13th/HSBRSIM/blob/4e0525a198352557ef816bd3a53566d6ffd4c164/docs/BATTLEGROUNDS_SIMULATOR_AUDIT_2026-05-25.md)
distinguishes registered scripts from incomplete effects. A registered method is
not evidence that the whole game rule is faithfully implemented.

## Implemented integration boundary

`simulator/firestone.mjs` initializes the reference service entirely from the
frozen local card snapshot. It never loads a live card database during a run.
It checks the snapshot checksum, requires verified combat pools, and rejects
inactive minions and heroes.

Normal random recruiting pools use the official **Tier 1–6 shop list**, rather
than the larger list containing special Tier 7 cards. Old definitions remain
available only as referenced tokens, goldens, enchantments, or scripted generated
cards. The generator does not offer them as starting minions.

The initial training task is **synthetic board positioning against a fixed
opponent**, using a conservative current-pool subset and Patchwerk contexts.
It does not model recruiting, hero selection, trinket or Dark Gift decisions,
eight-player placements, or the real distribution of human boards. Starting
positions exclude identified hand/history dependencies, initial threshold-state
dependencies, unregistered combat handlers, and unaudited random-summon closure.
Every exclusion is recorded in the generated coverage report. This eligibility
filter is a conservative screening process, not a formal proof of all effect
semantics.

Focused tests check deterministic combat, simultaneous damage, damage totals,
Divine Shield, taunt, input preservation, repeated seeded calls around another
simulation, inactive-card rejection, snapshot integrity, and a concrete current
Harmless Bonehead deathrattle. Evaluation uses freshly seeded combats after model
and baseline orders have been frozen.

The separate `refine_positions.mjs` policy uses the top three frozen neural
proposals together with the attack and taunt-last orders, then spends 128 new
combat simulations on each distinct proposal. Search ties follow neural rank.
Stored candidate labels are never consulted. The resulting selections and search
outcomes are saved before a further, independently seeded evaluation. This is a
combined neural-and-search policy: any measured gain is not evidence that the
neural network alone improves play. The evaluation report names the policy and
rejects reuse of label or search RNG streams.

The next full-player step is to connect an independently validated recruiting
engine to this combat adapter, then establish legal actions, pool conservation,
gold/triples/targeting, discover choices, current seasonal systems, shared lobby
state, and final placement rewards before starting full-game reinforcement
learning.
