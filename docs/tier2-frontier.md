# Current Tier-2 frontier

The pinned **36.4.2.251332 Solo** manifest contains **34 Tier-2 minions and
seven Tier-2 Tavern spells**. `reports/tier2_frontier.json` inventories all
41 cards, their exact client text and numeric tags, their registered pinned
HSBRSIM handlers, source-file SHA256 values and remaining integration work.
The inventory includes every current card. Handler registration establishes
that code exists; it does not establish faithful behavior.

The existing two-turn opening adapter still rejects a turn-one upgrade before
applying combat damage. No current pool, frozen training adapter, checkpoint,
or prior benchmark was changed. `tier2_pool_ready` and `full_game_ready` remain
false. Removing individual complex Tier-2 cards from the shop would change the
game and is not an acceptable way to lift this guard.

## A concrete repaired error

Pinned HSBRSIM implements Forest Rover's **Beetle-only** permanent Battlecry
as `ApplyRaceAura(BEAST, ...)`. That incorrectly buffs Forest Rover itself and
every other Beast, creating a particularly misleading Tier-2 buy valuation.
The old handler also represents golden Battlecry values as two normal-sized
triggers, which is incompatible with correct trigger counts and repeat effects.

The additive `tier2_beetle_game_type()` fixture in
`python/bg_ai/tier2_opening.py` supplies:

- A permanent player bonus restricted to normal and golden Beetle identities.
- Dynamic bonuses for existing, held and newly created Beetles. Golden Beetles
  receive the same player bonus once; their separate base stats stay intact.
- Current Forest Rover amounts read from validated client definitions: normal
  +2/+1, golden +4/+2 per genuine Battlecry trigger.
- The existing strongest-Brann multiplier, correct trigger accounting, and
  identical intrinsic amounts through explicit `TriggerBattlecry` actions.
- Ordinary/golden Rover Deathrattles that summon one/two normal Beetles with the
  current player bonus; selling the Rover does not erase that bonus.
- No mutation of external files, shared registries or old adapter classes.

This is a **recruitment effect repair fixture**, not a newly enabled training
pool. Its direct native combat entry point raises because permanent global
counter transport into/out of Firestone has not yet been integrated. It also
does not certify ordinary triples, arbitrary upstream card effects, generated
pool closures, a whole Tier-2 lobby, or policy playing strength. The old timed
recruitment wrapper remains mandatory when this effect is integrated into a
future complete adapter; no action macro or free interaction is introduced.

The [official current card gallery](https://hearthstone.blizzard.com/en-us/battlegrounds/)
and frozen authoritative client XML provide the card identity, scope and
amounts. Firestone's independently implemented `forest-rover.js` and
`beetle.js` handlers in the
[pinned published simulator package](https://registry.npmjs.org/@firestone-hs%2fsimulate-bgs-battle/1.1.750)
corroborate player-level Beetle bonuses and the absence of a general Beast
buff. These files are under `dist/cards/impl/minion/` in the package. Tests
use the installed pinned code rather than assuming a mutable upstream branch
is the same version.

## Why the full next shop needs more than a pool switch

| Requirement | Current Tier-2 examples |
| --- | --- |
| Persistent exact combat changes | Tarecgosa keeps combat gains; Eternal Knight needs deaths counted; Ancestral Automaton needs summon history. |
| Persistent private-hand changes | Winterfinner buffs a selected hand minion; Roadboar and Metallic Hunter generate cards during combat. |
| Complete generated-card pools | Tad, Chef's Choice, Patient Scout, Clever Castaway, Search Through Time and Lockbox. |
| Recruit damage and lifetime rules | Mind Muck, Lurking Lionfish/Fishbait, Lava Lurker, Hasty Excavation and temporary Magnetic transfers. |
| Entity/player counters | Patient Scout age, Baller improvements, Thaumaturgist spell count, Fodder refreshes and Winner's Bread's bound target. |
| Correct current shop behavior | Both minion tiers, both spell tiers, five-tribe matching, freeze/refill, independent spell slot and shared-pool conservation. |

Patient Scout is another concrete upstream defect: its Discover tier is
`min(6, initial_tier + game.turn - 1)`. A freshly bought Scout must not start
with a high-tier Discover simply because it is a later game turn. A future
repair must record the individual card's visible upgrades and separately
validate hand/Tavern lifetime and golden merging; this module does not invent
those unverified transitions.

The smallest faithful upgrade path therefore needs the full Tier-2 shop and
generated-effect closure plus a versioned sampled receipt carrying exact
persistent changes. Average damage/win probability cannot reconstruct those
changes. All legal branches must stay available and any unresolved branch
must abort the whole scenario with its cause recorded.

## Verification and reproduction

Focused Python tests exercise actual normal/golden play, strongest Brann,
retriggering, damaged Beetles, independent other auras, future tokens, hand
cards, opponent isolation, selling, Deathrattles and current-definition
rejection. The inventory test checks equality with the exact active pool.

```bash
PYTHONPATH=python python -m unittest discover -s tests -p test_tier2_opening.py
PYTHONPATH=python python - <<'PY'
import json
from pathlib import Path
from bg_ai.tier2_opening import build_frontier_manifest
report = build_frontier_manifest(Path('../research/HSBRSIM'))
Path('reports/tier2_frontier.json').write_text(json.dumps(report, indent=2) + '\n')
PY
```
