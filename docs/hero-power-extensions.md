# Current hero-power extension audit — 5 September 2026

`bg_ai.hsbrsim_heroes` adds Galewing's recruitment state machine to the pinned
external `hsrl2` engine. It closes one of the 21 missing **registry** entries in
the original audit, together with its three generated flightpath powers. It
does **not** establish that Galewing games, or the full game, are validated for
training. The remaining engine and generation-distribution gates still apply.

## Implemented transitions

| Power | Current behavior | Timing and action boundary |
|---|---|---|
| Dungar's Gryphon, `BG20_HERO_283p` | Choose Westfall, Ironforge, or Eastern Plaguelands | Zero gold; explicit power action followed by an explicit pending choice. The last completed path is excluded. |
| Westfall, `BG20_HERO_283p_t1` | Get a random current 1-cost Tavern spell | Next recruit turn, after one turn has elapsed. Grants a hand card; playing it consumes a separate timed action. |
| Ironforge, `BG20_HERO_283p_t2` | Gain 2 gold | After two turns; adds to that turn's income after the engine resets base gold. |
| Eastern Plaguelands, `BG20_HERO_283p_t3` | Discover a minion of the player's Tavern tier | After three turns; samples the tier at completion, opens an explicit discovery, and acquires the chosen card from the current shared pool. |

Blizzard's [25.2 patch notes](https://hearthstone.blizzard.com/en-us/news/23892225/25-2-patch-notes)
explicitly prohibit choosing the same flightpath consecutively and define the
two- and three-turn rewards. This restriction is absent from the base power's
short card text and would be easy to miss in a text-only implementation.
The [33.6 patch notes](https://hearthstone.blizzard.com/en-us/news/24232759/33-6-patch-notes)
replace Westfall's former minion buff with a 1-cost Tavern spell, leaving the
other paths unchanged. The frozen client XML and reference definitions for
build `251332` confirm the current text, zero costs, and 1/2/3-turn counters.

Westfall filters on **gold cost**, without adding an unsupported Tavern-tier
limit. A current Tier-4 spell costing 1 gold can therefore be awarded while the
player is at Tier 1. No historical Westfall buff remains in this implementation.

The spell is sampled uniformly from available current 1-cost definitions and
returned to the pool immediately on acquisition, following Blizzard's
[28.2 spell-pool rule](https://hearthstone.blizzard.com/en-gb/news/24008697/28-2-patch-notes).
The current card and its hand queue remain separate from pool occupancy.
Exact generation weights remain unverified. The unpatched native engine also
returns spells on casting: later casting therefore requires a validated cast
adapter to prevent a second return. These limits remain in the installation
report and keep full-game training closed. An empty candidate set raises an
explicit error instead of replacing the reward with a no-op.

## Integration

Build the database with `build_current_database` first. That bridge checks the
external checkout and builds a fresh database from current authoritative data.

```python
from bg_ai.hsbrsim_data import build_current_database
from bg_ai.hsbrsim_heroes import installed_current_hero_extensions

current = build_current_database(engine_root)
with installed_current_hero_extensions(current.db) as coverage:
    # Construct and finish the game inside this context.
    # Use the shared timed recruit wrapper for every player decision.
    ...
```

Installation verifies the power definitions before changing the registry and
restores every previous registry object when the context exits, including on
exceptions and under nested use. It does not change the external source files.
The public `hero_extension_state(hero)` supplies the current path, due turn,
last completed path, and pending-choice flag for future observations and traces.

The listener is scoped to the correct hero. Duplicate binding does not duplicate
rewards. A replayed choice cannot reset the clock. Eliminated heroes receive no
reward, and replacing an in-flight power cannot resurrect its old power or grant
its reward. Acquiring Galewing through a second/replaced-power mechanism, and
its Buddy interactions, are outside this base-hero integration scope.

The presently authorized recruitment fixture supports other heroes and early
turns; this extension does not automatically widen that fixture or the training
population.

## Tests executed

```bash
PYTHONPATH=python python -m unittest discover -s tests -p test_hsbrsim_heroes.py -v
```

**18 tests passed** against the actual pinned external engine and the freshly
built current database. Assertions cover exact turn delays, current reward IDs
and tiers, income ordering, explicit choices, hand overflow, pool consumption,
repeated-path restrictions, replaced/dead heroes, registry restoration, stale
definitions, and keeping the full-game gate closed. The suite skips with a
visible reason when the external dependency is absent; set `HSBRSIM_ROOT` to its
checkout path when it is stored elsewhere.

## Other audited candidates remain unregistered

| Hero | Concrete remaining issue |
|---|---|
| Cariel Roame | Current client data contains the stance tokens, contrary to the original upstream audit's claim that the options are absent. However, multiple +1/+2 Attack and Health variants coexist without an authoritative offering link. Selecting one set by name would guess the current improvement distribution. The passive and active portions must ship together. |
| Guff Runetotem | Current text specifies buying 20 tiers' worth of cards, including Tavern spells. Primary text does not settle whether threshold overshoot carries forward. Correct grantable Triple Reward support is also required; an immediate discovery is not equivalent. |
| Zephrys, the Great | The three uses and 3-gold cost are explicit. Pair-selection, empty-pool/no-pair behavior, and the engine's triple/reward transitions must be validated before the power is registered. |
| Varden Dawngrasp | Its individual-minion freeze needs real partial-freeze semantics. Upstream refresh only preserves a hero-wide frozen Tavern and ignores frozen entity tags. Freezing the entire shop would change the game. |
| Loh, the Living Legend | Attack progress and rewards earned in combat require validated persistent cross-combat state and grantable Triple Rewards. |
| Buttons and Marin | Depend on actual current Trinket eligibility, offers, prices, purchases, and effects. |
| Genn, Sir Finley, Master Nguyen | Need verified eligible replacement-power pools and correct lifecycle, including passive and multiple-power state. |
| Jim Raynor and Kerrigan | Require their generated unit/upgrade pools and dedicated progression mechanics. |
| Morchie and Murozond | Hero-specific Timewarp exceptions require a real Timewarp implementation despite the general mode being inactive. |
| Professor Putricide, Silas, Sire Denathrius, Tavish, The Great Akazamzarak, Tickatus | Custom creations, tickets, quests, combat delivery, secrets, or prize systems remain separately required. |

Two additional core errors surfaced while evaluating these powers: upstream
`_grant_triple_reward` derives reward tier from the minion's tier and immediately
opens a discovery; normal rewards should be hand cards tied to the player's
Tavern tier when awarded. Upstream `upgrade_tavern` also permits Tier 7 without
checking an unlock. These observations were sent to the recruit adapter and
rules-extension work; this hero module does not silently patch either method.
