# Current recruit effect extensions

`python/bg_ai/hsbrsim_effects.py` adds an original, reversible extension layer for
HSBRSIM `hsrl2` at `4e0525a198352557ef816bd3a53566d6ffd4c164`. It requires the
fresh database from `bg_ai.hsbrsim_data`; loading the upstream 36.2.2 definitions
fails validation before the registry changes. No upstream source is vendored or
modified.

The official [36.4.2 patch notes](https://hearthstone.blizzard.com/en-us/news/24296231/3642-patch-notes)
and the pinned client definitions establish the current changes. The reference
client snapshot is build 251332 at
[HearthSim/hsdata revision dee8a641](https://github.com/HearthSim/hsdata/tree/dee8a641ef8427cf853ca707c2e427e752a1e11f).
Firestone's pinned `@firestone-hs/simulate-bgs-battle` 1.1.750 implementations are
an additional independent implementation reference for the four rows below;
matching another simulator alone is not proof of client conformance.

| Effect | Corrected behavior in this extension | Boundary |
|---|---|---|
| Soul Rewinder | Each damage instance restores the actual pre-damage Health and armor snapshot; normal gains 2 Health, golden 4. Multiple Rewinders do not restore damage twice. | Requires upstream's explicit per-instance damage event; missing snapshots raise. |
| Eredar Escapist | Each 4 accumulated damage generates Corrupted Cupcakes; golden generates two. Excess damage carries into the next threshold. | Cards enter the hand through the engine queue and require a later player action to cast. No automatic spell play. |
| Jailbird Juggernaut | Rally creates a Golem with the source's full attack and maximum Health, doubled for golden, and attacks the declared target before the source. | Uses the external engine's immediate attack primitive; a full board prevents the extra attack. |
| Sanguine Champion | Each intrinsic Battlecry or Deathrattle improves Blood Gems by 2/1, or 4/2 for golden. | Standalone golden play remains blocked by the upstream dispatcher bug. The separate `installed_battlecry_rules` migration now provides a correct scoped Sanguine play path. |

The Golem is inserted immediately to the right of Juggernaut. Non-gem buffs are
included; changing the Blood Gem count or current Gem value cannot change the
Golem independently of the source's stats. This matches the full-stat change
and Firestone's use of `maxAttack`/`maxHealth` in its current Juggernaut handler.
Its prior implementation instead multiplied Gem count by today's Gem value,
which could be wrong even under the old card text.

The installer validates all affected texts, base stats, script parameters and
Eredar's generated-card link against the frozen reference definitions. Golden
multipliers use definition identity: the upstream factory can construct a golden
card ID without setting the entity's golden flag unless its caller also passes
`golden=True`. The extension does not claim to repair that factory globally.

```python
from bg_ai.hsbrsim_data import build_current_database
from bg_ai.hsbrsim_effects import installed_current_effects

built = build_current_database("../research/HSBRSIM")
with installed_current_effects(built.db) as installation:
    # Construct and run external hsrl2 games inside this scope.
    assert installation.full_game_ready is False
```

All registry entries are restored on normal exit or exception. Install before
creating entities, and keep the context active through their complete lifetime;
existing entities retain bound script classes. Differing rulesets require
separate processes because upstream's registry is process-global.

## Still deferred

- **Rimescale Priestess / Rime or Reason:** the exact set of eligible
  stat-giving Tavern spells has not been verified. Firestone's Rime or Reason
  currently calls its general random Tavern spell generator, so copying that
  implementation would not establish the necessary restriction. The current
  client text does not enumerate edge cases such as set-stats or consume spells.
- **Unmasked Identity:** the exact eligible replacement-power pool and cleanup /
  initialization lifecycle are not validated. A generic Discover over all known
  powers could offer inappropriate powers or leave previous passive listeners.
- **Polarization:** upstream magnetic inheritance stores attached card IDs but
  later invokes their hooks using the host's source identity. Scripts that read
  `source.card_id`, source counters or event listeners can therefore use the wrong
  definition/state. Randomly selecting only supported Mechs would alter the
  actual effect's distribution. Its offering window is a separate data/rules
  audit. The official [Dark Gifts developer insight](https://us.forums.blizzard.com/en/hearthstone/t/battlegrounds-developer-insight-dark-gifts/163606)
  describes the effect and offering rules but does not fix these engine gaps.

These IDs stay explicitly listed in `DEFERRED_EFFECTS`; this extension does not
register no-op or guessed handlers. Full-game training also continues to require
current seasonal offers, remaining hero powers, generated-card closure, correct
pool semantics and full-game conformance.

## Executed validation

```bash
HSBRSIM_ROOT=../research/HSBRSIM PYTHONPATH=python \
  python -m unittest discover -s tests -p 'test_hsbrsim_effects.py' -v
```

Sixteen focused integration tests passed against the real external engine and
fresh current client database. They cover current normal/golden effect amounts,
armor rewind and multiple listeners, overflow/remainder, card generation without
free casting, leaving-board listener cleanup, full hand queue behavior, Golem
position/stats/attack/full-board behavior, stale-data rejection and exception-safe
registry restoration. They also exercise current numeric changes in Tichondrius,
Ashen Corruptor, Mighty Dragonbreath, Sanguine Refiner and Utility Drone through
upstream handlers. Those examples verify the specified transitions, not every
interaction of those cards. Tests skip explicitly when the pinned external engine
is absent; a skipped integration suite is not a successful conformance gate.
