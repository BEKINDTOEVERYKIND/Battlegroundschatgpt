# Scarlet Survivor and Tavern-targeted Fleeing Fugitive

The opt-in `HandScarletOpeningEngine` resolves **Scarlet Survivor gaining
Divine Shield while in hand** and carries its one-time completion marker into
Firestone. It does not enable Tavern threshold or Tavern Fugitive branches,
and it is not used by the production trainer. Existing adapters and frozen
benchmarks are unchanged.

## Evidence and its limits

[Blizzard's 35.6 patch notes, June 2, 2026](https://hearthstone.blizzard.com/en-us/news/24276665)
explicitly state: “Fixed a bug where Scarlet Survivor did not gain Divine
Shield from handbuffs.” That establishes the owned-hand behavior directly.
The same notes separately address the timing of its shield relative to death;
that does not by itself settle combat-to-recruitment persistence.

The pinned 36.4.2.251332 client definitions of normal/golden Scarlet Survivor
both contain threshold `TAG_SCRIPT_DATA_NUM_1=6` and tag
`TRIGGER_WHEREVER_THIS_IS=1` (numeric tag1724). Its text specifies a once-only
threshold and a completed display state. These support preserving an explicit
completion marker when an owned hand card moves onto the board.

**The Tavern conclusion remains an inference.** The wherever tag is suggestive,
but it does not establish every interaction with an unowned shop entity,
controller identity, buy/steal transitions and buffs applied before ownership.
The new subclass continues to reject a Tavern Scarlet crossing the threshold
before changing its stats. No legal card or action is removed to avoid it.

Fleeing Fugitive's current official card text describes a Health gain when
the player casts a spell on it. [The 36.2.2 notes](https://news.blizzard.com/en-gb/article/24293284/36-2-2-patch-notes)
establish its current 5/2 base stats. Unlike Scarlet, its client definition does
not carry tag1724. Neither source directly establishes the unowned Tavern
trigger. Pinned HSBRSIM attaches its listener on summon, so it has no listener
on a fresh shop copy; this is code behavior, not independent evidence that
the live client behaves the same way. The Tavern target remains an explicit
unsupported branch. Normal owned-board casts are tested with current spell
amounts and remain separately timed cast/target commands.

## Implementation

`python/bg_ai/offboard_scarlet.py` adds a separate two-turn adapter that changes
only the supported Scarlet hand path. A handbuff reaching at least six Attack
grants the shield immediately and consumes the one-time marker. Playing that
card preserves its state; further buffs or re-entry do not reset the marker.
The new fixture retains all 22 current Tier-1 minions, all eight current Tier-1
Tavern spells and all existing finite-action timing rules.

For reproducible conformance tests, `ScarletHandFixtureSpec` can declare
permanent initial handbuffs. These are explicit synthetic initial-state
assumptions; this does not assert that the old Tier-1 opening has an effect
that generates such buffs. Replay forks reconstruct the declared initial
state and retain spent action budgets. No training coverage increase or
playing-strength gain is claimed from these tests.

`simulator/offboard-scarlet-firestone.mjs` extends the frozen outcome worker
without modifying it. Every Scarlet in the board/hand payload requires a
boolean `scarlet_trigger_consumed`. A consumed marker becomes both
Firestone's exact `abiityChargesLeft=0` field and its `BG35_814e` completion
enchantment. Existing visible shield state is preserved independently: a
consumed trigger is not permission to restore a lost shield.

Both request and receipt require
`effect_transport_version=current-hand-scarlet-opening-v1`. The Python adapter
rejects an old worker's receipt before applying damage. The existing full
request hash covers the extra fields. Fresh independently sampled combats
still determine the actual health/armor transition.

Pinned Firestone's published
[`@firestone-hs/simulate-bgs-battle` 1.1.750 package](https://registry.npmjs.org/@firestone-hs%2fsimulate-bgs-battle/1.1.750)
contains the relevant implementation at
`dist/cards/impl/minion/scarlet-survivor.js`. Its stat-change handler reads both
the charge field and completion enchantment. The tests execute that actual
handler and real sampled combats. The current Tier-1 combat closure has no
effect that raises another minion's Attack, so these marker checks are not
evidence that any old positioning/recruitment benchmark outcome was wrong.

## Remaining evidence needed

A useful raw client replay should record a Scarlet in the Tavern below six
Attack, a specific permanent or Spellcraft buff crossing six, its controller,
zone, attack and Divine Shield tags immediately afterward, and subsequent
buy/steal/freeze transitions. For Fugitive, the replay should record a player's
targeted spell cast on the unowned Tavern copy, separate the spell's own
Health buff from any trigger buff, and show whether acquisition changes the
result. Golden copies and repeated casts require separate checks once normal
behavior is established.

These questions cannot be settled by treating an absent upstream listener as
a no-op rule or by setting the shop entity's controller to the acting hero.
Until direct evidence resolves them, whole scenarios reaching either guard
must still be rejected and counted.

```bash
PYTHONPATH=python python -m unittest discover -s tests -p test_offboard_scarlet.py
node --test tests/offboard-scarlet-firestone.test.mjs
```
