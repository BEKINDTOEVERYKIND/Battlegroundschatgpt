# Complete current Tier-1 opening pool

The separate `hsbrsim_opening.py` curriculum offers **all 22 current Tier-1
minions and all 8 current Tier-1 Tavern spells**, filtered to exactly five lobby
tribes. It preserves the previous seven-card adapter and trained results.

The current pool identity is checked against the pinned client and manifest
before loading. A rotation changing the Tier-1 IDs blocks this module. Both tribe
memberships of Ominous Seer are preserved: it appears when Demon or Naga is in.
`MECH` is the canonical installed Firestone enum; `MECHANICAL` is accepted only as
an input alias and normalized. No absent tribe is silently inserted.

The module is a **first-recruit-turn fixture, with explicit unsupported
frontiers**. Pool completeness does not establish full opening rules conformance
or full-game playing strength. Eight synthetic Patchwerk players share the real
minion/spell pools. After every player finishes, the adapter applies opening
end-of-turn effects and removes unused Spellcraft before exposing a precombat
snapshot. No subsequent recruit turn, final placement or inferred MMR is emitted.

```python
from pathlib import Path
import json
from bg_ai.hsbrsim_opening import (
    OpeningFixtureSpec, build_opening_database, OpeningRecruitEngine,
    TimedOpeningEngine,
)
from bg_ai.turn_budget import TimingProfile

spec = OpeningFixtureSpec(valid_tribes=("BEAST", "MECH", "MURLOC", "NAGA", "PIRATE"))
current = build_opening_database(Path("/path/to/pinned/HSBRSIM"),
                                 valid_tribes=spec.valid_tribes)
rules = json.loads(Path("data/ruleset.json").read_text())
raw = OpeningRecruitEngine(current.db, rules, fixture=spec,
    provenance=current.provenance, timer_ms=lambda player_id, turn: 60_000)
# 60 seconds above is an explicit synthetic timer assumption, not client fact.
engine = TimedOpeningEngine.for_fixture(
    raw, TimingProfile.load("config/turn-budget.json"), rules)
observation = engine.reset(seed=123)
```

All commands use the same finite time/action budget as v1. Buys never auto-play.
Mini-Myrmidon's generated spell gets an explicit target action, fixing the
upstream silent no-target fizzle. Alliance Flag resolves cast → branch → target
→ buff and spell events, charging three commands and never auto-selecting either
choice. Discover options contain current visible text and stats, and generated
cards stay in hand until their own play. Lullabot can be played as a body or
Magnetized; attachment buffs, granted keywords and end-of-turn growth are
preserved, and selling the host returns its attached pool copy. Aureate Laureate
is correctly Golden in the Tavern, hand and battlefield; three copies stay three
Golden minions, produce no reward and return one pool copy each when sold.

The underlying engine's ordinary triple path is unsafe, so acquisitions that
would reach it raise an explicit error. This includes generated acquisitions
through River Skipper, Recruit a Trainee, Enchanted Lasso or A New Sprout.
River Skipper loops can accumulate gold even on turn one; the shared time budget
bounds them, while an eventual triple remains a declared frontier. Upgrading is
an actual command; refreshing afterwards raises because a complete Tier-2 pool
has not yet been included. Full-hand generation, pending-choice expiry, later
turns and seasonal mechanics also abort. These exceptions require discarding the
rollout, never masking an unsupported effect as vanilla stats.

All **22 current Tier-1 minions now have tested explicit play transitions**.
The upstream Wrath Weaver self-trigger was corrected using a
[public raw client replay](https://xml.firestoneapp.com/hearthstone/replay-logged-in/2025/12/15/4df7e210-6f1d-4b85-8946-f952d161fa31.xml.zip)
posted after Weaver became a Demon. In that replay, playing the first Weaver
does not buff itself or damage its hero; playing a second triggers only the
first. The local handler reproduces that ordering with **current** +2/+2 numbers,
not the replay's older +2/+1 values.
Molten Rock now uses an original no-self handler with current numerical values:
the same After-you-play semantics are independently implemented and tested by
[stone_ground_hearth_battles](https://github.com/JDBumgardner/stone_ground_hearth_battles/blob/05dba5d0c2c58b1c2f6a6011f5896e8d1b8f3617/hearthstone/simulator/core/card_pool.py),
including its [Rock-then-Cyclone test](https://github.com/JDBumgardner/stone_ground_hearth_battles/blob/05dba5d0c2c58b1c2f6a6011f5896e8d1b8f3617/tests/test_tavern.py).
That historic engine is evidence for trigger ordering, not a source of current stats.
Scarlet Survivor's off-board threshold
behavior and Tavern-targeted Fleeing Fugitive triggers also need client verification.
Reaching an off-board Scarlet threshold or selecting a Tavern Fugitive spell
target therefore aborts explicitly, while ordinary on-board effects remain tested.
Accordingly,
`coverage()["full_opening_rules_ready"]` remains false; a smoke-tested handler is
not promoted to complete rules conformance.

`simulator/opening-firestone.mjs` provides
`OpeningFirestoneCombat.evaluate({player, opponent, trials, seed})` for snapshots
from `raw.combat_snapshot(player_id)`. It uses the corrected Firestone v2 tribe
bridge and **the actual hand**, allowing Flighty Scout's hand-only Start of
Combat summon to affect results. It preserves existing stat buffs and their
permanent/temporary enchantment metadata. The real current deathrattle/Rally
implementations generate Skeletons, Microbots, Beetles and Bat tokens. Input
provenance, eligible IDs, five tribes, natural Golden identity and both hand
arrays are checked. Combat changes are explicitly not written into a later
recruitment state; that persistence integration remains future work.

Behavioral tests cover all 22 explicit plays and every
current Tier-1 spell's complete choice chain, dual-tribe and Mech offering,
Magnetic effects/pool return, Aureate identity, Spellcraft expiration, target
order, bounded choices, fork isolation and frontier failures. Firestone tests
exercise all 22 minions, actual token-producing combat, provenance rejection,
and a hand-only Flighty Scout changing an empty-board tie into a win.

Sources: current definitions use the pinned manifest/client evidence already
recorded by `hsbrsim_data`; the
[official Season 14 minion update](https://us.forums.blizzard.com/en/hearthstone/t/battlegrounds-season-14-minion-updates/163700)
lists the current changed minions, and the
[28.2 spell announcement](https://hearthstone.blizzard.com/en-gb/news/24008697/28-2-patch-notes)
establishes the dedicated Tavern spell slot and return-to-pool timing.
