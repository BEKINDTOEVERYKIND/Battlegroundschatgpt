# Two-turn opening transition

`python/bg_ai/opening_transition.py` adds a separate **two-recruit-turn,
two-combat fixture**. It uses the current 22 Tier-1 minions, eight Tier-1 Tavern
spells and five-tribe restrictions already checked by the opening database.
The old first-combat fixture and archived model/data semantics are unchanged.
This transition makes the next shop, a held Coin, Spellcraft expiration and
Southsea Busker's deferred income available to a real two-turn objective.
It does not establish full-game playing strength.

The original live recruitment entities remain in Python throughout combat.
`simulator/opening-transition-firestone.mjs` simulates separate copies, including
the actual private hands. One independently seeded sample per declared player
pair returns the winner and exact integer damage. Python verifies the complete
request's SHA256, eight-player pairing coverage, turn, seed, sample count and
damage consistency before applying the receipt. An aggregate win probability or
average damage is never used as the next game state.

The engine disables native `run_combat` and `end_recruit_phase`. Calling either
raises. End-of-turn effects are applied once before exporting the request, so
they cannot be replayed accidentally when combat completes.

| State | Transition into recruit turn two |
| --- | --- |
| Original battlefield and permanent buffs | Keep the same entities and listeners; combat copies, deaths and generated tokens do not overwrite them. |
| Flighty Scout in hand | Keep the original hand minion; its combat summon is a copy. |
| Health and armor | Apply the sampled loss to armor, then health, with the current opening Solo cap of five. |
| Frozen shop | Preserve retained entities and their buffs; refill to three minions plus a separate spell slot; consume the freeze. |
| Unfrozen shop | Return existing shop copies and draw the actual next shared-pool shop. |
| Saved hand cards | Preserve them, including unplayed Tavern Coins; unused Spellcraft cards expire before combat. |
| Spellcraft buffs | Actual per-target turn-start listeners remove temporary buffs, including buffs on retained frozen shop cards. |
| Fresh Spellcraft | Generate a new spell from each qualifying minion through the actual start-of-turn engine hook. |
| Gold | Reset to four, then execute actual queued deferred gains. Unspent gold is not banked. |
| Deferred gold feature | Sum the explicit `GainGold.amount` values, rather than counting scheduled actions. Unknown deferred action types abort. |
| Upgrade cost | Apply the real one-gold turn-start discount. A second-turn upgrade remains a legal action. |
| Activate and per-turn counters | Reset through the real start-of-turn hook. Permanent card counters and next-spell discounts persist. |
| Action budget | Start a separate clock for each player's second recruit turn from the supplied timer callback. Forking never refunds used time. |

This no-combat-writeback rule is scoped to the inspected current Tier-1 closure.
Glim Guardian and Tusked Camper grow themselves during combat; Rot Hide Gnoll's
bonus counts only this combat; Scarlet Survivor's combat-acquired shield is not
a general permanent combat-enchantment mechanism. None of the current Tier-1
cards requires carrying combat stat deltas or generated combat cards into the
next recruit phase. That statement must be reviewed before adding higher-tier
minions, heroes, Dark Gifts, Trinkets or other persistent combat effects.

The original hero health and public Tavern tiers are visible policy information.
Combat requests, complete opponent hands, pair sample seeds, receipt hashes,
pool inventory and future shops are **trusted evaluator data**. They never enter
`RecruitObservation.public_state` or the current player's private observation.
The explicit eight-player pairings are synthetic experiment assumptions; this
module does not certify Blizzard's matchmaking or hidden future-state sampler.

## API

Construct the same `OpeningFixtureSpec` and current database as in
`docs/opening-pool.md`, then use these new classes:

```python
from bg_ai.opening_transition import (
    TwoTurnOpeningRecruitEngine, TimedTwoTurnOpeningEngine,
)

raw = TwoTurnOpeningRecruitEngine(current.db, ruleset,
    fixture=spec, provenance=current.provenance,
    timer_ms=lambda player_id, turn: 60_000)
timed = TimedTwoTurnOpeningEngine.for_fixture(raw, timing_profile, ruleset)
observation = timed.reset(seed=123)
# Execute every explicit recruit command through timed.step(action).
# None means no current recruit observation; inspect raw.phase.
# All eight players ending produces raw.phase == "awaiting_combat".
request = raw.combat_request(
    pairings=[(0, 1), (2, 3), (4, 5), (6, 7)], seed=456)
# Send request as one JSON line to the worker below; parse its receipt.
observation = timed.advance_combat(receipt)
# First receipt starts turn two. Second receipt ends at phase "complete".
```

The 60-second value is an explicit synthetic timer assumption, not a claim about
the live client's turn schedule. Every buy, play, choice, position, cast,
activate, freeze and upgrade remains separately charged by the existing wrapper.

```bash
node simulator/opening-transition-firestone.mjs
```

The worker accepts one raw request JSON per line and returns one receipt JSON per
line. Failure returns `{"error": "..."}`. It remains alive for subsequent
independent requests. Pair seeds are reproducibly derived from the supplied
32-bit seed, pair index and player IDs. Root candidates may use matched seeds
where valid, but final evaluation must reserve independent seeds.

`fork()` reconstructs fresh callback closures by replaying the complete sequence
of recruit commands and sampled receipts. It does not call Firestone again or
use `deepcopy` on live engine closures. The time wrapper also copies its spent
budgets and closed-turn records. Forking is deterministic conditional on the
same hidden initial state; it is not itself sampling multiple hidden futures.

## Explicit remaining frontiers

All legal commands remain visible. An unsupported branch is rejected as a whole;
it must not be treated as a weaker legal move or retried until a favorable
outcome appears. An experiment that requires every candidate to complete must
report the resulting conditional scenario coverage and rejection counts.

- A turn-one upgrade requires a complete Tier-2 shop when turn two begins.
  That transition aborts before applying combat damage. A turn-two upgrade is
  playable, but a following Tier-2 refresh is still unsupported. The two-combat
  objective also understates benefits arriving after the second combat.
- Ordinary triples/rewards, generated full-hand queues, pending-choice timeout,
  off-board Scarlet Survivor thresholds and Tavern-targeted Fleeing Fugitive
  retain the existing opening guards.
- Applying Mini-Myrmidon's temporary buff to a Tavern Lullabot and then
  magnetizing it reaches a specifically guarded transfer frontier. Upstream
  flattens those stats into the host's permanent base stats. The new module
  rejects this legal magnetic action **before changing either zone**. Blizzard's
  23.2.2 fix explicitly establishes that this interaction must not make the
  Spellcraft buff permanent. The old one-combat engine remains unchanged because
  this error concerns later persistence.
- Eliminations, ghosts, turn three, Dark Gifts, Trinkets, different heroes and
  persistent combat effects outside the current Tier-1 closure remain outside
  this fixture. Full-game and full-opening conformance flags remain false.
- Scarlet Survivor's internal one-time threshold marker is not part of the old
  generic combat export. The inspected current Tier-1 combat closure has no
  effect that raises another minion's Attack, so no wrong reachable outcome was
  found here; broader pools require explicit marker transport.

## Evidence and checks

Current numbers and pool identities come from the pinned manifest/client hashes,
not historical simulators. The following primary sources support the underlying
persistent rules:

- [Introducing Battlegrounds](https://hearthstone.blizzard.com/en-us/news/23156373/introducing-hearthstone-battlegrounds): freezing preserves the current lineup for the next Tavern visit.
- [28.2 patch notes](https://hearthstone.blizzard.com/en-gb/news/24008697/28-2-patch-notes): the Tavern spell slot is additional to minions and obtained spells return to their pool.
- [Prepare for the Rise of the Naga](https://news.blizzard.com/en-gb/article/23784374/prepare-for-the-rise-of-the-naga): Spellcraft buffs last until the next recruit phase.
- [23.2.2 patch notes](https://news.blizzard.com/en-gb/article/23797162/23-2-2-patch-notes): corrected temporary Spellcraft becoming permanent through a Tavern Magnetic minion.
- [Announcing Battlegrounds Season 9](https://news.blizzard.com/en-us/article/24159389/announcing-battlegrounds-season-9): Solo opening damage cap starts at five and is removed in the top four.

Focused tests execute the actual pinned Python engine and installed Firestone.
They cover all 22 current Tier-1 minions crossing the boundary, real damage and
armor, retained frozen identities and independent spell capacity, shared-pool
conservation, saved cards, exact Busker income, Lullabot growth, Spellcraft
expiration/regeneration, Activate/counter reset, two-combat termination,
replayable forks and clocks, receipt rejection, private-state exclusion and
explicit unsupported transitions. These are fixture correctness checks, not a
playing-strength benchmark.

```bash
PYTHONPATH=python python -m unittest discover -s tests -p test_opening_transition.py
node --test tests/opening-transition-firestone.test.mjs
```
