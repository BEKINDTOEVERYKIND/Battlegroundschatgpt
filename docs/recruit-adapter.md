# Explicit recruitment adapter

`bg_ai.hsbrsim_adapter.EarlyRecruitFixtureEngine` now executes real `hsrl2`
recruitment commands against an external checkout pinned by
`bg_ai.hsbrsim_data.build_current_database`. It does not load the upstream
rotation's card database or modify the external checkout.

This is a **restricted early recruitment curriculum**, not a full-game Solo
simulator or placement learner. It runs eight synthetic players for at most two
recruit rounds, before turn-three Dark Gifts. The last round stops after all
players finish recruitment, before that round's combat/end effects. Earlier
rounds execute the upstream global phase exactly once, only after every player
has individually finished. No hand is automatically played at the end of a turn.

The fixture's exact current-card pool is seven minions (Crackling Cyclone, Risen
Rider, Razorfen Geomancer, Southsea Busker, Shell Collector, Intrepid Botanist,
Suspicious Prisonguard), three Tavern spells (Tavern Coin, Tavern Dish Banana,
Fortify), and generated Blood Gems. Its heroes are Patchwerk and optionally
Pyramad. Optional initial boards and hands are explicitly synthetic setups, not
unpaid plays. The reduced pool changes offer probabilities; results do not
measure full-pool game strength.

```python
from pathlib import Path
import json
from bg_ai.hsbrsim_data import build_current_database
from bg_ai.hsbrsim_adapter import (
    EarlyRecruitFixtureEngine, SUPPORTED_MINIONS, SUPPORTED_SPELLS,
)
from bg_ai.timed_recruiting import TimedRecruitEngine
from bg_ai.turn_budget import TimingProfile

ruleset = json.loads(Path("data/ruleset.json").read_text())
current = build_current_database(
    Path("/path/to/pinned/HSBRSIM"),
    fixture_minion_ids=SUPPORTED_MINIONS,
    fixture_spell_ids=SUPPORTED_SPELLS,
)
raw = EarlyRecruitFixtureEngine(
    current.db, ruleset, provenance=current.provenance,
    # An explicit curriculum timer assumption, NOT a factual client schedule:
    timer_ms=lambda player_id, turn: 60_000,
)
engine = TimedRecruitEngine.for_fixture(
    raw, TimingProfile.load("config/turn-budget.json"), ruleset,
)
observation = engine.reset(seed=123)
```

The production `TimedRecruitEngine(...)` constructor still requires successful
full-game conformance. Its separate `for_fixture(...)` entry checks the concrete
fixture type, scope, matching ruleset digest, and explicit `full_game_ready=False`.
Every observation includes the supplied player's remaining timer and the shared
`TurnBudget` state. The timer callback has signature `(player_id, turn) -> ms`;
it may tighten a clock but cannot refill spent time. There is no invented
round-duration table. Economically profitable loops still exhaust time/actions.

The action list contains separate buys, minion plays and placement positions,
spell casts, sells, refreshes, freeze/unfreeze toggles, Tavern upgrades, minion
moves, hero powers, activates, and pending choices/targets. All target and choice
options are explicit; policy actions never use a random or first-choice fallback.
A spell or Activate followed by a target selection costs two actions in this
conservative input model. Only battlefield minions can be sold. Upgrades stop at
tier two in the fixture. Pending choice expiry aborts the rollout: real client
choice-timeout behavior has not been verified.

Observations contain the player's own board, hand, shop, gold, counters and
choices plus public hero health/armor/tier. They exclude opposing shops, hands,
current private boards, pool counts, hidden future draws, and engine RNG state.
Entity IDs are local opaque references, independent of card-table indices.
Numbers in current effect-text placeholders are resolved before feature encoding.
The trusted `player_board(player_id)` evaluator export must not be fed back into
a policy to reveal another player's private state.

`engine.fork()` replays actions from reset into fresh engine objects, then copies
the already-spent per-player budgets. Python `deepcopy` is deliberately avoided:
upstream callbacks close over mutable heroes and games and would otherwise point
into the original branch. Forking a pending target/Choose One is behaviorally
tested to leave its parent untouched.

The local fixture compatibility class corrects four specific upstream issues:
manual refresh replaces a frozen shop; freezing is a toggle; a frozen shop fills
its own missing minion slots without counting the dedicated Tavern-spell slot;
and a purchased Tavern spell immediately returns to its pool and is not returned
again when played. The dedicated spell slot and return timing follow Blizzard's
[28.2 announcement](https://hearthstone.blizzard.com/en-gb/news/24008697/28-2-patch-notes).
The verified targeted fixture buffs can also target Tavern minions.

Encountering an unsupported card, seasonal state, triple/golden transition,
full-hand generated-card queue, or unknown target payload raises an error.
Registration of other upstream scripts does not permit their use here. Full-game
placements remain unavailable, and the wider engine audit remains blocked.

Tests in `tests/test_hsbrsim_adapter.py` exercise the actual pinned engine: explicit
Gem and Activate chains, choices, spell pool accounting, freeze/refresh behavior,
eight-player round closure, no hand autoplay, separate budget charging, timeout
errors, hidden-state exclusion, replay branch isolation, and unsupported effects.
Set `HSBRSIM_ROOT` to an exact clean checkout when it is outside the conventional
`../research/HSBRSIM` location. These tests skip with an explicit reason when no
external checkout is supplied; a skipped engine test is not conformance evidence.
