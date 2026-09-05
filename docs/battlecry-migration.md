# Battlecry and Choose One migration

The external `hsrl2` engine doubles golden Battlecry execution globally, while
many scripts return normal-card amounts to compensate. This distorts the number
of Battlecries triggered, combines incorrectly with Brann, and changes
retriggered golden effects. Its generic play path also sends Choose One effects
through the Battlecry repeat loop.

`bg_ai.hsbrsim_battlecries` fixes these paths on one Game instance and explicitly
migrates the following current card families. Golden strength is entirely inside
the card's intrinsic effect; golden quality itself never adds a trigger.

| Card family | Normal intrinsic effect | Golden intrinsic effect | Classification |
|---|---|---|---|
| Razorfen Geomancer | Generate 2 Blood Gems | Generate 4 Blood Gems | Battlecry; current Tier 1 |
| Southsea Busker | Gain 1 Gold next turn | Gain 2 Gold next turn | Battlecry; current Tier 1 |
| Ominous Seer | Next Tavern spell purchase costs 1 less | Costs 2 less | Battlecry; current Tier 1 |
| Shell Collector | Generate 1 Tavern Coin | Generate 2 Tavern Coins | Battlecry; current Tier 2 |
| Sanguine Champion | Improve Blood Gems by 2/1 | Improve Blood Gems by 4/2 | Battlecry and separate Deathrattle |
| Intrepid Botanist | Choose permanent Tavern spell +1 Attack or +1 Health | Choose +2 Attack or +2 Health | Choose One, once; zero Battlecry triggers |

That is **12 explicit card IDs**, covering every current Tier 1 Battlecry family
and its golden version. These amounts and classifications are checked against
build 251332's current client text. Literal-text implementations reject changed
wording; Sanguine Champion also checks its current numeric script parameters.
The definitions come from the pinned
[HearthSim/hsdata snapshot](https://github.com/HearthSim/hsdata/tree/dee8a641ef8427cf853ca707c2e427e752a1e11f),
joined to Blizzard's current Solo gallery. The [36.4.2 patch notes](https://hearthstone.blizzard.com/en-us/news/24296231/3642-patch-notes)
independently establish the new Sanguine Champion values.

## Dispatch and card generation

For a true Battlecry, the dispatcher executes its entire intrinsic effect once
per applicable trigger. Each execution increments the hero's Battlecry counter
once and emits one `battlecry_trigger` event. A normal golden Razorfen therefore
generates four Gems but only one trigger. With golden Brann it generates twelve
Gems across three triggers.

[Brann's official card page](https://hearthstone.blizzard.com/en-us/battlegrounds/2949-brann-bronzebeard?tier=5&type=minion)
and current normal/golden client text specify two/three triggers. The dispatcher
uses the strongest active Brann/Moira value, taking a maximum across copies;
multiple Branns are not added together. Dead and silenced sources do not apply.
An unrecognized repeat-source tag fails before a Battlecry minion enters play.
These are the four explicitly recognized Brann/Moira IDs; other seasonal/hero
repeat mechanics still need their own migration.

Choose One resolves only the selected branch once, with no Battlecry counter or
Brann multiplication. A separately supplied Choose Both state resolves each
branch once. Attempting to retrigger Botanist as a Battlecry does nothing and
does not open another choice. Ordinary minion plays likewise produce no
spurious Battlecry events.

Generated Gems and Coins enter the hand and require player actions to play.
Southsea Busker schedules its Gold for the following recruit turn. A generated
Coin is not a reserved Tavern offer, so its generation never decreases the
spell pool. The surrounding current recruit adapter must also apply the correct
Tavern-spell purchase/play pool lifecycle; the intrinsic handler alone does not
repair every spell operation in the old engine.

## Integration

```python
from bg_ai.hsbrsim_rules import installed_recruit_rules
from bg_ai.hsbrsim_battlecries import installed_battlecry_rules, dispatch_battlecry

with installed_recruit_rules(game):
    with installed_battlecry_rules(game):
        # Ordinary triples can now play the explicitly migrated golden effects.
        # Actual policy actions still go through the timed recruit wrapper.
        ...
```

The inner context binds migrated scripts to newly created and existing matching
entities, installs the new post-summon play dispatcher, and routes native
`TriggerBattlecry` actions passed through `game.run_actions` into
`dispatch_battlecry`. It does not mutate the process-global registry. Original
instance methods and existing bindings are restored on exit or exception. Keep
both contexts active for the complete game lifetime.

The triple module's default golden Battlecry/Choose One guard permits only the
IDs advertised by this installed version-one dispatcher. Without the dispatcher,
the guard retains its original block. The earlier training fixture and completed
models are unchanged.

Unmigrated Battlecry/Choose One cards, unknown repeat sources, and custom
Battlecry overrides fail explicitly. The broader Dark Gift/Echoing Voice,
magnetic inheritance and other direct script-hook callers still need separate
migration; they are not certified by fixing ordinary play and the native
TriggerBattlecry action. The extension reports `full_game_ready=False`.

## Executed validation

```bash
HSBRSIM_ROOT=../research/HSBRSIM PYTHONPATH=python \
  python -m unittest discover -s tests -p 'test_hsbrsim_battlecries.py' -v
```

**26 real-engine integration tests pass.** Every normal/golden migrated family
has a concrete effect assertion. Tests also cover complete current Tier 1
Battlecry coverage; multiple normal/golden Branns; delayed versus immediate
Gold; discount consumption; manual Choose One; Choose Both; native golden
Battlecry retriggers; missing/unmigrated effects; no-op prevention; no false
Battlecry events; instance/registry isolation; and a real three-copy Razorfen
triple whose golden play grants four Gems plus a separate playable Triple
Reward. The prior triple foundation's 16 tests continue to pass.
