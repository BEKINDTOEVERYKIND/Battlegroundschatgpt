# Turn timers and finite action budgets

Every intentional recruit action must consume a shared budget for that player
and turn. Generating gold or cards cannot generate more time. Ending the turn,
changing plans, interleaving opponents, or opening a Discover must not reset it.

The default `config/turn-budget.json` permits **one complete command per second**
and reserves five seconds before the supplied recruit timer expires. This is a
configurable training assumption, not a measured claim about the user's APM or
client animation speed. Mouse movement and clicks within one complete command
are not separate actions.

The hard upper bound is:

`floor(max(0, available_seconds - 5) * actions_per_second)`

Examples of supplied available time, not a claimed game-round schedule:

| Available time | Maximum commands at the default rate |
|---|---:|
| 30 seconds | 25 |
| 60 seconds | 55 |
| 100 seconds | 95 |

Actions also consume virtual time. Each costs at least one second by default.
The profile adds 0.25 seconds for buying/selling, 0.5 for playing, refreshing,
upgrading, spells, hero powers, and Activate, and one second for choices and
Trinket purchases. Moving and freezing have no added delay. Consequently an
animation-heavy turn normally has fewer actions than the hard count limit.
These allowances need calibration against client traces; they do not model
every possible animation chain. An observed shorter remaining timer further
reduces the budget, and a larger repeated timer reading cannot refill it.

## Action semantics

Buying, selling, playing, rerolling, freezing, upgrading, casting a spell,
using a hero power, activating, selecting a choice, purchasing a Trinket, and
dragging a minion each consume a command. A targeted play includes its target;
a resulting Discover requires an additional `choose` command. A macro such as
"buy, play, choose, sell" must expand into four actions. Adapter action kinds
must map explicitly to the profile; unknown kinds fail instead of being free.

Automatic card triggers are not extra player inputs. They still require the
engine's independent effect-resolution loop safeguard, and their elapsed
animation time must be reflected in supplied player time where available.
End-turn is permitted without charging another player command. At exhaustion,
the wrapper calls the adapter's expiry handler; it never auto-plays the hand.
Unresolved choices require validated client timeout behavior, not an invented
random or optimal selection after the budget has expired.

## Implementation and present scope

`TurnBudget` implements the clock. `TimedRecruitEngine` applies it at every
recruit transition, masks unaffordable legal actions, records time/action traces,
and exposes remaining time and commands in `RecruitObservation.action_budget`.
Every player has a separate clock, preserved across interleaved observations.
There are no wall-clock sleeps in simulation. Faster CPUs cannot enable extra
commands. Branching a planning search must copy the current clock, not create a
fresh turn budget; only the selected actual transition spends the real budget.

The wrapper requires a validated full-game engine plus explicit conformance for
actual player turn windows and pending-choice expiry. HSBRSIM currently provides
neither, so this adds working, tested enforcement infrastructure without claiming
that full-game self-play is ready. Its internal 20,000-effect guard does not cap
deliberate economic loops. Its older fixed-action environment also auto-plays
hand cards and has a separate plan budget, so it is not reused here.

The positioning recommendation accepts supplied remaining recruit time, filters
orders by the number of actual drag commands required, and returns a feasible
drag plan. Existing snapshot training and benchmark results were not retrained
with turn timing and must not be relabeled as time-constrained full-game results.
The recommendation is an offline advisor; its search runtime is not a validated
live deployment latency guarantee.

For eventual actor-critic training, feed the time/action features into both
policy and value heads; record the timing profile hash with episodes and
checkpoints. Train and evaluate under the same execution limits. Compare slower
and faster profiles separately, and retain clocks when replaying plans. Do not
claim strength at one APM setting from evaluation at another.

## Timer sources and acceptance tests

The adapter must provide the player's actual remaining recruit window. A generic
round lookup can become obsolete: Blizzard has previously changed timer lengths
in [33.2.2](https://hearthstone.blizzard.com/en-us/news/24223662/33-2-2-patch-notes)
and [34.4.2](https://hearthstone.blizzard.com/en-us/news/24244400/34-4-2-patch-notes).
An exact current schedule and current pending-Discover timeout rules were not
established from primary sources. Missing timing therefore blocks the adapter.

Tests exercise finite buy/sell loops, expiry with pending choices, zero-time
turns, separate/interleaved players, shortening and non-refilling clocks,
affordability masks, and minimum-drag positioning plans.
