# Ordinary triples and playable rewards

`bg_ai.hsbrsim_rules` repairs the basic triple transition in one external `hsrl2`
Game instance. It does not modify upstream files or certify a whole-game trainer.
Use it with the fresh current database from `bg_ai.hsbrsim_data`.

## Rule evidence

Blizzard's [Battlegrounds introduction](https://hearthstone.blizzard.com/en-us/news/23156373/introducing-hearthstone-battlegrounds)
says three matching minions combine into a golden version that retains buffs.
Playing that golden minion grants a reward card in hand; the reward discovers
from the next Tavern Tier. The current client reward card is
`TB_BaconShop_Triples_01`, whose text names a specific Tier.

The [32.4.2 patch notes](https://news.blizzard.com/en-gb/article/24205944/32-4-2-patch-notes)
add a 3-Gold pouch when a Triple Reward has no valid minions to Discover. The
current client defines `BG27_Anomaly_574t2` as that consolation spell. It grants
Gold when played, so the implementation puts the pouch into hand and requires
another player action.

## Corrected sequence

1. The third ordinary matching minion combines with the other two. The golden
   always enters the hand, including when earlier copies occupied the board.
2. The player can upgrade before playing the golden. Playing it grants a
   separate reward spell whose tier uses the player's Tavern Tier at that time,
   capped at six for ordinary Solo rewards. The original minion's tier is irrelevant.
3. The player plays the reward spell as a separate action. Its saved tier does
   not change if the player upgraded after receiving it.
4. A manual `triple_reward` pending choice presents up to three eligible distinct
   minions from that tier. The chosen minion consumes one actual pool copy and
   enters the hand, where it can itself complete a triple.
5. If there are no eligible minions, a playable consolation pouch enters the
   hand instead. Neither reward generation nor pouch generation automatically
   selects or plays another card.

The exact card-instance tier is stored in `_bg_ai_triple_reward_tier` and the
entity's `GameTag.TECH_LEVEL`, so an adapter can expose it to the learner. Reward
sampling uses the engine's seeded sampler and current, nonempty, lobby-filtered
minion pool. Offering a choice does not remove pool copies; resolving it does.
The callback prevents duplicate resolution and raises if a selected copy became
unavailable. This does not independently validate all Discover distribution or
simultaneous-player reservation rules.

## API and lifecycle

```python
from bg_ai.hsbrsim_data import build_current_database
from bg_ai.hsbrsim_rules import installed_recruit_rules, grant_triple_reward
from hsrl2.game import Game

built = build_current_database("../research/HSBRSIM")
game = Game(heroes, built.db, seed=17)
with installed_recruit_rules(game):
    # Normal purchases and generated minions use the repaired triple check.
    # A separately audited hero/effect can also grant a reward directly:
    reward = grant_triple_reward(game, heroes[0], source="audited-effect-id")
    # The action adapter must enumerate playing this spell and choosing a result.
```

`grant_triple_reward` also accepts an explicit `reward_tier` for separately
audited effects. It binds the reward's script directly, so the helper does not
require global registry mutation. Creating a raw reward by ID without setting
its frozen tier is insufficient and fails loudly when its effect is invoked.

`installed_recruit_rules(game)` reversibly overrides only that instance's
triple check, reward grant and minion-play guard. Keep its context active for the
whole game. Another Game instance remains unchanged, and original instance
attributes are restored even when an exception occurs. The existing timed
adapter must still count every play and choice; these rules do not provide a
new timer or refill the existing budget.

## Supported merge state and remaining boundaries

Ordinary permanent numeric buffs and the supported bonus keywords survive the
merge. Buff objects are copied without firing fresh gain-stat events. The new
golden's printed stats are used once; hero-wide race auras apply once rather
than being added for each consumed copy. Blood Gem buffs retain their identity
and the played-Gem count is summed. Consumed entities' listeners are unregistered.
The three existing pool copies stay out of the pool; selling the resulting
ordinary triple returns exactly three base copies.

The merge validates participants before removing them. It raises for Magnetic
attachments, Dark Gifts, custom hooks, unknown per-minion counters or tags,
temporary buffs, modified base/set-stats, substitute-Elemental triples, alternate
triple thresholds/rewards and unaudited generated-token triples. These are
specific implementation gaps, not permission to remove those mechanics from the
actual game. In a rollout, an unsupported transition invalidates the rollout;
a surrounding action transaction remains responsible for rolling back any
purchase that occurred before the merge check.

Golden Battlecry and Choose One play are blocked by default before hand/board mutation.
The explicitly migrated IDs in `installed_battlecry_rules(game)` are permitted
while that version-one dispatcher is installed; see `battlecry-migration.md`.
Upstream doubles golden Battlecry executions globally, and many individual
scripts compensate by using normal-card values. Removing that multiplier without
migrating the individual handlers would break other effects. Simple golden
minions without that play-effect path can exercise the repaired reward sequence.
This module consequently reports `full_game_ready=False`.

## Executed validation

```bash
HSBRSIM_ROOT=../research/HSBRSIM PYTHONPATH=python \
  python -m unittest discover -s tests -p 'test_hsbrsim_rules.py' -v
```

Sixteen tests pass against the actual pinned external engine and fresh client
DB. They cover mixed hand/board combining, no premature reward or Discover,
Tavern-tier selection and a frozen reward tier, the Tier-six cap, manual choice
and one-copy acquisition, empty-pool pouch play, small pools/lobby restrictions,
buff preservation without extra gain events, listener removal, three-copy pool
conservation, Discover completing a new triple, unsupported-merge rejection,
golden Battlecry/Choose One rejection, missing-tier failure and instance
restoration. Integration tests explicitly skip if the external checkout is
absent; a skip does not certify these transitions.
