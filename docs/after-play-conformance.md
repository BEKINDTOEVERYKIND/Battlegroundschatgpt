# After-play self-trigger evidence

Wrath Weaver must not trigger from its own play. This is directly observed after
its change to the Demon type, rather than inferred from its former neutral type.

Blizzard changed Wrath Weaver from Neutral to Demon in the
[34.2 patch notes](https://hearthstone.blizzard.com/en-sg/news/24244423/34-2-patch-notes-battlegrounds-arena-and-gameplay-updates).
A player then publicly linked a Demon match on
[December 15, 2025](https://us.forums.blizzard.com/en/hearthstone/t/bgs-i-did-my-best-as-far-as-i-think-i-could-have-but/155406).
Its [public replay](https://replays.firestoneapp.com/?reviewId=4df7e210-6f1d-4b85-8946-f952d161fa31)
and [raw XML archive](https://xml.firestoneapp.com/hearthstone/replay-logged-in/2025/12/15/4df7e210-6f1d-4b85-8946-f952d161fa31.xml.zip)
identify build 231720. Both relevant `BGS_004` entities explicitly carry
`CARDRACE=15` (Demon).

| Play timestamp in the XML | Played Weaver | Played card before → after | Existing Weaver | Hero armor |
| --- | --- | --- | --- | --- |
| `24:05:56.613797` | 1831 | 1/4 → 1/4; no own trigger | None | 6 → 6 |
| `24:05:59.715027` | 1833 | 1/4 → 1/4; no own trigger | 1831 triggers once, gaining +2/+1 to become 3/5 | 6 → 5 |

The second PLAY block contains a TRIGGER block for existing entity 1831, creates
`BGS_004e` attached to 1831, then updates that entity's Attack and Health. There is
no corresponding trigger or stat mutation for newly played entity 1833. The
first play likewise contains no self-trigger, stat mutation, or hero damage.
Before/after armor and pre-play zones were checked by folding the full replay's
entity tags in order; their initial values are not all retained in the excerpt.

The checked-in [XML excerpt](../tests/fixtures/conformance/wrath_weaver_demon_after_play.xml)
contains the two exact `FullEntity` definitions and the two exact PLAY blocks,
with an added `HSReplayExcerpt` wrapper. Each retained fragment was checked
byte-for-byte against the downloaded XML. Player names, account identifiers,
and unrelated gameplay are omitted. This is evidence, not a complete replay or
training input. The [machine-readable report](../reports/after_play_conformance.json)
records source URLs, archive/XML/excerpt SHA-256 hashes, observations, and limits.

The source-specific correction is an identity guard: ignore an after-play event
when the played entity is the effect's owning minion. Another copy already on
the board must still trigger. Apply this to the normal Weaver handler and test
both an empty-board first play and a second-copy play. Preserve current
build-251332 parameters: base 1/3 and +2/+2 per trigger. Historical replay
parameters 1/4 and +2/+1 must never enter the current training pool. This evidence
does not cover a golden Weaver or establish every other after-play effect's
timing.

Molten Rock has corroborating evidence from an independent simulator, with a
different confidence level. At commit
`05dba5d0c2c58b1c2f6a6011f5896e8d1b8f3617`,
[stone_ground_hearth_battles excludes the played card itself](https://github.com/JDBumgardner/stone_ground_hearth_battles/blob/05dba5d0c2c58b1c2f6a6011f5896e8d1b8f3617/hearthstone/simulator/core/card_pool.py#L1909-L1920).
Its [test plays Molten Rock followed by Crackling Cyclone](https://github.com/JDBumgardner/stone_ground_hearth_battles/blob/05dba5d0c2c58b1c2f6a6011f5896e8d1b8f3617/tests/test_tavern.py#L2378-L2393)
and expects exactly one Health gain. This supports the same identity guard for
Molten Rock's unchanged health-only after-Elemental-play condition, using current
stats. It is independent engine corroboration, not direct observation of Molten
Rock in build 251332.

The pinned HSBRSIM upstream registers these listeners during summon and then
dispatches `card_played`; its comments assert self-triggering without citing
evidence. Wrath Weaver's observed replay contradicts that assertion. Unbound
Tempest and any other handlers sharing that assumption need individual checks;
this finding does not authorize a blanket engine-wide change.
