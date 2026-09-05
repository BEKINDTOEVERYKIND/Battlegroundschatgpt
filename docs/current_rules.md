# Current rules and data evidence

Observed **5 September 2026**, for **Solo Battlegrounds**, Season 14,
**Dark Gifts of Dalaran**, client **36.4.2.251332**.

The verified combat pool is available for training. The complete game's rules
are **not yet verified or implemented**. In particular, the unverified Trinket
pool and incomplete offering probabilities must not be represented as a fully
working season. `data/ruleset.json` records these distinctions explicitly.

## Sources and version pinning

The official [36.4.2 patch notes](https://hearthstone.blizzard.com/en-us/news/24296231/3642-patch-notes)
were published on 3 September. They change minion statistics and tiers, hero
armor, Tavern spells, and Trinkets. The snapshot checks concrete patch
sentinels including Hoarding Hyena, Goldrinn, Private Investigator, Cagey
Conjurer, Unleashed Mana Surge, Tasty Lobster, and Natural Blessing. Sanctify
must not appear in the Tavern spell pool.

Current [Blizzard known issues](https://us.forums.blizzard.com/en/hearthstone/t/364-highlights-known-issues/153567)
list Gentle Djinni as banned. It is absent from both the current client pool
and official Solo gallery. Returning in an earlier patch does not override a
subsequent ban.

Client definitions come from
[HearthSim/hsdata commit dee8a64](https://github.com/HearthSim/hsdata/commit/dee8a641ef8427cf853ca707c2e427e752a1e11f),
whose declared build is 36.4.2.251332. The pinned
[CardDefs.Bacon.xml](https://raw.githubusercontent.com/HearthSim/hsdata/dee8a641ef8427cf853ca707c2e427e752a1e11f/CardDefs.Bacon.xml)
includes numeric tags and canonical CardIDs as well as DBF IDs. Its compressed
original is preserved in `data/source/CardDefs.Bacon.xml.gz`.

The [official card gallery](https://hearthstone.blizzard.com/en-us/battlegrounds)
uses a public JSON endpoint. Its actual Solo mode parameter is **`solos`**:

```text
https://hearthstone.blizzard.com/en-us/api/cards?gameMode=battlegrounds&bgGameMode=solos&bgCardType=minion&pageSize=450&page=1&locale=en_US
```

Separate queries use `minion`, `hero`, `spell`, `trinket`, and `darkGift`.
All pages must be retrieved; `cardCount`, uniqueness, and client identity
matching are checked. Gallery numeric `id` means DBF ID. It is mapped to a
canonical string CardID before it can enter a training whitelist.

## Verified pool and unresolved records

| Component | Official Solo gallery | Training status |
| --- | ---: | --- |
| Minions | 246 | Verified; 234 normal-shop minions and 12 Tier 7 minions |
| Heroes | 116 | Verified identity, base health, armor, and associated power |
| Tavern spells | 71 | Verified identity, cost, tier, and pool membership |
| Dark Gifts | 43 | Verified identity and text; full offering procedure remains incomplete |
| Lesser Trinkets | 109 | 108 candidates after a confirmed removal; inactive until discrepancy review |
| Greater Trinkets | 120 | 120 candidates plus one unresolved client record; inactive until discrepancy review |

Minion, hero, and Tavern spell gallery pools match their client eligibility
tags exactly after removing Duos-exclusive entries. Differences before the
Solo filter are 28 minions, five heroes, and three Tavern spells.

Tier 7 cards can occur through special acquisition rules. Their presence in
the card gallery does **not** make them normal shop offerings. Ordinary shops
must use `active.shop_minion_ids` and the player's Tavern Tier. The broader
`active.minion_ids` is suitable for examining possible combat boards.

Aureate Laureate (`BG32_236`) is an exceptional legal shop minion that is
always Golden. Rejecting every `premium=true` card would incorrectly remove
it. An upgraded variant's `battlegroundsNormalDbfId` distinguishes it from a
normal pool entity.

Two source discrepancies are recorded in `data/source_discrepancies.json`:

* **Blessing Portrait** (`BG32_MagicItem_894`) is still present in the gallery
  and client eligibility tags despite the 36.4.2 removal. The dated official
  removal takes precedence, so it is excluded.
* **Sphere of Memory** (`BG36_MagicItem_372`) is marked eligible in the client
  but absent from the gallery. Its eligibility has not been resolved. Both
  active Trinket lists therefore remain empty; candidates are separate.

The patch also removes Lesser Copper Coil, Cowrie Necklace, Deathwhisper
Sticker, and Greater Coral Spear. The current gallery already omits those
specific versions. Greater Copper Coil remains distinct from its removed
Lesser version.

## Current seasonal systems

[Season 14's announcement and patch](https://hearthstone.blizzard.com/en-us/news/24290432)
confirm that **Dark Gifts and Trinkets coexist**. Activate, Fishbait, and
Lockboxes are also relevant. Activate abilities are usable during recruitment
and require their stated Gold cost; their normal limit is once per turn.
Fishbait allows attacks during recruitment, so an engine restricted to combat
attacks cannot reproduce this season. Lockboxes delay access to Golden
minions and interact with ways of opening them sooner.

Dark Discovery unlocks on turn 3, costs 3 Gold, presents three minion/gift
pairs, and has limits of one use per turn and three per game. Card and gift
eligibility changes with the turn. The [developer's offering specification](https://us.forums.blizzard.com/en/hearthstone/t/battlegrounds-developer-insight-dark-gifts/163606)
describes minion tier windows, type-dependent restrictions, exceptions for
Tras'tath, and nonuniform gift rarity. The launch specification alone is
insufficient: the [36.2.1 hotfix](https://us.forums.blizzard.com/en/hearthstone/t/3621-hotfix-patch/164300)
and [36.2.2 patch](https://hearthstone.blizzard.com/en-us/news/24293284)
change eligibility windows, effects, and exclusions.

`data/dark_gift_offering_rules.json` contains the current client turn-window
tags and separately marks the incomplete parts. It is not an executable
claim of exact offering probabilities. Unknown probabilities must not be
silently replaced by uniform sampling for a claimed faithful trainer.

Global Quests, Buddies, Anomalies, and the Timewarped Tavern are not enabled
as seasonal systems in this snapshot. This does not mean their every card
is impossible: hero-specific access, such as E.T.C.'s Buddies or Sire
Denathrius's Quest, requires explicit generation rules. A hero-specific
exception does not enable a retired system for the whole lobby.

## Dataset contract

`data/reference_cards.json` is a list of current-client reference objects.
Fields needed by the Firestone adapter include `id`, `dbfId`, `type`, `set`,
`attack`, `health`, `techLevel`, `races`, `mechanics`, `tags`, `premium`,
`battlegroundsNormalDbfId`, and `battlegroundsPremiumDbfId`. Hero records also
carry `heroPowerDbfId`; script parameters are retained. `rawTags` preserves
unknown numeric tags rather than guessing their meanings.

The reference list retains 5,613 Battlegrounds client entities, including
old definitions, tokens, Golden variants, enchantments, and hero-specific
exceptions. **Reference presence grants no gameplay eligibility.** Only
manifest whitelists enable normal recruitment or drafting. `isBaconPool` is
reset from the verified active list; retired and Duos-only minions cannot
be recruited merely because their definitions are available.

The enum name mapping is derived from the MIT-licensed
[`@firestone-hs/reference-data` 3.0.196](https://www.npmjs.com/package/@firestone-hs/reference-data/v/3.0.196).
Numeric tags are authoritative when a newly introduced enum lacks a name.
`ReferencedTag` entries are distinct from actual card mechanics: mentioning
Deathrattle in a card's text does not itself give that card a Deathrattle.

The manifest's `status=blocked` and `snapshot_verified=false` apply to the
incomplete full snapshot. `combat_pool_verified=true` authorizes the
verified combat slice only. `fullgame_supported=false` prevents confusing
metadata verification with a complete game engine.

Every frozen file has a SHA-256 entry in `checksums`. These paths are
relative to **`data/`**, not the repository root. Training artifacts should
record the complete ruleset and reference data hashes.

## Reproduction and rotation workflow

```bash
python scripts/sync_ruleset.py --check
python scripts/sync_ruleset.py --offline
python scripts/sync_ruleset.py --refresh
```

`--check` performs read-only integrity and eligibility validation. With no
arguments the script also checks only. `--offline` regenerates the frozen
snapshot from its saved inputs and preserves the observation date.

`--refresh` downloads into an isolated timestamped candidate directory. It
never overwrites the trainer's current snapshot. A changed live build fails
closed and records the observed version for review. Even a same-build
candidate needs fresh official patch and ban review; it cannot declare
itself trainable based only on unchanged client version numbers.

Promotion of a new rotation requires reviewing additions/removals and
effect changes, updating the pinned client commit, reconciling client and
gallery data, auditing simulator support, and rerunning validation. This
preserves the reusable representations and engine interface while keeping
outdated cards out of the next trainer.

## Live preflight before current-game training

```bash
python scripts/check_live_ruleset.py
python scripts/check_live_ruleset.py --report reports/live-preflight.json
python scripts/check_live_ruleset.py --allow-historical
```

The read-only live preflight is separate from frozen-file integrity. It
compares six live sources with `config/live-source-guard.json`: HearthSim's
current declared client version; the official Solo minion, hero, Tavern
spell, and Dark Gift galleries; and the maintained first official post of
the current [known-issues topic](https://us.forums.blizzard.com/en/hearthstone/t/153567.json).

Gallery comparison covers gameplay text, identity, statistics, costs,
tiers, types, keywords, related cards, and Battlegrounds eligibility fields.
Image URLs and other presentation fields are omitted. Arrays and card order
are normalized, so a CDN or sorting change does not invalidate the ruleset.
Incomplete pagination and Duos-only leakage fail the check.

For the known-issues topic, the baseline records its title, first-post ID,
author, edit time, and SHA-256 of the actual `cooked` post content retrieved
on 5 September. It does not hash the comment stream. A ban or maintained-post
edit can therefore block training even if the client build number remains
unchanged; ordinary user replies do not create noise. The first live check
matched all six sources at **09:09:56 UTC on 5 September 2026**.

The guard is tied to the reviewed manifest and reference-data hashes.
Changed sources, unavailable sources, or changed frozen inputs cause a
failure with an evidence report. No failed check changes the baseline or
automatically approves a replacement. The explicit `--allow-historical`
option, also accepted as `--offline`, skips network freshness and always
returns `current=false` and `status=historical`.

This establishes agreement with the watched public sources at check time.
It cannot establish a server-side change that Blizzard has not yet reflected
there, and a separate new announcement may precede the maintained topic's
update. New patch and hotfix announcements still require review. Long runs
should check again before promoting a model as current. Passing freshness
does not remove the existing full-game support and Trinket-pool blockers.
