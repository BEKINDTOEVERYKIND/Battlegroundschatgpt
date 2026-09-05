# Current Trinket integration — 5 September 2026

`bg_ai.trinkets` implements **39 recruitment effect IDs**, four-option offering constraints, payment, pending discoveries, and a mutation bridge into the external `hsrl2` engine. It does not authorize full-game training or silently substitute this subset for the current seasonal pool.

## Source resolution

**Sphere of Memory (`BG36_MagicItem_372`)** appears in Blizzard's [Season 14 introduction](https://us.forums.blizzard.com/en/hearthstone/t/battlegrounds-season-14-trinket-updates/163710), and the [18 August balance patch](https://hearthstone.blizzard.com/en-gb/news/24293284/36-2-2-patch-notes) increases its end-of-turn copies from two to three. The pinned current client agrees: cost 2, three copies. The latest [maintained issues notice](https://us.forums.blizzard.com/en/hearthstone/t/364-highlights-known-issues/153567) does not list a Trinket ban. Consequently, this module offers an **explicit inference from official history** that Sphere remains current despite its gallery omission. Enable it with `resolve_sphere_from_official_history=True`; the original frozen manifest is not rewritten.

**Blessing Portrait (`BG32_MagicItem_894`) remains excluded**, because the [3 September patch](https://hearthstone.blizzard.com/en-us/news/24296231/3642-patch-notes) explicitly removes it. A stale gallery entry cannot override that removal.

Blizzard's [offering explanation](https://us.forums.blizzard.com/en/hearthstone/t/battlegrounds-developer-insights-updated-trinket-rules/158955) supplies these implemented constraints: four options on turns 6/9; established-type thresholds 2/3; neutral, affordable and most-common-type guarantees; a two-gold pivot discount; ownership exclusion; and several named restrictions. It also says generated Trinket cards wait for hand space. Exact individual eligibility flags and offering weights are not fully published.

## Working effects

| Trinket | Current IDs | Base cost | Recruitment transition |
|---|---|---:|---|
| Book of Medivh | `BG30_MagicItem_420`, `BG30_MagicItem_420t` | 1 / 2 | One/two explicit spell discoveries on acquisition and subsequent turn starts |
| Goldenizer Supply | `BG30_MagicItem_435` | 0 | Protected `BG26_813t` grant every third turn end |
| Comfy Coffin | `BG30_MagicItem_547`, `BG30_MagicItem_547t` | 1 / 4 | Permanent Undead attack from Tavern spell events |
| Bartend-o-Tron's Oilcan | `BG30_MagicItem_705` | 2 | Upgrade discount immediately and each subsequent turn start |
| Goblin Wallet | `BG30_MagicItem_847` | 1 | Raises future income cap at turn end |
| Nerglish Phrasebook | `BG30_MagicItem_914`, `BG30_MagicItem_914t` | 1 / 1 | Buffs the first minion remaining in hand after a play |
| Bob's Tip Jar | `BG30_MagicItem_996` | 0 | Immediate gold and income-cap increase |
| Enigmatic Headstone | `BG32_MagicItem_276` | 1 | Permanent Undead attack at turn end |
| Stuffed Coin Purse | `BG35_MagicItem_814` | 0 | One payout upon reaching Tier 6 |
| Mysterious Orb | `BG35_MagicItem_818` | 0 | Immediate gold and explicit next-offering stage override |
| Frigid Blossom | `BG36_MagicItem_300` | 0 | Upgrade discount after refresh |
| Wand of Divination | `BG36_MagicItem_307` | 1 | Targeted-spell counter and gold payout; counter persists between turns |
| Defensive Shell | `BG36_MagicItem_811` | 1 | Divine Shield for the first minion played that turn |
| Lorewalker Scroll | `BG30_MagicItem_422`, `BG30_MagicItem_422t` | 2 / 2 | Current 4/4 or 10/10 bonus to a spell's minion target |
| Sphere of Memory | `BG36_MagicItem_372` | 2 | Three protected copies of the last Tavern spell; history persists between turns |

The second batch adds these twenty current effect IDs:

| Trinket | Current IDs | Base cost | Recruitment transition |
|---|---|---:|---|
| Butcher's Sickle | `BG30_MagicItem_406` | 3 | Protected Butchering grant immediately and at subsequent starts |
| Wisdomball Supply | `BG31_MagicItem_903` | 2 | Protected Knockoff Wisdomball grant immediately and at subsequent starts |
| Nomi Sticker | `BG30_MagicItem_544`, `BG30_MagicItem_544t` | 0 / 4 | Current and future Tavern Elementals gain 3/2 or 5/5 per Elemental played |
| Great Boar Sticker | `BG30_MagicItem_988`, `BG30_MagicItem_988t` | 0 / 2 | Adds gem bonuses and grants three/five gems to hand or queue |
| Water Wheel | `BG35_MagicItem_851` | 1 | Spell generation on Elemental plays, twice per turn |
| Secret Schematic | `BG36_MagicItem_840` | 0 | Spell generation after Mech purchases, including free purchases |
| Cookie's Stirring Rod | `BG36_MagicItem_850` | 2 | Two spells every fifth Murloc played; progress persists between turns |
| Demonic Bloodletter | `BG36_MagicItem_800` | 3 | Permanent Tavern stat increases from Tavern spell casts |
| Dragonwing Glider | `BG30_MagicItem_900` | 1 | Random Dragon stat increase from explicit hand plays |
| Booty Bay Brew | `BG30_MagicItem_924`, `BG30_MagicItem_924t` | 3 / 0 | Two distinct Pirates receive current bonuses per Gold spent |
| Warcry Totem | `BG36_MagicItem_202` | 3 | Pure buy-price quotes and allowance consumption on successful Battlecry purchases |
| Glowing Crystal | `BG36_MagicItem_220` | 1 | Gold at subsequent starts using distinct minion-type representatives |
| Recycling Sticker | `BG32_MagicItem_888` | 5 | Free Refresh credit after Elemental plays |
| Worn Treasure Map | `BG32_MagicItem_428` | 0 | One ten-gold payout after two subsequent starts |
| Ornate Clock | `BG32_MagicItem_271` | 0 | Immediate gold and the normal Greater offering moved to next turn |
| Rockin' Music Box | `BG30_MagicItem_430` | 1 | Pool-accounted Battlecry minion generation on purchase and subsequent starts |
| Lava Lamp | `BG30_MagicItem_951` | 2 | Pool-accounted Elemental generation every seventh sale |

Costs and script parameters come from the pinned 251332 definitions. Generated tokens remain dependencies, not shop permissions. Buying an unsupported effect fails before payment. Duplicate event IDs and repeated turn-ending calls fail, preventing accidental repeated economy rewards.

## Connecting a recruit adapter

1. Build the authoritative database using `bg_ai.hsbrsim_data.build_current_database`; old `CardDB.load` misclassifies some Battleground spell definitions.
2. Construct `HSRLTrinketHost(game, hero, spell_ids=current_verified_spell_ids, minion_ids=explicit_generation_ids)`, then `TrinketController` with explicit `allowed_ids`. Keep the chosen subset in experimental provenance.
3. Feed `start_turn` after base income reset and before board start-of-turn effects. Install the observed or explicitly experimental offering.
4. Surface `buy_trinket` and every `pending_choice` as individual legal actions. The shared `TimedRecruitEngine` must charge each action; a discovery is never automatically selected or played. A pending choice blocks ordinary actions and turn advancement.
5. Deliver each play, spell, refresh, buy, sell, Gold-spend and tier-change event once. `quote_minion_buy` changes no state; after a successful engine purchase, `on_minion_bought` checks the paid price and consumes the allowance. The adapter must apply the quoted price to its own purchase path. Health-price replacement interactions explicitly fail.
6. `on_card_played` is separate from `on_spell_cast`: automatically cast spells do not trigger hand-play effects. Multiple hooks from one action use distinct event IDs, such as `action17:card_played` and `action17:minion_played`. They remain one timed player action. `buy_trinket` processes Gold-spend effects internally; engine echoes marked `source_kind="trinket_purchase"` do not apply those effects again.
7. Query `scheduled_offering()` when a turn starts; this honors Ornate Clock and Mysterious Orb without refilling either turn's action budget. Call `end_turn` at the Trinket phase, before minion end-of-turn effects. Call `host.flush_generated_cards()` when an action frees hand space.

The host mutates actual `hsrl2` gold, upgrade cost, income tags, gem bonuses, free Refresh credits, minion buffs, keywords, Tavern buff descriptors, shared minion pool, hand and queue. Free Refresh means no Gold charge; the Refresh still costs a timed action. Tavern buffs match authoritative dual types and do not double-apply to a frozen retained minion. Queued minions reserve a shared-pool copy immediately and check triples when hand space opens. Discovery resolution fires the engine's `card_discovered` event. The protected queue bridge deliberately rejects mixed-source queue cases that need the integrated adapter's common FIFO handling.

`offer_trinkets` requires an explicit candidate list and `OfferingAssumptions`, including a weight for every candidate. It uses bounded rejection sampling under declared constraints; it does not claim Blizzard's probability distribution. Neutral/menagerie and multi-affiliation metadata, hero-power-specific eligibility, and type-tie selection still need current-client trace validation. The default catalog leaves individual typed restrictions unknown rather than interpreting a discount tag as proven eligibility. Random minion generation uses the caller's explicit eligible pool, filters remaining copies and lobby types, and samples card identities uniformly. Those eligibility/weight assumptions require client validation; these tests establish shared-pool conservation, not Blizzard's hidden generation distribution. Glowing Crystal uses maximum one-minion-per-type matching so a lone dual-type minion or Amalgam does not create multiple representatives.

## Validation and remaining work

Fifty-five focused tests passed, including ten using the pinned external engine and freshly constructed current database. The tests exercise current prices, rejected unsupported purchases, pending choices, full-hand queuing, counters across turns, persistent buffs, duplicate-event rejection, and 100 seeded constrained-offering samples. The second batch additionally verifies actual zero/zero/three-Gold Warcry purchases, gem bonuses applied by the engine, frozen/new Tavern buff behavior, pool reservation while a minion waits for hand space, a queued third copy forming a triple without a second pool charge, and free Refresh credit consumption.

There are **190 remaining effect IDs** relative to the 228 existing candidates plus Sphere. Combat-event persistence, replacement/stacking interactions, mixed generation queues, individual offering metadata and weights, and real-client timeout behavior remain unvalidated. Combat spell events explicitly raise. These limitations keep `full_game_ready=False`; the module is usable implementation progress rather than a complete Season 14 environment.
