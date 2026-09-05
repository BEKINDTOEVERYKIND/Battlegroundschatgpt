# Current recruit database bridge

The data bridge now constructs a **fresh `hsrl2.CardDB` from the frozen current
client definitions**. It never loads HSBRSIM's historical `data/bg_cards.json`
and never relies on its duplicate-registration merge, which retains old numeric
values. The external checkout is read only; no upstream source is vendored.

The checked snapshot is Solo patch **36.4.2**, build **251332**. The reference JSON
and compressed CardDefs XML must match the checksums in `data/ruleset.json`.
Their 5,613 entity identities, complete numeric tags, English names and text are
then cross-checked. Contradictory flattened statistics, a missing required field,
a broken current golden linkage, or an unresolved reachable structural reference
fails the export. Changing a JSON file and updating its checksum cannot bypass
the separate XML consistency check.

| Definition/pool | Verified count | How the bridge treats it |
|---|---:|---|
| Current minions | 246 | Validate base attack, health, tier and reciprocal golden links |
| Ordinary shop minions | 234 | Exactly `active.shop_minion_ids` becomes `is_pool_minion` |
| Special tier-7 minions | 12 | Keep definitions available; exclude ordinary shop and refresh |
| Current heroes | 116 | Current health/armor and exact linked power IDs |
| Linked hero powers | 116 | Must exactly equal the manifest's active power set |
| Tavern spells | 71 | Exactly `active.tavern_spell_ids` becomes `is_pool_spell` |
| Dark Gifts | 43 | Exactly the active gift IDs receive the `dark_gift` flag |
| Candidate Trinkets | 228 | Preserve reference definitions; zero active Trinkets |
| Reachable structural definitions | 1,099 | All explicit reachable card references resolve |

Reference, token, buddy, enchantment, golden and retired definitions do not gain
shop permission by being present. A reference to a buddy is not permission for a
general buddy mode. The dependency graph records golden/base, hero/power,
Spellcraft, evolution, guide-related and companion links. It is **structural
coverage**, not proof that every dynamically generated card in every script is
legal or that its effect is implemented. Generation legality and script-level
closure remain separate full-game gates.

## Numeric and keyword behavior

All six client script-number slots are retained, including explicit zeroes.
HSBRSIM's four native slots receive the first four; slots five and six are
available in `CardDef.raw`. Goldrinn is now tier 5 and 7/7, Private Investigator
is 5/6, and all hero armor values come from this snapshot. An absent client armor
or cost tag becomes zero, with the absence recorded in the exported metadata.
A minion's client mana cost is distinct from the engine's ordinary 3-gold buy
operation. The bridge does not change that buy rule.

Keyword flags use tags on the card itself. Referenced keywords are preserved
separately and cannot confer a keyword on their source: Fortify grants Taunt but
does not have Taunt. Avenge thresholds use the explicit threshold in the current
text, resolving a script-number placeholder when present. In particular,
Onyxia's Avenge threshold is 4; her first script parameter is the Whelp's 1/1
stats and must not become the threshold. The precise text “Also damages adjacent
minions.” supplies the few cleave definitions with no dedicated client tag;
other adjacent-damage effects are not treated as cleave.

Two compatibility issues are explicit:

- The client represents Tavern spells as type 42. Pinned `hsrl2.CardType` lacks
  that enum and would otherwise construct an `INVALID` definition. The bridge
  projects 42 to the engine's generic spell type 5; `source_card_type` and the
  complete numeric source tags preserve the original type.
- The engine has a single-race field. Ten current minions have two tribes. Their
  complete tribe lists remain in `CardDef.raw['races']`, but this does not repair
  downstream tribe checks. Full-game readiness remains false; restricted
  fixtures must avoid these cards until the engine handles both tribes.

Aureate Laureate is deliberately still an ordinary shop entry. Its own premium
flag describes its always-Golden effect; it is not a separate triple-golden
catalog entry. Ordinary golden definitions are recognized through their base
link rather than by indiscriminately excluding every premium flag.

## Use

Data-only export requires no external engine:

```bash
python scripts/export_recruit_data.py \
  --report-out reports/current_recruit_data.json \
  --definitions-out /tmp/current-hsrl2-cards.json
```

Construct and check the real external database:

```bash
python scripts/export_recruit_data.py \
  --engine-root ../research/HSBRSIM \
  --report-out reports/current_recruit_data.json
```

Before any external import, `verify_engine_checkout` requires clean Git state
and exact revision `4e0525a198352557ef816bd3a53566d6ffd4c164`. It also rejects an
already imported `hsrl2` from another location. Both the definitions and the
external tree remain untouched. Export success is exit code 0; the resulting
report still says `full_game_ready: false` and lists the remaining limitations.

The Python entry point is:

```python
from pathlib import Path
from bg_ai.hsbrsim_data import build_current_database

current = build_current_database(Path('../research/HSBRSIM'))
db = current.db
provenance = current.provenance
```

For explicitly restricted engine experiments, `fixture_minion_ids` and
`fixture_spell_ids` may narrow the already verified current pools. Inactive or
duplicate IDs are rejected. The result records the exact fixture pools and
`fixture_only: true`; it never claims to be a complete current-game environment.
A fixture must still validate handlers, targets and generated-card transitions.

## Executed verification

```text
PYTHONPATH=python python -m unittest discover -s tests -p test_hsbrsim_data.py -v
Ran 12 tests in 7.903s
OK
```

The external integration test ran in a child process and checked all 5,613 real
`CardDef` objects against their source attack, health, tier, cost, armor and all
four native script-number slots. It checked exact shop/spell/gift pools and a
restricted fixture, and rejected an inactive fixture ID. Other tests cover
rehash-resistant contradictory data, missing source fields, forbidden candidate
promotion, referenced-keyword confusion, Avenge threshold confusion, and
wrong/dirty external revisions before import. CI can run the data checks without
an external checkout; set `HSBRSIM_ROOT` to enable the separate external test.
