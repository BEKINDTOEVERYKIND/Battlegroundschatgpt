# Two-turn visible enchantment lifetimes

The v1 recruitment vector used current Attack and Health but omitted the
lifetimes of buffs already visible on owned cards and in the Tavern. For
example, Mini-Trident's +2 Attack expires at the next recruit phase. Identical
current stats built from permanent buffs have a different second-turn value.

`bg_ai.two_turn_features_v2` adds a separate 1,311-input schema. Its first 1,129
inputs are exactly the frozen `two_turn_features` v1 vector. The 182 appended
inputs describe each ordered board slot (7), hand slot (10), Tavern slot (7),
and the candidate action's source and target. Each view has these fields:

| Field | Meaning |
| --- | --- |
| Present | The visible slot contains a card. |
| Temporary Attack / Health | Sum of explicitly recorded temporary stat enchantments, divided by 20. |
| Permanent Attack / Health | Sum of explicitly recorded permanent stat enchantments, divided by 20. |
| Expires next recruit | At least one audited enchantment expires at the next recruit phase. |
| Discard at recruit end | An uncast generated Spellcraft spell leaves hand at recruit end. |

Permanent enchantment totals are not total permanent stats: intrinsic stats,
auras and copied Magnetic base stats retain their existing representation.
Current stats remain in the exact old prefix. Card and source IDs are used to
locate visible entities and validate audited lifetime semantics; IDs, names,
random seeds, future shops and combat receipts do not become model inputs.

## Integration and scope

```python
from bg_ai.two_turn_features_v2 import (
    enrich_two_turn_observation, encode_two_turn_actions,
    warm_start_two_turn_ranker,
)

visible = enrich_two_turn_observation(timed_observation)
features = encode_two_turn_actions(visible)
migrated = warm_start_two_turn_ranker(frozen_v1_ranker)
```

Enrichment reads only the existing visible `enchantments` and
`temporary_spellcraft` card records. It returns a copy, preserves timing and the
legal action mask, and adds versioned lifetime metadata. It requires the
audited two-turn opening scope. Encoding without metadata, with missing raw
records, with stale metadata or with an unknown temporary source fails closed.
Empty zones and zero enchantments must be explicit.

Mini-Trident is the only supported temporary recruit enchantment source in this
opening pool. Its callback expires the exact stat buff at the next recruit
phase, including on a retained frozen Tavern target. Supported permanent sources
are the opening cards and the audited primitive that leaves `source_id` empty.
No supported opening recruit effect grants a temporary keyword. Such grants
need an audited exporter and a new schema before training; this version does
not infer their lifetimes from a minion's current keyword flags.

The helper does not expand engine coverage. Ordinary triples, temporary
Spellcraft Magnetic attachment, later tiers and later seasonal turns retain
their existing explicit frontiers.

## Transfer and validation

Migration accepts only the exact frozen 1,129-feature two-turn v1 Ranker. It
deep-copies shared weights, normalization, Adam moments, optimizer step and RNG
state. Appended input weights and moments start at zero, their means at zero
and their scales at one, preserving predictions before additional training.
Existing checkpoints and feature modules remain unchanged.

Tests cast real Mini-Trident plus Tavern Dish Banana on one current minion,
and real Alliance Flag plus Blood Gem on its counterpart. The resulting current
stats match, while the first has +2 temporary Attack. In a shared observation
context, the old vectors alias those actual card states and the new vectors
distinguish them. This isolates a representation regression; it does not claim
the complete trajectories, including their leftover hands, are equal. Tests
also exercise real replay forks, immutable parent state, time/legal-mask
preservation, missing data rejection, hidden-field exclusion and checkpoint
migration/serialization. These checks establish representation correctness,
not playing-strength improvement.
