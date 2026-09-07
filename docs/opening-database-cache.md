# Opening database setup performance

The September 7 behavior-cloning run repeatedly reconstructs the current card
database when it encounters a new five-tribe combination. Its 512 training
seeds contain 227 distinct combinations. `build_opening_database` calls the
full verified export once to check the Tier 1 pool, then
`build_current_database` calls it again before constructing 5,613 definitions.
That repeats XML parsing, source validation and definition export twice per
new combination. This finding does not involve held-out policy outcomes.

The standalone audit in `scripts/audit_opening_database_cache.py` measured:

| Operation | Mean time |
| --- | --- |
| Original database build for an unseen tribe combination | 2.646 seconds |
| Build from a cached verified export, with fresh definitions | 0.315 seconds |
| Speedup of the repeated build, excluding initial export | 8.4 times |

The benchmark overlapped a live training process; these are measured local
timings, not a hardware-independent guarantee. Three performance seeds
(`710041`, `710042`, `710043`) produced exactly equal complete two-turn
trajectories in the original and cached factories. Equality includes setup
provenance, every visible observation and command, learner features, budget
traces and actual Firestone combat receipts. Repeated runs also matched.
`reports/opening-database-cache-audit.json` preserves their hashes and timings.

`bg_ai.policy_factory_cached.CachedPolicyFactory` packages this optimization
for future runs. It caches the fully validated export as immutable JSON bytes,
then decodes fresh nested dictionaries and fresh `CardDef` objects for every
new view. Source checksums are taken from the verified export's provenance;
both fresh and reused views recheck the source files and the clean pinned
external checkout. It preserves original definition insertion order, scoped
pool flags and provenance. Each game still gets independent RNG, pool counts,
entities, counters and callbacks.

The new factory keeps at most 16 scoped databases by default, evicting the
least recently used view. A running game retains its own database reference.
Rebuilding an evicted view does not change its seed or insertion order. This
bounds retained database memory without sharing mutable card dictionaries.
The original behavior-cloning factory and its executing experiment remain
unchanged. Production conformance tests include replay after eviction at a
one-view capacity.

Sharing the upstream `CardDef` objects directly would require additional
protection. Although their dataclass fields are frozen, `raw` is an ordinary
mutable dictionary, and upstream duplicate registration deliberately mutates
existing definitions through `object.__setattr__`. No audited opening gameplay
path writes those definitions, but this optimization avoids relying on that
assumption. It also retains correctly scoped flags on each definition:
overriding only the pool-list methods would be insufficient because pool
release code consults `db.get(card_id).is_pool_minion` and `is_pool_spell`.

A separate single-seed archive benchmark found no comparable compression
bottleneck: a warm expert episode took 0.117 seconds to play and 0.031 seconds
to encode and gzip; practical evaluation took 0.065 plus 0.015 seconds; an
untrained 64-hidden-unit actor took 0.108 plus 0.016 seconds. That benchmark
measures execution time only. It does not select or assess a trained policy.
The archive format and compression level are unchanged.
