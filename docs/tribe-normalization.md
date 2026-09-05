# Positioning tribe normalization and frozen benchmark coverage

The corrected positioning adapter uses `MECH`, matching the frozen client card
definitions and Firestone reference package 3.0.196 (`Race.MECH === 17`). That
package has no `Race.MECHANICAL`. The original adapter used `MECHANICAL` in lobby
names, while minions retained `MECH`. This excluded every single-tribe Mech from
the original sampler and passed an undefined tribe value into the combat context.
Dual Mechs could still enter a lobby through their other tribe.

The audit in [`positioning_coverage_audit.json`](../reports/positioning_coverage_audit.json)
counts each scenario's first candidate board and opponent once. Candidate
permutations do not inflate counts. A dual minion contributes to both race totals.

| Frozen dataset | Scenarios | Entity appearances | Distinct minions actually observed | Nominally eligible |
| --- | ---: | ---: | ---: | ---: |
| `20260905-positioning-v1` | 6,000 | 66,244 | 123 | 138 |
| `20260905-holdout-v2` | 1,000 | 11,064 | 123 | 138 |

Both datasets omit the same 15 eligible single-tribe Mechs. Their only observed
Mechs are Prosthetic Hand (473 training-dataset appearances, 113 holdout) and
Gearfin (389 and 75). The report lists every observed and missing card, raw lobby
labels, source-race counts, and compressed/uncompressed dataset hashes. It also
checks exact regeneration of the first ten scenarios of each dataset, excluding
stored labels and the newly explicit version fields. These files are unchanged;
their old scores describe the legacy adapter and sampled distribution.

## Explicit adapter versions

| API or metadata | Semantics |
| --- | --- |
| `FirestoneCombat` without an option | `legacy-v1`, dataset schema 1; original behavior for existing imports |
| `FirestoneCombatV2` | `firestone-combat-v2-tribes`, dataset schema 2 |
| `FirestoneCombat.fromFiles(cards, rules, {adapterVersion})` | Explicit choice; unknown versions are rejected |
| No `adapterVersion` in a frozen row | Legacy, never inferred to be corrected from its date or filename |

V2 normalizes `MECHANICAL` to `MECH` in reference definitions supplied to
`AllCardsService`, generated entities, supplied combat entities, and lobby
validation. A dual keeps both races and is legal if either appears in the lobby.
Alias duplicates do not create two distinct lobby tribes. Unknown tribe names
are rejected. The frozen source JSON/XML is not rewritten, and its original hash
remains the provenance identity.

`makeEntity(card, id, bonus, {adapterVersion})` follows the same version choice;
its default remains legacy for existing imports. `CURRENT_TRIBES` contains
canonical names. The original `TRIBES` export remains available for legacy
reproduction. `engine.versionMetadata()` supplies both `adapterVersion` and
`datasetSchemaVersion` for new rows and reports.

## Generation, learning, and evaluation

`scripts/generate_positions.mjs` defaults to V2. Explicit
`--adapter-version legacy-v1` reproduces the former distribution. Both new rows
and generation manifests record the adapter and dataset schema version.

`scripts/train_positioning.py` checks the manifest and every row before training.
An explicit requested version must match. Reused unversioned data stays legacy;
new generation defaults to V2. Shards must agree on both versions. The
generation-only path checks live source freshness again before archiving.

Frozen selections carry version provenance. Search, independent evaluation, and
the recommendation CLI select the adapter from each scenario's metadata and
reject contradictory version claims. An explicit V2 re-evaluation of legacy
boards is a separate audit: preserve original provenance and record the changed
semantics in a new result. Do not replace old labels or relabel old runs.

Shared model features already recognize both Mech spellings, so an old checkpoint
may intentionally warm-start a new V2 run. This is transfer learning across an
adapter correction; its old evaluation does not establish V2 performance. Use
fresh corrected labels and an independent corrected evaluation before drawing
that conclusion. Normalized tribe membership alone does not verify every newly
reachable effect or enable full-game self-play.
