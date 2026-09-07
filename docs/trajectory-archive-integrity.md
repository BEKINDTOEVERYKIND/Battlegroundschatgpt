# Trajectory archive integrity

The first 512-episode BC run's original expert archive is incomplete: strict gzip
decoding fails, and independent streaming recovery finds only 493 complete
records, covering 3,851 of the 3,999 recorded training decisions. The final 19
trajectories are absent, not merely the gzip footer. The checksum in the original
manifest matches these incomplete bytes, so a checksum alone did not establish
data completeness. The original is preserved as diagnostic evidence. Exact
counts, missing seeds, coverage and hashes are in
`reports/bc-expert-archive-integrity.json`.

The archived executing trainer correctly exits its gzip-writing context before
creating the manifest. Ordinary per-record `flush()` calls do not explain the
missing final records. File modification/change times are consistent with a
live-path replacement or stale snapshot before the writer finished, but the
cause remains unproven. A footer-only repair would fabricate completeness and
must not be used. Reconstruction requires exact agreement with all recoverable
records and independent reproduction of the trained checkpoints.

## Writer for future runs

The separate `bg_ai.verified_trajectory_archive` module is now integrated into
future executions of the BC trainer. Integration waited until deterministic
recovery reproduced all four original checkpoint byte hashes. The original
archives/execution snapshots remain unchanged, and no executing trainer was
modified during the recovery or gradient run.

```python
from bg_ai.verified_trajectory_archive import VerifiedTrajectoryWriter

accepted_seeds = []
with VerifiedTrajectoryWriter(output / "expert_trajectories.jsonl.gz") as writer:
    for trajectory in complete_trajectories:
        writer.write(trajectory)
        accepted_seeds.append(trajectory["seed"])
    manifest = writer.finish(accepted_seeds)
# Only after finish succeeds may fitting begin.
```

The writer uses a private, exclusive temporary name and tracks its open-file
identity. Replacing that path while the writer is active raises an error. Each
trajectory must be explicitly complete and include both ordered combat
receipts. Finalization closes gzip, flushes and fsyncs the underlying file, then
strictly rereads every JSONL record and the gzip footer. Exact ordered seeds and
record counts must match the completed generation plan.

Only verified bytes are published, through a same-directory atomic hard link
that cannot overwrite existing evidence. The final path is then independently
reopened and verified before the method returns. Directory metadata is synced.
A partial or replaced temporary archive remains available for diagnosis; leaving
the context without successful explicit finalization fails instead of silently
allowing model fitting. These checks detect incomplete or substituted archives;
they do not claim to diagnose the unresolved original filesystem event.

Focused tests cover complete publication, replacement of an open temporary path,
a valid gzip file missing an expected trajectory, a missing footer after a
complete JSONL line, and immutable destinations/unfinalized contexts.
