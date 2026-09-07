# Preserve large training archives in bounded Git blobs

Completed runs can store large archives as ordered **8 MiB binary chunks**.
The package preserves the exact original archive bytes, including gzip headers;
it does not decompress, recompress, inspect evaluations, or change model JSON.
Original archives are always kept locally.

**Byte preservation is not dataset validation.** An interrupted or corrupt gzip
can be preserved exactly as diagnostic evidence. Successful packing or unpacking
does not establish that an archive decompresses, contains the declared number of
episodes, or reproduces a model. Keep the run's separate integrity/recovery audit
alongside the package, and use only independently validated data for training.

After a run has produced `results.json`, run from the repository root:

```bash
python scripts/package_training_evidence.py pack runs/RUN_NAME
```

By default, the command scans only archive files directly inside that run directory:
`.gz`, `.zip`, `.tar`, `.xz`, `.bz2`, and `.zst`. Files larger than 8 MiB receive
parts in `evidence_parts/`. Archives of 8 MiB or less, empty archives, model
JSON and other files remain untouched. It writes
`training_evidence_manifest.json`, which records each original filename,
byte count and SHA-256, followed by each part's ordered path, byte count and
SHA-256. The last part can be smaller than 8 MiB.

To include a recovered archive in an explicitly named subdirectory:

```bash
python scripts/package_training_evidence.py pack runs/RUN_NAME --archive-subdirectory recovered_expert_data
```

Repeat `--archive-subdirectory` to select more directories. Each selected
directory is scanned directly; there is no implicit recursive scan. Paths must
be distinct, safe relative directories inside the completed run and cannot
contain symlinks. The completion check still uses the root run's `results.json`.
Original and part paths preserve the selected directory structure, including
when separate directories contain archives with the same basename. Unpacking
recreates those relative directories automatically; no extra option is needed.

Commit the parts and manifest along with the run's other artifacts. A per-run
`.gitignore` can list the packaged original archive names so Git stores one
copy of their bytes. The packaging tool does not edit ignore rules or delete
anything. Avoid ignoring small archives that have no manifest entry.

After cloning the repository, reconstruct and verify all packaged originals:

```bash
python scripts/package_training_evidence.py unpack runs/RUN_NAME
```

To restore into another directory:

```bash
python scripts/package_training_evidence.py unpack runs/RUN_NAME --output-directory /path/to/restored
```

Unpacking checks every part's size and checksum and every reconstructed
archive's full checksum before publishing any reconstructed archive. Existing
files are reused only when their full bytes match; mismatching files remain
untouched and produce an error. Restoration uses bounded reads and temporary
files, so it does not assemble all archives in memory. Publication creates
files without replacing an existing destination.

Repeated packaging is idempotent when the sources, manifest and existing parts
match. It can recreate a missing part from the retained original. It refuses
to rewrite an incompatible manifest or conflicting part. Sources that change
while packaging invalidate that attempt. A run still being written has no
completion marker and cannot be packaged.

Manifest validation rejects duplicate JSON keys, duplicate archive/part names,
wrong part ordering, inconsistent sizes, malformed hashes, traversal, absolute
paths, Windows device/stream aliases and symlinks. On platforms supporting
directory descriptors, reads and publication walk parents without following
symlinks. Keep the package directory under your control while operating on it.

The optional `--chunk-size` argument accepts smaller byte counts for integrity
fixtures and changes the packaging threshold accordingly; it never permits
chunks larger than 8 MiB. `tests/test_training_evidence_package.py` exercises
lossless gzip restoration, threshold boundaries, corruption, source mutation,
unmatched destinations, malformed manifests, symlinks and publication races
using temporary files. No real training holdout is needed for those tests.
