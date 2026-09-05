"""Offline ruleset integrity and training provenance, independent of card rotation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class SnapshotError(ValueError):
    """The requested snapshot cannot safely be used for the claimed scope."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def load_snapshot(path: str | Path, *, scope: str = "positioning") -> dict:
    """Validate an immutable source snapshot, without claiming engine completeness.

    Historical snapshots remain usable as explicitly historical snapshots. Live
    freshness must be re-established by the source sync command, not by changing
    a timestamp in a training run.
    """
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if scope not in {"positioning", "full_game"}:
        raise SnapshotError(f"Unknown training scope: {scope}")
    if manifest.get("mode") != "solo":
        raise SnapshotError("This trainer supports Solo snapshots only")
    if not manifest.get("ruleset_id") or not manifest.get("patch") or not manifest.get("build"):
        raise SnapshotError("Missing ruleset identity, patch, or client build")
    verified = manifest.get("status") == "verified"
    if not verified and not (scope == "positioning" and manifest.get("combat_pool_verified") is True):
        raise SnapshotError("Source snapshot is unverified: " + str(manifest.get("blockers", [])))
    active = manifest.get("active", {})
    if not active.get("minion_ids"):
        raise SnapshotError("No verified active minion pool")
    for pool, identifiers in active.items():
        if not isinstance(identifiers, list) or any(not isinstance(i, str) for i in identifiers):
            raise SnapshotError(f"{pool} must contain canonical string card IDs")
        if len(set(identifiers)) != len(identifiers):
            raise SnapshotError(f"Duplicate IDs in {pool}")
    checksums = manifest.get("checksums", {})
    if not checksums:
        raise SnapshotError("Snapshot has no file checksums")
    for relative, expected in checksums.items():
        target = (path.parent / relative).resolve()
        if not target.is_relative_to(path.parent):
            raise SnapshotError(f"Checksum path escapes snapshot directory: {relative}")
        if not target.is_file() or file_sha256(target) != expected:
            raise SnapshotError(f"Snapshot content changed or is missing: {relative}")
    if scope == "full_game":
        # A verified catalogue says nothing about correct game transitions.
        raise SnapshotError(
            "Full-game training is not enabled: the recruit engine must pass its "
            "current-ruleset conformance and coverage gate first"
        )
    return manifest


def training_provenance(snapshot_path: str | Path, data_path: str | Path, *, scope: str) -> dict:
    snapshot = load_snapshot(snapshot_path, scope=scope)
    return {
        "schema_version": 1,
        "scope": scope,
        "ruleset_id": snapshot["ruleset_id"],
        "ruleset_sha256": file_sha256(snapshot_path),
        "dataset_sha256": file_sha256(data_path),
        "patch": snapshot["patch"],
        "build": snapshot["build"],
        "source_verified_at": snapshot.get("verified_at"),
        "reference_checksums": snapshot["checksums"],
        "limitations": [
            "Positioning on the generated state distribution, not full-game self-play",
            "No human MMR or real-world win-rate claim",
        ],
    }
