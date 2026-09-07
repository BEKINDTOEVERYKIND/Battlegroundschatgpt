"""Future-run writer: publish complete, verified trajectory archives atomically.

This module is deliberately separate from the frozen/executing trainers. A
caller may begin fitting only after finish() returns its verification manifest.
Missing rows, incomplete gzip data and replacement of the live temporary path
are fatal. No code here repairs or modifies previously recorded archives.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile


class ArchiveIntegrityError(RuntimeError):
    pass


def _trajectory_seed(record):
    if (not isinstance(record, dict) or type(record.get("seed")) is not int
            or record["seed"] < 0 or record.get("complete") is not True):
        raise ArchiveIntegrityError("Only explicitly complete, seeded trajectories may be archived")
    combats = record.get("combats")
    if (not isinstance(combats, list) or len(combats) != 2
            or [c.get("receipt", {}).get("turn") for c in combats] != [1, 2]):
        raise ArchiveIntegrityError("Trajectory must contain both ordered combat receipts")
    return record["seed"]


def verify_trajectory_archive(path, expected_seeds):
    """Strictly decode every record/footer and verify the exact ordered seed plan."""
    expected_seeds = tuple(expected_seeds)
    if (any(type(seed) is not int or seed < 0 for seed in expected_seeds)
            or len(set(expected_seeds)) != len(expected_seeds)):
        raise ArchiveIntegrityError("Expected trajectory seeds must be unique nonnegative integers")
    decoded_digest, observed = hashlib.sha256(), []
    try:
        with gzip.open(path, "rb") as stream:
            for line in stream:
                if not line.endswith(b"\n"):
                    raise ArchiveIntegrityError("Archive contains an unterminated JSONL record")
                decoded_digest.update(line)
                record = json.loads(line)
                observed.append(_trajectory_seed(record))
    except (EOFError, gzip.BadGzipFile, UnicodeError, json.JSONDecodeError) as error:
        raise ArchiveIntegrityError(f"Strict trajectory archive decoding failed: {error}") from error
    if tuple(observed) != expected_seeds:
        raise ArchiveIntegrityError("Archived trajectory count/order/seeds differ from the completed generation plan")
    path = Path(path)
    compressed_digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            compressed_digest.update(block)
    return {"records": len(observed), "seeds": observed, "gzip_footer_verified": True,
            "archive_bytes": path.stat().st_size, "archive_sha256": compressed_digest.hexdigest(),
            "decoded_sha256": decoded_digest.hexdigest()}


class VerifiedTrajectoryWriter:
    """Write a private temporary stream; finish, fsync, verify, then publish.

    The destination must not already exist. Publication uses a same-directory
    hard link with no-overwrite semantics, then removes the temporary name.
    Replaced or corrupt temporary evidence is retained on failure for diagnosis.
    A successful finish is required explicitly; merely leaving the context does
    not silently publish an unverified archive.
    """

    def __init__(self, destination):
        self.destination = Path(destination)
        if self.destination.exists():
            raise FileExistsError("Trajectory evidence is immutable; destination already exists")
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=f".{self.destination.name}.", suffix=".pending",
                                    dir=self.destination.parent)
        self.temporary_path = Path(name)
        self._raw = os.fdopen(fd, "w+b")
        self._identity = (os.fstat(fd).st_dev, os.fstat(fd).st_ino)
        self._gzip = gzip.GzipFile(filename="", mode="wb", fileobj=self._raw, mtime=0)
        self._text = io.TextIOWrapper(self._gzip, encoding="utf8", newline="\n")
        self._seeds = []
        self._seed_set = set()
        self._decoded_digest = hashlib.sha256()
        self._finished = False

    def __enter__(self):
        return self

    def _check_temporary_identity(self):
        try:
            stat = self.temporary_path.stat()
        except FileNotFoundError as error:
            raise ArchiveIntegrityError("The live trajectory temporary path disappeared") from error
        if (stat.st_dev, stat.st_ino) != self._identity:
            raise ArchiveIntegrityError("The live trajectory temporary path was replaced while its writer was open")

    def write(self, record):
        if self._finished or self._text.closed:
            raise ArchiveIntegrityError("Cannot append after trajectory archive finalization")
        seed = _trajectory_seed(record)
        if seed in self._seed_set:
            raise ArchiveIntegrityError("Trajectory seed already occurred in this archive")
        self._check_temporary_identity()
        line = json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n"
        self._text.write(line)
        self._text.flush()
        self._decoded_digest.update(line.encode("utf8"))
        self._seeds.append(seed)
        self._seed_set.add(seed)

    def finish(self, expected_seeds):
        if self._finished:
            raise ArchiveIntegrityError("Trajectory archive has already been published")
        expected_seeds = tuple(expected_seeds)
        if tuple(self._seeds) != expected_seeds:
            raise ArchiveIntegrityError("Writer records disagree with the completed generation plan")
        self._check_temporary_identity()
        self._text.close()  # Gzip footer is written; the explicitly owned raw fd stays open.
        self._raw.flush()
        os.fsync(self._raw.fileno())
        self._check_temporary_identity()
        result = verify_trajectory_archive(self.temporary_path, expected_seeds)
        if result["decoded_sha256"] != self._decoded_digest.hexdigest():
            raise ArchiveIntegrityError("Archived record contents differ from the accepted trajectories")
        self._check_temporary_identity()
        os.link(self.temporary_path, self.destination)  # Atomically fail if another writer published first.
        try:
            stat = self.destination.stat()
            if (stat.st_dev, stat.st_ino) != self._identity:
                raise ArchiveIntegrityError("Published trajectory identity differs from its verified stream")
            # Reopen the published path: fitting is gated on the bytes readers
            # will actually see, rather than the writer's in-memory counts.
            confirmed = verify_trajectory_archive(self.destination, expected_seeds)
            if confirmed != result:
                raise ArchiveIntegrityError("Published trajectory bytes differ from their verified temporary stream")
            directory_fd = os.open(self.destination.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except Exception:
            # Retain both names as failed evidence; caller must not begin fit.
            raise
        self.temporary_path.unlink()
        self._finished = True
        self._raw.close()
        return dict(result, publication="verified-atomic-no-overwrite", path=str(self.destination))

    def __exit__(self, kind, value, traceback):
        try:
            if not self._text.closed:
                self._text.close()
        finally:
            if not self._raw.closed:
                self._raw.close()
        if kind is None and not self._finished:
            raise ArchiveIntegrityError("Archive context ended without successful explicit finish; fitting must not proceed")
        return False
