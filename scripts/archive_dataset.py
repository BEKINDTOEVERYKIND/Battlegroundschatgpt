#!/usr/bin/env python3
"""Persist large replay datasets as deterministic gzip, with both content hashes."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def archive(path: Path) -> dict:
    target = Path(str(path) + ".gz")
    with path.open("rb") as source, target.open("wb") as output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0, compresslevel=9) as encoded:
            shutil.copyfileobj(source, encoded)
    expected = digest(path)
    h = hashlib.sha256()
    with gzip.open(target, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    if h.hexdigest() != expected:
        raise ValueError("Compressed dataset failed the round-trip checksum")
    record = {"format": "gzip-jsonl", "compressed_file": target.name,
              "compressed_sha256": digest(target), "uncompressed_file": path.name,
              "uncompressed_sha256": expected, "uncompressed_bytes": path.stat().st_size,
              "compressed_bytes": target.stat().st_size}
    (path.parent / "dataset_archive.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    print(json.dumps(archive(args.input)))
