#!/usr/bin/env python3
"""Losslessly archive recruit traces with a shared provenance registry.

Large identical provenance objects are replaced with a content-addressed pointer.
Restoration must reproduce the exact original uncompressed JSONL SHA-256 before
any original file is considered redundant. The gzip stream uses mtime=0 and no
filename, so compression itself is deterministic.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def lines(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf8") as handle:
        yield from handle


def restored_lines(directory):
    directory = Path(directory)
    record = json.loads((directory / "scenario_archive.json").read_text())
    registry_file = directory / record["registry_file"]
    compact_file = directory / record["compact_file"]
    if digest(registry_file) != record["registry_sha256"]:
        raise ValueError("Recruit archive or provenance registry checksum mismatch")
    if compact_file.exists():
        if digest(compact_file) != record["compact_sha256"]:
            raise ValueError("Recruit archive or provenance registry checksum mismatch")
        source = gzip.open(compact_file, "rt", encoding="utf8")
    else:
        chunks = []
        for part in record.get("compact_parts", []):
            content = (directory / part["file"]).read_bytes()
            if len(content) != part["bytes"] or hashlib.sha256(content).hexdigest() != part["sha256"]:
                raise ValueError("Recruit archive part checksum mismatch")
            chunks.append(content)
        content = b"".join(chunks)
        if hashlib.sha256(content).hexdigest() != record["compact_sha256"]:
            raise ValueError("Recruit archive parts do not reconstruct the declared gzip")
        source = io.TextIOWrapper(gzip.GzipFile(fileobj=io.BytesIO(content)), encoding="utf8")
    registry = json.loads(registry_file.read_text())
    try:
        for line in source:
            row = json.loads(line)
            item = row.get("metadata", {}).get("engine_provenance")
            if isinstance(item, dict) and set(item) == {"$provenance_sha256"}:
                row["metadata"]["engine_provenance"] = registry[item["$provenance_sha256"]]
            yield json.dumps(row, separators=(",", ":")) + "\n"
    finally:
        source.close()


def read_rows(directory):
    directory = Path(directory)
    source = directory / "scenarios.jsonl.gz"
    if source.exists():
        sidecar = directory / "scenario_archive.json"
        expected = json.loads(sidecar.read_text())["original_uncompressed_sha256"] if sidecar.exists() else None
        actual = hashlib.sha256()
        for line in lines(source):
            actual.update(line.encode("utf8"))
            yield json.loads(line)
        if expected is not None and actual.hexdigest() != expected:
            raise ValueError("Original recruit trace checksum mismatch")
        return
    record = json.loads((directory / "scenario_archive.json").read_text())
    actual = hashlib.sha256()
    for line in restored_lines(directory):
        actual.update(line.encode("utf8"))
        yield json.loads(line)
    if actual.hexdigest() != record["original_uncompressed_sha256"]:
        raise ValueError("Restored recruit trace checksum mismatch")


def source_archive_hash(directory):
    directory = Path(directory)
    sidecar = directory / "scenario_archive.json"
    if sidecar.exists():
        # Both restored and original gzip containers are accepted only when
        # read_rows validates their exact declared uncompressed trace bytes.
        return json.loads(sidecar.read_text())["original_compressed_sha256"]
    return digest(directory / "scenarios.jsonl.gz")


def archive(directory):
    directory = Path(directory)
    original = directory / "scenarios.jsonl.gz"
    compact = directory / "scenarios.compact.jsonl.gz"
    registry_file = directory / "scenario_provenance.json"
    registry = {}
    preregistration = directory / "preregistration.json"
    if preregistration.exists():
        declared = json.loads(preregistration.read_text()).get("provenance")
        if isinstance(declared, dict):
            key = hashlib.sha256(json.dumps(declared, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            registry[key] = declared
    raw_digest = hashlib.sha256()
    raw_bytes = count = 0
    with compact.open("wb") as file:
        with gzip.GzipFile(filename="", mode="wb", fileobj=file, mtime=0, compresslevel=9) as zipped:
            for line in lines(original):
                encoded = line.encode("utf8")
                raw_digest.update(encoded)
                raw_bytes += len(encoded)
                row = json.loads(line)
                value = row.get("metadata", {}).get("engine_provenance")
                if isinstance(value, dict):
                    key = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                    if key in registry and registry[key] != value:
                        raise ValueError("Provenance content hash collision")
                    registry[key] = value
                    row["metadata"]["engine_provenance"] = {"$provenance_sha256": key}
                zipped.write((json.dumps(row, separators=(",", ":")) + "\n").encode("utf8"))
                count += 1
    registry_file.write_text(json.dumps(registry, indent=2) + "\n")
    record = {"schema_version": 1, "format": "provenance-deduplicated-jsonl-gzip",
        "rows": count, "original_compressed_file": original.name,
        "original_compressed_sha256": digest(original),
        "original_uncompressed_sha256": raw_digest.hexdigest(), "original_uncompressed_bytes": raw_bytes,
        "compact_file": compact.name, "compact_sha256": digest(compact), "compact_bytes": compact.stat().st_size,
        "registry_file": registry_file.name, "registry_sha256": digest(registry_file),
        "restoration": "Original JSONL bytes verified; gzip container header is regenerated deterministically"}
    if compact.stat().st_size > 6000000:
        record["compact_parts"] = []
        with compact.open("rb") as handle:
            index = 0
            while chunk := handle.read(4000000):
                part = directory / (compact.name + f".part-{index:03d}")
                part.write_bytes(chunk)
                record["compact_parts"].append({"file": part.name, "bytes": len(chunk), "sha256": digest(part)})
                index += 1
    (directory / "scenario_archive.json").write_text(json.dumps(record, indent=2) + "\n")
    restored = hashlib.sha256()
    for line in restored_lines(directory):
        restored.update(line.encode("utf8"))
    if restored.hexdigest() != raw_digest.hexdigest():
        raise ValueError("Compact recruit archive failed exact JSONL round-trip")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--restore", type=Path)
    args = parser.parse_args()
    if args.restore:
        expected = json.loads((args.directory / "scenario_archive.json").read_text())["original_uncompressed_sha256"]
        actual = hashlib.sha256()
        args.restore.parent.mkdir(parents=True, exist_ok=True)
        with args.restore.open("wb") as file:
            zipped = gzip.GzipFile(filename="", mode="wb", fileobj=file, mtime=0, compresslevel=9) if args.restore.suffix == ".gz" else file
            try:
                for line in restored_lines(args.directory):
                    encoded = line.encode("utf8")
                    actual.update(encoded)
                    zipped.write(encoded)
            finally:
                if zipped is not file:
                    zipped.close()
        if actual.hexdigest() != expected:
            raise ValueError("Restored output checksum mismatch")
        print(json.dumps({"restored": str(args.restore), "uncompressed_sha256": expected}))
    else:
        print(json.dumps(archive(args.directory)))

if __name__ == "__main__":
    main()
