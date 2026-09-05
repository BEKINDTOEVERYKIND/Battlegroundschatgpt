#!/usr/bin/env python3
"""Restore all attempted checkpoints from verified repository-sized archive parts."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "runs/20260905-positioning-v2/candidate_archives.json")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    root = args.manifest.parent.resolve()
    out = (args.out or root).resolve()
    pieces = []
    for part in manifest["parts"]:
        path = (root / part["file"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Invalid archive part path")
        data = path.read_bytes()
        if len(data) != part["bytes"] or hashlib.sha256(data).hexdigest() != part["sha256"]:
            raise ValueError("Archive part checksum mismatch")
        pieces.append(data)
    data = b"".join(pieces)
    if hashlib.sha256(data).hexdigest() != manifest["sha256"]:
        raise ValueError("Combined archive checksum mismatch")
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for row in manifest["members"]:
            target = (out / row["member"]).resolve()
            if not target.is_relative_to(out):
                raise ValueError("Invalid member path")
            raw = archive.extractfile(row["member"]).read()
            if hashlib.sha256(raw).hexdigest() != row["sha256"]:
                raise ValueError("Checkpoint checksum mismatch")
            if target.exists() and target.read_bytes() != raw:
                raise ValueError(f"Refusing to overwrite a changed checkpoint: {target.name}")
            target.write_bytes(raw)
    print(json.dumps({"restored": len(manifest["members"]), "directory": str(out)}))
