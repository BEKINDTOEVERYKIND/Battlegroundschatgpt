#!/usr/bin/env python3
"""Fetch the pinned external recruit dependency without changing existing work."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from bg_ai.hsbrsim_data import verify_engine_checkout
from bg_ai.recruiting import HSBRSIM_REVISION, HSBRSIM_URL


def prepare(destination: Path) -> Path:
    destination = destination.resolve()
    if destination.exists():
        # Never reset, clean, or overwrite a user's checkout.
        return verify_engine_checkout(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "--quiet", str(destination)], check=True)
    subprocess.run(["git", "-C", str(destination), "remote", "add", "origin", HSBRSIM_URL], check=True)
    subprocess.run(["git", "-C", str(destination), "fetch", "--quiet", "--depth=1", "origin", HSBRSIM_REVISION], check=True)
    subprocess.run(["git", "-C", str(destination), "checkout", "--quiet", "--detach", "FETCH_HEAD"], check=True)
    return verify_engine_checkout(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-root", type=Path, default=ROOT.parent / "research/HSBRSIM")
    args = parser.parse_args()
    try:
        root = prepare(args.engine_root)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"Recruit dependency preparation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"engine_root": str(root), "revision": HSBRSIM_REVISION,
                      "checkout_clean": True, "full_game_ready": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
