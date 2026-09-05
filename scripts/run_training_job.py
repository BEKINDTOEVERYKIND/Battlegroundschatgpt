#!/usr/bin/env python3
"""Run one declared, finite corrected-pool scratch/warm-start experiment."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from bg_ai.learning import _paired_summary
from bg_ai.positioning_versions import CURRENT_ADAPTER_VERSION
from bg_ai.provenance import file_sha256


LIMITS = {"scenarios": (100, 12000), "candidates": (5, 16),
          "label_trials": (32, 256), "evaluation_trials": (128, 2048),
          "epochs": (1, 80), "hidden": (8, 64), "workers": (1, 8),
          "seed": (1, 2**31 - 1), "model_seed": (1, 2**31 - 1)}


def validate_config(config: dict, root: Path = ROOT) -> dict:
    expected = {*LIMITS, "experiment", "adapter_version", "warm_start", "warm_start_sha256"}
    if set(config) != expected:
        raise ValueError("Training configuration has missing or unknown fields")
    if not isinstance(config["experiment"], str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", config["experiment"]):
        raise ValueError("Invalid experiment identifier")
    if config["adapter_version"] != CURRENT_ADAPTER_VERSION:
        raise ValueError("This experiment requires corrected current adapter semantics")
    for name, (minimum, maximum) in LIMITS.items():
        if type(config[name]) is not int or not minimum <= config[name] <= maximum:
            raise ValueError(f"{name} must be an integer in {minimum}..{maximum}")
    checkpoint = (root / config["warm_start"]).resolve()
    if not checkpoint.is_relative_to(root.resolve()) or not checkpoint.is_file():
        raise ValueError("Warm start must be an existing repository checkpoint")
    if file_sha256(checkpoint) != config["warm_start_sha256"]:
        raise ValueError("Warm-start checkpoint hash differs from preregistration")
    return dict(config)


def commands(config: dict, out: Path) -> list[tuple[str, list[str]]]:
    base = [sys.executable, "scripts/train_positioning.py", "--adapter-version", config["adapter_version"],
            "--count", str(config["scenarios"]), "--candidates", str(config["candidates"]),
            "--trials", str(config["label_trials"]), "--eval-trials", str(config["evaluation_trials"]),
            "--epochs", str(config["epochs"]), "--hidden", str(config["hidden"]),
            "--workers", str(config["workers"]), "--seed", str(config["seed"]),
            "--model-seed", str(config["model_seed"])]
    data = str(out / "generation/positions.jsonl")
    return [("generate", base + ["--out", str(out / "generation"), "--only-generate"]),
            ("scratch", base + ["--out", str(out / "scratch"), "--data", data]),
            ("warm", base + ["--out", str(out / "warm"), "--data", data,
                             "--warm-start", str(ROOT / config["warm_start"])])]


def compare(out: Path) -> dict:
    arms = {}
    for name in ("scratch", "warm"):
        rows = [json.loads(line) for line in (out / name / "fresh_evaluation.jsonl").read_text().splitlines() if line]
        ids = [row["scenario_id"] for row in rows]
        if not rows or len(set(ids)) != len(ids):
            raise ValueError("Invalid evaluation scenario IDs")
        arms[name] = {row["scenario_id"]: row for row in rows}
    if set(arms["scratch"]) != set(arms["warm"]):
        raise ValueError("Scratch and warm evaluation scenarios differ")
    scores = {"scratch": [], "warm": []}
    for scenario in sorted(arms["scratch"]):
        left, right = (arms[name][scenario] for name in ("scratch", "warm"))
        if left["split"] != "test" or right["split"] != "test":
            raise ValueError("Comparison requires held-out test scenarios")
        for key in ("combatSeed", "trials", "adapterVersion", "cardsSha256", "rulesetSha256"):
            if left["metadata"].get(key) != right["metadata"].get(key):
                raise ValueError(f"Comparison has different {key}")
        for name in scores:
            scores[name].append(arms[name][scenario]["scores"]["model"]["score"])
    delta = np.asarray(scores["warm"]) - np.asarray(scores["scratch"])
    return {"scope": "corrected-pool positioning transfer; not a subsequent rotation or full-game test",
            "scenarios": len(delta), "mean_scores": {k: float(np.mean(v)) for k, v in scores.items()},
            "warm_minus_scratch": _paired_summary(delta, np.random.default_rng(202609057), 10000),
            "policies": "both use learned proposals plus identical finite simulation search",
            "interpretation": "No automatic model promotion; equal epoch counts do not establish faster adaptation."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/training-job.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = validate_config(json.loads(args.config.read_text()))
    out = args.out.resolve()
    if out.exists():
        parser.error("Output must be new; existing experiments are never overwritten")
    plan = commands(config, out)
    if args.dry_run:
        print(json.dumps({"config": config, "commands": plan}, indent=2))
        return 0
    out.mkdir(parents=True)
    record = {"config": config, "config_sha256": file_sha256(args.config),
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "github_run_id": os.environ.get("GITHUB_RUN_ID"), "commands": plan,
              "status": "running", "completed_stages": []}
    record_path = out / "job.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    started = time.monotonic()
    try:
        for name, command in plan:
            stage_start = time.monotonic()
            print(f"Starting {name}", flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
            record["completed_stages"].append({"stage": name, "seconds": time.monotonic() - stage_start})
            record_path.write_text(json.dumps(record, indent=2) + "\n")
        comparison = compare(out)
        (out / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
        record["status"] = "completed"
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic() - started
        record_path.write_text(json.dumps(record, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
