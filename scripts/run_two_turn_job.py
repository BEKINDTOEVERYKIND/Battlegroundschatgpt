#!/usr/bin/env python3
"""Run one declared finite improvement of the two-turn recruitment policy."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from bg_ai.provenance import file_sha256
from bg_ai.learning import Ranker
from bg_ai.two_turn_features import TWO_TURN_FEATURE_NAMES, TWO_TURN_FEATURE_VERSION, two_turn_schema_id
from bg_ai.turn_budget import TimingProfile

LIMITS = {"trajectories": (10, 200), "label_samples": (1, 16),
          "evaluation_episodes": (2, 200), "evaluation_samples": (1, 32),
          "seed": (1, 2**31 - 1), "hidden": (8, 64), "timeout_seconds": (60, 6600)}


def validate_config(config: dict, root: Path = ROOT) -> dict:
    fields = {*LIMITS, "experiment", "epochs", "policy_checkpoint", "policy_checkpoint_sha256", "continuation_policy"}
    if set(config) != fields:
        raise ValueError("Two-turn job configuration has missing or unknown fields")
    if not isinstance(config["experiment"], str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", config["experiment"]):
        raise ValueError("Invalid experiment identifier")
    if config["continuation_policy"] not in ("practical", "checkpoint"):
        raise ValueError("Continuation policy must be explicitly declared")
    for name, (minimum, maximum) in LIMITS.items():
        if type(config[name]) is not int or not minimum <= config[name] <= maximum:
            raise ValueError(f"{name} must be an integer in {minimum}..{maximum}")
    epochs = config["epochs"]
    if (not isinstance(epochs, list) or not 1 <= len(epochs) <= 4 or
            any(type(e) is not int or not 1 <= e <= 80 for e in epochs) or epochs != sorted(set(epochs))):
        raise ValueError("Epochs must be increasing distinct integers in 1..80")
    if not isinstance(config["policy_checkpoint"], str):
        raise ValueError("Policy checkpoint must be a repository path")
    checkpoint = (root / config["policy_checkpoint"]).resolve()
    if not checkpoint.is_relative_to(root.resolve()) or not checkpoint.is_file():
        raise ValueError("Policy checkpoint must be an existing repository file")
    if file_sha256(checkpoint) != config["policy_checkpoint_sha256"]:
        raise ValueError("Policy checkpoint differs from its declared SHA256")
    model = Ranker.load(checkpoint, TWO_TURN_FEATURE_NAMES)
    if (model.metadata.get("feature_version") != TWO_TURN_FEATURE_VERSION or
            model.metadata.get("two_turn_feature_schema_sha256") != two_turn_schema_id() or
            model.metadata.get("ruleset_file_sha256") != file_sha256(root / "data/ruleset.json") or
            model.metadata.get("timing_profile_sha256") != TimingProfile.load(root / "config/turn-budget.json").fingerprint):
        raise ValueError("Policy schema, ruleset or action timing differs from this job")
    if model.params["w1"].shape[1] != config["hidden"]:
        raise ValueError("Hidden width must match the warm-start policy")
    return dict(config)


def command(config: dict, out: Path, engine_root: Path) -> list[str]:
    return [sys.executable, "scripts/train_two_turn_recruit.py", "--engine-root", str(engine_root),
            "--out", str(out), "--policy-checkpoint", str(ROOT / config["policy_checkpoint"]),
            "--continuation-policy", config["continuation_policy"],
            "--trajectories", str(config["trajectories"]), "--label-samples", str(config["label_samples"]),
            "--evaluation-episodes", str(config["evaluation_episodes"]),
            "--evaluation-samples", str(config["evaluation_samples"]), "--seed", str(config["seed"]),
            "--hidden", str(config["hidden"]), "--epochs", *map(str, config["epochs"])]


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/two-turn-training-job.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--engine-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = validate_config(json.loads(args.config.read_text()))
    out = args.out.resolve()
    if out.exists():
        parser.error("Output must be a new directory; previous results are never overwritten")
    plan = command(config, out, args.engine_root.resolve())
    if args.dry_run:
        print(json.dumps({"config": config, "command": plan}, indent=2))
        return 0
    out.mkdir(parents=True)
    source_paths = ["scripts/train_two_turn_recruit.py", "scripts/run_two_turn_job.py",
                    "python/bg_ai/opening_transition.py", "python/bg_ai/two_turn_features.py",
                    "simulator/opening-transition-firestone.mjs"]
    record = {"config": config, "config_sha256": file_sha256(args.config),
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source_sha256": {p: file_sha256(ROOT / p) for p in source_paths},
              "github_run_id": os.environ.get("GITHUB_RUN_ID"), "command": plan,
              "status": "running", "policy_promoted": False}
    record_path = out / "runner.json"
    write_json(record_path, record)
    started = time.monotonic()
    try:
        subprocess.run(plan, cwd=ROOT, check=True, timeout=config["timeout_seconds"],
                       env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"))
        results = json.loads((out / "results.json").read_text())
        summary = {key: results[key] for key in (
            "scope", "training_trajectory_attempts", "accepted_training_trajectories", "decision_splits",
            "generation_failures", "generation_failure_reasons", "evaluation_episodes",
            "evaluation_attempts", "evaluation_failures", "scores", "paired_model_minus_practical", "timing_budget_violations",
            "model_action_counts", "current_snapshot_verified", "policy_promoted", "limitations")}
        print("TWO_TURN_RESULT " + json.dumps(summary, allow_nan=False), flush=True)
        if summary_path := os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(summary_path, "a", encoding="utf-8") as stream:
                stream.write("## Completed two-turn policy experiment\n\n```json\n" +
                             json.dumps(summary, indent=2, allow_nan=False) + "\n```\n")
        record.update(status="completed", results_sha256=file_sha256(out / "results.json"))
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic() - started
        write_json(record_path, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
