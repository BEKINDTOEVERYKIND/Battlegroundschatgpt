#!/usr/bin/env python3
"""Predeclared train/validation-only sweep for the gap-weighted positioning loss.

The existing final-test split is discarded before its candidates/features/scores
are processed. This script does not open any fresh evaluation files. A separate
new test distribution/RNG run must assess the selected checkpoint afterward.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time

# Small per-scenario matrix multiplies should not create nested BLAS thread pools.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from bg_ai.features import feature_schema_id
from bg_ai.learning import (Ranker, TRAINING_OBJECTIVE, LEGACY_TRAINING_OBJECTIVE,
                            _paired_summary, jsonl_lines, load_dataset)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validation_baselines(data_path):
    """Extract only fixed heuristic indices on validation scenarios."""
    result = {}
    for line in jsonl_lines(data_path):
        row = json.loads(line)
        if row["split"] != "validation":
            continue
        original = row["candidates"][0]["board"]
        orders = {tuple(c["entityId"] for c in candidate["board"]): i
                  for i, candidate in enumerate(row["candidates"])}
        chosen = {"random": 0}
        for name, board in {
            "attack": sorted(original, key=lambda c: -c["attack"]),
            "health": sorted(original, key=lambda c: -c["health"]),
            "taunt_last": sorted(original, key=lambda c: (bool(c.get("taunt", False)), -c["attack"])),
        }.items():
            chosen[name] = orders[tuple(c["entityId"] for c in board)]
        result[row["scenario_id"]] = chosen
    return result


def validation_report(model, scenarios, baselines, seed):
    names = ("model", "random", "attack", "health", "taunt_last")
    scores = {name: [] for name in names}
    selected = []
    for scenario in scenarios:
        index = int(np.argmax(model.predict(scenario.features)))
        selected.append({"scenario_id": scenario.scenario_id, "candidate_index": index})
        scores["model"].append(scenario.scores[index])
        for name in names[1:]:
            scores[name].append(scenario.scores[baselines[scenario.scenario_id][name]])
    arrays = {name: np.asarray(values) for name, values in scores.items()}
    rng = np.random.default_rng(seed)
    return {"split": "validation", "scenario_count": len(scenarios),
            "mean_scores": {name: float(values.mean()) for name, values in arrays.items()},
            "paired_model_minus_baseline": {
                name: _paired_summary(arrays["model"] - arrays[name], rng, 2000) for name in names[1:]},
            "selected_candidates": selected,
            "limitations": "Validation selects among candidates; its score/CI is not a final held-out strength claim"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reference", help="Optional legacy checkpoint, validation-only reference")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 4:
        parser.error("--workers must be 1..4")
    output = Path(args.out)
    if output.exists():
        parser.error("--out must be a new directory; all candidate artifacts are retained")
    output.mkdir(parents=True)
    started = time.time()
    dataset_hash = sha256(args.data)
    archive_path = Path(args.data).parent / "dataset_archive.json"
    if str(args.data).endswith(".gz") and archive_path.exists():
        archive = json.loads(archive_path.read_text())
        if archive.get("compressed_file") != Path(args.data).name or archive.get("compressed_sha256") != dataset_hash:
            raise ValueError("Compressed dataset does not match its frozen archive manifest")
    plan = {"objective": TRAINING_OBJECTIVE,
            "grid": {"hidden": [16, 32], "tie_tolerance": [0.0, 0.02], "epochs": [5, 10, 20, 40]},
            "seed": args.seed, "learning_rate": 0.003, "batch_size": 16, "l2": 0.0001,
            "dataset": str(Path(args.data).resolve()), "dataset_sha256": dataset_hash,
            "selection_rule": "Maximum validation mean combat score; exact ties prefer smaller hidden layer, then earlier epoch, then smaller tie tolerance",
            "allowed_splits": ["train", "validation"], "test_used": False,
            "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in
                              (ROOT / "python/bg_ai/learning.py", ROOT / "python/bg_ai/features.py", Path(__file__).resolve())},
            "legacy_objective": LEGACY_TRAINING_OBJECTIVE,
            "legacy_note": "Original v1 checkpoint predates objective metadata and used within-scenario normalized absolute gaps; original artifacts remain unchanged"}
    (output / "predeclared_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"event": "load_train_validation_only", "dataset_sha256": dataset_hash}), flush=True)
    dataset = load_dataset(args.data, allowed_splits=("train", "validation"))
    validation = dataset.split("validation")
    if not dataset.split("train") or not validation or dataset.split("test"):
        raise ValueError("Require train and validation only")
    baselines = validation_baselines(args.data)
    plan.update({"training_scenarios": len(dataset.split("train")), "validation_scenarios": len(validation),
                 "feature_count": len(dataset.feature_names), "feature_schema_id": feature_schema_id(dataset.feature_names)})
    (output / "dataset_provenance.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"event": "loaded", "train": plan["training_scenarios"],
                      "validation": len(validation), "seconds": time.time() - started}), flush=True)
    if args.reference:
        reference = Ranker.load(args.reference, dataset.feature_names)
        report = validation_report(reference, validation, baselines, args.seed)
        report.update({"checkpoint_sha256": sha256(args.reference),
                       "training_objective": reference.metadata["training_objective"]})
        (output / "legacy_reference_validation.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"event": "legacy_validation", "mean_scores": report["mean_scores"]}), flush=True)

    lock = threading.Lock()
    log = (output / "epoch_progress.jsonl").open("w")
    def emit(row):
        with lock:
            encoded = json.dumps(row, allow_nan=False)
            log.write(encoded + "\n")
            log.flush()
            print(encoded, flush=True)

    def train_trajectory(settings):
        hidden, tolerance = settings
        trajectory = f"hidden{hidden}-tie{tolerance:g}"
        model = Ranker(dataset.feature_names, hidden=hidden, seed=args.seed)
        model.metadata["candidate_sweep"] = {"dataset_sha256": dataset_hash,
                                              "selection_split": "validation", "test_used": False}
        history, candidates = [], []
        for epoch in range(1, 41):
            epoch_started = time.time()
            loss = model.fit(dataset, epochs=1, learning_rate=0.003, batch_size=16,
                             l2=0.0001, tie_tolerance=tolerance, objective=TRAINING_OBJECTIVE)[0]
            history.append(loss)
            emit({"event": "epoch", "trajectory": trajectory, "epoch": epoch,
                  "gap_weighted_pair_loss": loss, "seconds": time.time() - epoch_started,
                  "elapsed_seconds": time.time() - started})
            if epoch not in (5, 10, 20, 40):
                continue
            name = f"{trajectory}-epoch{epoch:02d}"
            checkpoint = output / f"{name}.json"
            model.save(checkpoint)
            report = validation_report(model, validation, baselines, args.seed)
            record = {"candidate": name, "hidden": hidden, "tie_tolerance": tolerance,
                      "epochs": epoch, "seed": args.seed, "objective": TRAINING_OBJECTIVE,
                      "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256(checkpoint),
                      "validation_mean_score": report["mean_scores"]["model"],
                      "validation_mean_scores": report["mean_scores"],
                      "paired_model_minus_baseline": report["paired_model_minus_baseline"],
                      "loss_history": list(history), "test_used": False}
            (output / f"{name}.validation.json").write_text(json.dumps(report, indent=2) + "\n")
            (output / f"{name}.metadata.json").write_text(json.dumps(record, indent=2) + "\n")
            candidates.append(record)
            emit({"event": "candidate", "candidate": name, "validation_mean_score": record["validation_mean_score"],
                  "delta_attack": report["paired_model_minus_baseline"]["attack"]["mean"],
                  "delta_taunt_last": report["paired_model_minus_baseline"]["taunt_last"]["mean"]})
        return candidates

    settings = [(hidden, tolerance) for hidden in (16, 32) for tolerance in (0.0, 0.02)]
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            records = [record for group in pool.map(train_trajectory, settings) for record in group]
    finally:
        log.close()
    records.sort(key=lambda row: (-row["validation_mean_score"], row["hidden"], row["epochs"], row["tie_tolerance"]))
    winner = records[0]
    selected = output / "selected_positioning.json"
    shutil.copyfile(winner["checkpoint"], selected)
    summary = {"selection_rule": plan["selection_rule"], "selected": winner,
               "selected_checkpoint": str(selected.resolve()), "selected_checkpoint_sha256": sha256(selected),
               "candidate_count": len(records), "candidates_ranked_by_validation": records,
               "dataset_sha256": dataset_hash, "test_used": False,
               "elapsed_seconds": time.time() - started,
               "next_gate": "Freeze this checkpoint; evaluate on new independent test scenarios/RNG without selecting another model on their results"}
    (output / "selection.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"event": "selected", "candidate": winner["candidate"],
                      "validation_mean_score": winner["validation_mean_score"],
                      "checkpoint": str(selected), "checkpoint_sha256": summary["selected_checkpoint_sha256"],
                      "elapsed_seconds": summary["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
