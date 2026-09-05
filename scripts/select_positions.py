#!/usr/bin/env python3
"""Freeze held-out model and heuristic choices before independent simulation."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from bg_ai.features import feature_schema_id
from bg_ai.learning import Ranker, jsonl_lines, load_dataset
from bg_ai.positioning_versions import adapter_version_for_metadata


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_selections(data_path, checkpoint_path, split="test"):
    dataset = load_dataset(data_path)
    model = Ranker.load(checkpoint_path, dataset.feature_names)
    wanted_splits = ("validation", "test") if split == "heldout" else (split,)
    raw = {row["scenario_id"]: row for row in
           (json.loads(line) for line in jsonl_lines(data_path) if line.strip())
           if row["split"] in wanted_splits}
    trained = set(model.metadata.get("training_scenario_ids", []))
    provenance = {"checkpoint_sha256": sha256(checkpoint_path),
                  "dataset_sha256": sha256(data_path),
                  "feature_schema_id": feature_schema_id(model.feature_names),
                  "selection_rule": "argmax frozen model; ties use first candidate",
                  "primary_baseline": "attack",
                  "all_baselines": ["random", "attack", "health", "taunt_last"],
                  "scope": "synthetic legal-pool combat positioning benchmark only"}
    selections = []
    for scenario in dataset.scenarios:
        if scenario.split not in (("validation", "test") if split == "heldout" else (split,)):
            continue
        if scenario.scenario_id in trained:
            raise ValueError("Cannot freeze a held-out selection for a training scenario")
        source = raw[scenario.scenario_id]
        candidates = source["candidates"]
        original = candidates[0]["board"]
        ids = [c["entityId"] for c in original]
        if len(set(ids)) != len(ids):
            raise ValueError("Minion entityId must be unique within a board")
        orders = {tuple(c["entityId"] for c in candidate["board"]): i
                  for i, candidate in enumerate(candidates)}
        baselines = {
            "attack": sorted(original, key=lambda c: -c["attack"]),
            "health": sorted(original, key=lambda c: -c["health"]),
            "taunt_last": sorted(original, key=lambda c: (bool(c.get("taunt", False)), -c["attack"])),
        }
        ranks = np.argsort(-model.predict(scenario.features), kind="stable").tolist()
        chosen = {"model": ranks[0], "random": 0}
        for name, board in baselines.items():
            key = tuple(c["entityId"] for c in board)
            if key not in orders:
                raise ValueError(f"{scenario.scenario_id}: required baseline {name} is absent from candidates")
            chosen[name] = orders[key]
        selections.append({"scenario_id": scenario.scenario_id, "split": scenario.split,
                           "selections": chosen,
                           "ranked_candidates": ranks,
                           "provenance": {**provenance,
                               "adapter_version": adapter_version_for_metadata(source["metadata"]),
                               "dataset_schema_version": source["metadata"].get("datasetSchemaVersion", 1),
                               "label_combat_seed": source["metadata"]["combatSeed"],
                               "cards_sha256": source["metadata"]["cardsSha256"],
                               "ruleset_sha256": source["metadata"]["rulesetSha256"]}})
    if not selections:
        raise ValueError(f"No {split} scenarios")
    return selections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", "--input", required=True, dest="data")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("validation", "test", "heldout"), default="test")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = freeze_selections(args.data, args.checkpoint, args.split)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise ValueError("Frozen selection output already exists; use a new filename")
    target.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
    print(json.dumps({"selected": len(rows), "split": args.split, "output": str(target),
                      "selections_sha256": sha256(target),
                      "checkpoint_sha256": rows[0]["provenance"]["checkpoint_sha256"]}))


if __name__ == "__main__":
    main()
