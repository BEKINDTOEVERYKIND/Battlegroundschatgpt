#!/usr/bin/env python3
"""Reproduce a finite, provenance-bound positioning training/evaluation run."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from bg_ai.provenance import file_sha256, load_snapshot, training_provenance
from bg_ai.positioning_versions import ADAPTER_VERSIONS, CURRENT_ADAPTER_VERSION, adapter_version_for_metadata
from archive_dataset import archive
from check_live_ruleset import check_live_ruleset


def validate_generation_metadata(generation: dict, data: Path, root: Path = ROOT,
                                 expected_adapter_version: str | None = None) -> str:
    """Bind reused labels to the exact cards and installed simulator versions."""
    version = adapter_version_for_metadata(generation)
    if expected_adapter_version is not None and version != expected_adapter_version:
        raise ValueError("Requested adapter differs from dataset generation")
    if generation.get("datasetSha256") != file_sha256(data):
        raise ValueError("Dataset content does not match its generation manifest")
    if generation.get("rulesetSha256") != file_sha256(root / "data/ruleset.json"):
        raise ValueError("Dataset belongs to a different ruleset")
    if generation.get("cardsSha256") != file_sha256(root / "data/reference_cards.json"):
        raise ValueError("Dataset labels use a different reference-card snapshot")
    for field, package in (("engine", "@firestone-hs/simulate-bgs-battle"),
                           ("referencePackage", "@firestone-hs/reference-data")):
        installed = json.loads((root / "node_modules" / package / "package.json").read_text())["version"]
        if generation.get(field) != installed:
            raise ValueError(f"Dataset {field} version {generation.get(field)!r} differs from installed {installed!r}")
    if generation.get("rejected", 0):
        raise ValueError("Investigate rejected simulation scenarios before training; see generation manifest")
    opener = gzip.open if str(data).endswith(".gz") else open
    with opener(data, "rt") as stream:
        for index, line in enumerate(stream):
            if line.strip() and adapter_version_for_metadata(json.loads(line).get("metadata", {})) != version:
                raise ValueError(f"Dataset row {index + 1} adapter differs from generation manifest")
    return version


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="A new run directory")
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--trials", type=int, default=128)
    parser.add_argument("--eval-trials", type=int, default=1024)
    parser.add_argument("--candidates", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--model-seed", type=int, default=17)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--only-generate", action="store_true")
    parser.add_argument("--split", choices=("auto", "train", "validation", "test"), default="auto")
    parser.add_argument("--allow-historical", action="store_true", help="Explicit offline reproduction; never reported as current")
    parser.add_argument("--objective", choices=("gap-weighted", "legacy-normalized"), default="gap-weighted")
    parser.add_argument("--tie-tolerance", type=float, default=0.02)
    parser.add_argument("--pure-model", action="store_true", help="Evaluate the network without simulation search")
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--data", type=Path, help="Use an existing, hash-checked generated dataset")
    parser.add_argument("--adapter-version", choices=ADAPTER_VERSIONS,
                        help="New generation defaults to corrected V2; reused data keeps its recorded version")
    args = parser.parse_args()
    if args.workers < 1 or args.count < 10:
        parser.error("Require at least one worker and ten scenarios")
    if not 5 <= args.candidates <= 24:
        parser.error("Require 5..24 candidates so all predeclared baseline orders can be included")
    os.chdir(ROOT)
    out = Path(args.out).resolve()
    if out.exists():
        parser.error("--out must be new; existing runs are never overwritten")
    load_snapshot(ROOT / "data/ruleset.json")
    live_preflight = check_live_ruleset(root=ROOT, allow_historical=args.allow_historical)
    out.mkdir(parents=True)
    (out / "live_preflight.json").write_text(json.dumps(live_preflight, indent=2) + "\n")
    started = time.time()
    env = dict(os.environ, PYTHONPATH=str(ROOT / "python"), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    commands = []

    def run(command: list[str], label: str):
        commands.append(command)
        print(label, flush=True)
        with (out / f"{label}.log").open("w") as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

    data = args.data.resolve() if args.data else out / "positions.jsonl"
    if not args.data:
        workers = min(args.workers, args.count)
        shard_paths = [out / f"positions-{i:02d}.jsonl" for i in range(workers)]
        def generate(i):
            start = args.count * i // workers
            stop = args.count * (i + 1) // workers
            run(["node", "scripts/generate_positions.mjs", "--count", str(stop-start),
                 "--start", str(start), "--trials", str(args.trials),
                 "--candidates", str(args.candidates), "--seed", str(args.seed),
                 "--split", args.split, "--adapter-version", args.adapter_version or CURRENT_ADAPTER_VERSION,
                 "--out", str(shard_paths[i])], f"generate-{i:02d}")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(generate, range(workers)))
        shards = [json.loads(Path(str(p) + ".meta.json").read_text()) for p in shard_paths]
        invariant = ("engine", "referencePackage", "cardsSha256", "rulesetSha256", "trials", "candidates", "seed", "adapterVersion", "datasetSchemaVersion")
        for shard, shard_path in zip(shards, shard_paths):
            if any(shard[k] != shards[0][k] for k in invariant):
                raise ValueError("Generation shards disagree on immutable settings")
            if shard["datasetSha256"] != file_sha256(shard_path):
                raise ValueError("Generation shard checksum mismatch")
        with data.open("wb") as stream:
            for shard_path in shard_paths:
                stream.write(shard_path.read_bytes())
        combined = dict(shards[0], start=0, workers=workers,
                        attempted=sum(s["attempted"] for s in shards),
                        written=sum(s["written"] for s in shards),
                        rejected=sum(s["rejected"] for s in shards),
                        combats=sum(s["combats"] for s in shards),
                        elapsedSeconds=time.time()-started,
                        failures=[f for s in shards for f in s["failures"]],
                        datasetSha256=file_sha256(data), shards=shards)
        Path(str(data) + ".meta.json").write_text(json.dumps(combined, indent=2) + "\n")
        first_coverage = Path(str(shard_paths[0]) + ".coverage.json")
        if first_coverage.exists():
            Path(str(data) + ".coverage.json").write_bytes(first_coverage.read_bytes())
        for shard_path in shard_paths:
            shard_path.unlink()  # Exact content retained, in order, in the combined dataset.
    generation = json.loads(Path(str(data) + ".meta.json").read_text())
    adapter_version = validate_generation_metadata(generation, data, expected_adapter_version=args.adapter_version)
    if args.only_generate:
        live_postflight = check_live_ruleset(root=ROOT, allow_historical=args.allow_historical)
        (out / "live_postflight.json").write_text(json.dumps(live_postflight, indent=2) + "\n")
        archive(data)
        (out / "generation_commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        print(json.dumps({"data": str(data), "scope": "label generation only", "scenarios": generation["written"]}))
        return 0
    checkpoint = out / "positioning.json"
    command = [sys.executable, "-m", "bg_ai.learning", "train", "--data", str(data),
               "--checkpoint", str(checkpoint), "--epochs", str(args.epochs),
               "--hidden", str(args.hidden), "--seed", str(args.model_seed),
               "--objective", args.objective, "--tie-tolerance", str(args.tie_tolerance),
               "--report", str(out / "training.json")]
    if args.warm_start:
        command += ["--warm-start", str(args.warm_start.resolve())]
    run(command, "train")
    provenance = training_provenance(ROOT / "data/ruleset.json", data, scope="positioning")
    provenance.update({"generation": generation, "adapter_version": adapter_version, "model_seed": args.model_seed,
                       "checkpoint_sha256": file_sha256(checkpoint),
                       "warm_start_sha256": file_sha256(args.warm_start) if args.warm_start else None})
    selections = out / "frozen_selections.jsonl"
    proposals = selections if args.pure_model else out / "neural_selections.jsonl"
    run([sys.executable, "scripts/select_positions.py", "--data", str(data),
         "--checkpoint", str(checkpoint), "--split", "test", "--out", str(proposals)], "freeze")
    if not args.pure_model:
        run(["node", "scripts/refine_positions.mjs", "--input", str(data),
             "--proposals", str(proposals), "--adapter-version", adapter_version, "--out", str(selections)], "search")
    fresh = out / "fresh_evaluation.jsonl"
    run(["node", "scripts/evaluate_positions.mjs", "--input", str(data),
         "--selections", str(selections), "--trials", str(args.eval_trials),
         "--seed", "982451653", "--adapter-version", adapter_version, "--out", str(fresh)], "evaluate")
    run([sys.executable, "scripts/report_fresh_evaluation.py", "--input", str(fresh),
         "--selections", str(selections), "--out", str(out / "evaluation.json")], "report")
    provenance.update({"elapsed_seconds": time.time() - started, "commands": commands,
                       "fresh_evaluation_sha256": file_sha256(fresh),
                       "evaluation_report_sha256": file_sha256(out / "evaluation.json")})
    provenance["dataset_archive"] = archive(data)
    provenance["live_preflight"] = live_preflight
    provenance["live_postflight"] = check_live_ruleset(root=ROOT, allow_historical=args.allow_historical)
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"run": str(out), "elapsed_seconds": provenance["elapsed_seconds"],
                      "scope": "positioning only", "report": str(out / "evaluation.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
