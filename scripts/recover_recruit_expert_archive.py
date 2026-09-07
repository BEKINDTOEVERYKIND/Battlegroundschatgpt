#!/usr/bin/env python3
"""Recover a truncated expert archive by deterministic regeneration and refit.

Original files remain untouched. Success requires exact JSON equality with every
recoverable original trajectory, all planned training decisions, and byte-for-
byte equality with every original checkpoint. No holdout trajectory is consumed.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from bg_ai.learning import Dataset, Ranker
from bg_ai.policy_factory_cached import CachedPolicyFactory
from bg_ai.policy_learning import fit_policy
from bg_ai.turn_budget import TimingProfile
from bg_ai.two_turn_features_v2 import TWO_TURN_FEATURE_NAMES, TWO_TURN_FEATURE_VERSION, two_turn_schema_id
from bg_ai.choice_timing import CHOICE_TIMING_VERSION
from train_recruit_policy import CombatBridge, SCOPE, play_episode, trajectory_scenarios, write_line


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def canonical(row):
    return json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def extract_prefix(original, output):
    decoder, buffer = zlib.decompressobj(31), b""
    count, decoded_bytes = 0, 0
    decoded_hash = hashlib.sha256()
    with original.open("rb") as source, gzip.open(output, "wb") as sink:
        while chunk := source.read(128 * 1024):
            decoded = decoder.decompress(chunk)
            decoded_hash.update(decoded)
            decoded_bytes += len(decoded)
            buffer += decoded
            lines = buffer.split(b"\n")
            buffer = lines.pop()
            for line in lines:
                json.loads(line)  # Reject partial/non-JSON evidence; never fill gaps.
                sink.write(line + b"\n")
                count += 1
    if buffer or decoder.unused_data:
        raise ValueError("Original archive is not a clean complete-JSONL prefix")
    return {"complete_original_rows": count, "original_stream_complete": decoder.eof,
            "decoded_prefix_bytes": decoded_bytes, "decoded_prefix_sha256": decoded_hash.hexdigest(),
            "strict_prefix_archive_sha256": sha(output)}


def verify_loaded_sources(sources):
    verified, additional = {}, {}
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if not path.is_relative_to(ROOT) or path.suffix != ".py" or path == Path(__file__).resolve():
            continue
        relative = path.relative_to(ROOT).as_posix()
        value = sha(path)
        if relative in sources:
            if value != sources[relative]["sha256"]:
                raise ValueError("A loaded original training dependency changed: " + relative)
            verified[relative] = value
        elif relative in {"python/bg_ai/policy_factory_cached.py", "scripts/archive_recruit_runs.py"}:
            additional[relative] = value
        else:
            raise ValueError("Unrecorded loaded repository dependency: " + relative)
    for relative, record in sources.items():
        if relative.startswith("simulator/"):
            if sha(ROOT / relative) != record["sha256"]:
                raise ValueError("Original simulator source changed: " + relative)
            verified[relative] = record["sha256"]
    return {"verified_original_dependencies": verified, "additional_recovery_dependencies": additional}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--engine-root", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    if not (run / "results.json").exists():
        raise ValueError("Only completed original runs are eligible")
    out = run / "recovered_expert_data"
    if out.exists() and any(out.iterdir()):
        raise ValueError("Recovery destination must be new or empty")
    out.mkdir(exist_ok=True)
    started = time.monotonic()
    original = run / "expert_trajectories.jsonl.gz"
    source_sha = sha(original)
    original_manifest = json.loads((run / "expert_data_manifest.json").read_text())
    prereg = json.loads((run / "preregistration.json").read_text())
    if source_sha != original_manifest["archive_sha256"]:
        raise ValueError("Original archive changed from its recorded manifest")
    with gzip.open(run / "execution_sources.json.gz", "rt", encoding="utf8") as stream:
        sources = json.load(stream)
    if {path: record["sha256"] for path, record in sources.items()} != prereg["source_sha256"]:
        raise ValueError("Original source inventory changed")
    for record in sources.values():
        if hashlib.sha256(record["content"].encode()).hexdigest() != record["sha256"]:
            raise ValueError("Archived source content checksum mismatch")
    dependencies = verify_loaded_sources(sources)
    expected = prereg["expert_data"]
    profile = TimingProfile.load(ROOT / "config/turn-budget.json")
    if (expected["ruleset_file_sha256"] != sha(ROOT / "data/ruleset.json")
            or expected["reference_cards_sha256"] != sha(ROOT / "data/reference_cards.json")
            or expected["timing_profile_sha256"] != profile.fingerprint
            or expected["feature_schema_sha256"] != two_turn_schema_id()
            or expected["choice_timing_version"] != CHOICE_TIMING_VERSION):
        raise ValueError("Recovery rules, data, schema or action timing changed")
    if original_manifest["failures"]:
        raise ValueError("This exact recovery requires an originally complete accepted expert seed plan")
    prefix = extract_prefix(original, out / "original_complete_prefix.jsonl.gz")
    registration = {"status": "running", "original_archive_sha256": source_sha,
                    "original_manifest_sha256": sha(run / "expert_data_manifest.json"),
                    "original_preregistration_sha256": sha(run / "preregistration.json"),
                    "execution_sources_sha256": sha(run / "execution_sources.json.gz"),
                    "recovery_script_sha256": sha(Path(__file__)), "prefix": prefix,
                    "holdout_trajectories_read": False, "source_dependencies": dependencies,
                    "numeric_threads": {name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")}}
    write(out / "recovery_manifest.json", registration)
    factory = CachedPolicyFactory(args.engine_root.resolve(), json.loads((ROOT / "data/ruleset.json").read_text()), profile)
    bridge = CombatBridge(out / "firestone-worker.log")
    scenarios, compared, identical_lines, trajectory_seeds = [], 0, 0, []
    try:
        with gzip.open(out / "original_complete_prefix.jsonl.gz", "rb") as prefix_stream, \
                gzip.open(out / "expert_trajectories.jsonl.gz", "wt", encoding="utf8") as archive:
            for index, seed in enumerate(prereg["seed_plan"]["train"]):
                trajectory = play_episode(factory, bridge, seed, sample=0, collect=True)
                serialized = json.dumps(trajectory, separators=(",", ":"), allow_nan=False).encode() + b"\n"
                if index < prefix["complete_original_rows"]:
                    line = prefix_stream.readline()
                    original_row = json.loads(line)
                    if canonical(trajectory) != canonical(original_row):
                        write(out / "prefix_mismatch.json", {"index": index, "seed": seed,
                            "original_canonical_sha256": hashlib.sha256(canonical(original_row)).hexdigest(),
                            "regenerated_canonical_sha256": hashlib.sha256(canonical(trajectory)).hexdigest()})
                        raise ValueError("Regenerated trajectory differs from preserved original evidence")
                    compared += 1
                    identical_lines += serialized == line
                write_line(archive, trajectory)
                scenarios.extend(trajectory_scenarios(trajectory, verify=True))
                trajectory_seeds.append(seed)
                if (index + 1) % 16 == 0 or index + 1 == len(prereg["seed_plan"]["train"]):
                    progress = {"stage": "regenerate_expert", "completed": index + 1,
                        "total": len(prereg["seed_plan"]["train"]), "decisions": len(scenarios),
                        "original_rows_matched": compared, "elapsed_seconds": time.monotonic() - started}
                    write(out / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
            if prefix_stream.read():
                raise ValueError("Original prefix contains extra unplanned trajectories")
        if (trajectory_seeds != expected["attempted_seeds"] or len(scenarios) != original_manifest["decisions"]
                or len(trajectory_seeds) != original_manifest["accepted_trajectories"]):
            raise ValueError("Regenerated decision or accepted-trajectory counts differ")
        dataset = Dataset(scenarios, TWO_TURN_FEATURE_NAMES)
        checks = []
        for width in prereg["candidate_models"]["hidden"]:
            model = Ranker(TWO_TURN_FEATURE_NAMES, hidden=width, seed=prereg["candidate_models"]["model_seed"])
            prior = 0
            for epoch in prereg["candidate_models"]["epochs"]:
                fit_policy(model, dataset, epochs=epoch - prior)
                prior = epoch
                model.metadata.update(full_game_ready=False, scope=SCOPE, feature_version=TWO_TURN_FEATURE_VERSION,
                    feature_schema_sha256=two_turn_schema_id(), ruleset_file_sha256=expected["ruleset_file_sha256"],
                    timing_profile_sha256=profile.fingerprint, choice_timing_version=CHOICE_TIMING_VERSION,
                    collection_seed=expected["seed"], reserved_evaluation_seeds=prereg["seed_plan"]["validation"] + prereg["seed_plan"]["test"],
                    training_target="binary preferred teacher action set; imitation only")
                name = f"policy-h{width}-e{epoch}.json"
                model.save(out / name)
                record = {"checkpoint": name, "original_sha256": sha(run / name),
                          "recovered_sha256": sha(out / name)}
                record["byte_identical"] = record["original_sha256"] == record["recovered_sha256"]
                checks.append(record)
                write(out / "checkpoint_reproduction.json", {"checkpoints": checks})
                print(json.dumps({"stage": "refit_checkpoint", **record}), flush=True)
                if not record["byte_identical"]:
                    raise ValueError("Regenerated expert data did not reproduce the original checkpoint bytes")
        # A reusable manifest is created only after every required identity check.
        if sha(original) != source_sha or verify_loaded_sources(sources) != dependencies:
            raise ValueError("Original evidence or loaded source changed during recovery")
        manifest = dict(original_manifest, archive_sha256=sha(out / "expert_trajectories.jsonl.gz"))
        write(out / "expert_data_manifest.json", manifest)
        registration.update(status="verified_recovery", recovered_archive_sha256=manifest["archive_sha256"],
            regenerated_trajectories=len(trajectory_seeds), regenerated_decisions=len(scenarios),
            original_rows_json_identical=compared, original_rows_byte_identical=identical_lines,
            checkpoint_reproduction=checks, elapsed_seconds=time.monotonic() - started,
            interpretation="Original truncated bytes preserved; separate deterministic recovery matches every available original row and all original checkpoint bytes. No holdout trajectories used.")
        write(out / "recovery_manifest.json", registration)
        print(json.dumps(registration), flush=True)
    except BaseException as error:
        registration.update(status="failed", error=f"{type(error).__name__}: {error}", elapsed_seconds=time.monotonic() - started)
        write(out / "recovery_manifest.json", registration)
        raise
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
