#!/usr/bin/env python3
"""Preserve one declared successful training artifact; never rerun or promote it.

The companion workflow downloads the ZIP with actions/download-artifact v8.
This script checks its exact previously recorded digest before inspecting or
extracting paths. Dataset gzip is retained once; logs become deterministic gzip.
The resulting manifest accounts for every original byte and representation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "BEKINDTOEVERYKIND/Battlegroundschatgpt"
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_FILE_BYTES = 400 * 1024 * 1024
MAX_TOTAL_BYTES = 600 * 1024 * 1024
MAX_SAVED_FILE_BYTES = 95 * 1024 * 1024
MAX_JSON_BYTES = 32 * 1024 * 1024
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
TWO_TURN_WORKFLOW = ".github/workflows/train-two-turn.yml"
MAX_TRACE_LINE_BYTES = 256 * 1024 * 1024
MAX_TRACE_TOTAL_BYTES = 8 * 1024 * 1024 * 1024


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> dict:
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError(f"JSON is too large: {path.name}")
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON value: {value}")
    value = json.loads(path.read_text(), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")
    return value


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def validate_config(config: dict) -> dict:
    fields = {"repository", "run_id", "source_commit", "source_branch", "source_workflow",
              "artifact_id", "artifact_name", "artifact_sha256", "artifact_bytes",
              "artifact_files", "destination"}
    if set(config) != fields:
        raise ValueError("Ingest configuration has missing or unknown fields")
    if config["repository"] != REPOSITORY or config["source_branch"] != "main":
        raise ValueError("Only this repository's main-branch training is eligible")
    if config["source_workflow"] not in (".github/workflows/train.yml", TWO_TURN_WORKFLOW):
        raise ValueError("Unexpected source workflow")
    if not isinstance(config["source_commit"], str) or not re.fullmatch(r"[0-9a-f]{40}", config["source_commit"]):
        raise ValueError("Invalid source commit")
    for key, maximum in (("run_id", 10**15), ("artifact_id", 10**15),
                         ("artifact_bytes", MAX_ARCHIVE_BYTES), ("artifact_files", 128)):
        if type(config[key]) is not int or not 1 <= config[key] <= maximum:
            raise ValueError(f"Invalid {key}")
    if not isinstance(config["artifact_sha256"], str) or not HEX64.fullmatch(config["artifact_sha256"]):
        raise ValueError("Invalid artifact SHA256")
    prefix = "two-turn-training" if config["source_workflow"] == TWO_TURN_WORKFLOW else "training"
    if not isinstance(config["artifact_name"], str) or not re.fullmatch(
            rf"{prefix}-{config['run_id']}-[1-9][0-9]*", config["artifact_name"]):
        raise ValueError("Artifact name does not identify the declared run")
    if not isinstance(config["destination"], str) or not re.fullmatch(
            r"runs/[0-9]{8}-[a-z0-9][a-z0-9-]{0,79}", config["destination"]):
        raise ValueError("Destination must be a single named experiment under runs/")
    return dict(config)


def validate_source(config: dict, run: dict, artifact: dict) -> None:
    expected_run = {"id": config["run_id"], "head_sha": config["source_commit"],
                    "head_branch": "main", "path": config["source_workflow"],
                    "status": "completed", "conclusion": "success"}
    if any(run.get(key) != value for key, value in expected_run.items()):
        raise ValueError("Source run is not the declared successful main-branch training")
    if run.get("repository", {}).get("full_name") != config["repository"]:
        raise ValueError("Source run belongs to another repository")
    if run.get("head_repository", {}).get("full_name") != config["repository"]:
        raise ValueError("Source run used a different head repository")
    expected_artifact = {"id": config["artifact_id"], "name": config["artifact_name"],
                         "size_in_bytes": config["artifact_bytes"], "expired": False}
    if any(artifact.get(key) != value for key, value in expected_artifact.items()):
        raise ValueError("Artifact identity, size, or expiry differs from declaration")
    parent = artifact.get("workflow_run", {})
    if (parent.get("id") != config["run_id"] or parent.get("head_sha") != config["source_commit"]
            or parent.get("head_branch") != "main"):
        raise ValueError("Artifact is attached to a different source run")
    # Older API records may omit digest; the downloaded ZIP is always verified.
    if artifact.get("digest") not in (None, "sha256:" + config["artifact_sha256"]):
        raise ValueError("GitHub artifact digest differs from the recorded upload digest")


def api_json(path: str, token: str) -> dict:
    request = urllib.request.Request("https://api.github.com" + path, headers={
        "Authorization": "Bearer " + token, "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "battlegrounds-training-result-ingest"})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = response.read(2 * 1024 * 1024 + 1)
    if len(payload) > 2 * 1024 * 1024:
        raise ValueError("Unexpectedly large GitHub metadata")
    return json.loads(payload)


def prepare(config: dict, state_dir: Path) -> None:
    token = os.environ.get("GH_TOKEN")
    if not token:
        raise ValueError("GH_TOKEN is required only for authenticated metadata reads")
    if os.environ.get("GITHUB_REPOSITORY") != config["repository"]:
        raise ValueError("Workflow repository differs from ingest configuration")
    prefix = "/repos/" + config["repository"] + "/actions/"
    run = api_json(prefix + "runs/" + str(config["run_id"]), token)
    artifact = api_json(prefix + "artifacts/" + str(config["artifact_id"]), token)
    validate_source(config, run, artifact)
    state_dir.mkdir(parents=True, exist_ok=False)
    write_json(state_dir / "source.json", {"config": config, "run": run, "artifact": artifact})
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as stream:
        for key in ("run_id", "artifact_id", "destination", "source_commit"):
            stream.write(f"{key}={config[key]}\n")
    print(json.dumps({"validated_source_run": config["run_id"], "artifact": config["artifact_id"],
                      "status": "completed", "conclusion": "success"}))


def allowed_file(name: str, workflow: str = ".github/workflows/train.yml") -> bool:
    if workflow == TWO_TURN_WORKFLOW:
        return name in {"runner.json", "preregistration.json", "results.json", "progress.json",
                        "live_preflight.json", "live_postflight.json", "validation_sweep.json",
                        "frozen_evaluation_plan.json", "generation_failures.json", "evaluation_failures.json",
                        "selected_model.json", "training_trajectories.jsonl.gz",
                        "evaluation_trajectories.jsonl.gz", "firestone-worker.log"} or bool(
                            re.fullmatch(r"model-h(?:[8-9]|[1-5][0-9]|6[0-4])-e(?:[1-9]|[1-7][0-9]|80)\.json", name))
    if name in {"job.json", "comparison.json"}:
        return True
    if re.fullmatch(r"generation/(?:live_preflight|live_postflight|generation_commands|dataset_archive)\.json", name):
        return True
    if re.fullmatch(r"generation/positions(?:-[0-7][0-9])?\.jsonl\.(?:meta|coverage)\.json", name):
        return True
    if name in {"generation/positions.jsonl", "generation/positions.jsonl.gz"}:
        return True
    if re.fullmatch(r"generation/generate-[0-7][0-9]\.log", name):
        return True
    return bool(re.fullmatch(
        r"(?:scratch|warm)/(?:live_preflight\.json|positioning\.json|training\.json|"
        r"provenance\.json|evaluation\.json|(?:frozen_selections|neural_selections|fresh_evaluation)\.jsonl|"
        r"(?:train|freeze|search|evaluate|report)\.log)", name))


def safe_member(info: zipfile.ZipInfo, workflow: str = ".github/workflows/train.yml") -> str:
    name = info.filename
    path = PurePosixPath(name)
    if (not name or "\\" in name or "\x00" in name or path.is_absolute()
            or any(part in ("", ".", "..") for part in name.split("/")) or str(path) != name):
        raise ValueError("Unsafe archive path")
    mode = info.external_attr >> 16
    file_type = stat.S_IFMT(mode)
    if info.is_dir() or file_type not in (0, stat.S_IFREG):
        raise ValueError("Archive must contain only regular files, never symlinks")
    if info.flag_bits & 1 or not 0 <= info.file_size <= MAX_FILE_BYTES:
        raise ValueError("Encrypted or oversized artifact member")
    if not allowed_file(name, workflow):
        raise ValueError(f"Unexpected artifact path: {name}")
    return name


def extract_verified(archive: Path, config: dict, target: Path) -> dict:
    if archive.is_symlink() or not archive.is_file():
        raise ValueError("Artifact must be a regular ZIP file")
    if archive.stat().st_size != config["artifact_bytes"] or digest(archive) != config["artifact_sha256"]:
        raise ValueError("Original artifact ZIP size/SHA256 mismatch")
    inventory = {}
    with zipfile.ZipFile(archive) as zipped:
        members = zipped.infolist()
        if len(members) != config["artifact_files"] or len(members) > 128:
            raise ValueError("Original artifact file count differs from declaration")
        if sum(info.file_size for info in members) > MAX_TOTAL_BYTES:
            raise ValueError("Artifact exceeds bounded total uncompressed size")
        names = [safe_member(info, config["source_workflow"]) for info in members]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate archive member")
        # Validate every member before creating any extracted file.
        for name, info in zip(names, members):
            output = target / name
            output.parent.mkdir(parents=True, exist_ok=True)
            count = 0
            h = hashlib.sha256()
            with zipped.open(info) as source, output.open("xb") as sink:
                while chunk := source.read(1024 * 1024):
                    count += len(chunk)
                    if count > info.file_size:
                        raise ValueError("ZIP member expanded beyond declared size")
                    h.update(chunk)
                    sink.write(chunk)
            if count != info.file_size:
                raise ValueError("Truncated ZIP member")
            inventory[name] = {"bytes": count, "sha256": h.hexdigest()}
    return inventory


def assert_hash(path: Path, expected: str, context: str) -> None:
    if not isinstance(expected, str) or not HEX64.fullmatch(expected) or digest(path) != expected:
        raise ValueError(f"Checksum mismatch: {context}")


def source_bytes(root: Path, commit: str, path: str) -> bytes:
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or not re.fullmatch(r"[A-Za-z0-9_.\-/]+", path):
        raise ValueError("Invalid declared source object")
    if PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts:
        raise ValueError("Unsafe declared source path")
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=root)


def read_list(path: Path) -> list:
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("JSON list exceeds size bound")
    result = json.loads(path.read_text())
    if not isinstance(result, list):
        raise ValueError("Expected a JSON list")
    return result


def trace_rows(path: Path):
    """Read one bounded episode at a time without materializing the dataset."""
    total = 0
    with gzip.open(path, "rb") as stream:
        while line := stream.readline(MAX_TRACE_LINE_BYTES + 1):
            total += len(line)
            if len(line) > MAX_TRACE_LINE_BYTES or total > MAX_TRACE_TOTAL_BYTES:
                raise ValueError("Trace expansion exceeds declared resource bounds")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("Expected a complete episode object")
            yield row


def equal_number(actual, expected, context: str) -> None:
    if (type(actual) not in (int, float) or type(expected) not in (int, float)
            or not math.isfinite(actual) or not math.isfinite(expected) or abs(actual - expected) > 1e-12):
        raise ValueError(f"Raw evidence differs from {context}")


def validate_two_turn_payload(source: Path, config: dict, root: Path) -> dict:
    runner = read_json(source / "runner.json")
    results = read_json(source / "results.json")
    registration = read_json(source / "preregistration.json")
    plan = read_json(source / "frozen_evaluation_plan.json")
    commit = config["source_commit"]
    original_config_bytes = source_bytes(root, commit, "config/two-turn-training-job.json")
    original_config = json.loads(original_config_bytes)
    if (runner.get("status") != "completed" or runner.get("source_commit") != commit
            or str(runner.get("github_run_id")) != str(config["run_id"])
            or runner.get("config") != original_config
            or runner.get("config_sha256") != hashlib.sha256(original_config_bytes).hexdigest()
            or original_config.get("experiment") != PurePosixPath(config["destination"]).name
            or runner.get("policy_promoted") is not False):
        raise ValueError("Runner differs from the exact completed source configuration")
    assert_hash(source / "results.json", runner.get("results_sha256"), "runner result")
    expected_paths = {"scripts/train_two_turn_recruit.py", "scripts/run_two_turn_job.py",
                      "python/bg_ai/opening_transition.py", "python/bg_ai/two_turn_features.py",
                      "simulator/opening-transition-firestone.mjs"}
    for record, paths in ((runner, expected_paths), (registration, expected_paths - {"scripts/run_two_turn_job.py"})):
        if set(record.get("source_sha256", {})) != paths:
            raise ValueError("Source provenance is incomplete")
        for path in paths:
            if hashlib.sha256(source_bytes(root, commit, path)).hexdigest() != record["source_sha256"][path]:
                raise ValueError("Recorded source differs from exact training commit")
    checkpoint_hash = hashlib.sha256(source_bytes(root, commit, original_config["policy_checkpoint"])).hexdigest()
    if (checkpoint_hash != original_config["policy_checkpoint_sha256"]
            or registration.get("policy_checkpoint_sha256") != checkpoint_hash):
        raise ValueError("Behavior checkpoint differs from exact source checkpoint")
    for field, path in (("ruleset_file_sha256", "data/ruleset.json"),
                        ("reference_cards_sha256", "data/reference_cards.json")):
        if registration.get(field) != hashlib.sha256(source_bytes(root, commit, path)).hexdigest():
            raise ValueError("Training snapshot differs from the source commit")
    for registered, configured in (("seed", "seed"), ("trajectory_attempts", "trajectories"),
                                   ("evaluation_episodes", "evaluation_episodes"),
                                   ("evaluation_samples_per_policy", "evaluation_samples"),
                                   ("label_samples_per_candidate", "label_samples"),
                                   ("continuation_policy", "continuation_policy")):
        if registration.get(registered) != original_config[configured]:
            raise ValueError("Preregistration differs from source configuration")
    if (registration.get("full_game_ready") is not False or results.get("full_game_ready") is not False
            or results.get("policy_promoted") is not False or results.get("current_snapshot_verified") is not True):
        raise ValueError("Unexpected readiness, promotion or snapshot claim")
    for phase in ("preflight", "postflight"):
        live = read_json(source / f"live_{phase}.json")
        if live.get("current") is not True or live.get("network_checked") is not True or live.get("failures") != []:
            raise ValueError("Original live ruleset check did not pass")
    assert_hash(source / "selected_model.json", plan.get("checkpoint_sha256"), "selected checkpoint")
    if results.get("checkpoint_sha256") != plan["checkpoint_sha256"]:
        raise ValueError("Result refers to another checkpoint")
    sweep = read_list(source / "validation_sweep.json")
    if [item.get("epochs") for item in sweep] != original_config["epochs"]:
        raise ValueError("Checkpoint sweep differs from the preregistered epochs")
    for item in sweep:
        if item.get("checkpoint") != f"model-h{original_config['hidden']}-e{item['epochs']}.json":
            raise ValueError("Unexpected candidate checkpoint path")
        equal_number(item.get("validation_mean"), item.get("validation_mean"), "validation score")
    best = max(sweep, key=lambda item: item["validation_mean"])
    # Loading and saving the selected candidate may reorder JSON keys but cannot
    # alter its weights, optimizer, provenance or reserved evaluation seeds.
    if read_json(source / best["checkpoint"]) != read_json(source / "selected_model.json"):
        raise ValueError("Selected checkpoint differs from validation selection")
    policies = ("model", "practical_heuristic")
    expected_seeds = [original_config["seed"] + 10000000 + i * 1009
                      for i in range(original_config["evaluation_episodes"])]
    if (plan.get("seeds") != expected_seeds or plan.get("samples_per_policy") != original_config["evaluation_samples"]
            or plan.get("policies") != list(policies) or plan.get("selection_complete_before_test") is not True):
        raise ValueError("Frozen evaluation plan differs from the declared test")
    eval_failures = read_list(source / "evaluation_failures.json")
    if results.get("evaluation_failures") != eval_failures:
        raise ValueError("Evaluation rejection report differs from preserved failures")
    episode_scores, action_counts, violations, accepted = [], Counter(), 0, set()
    for row in trace_rows(source / "evaluation_trajectories.jsonl.gz"):
        seed = row["seed"]
        if seed not in expected_seeds or seed in accepted or row.get("episode_id") != f"two-turn-test-{seed}":
            raise ValueError("Evaluation contains repeated or undeclared held-out episodes")
        accepted.add(seed)
        if set(row["policies"]) != set(policies):
            raise ValueError("Evaluation episode is not a complete paired comparison")
        scores = {}
        for policy in policies:
            record = row["policies"][policy]
            if len(record["rollouts"]) != plan["samples_per_policy"]:
                raise ValueError("Evaluation sample count differs from frozen plan")
            sampled = []
            for rollout in record["rollouts"]:
                receipts = rollout["receipts"]
                if [receipt["turn"] for receipt in receipts] != [1, 2]:
                    raise ValueError("Evaluation omitted a sampled combat")
                combat_scores = []
                for receipt in receipts:
                    own = [outcome for outcome in receipt["outcomes"]
                           if 0 in (outcome["player_a"], outcome["player_b"])]
                    if len(own) != 1 or own[0]["winner_id"] not in (None, own[0]["player_a"], own[0]["player_b"]):
                        raise ValueError("Invalid player-zero sampled combat outcome")
                    combat_scores.append(.5 if own[0]["winner_id"] is None else float(own[0]["winner_id"] == 0))
                score = math.fsum(combat_scores) / 2
                equal_number(rollout["score"], score, "sampled rollout score")
                sampled.append(score)
                for trace in rollout["timing_trace"]:
                    if trace["player_id"] == 0 and trace["event"] == "action":
                        violations += int(trace["after"]["remaining_ms"] < 0 or trace["after"]["remaining_actions"] < 0)
                if policy == "model":
                    action_counts.update(command["action"]["kind"] for command in rollout["commands"] if command["player_id"] == 0)
            scores[policy] = math.fsum(sampled) / len(sampled)
            equal_number(record["score"], scores[policy], "episode policy mean")
        episode_scores.append(scores)
    failed_seeds = [failure["seed"] for failure in eval_failures]
    if (not accepted or len(failed_seeds) != len(set(failed_seeds)) or accepted & set(failed_seeds)
            or accepted | set(failed_seeds) != set(expected_seeds)
            or results.get("evaluation_episodes") != len(accepted)
            or results.get("evaluation_attempts") != len(expected_seeds)):
        raise ValueError("Evaluation attempts are not completely accounted for")
    for policy in policies:
        equal_number(results["scores"][policy], math.fsum(row[policy] for row in episode_scores) / len(episode_scores), "reported policy mean")
    differences = [row["model"] - row["practical_heuristic"] for row in episode_scores]
    equal_number(results["paired_model_minus_practical"]["mean_delta"], math.fsum(differences) / len(differences), "paired mean delta")
    if results.get("model_action_counts") != dict(action_counts) or results.get("timing_budget_violations") != violations:
        raise ValueError("Action counts or timing violations differ from raw trajectories")
    trajectories, decisions, indices, seen_states = Counter(), Counter(), set(), {}
    for row in trace_rows(source / "training_trajectories.jsonl.gz"):
        index = row["index"]
        split = "validation" if index % 5 == 4 else "train"
        if (type(index) is not int or not 0 <= index < original_config["trajectories"] or index in indices
                or row["episode_id"] != f"two-turn-{original_config['seed'] + index * 1009}" or row["split"] != split):
            raise ValueError("Training trajectory identity or split differs from source plan")
        indices.add(index)
        trajectories[split] += 1
        for decision in row["decisions"]:
            if decision["split"] != split or decision["episode_id"] != row["episode_id"]:
                raise ValueError("Related decisions crossed episode splits")
            fingerprint = decision["state_fingerprint"]
            if fingerprint in seen_states and seen_states[fingerprint] != split:
                raise ValueError("An equivalent visible state crossed dataset splits")
            seen_states[fingerprint] = split
            if len(decision["candidates"]) >= 2:
                decisions[split] += 1
    failures = read_list(source / "generation_failures.json")
    failed_indices = [failure["index"] for failure in failures]
    if (len(set(failed_indices)) != len(failed_indices) or indices & set(failed_indices)
            or indices | set(failed_indices) != set(range(original_config["trajectories"]))
            or results.get("accepted_training_trajectories") != len(indices)
            or results.get("training_trajectory_attempts") != original_config["trajectories"]
            or results.get("trajectory_splits") != dict(trajectories) or results.get("decision_splits") != dict(decisions)
            or results.get("generation_failures") != len(failures)
            or results.get("generation_failure_reasons") != dict(Counter(failure["error"] for failure in failures))):
        raise ValueError("Training counts or rejections differ from raw trajectories")
    return {"experiment_kind": "two_turn_recruitment", "results": results,
            "elapsed_seconds": runner.get("elapsed_seconds"),
            "raw_verified": {"evaluation_episodes": len(accepted), "training_trajectories": len(indices),
                             "sampled_combat_scores": True, "model_action_counts": dict(action_counts),
                             "timing_budget_violations": violations, "exact_source_commit": commit},
            "interpretation": "Completed two-turn experiment only; the poor checkpoint is preserved as evidence and is not promoted."}


def validate_payload(source: Path, config: dict, root: Path = ROOT) -> dict:
    if config["source_workflow"] == TWO_TURN_WORKFLOW:
        return validate_two_turn_payload(source, config, root)
    job = read_json(source / "job.json")
    if (job.get("status") != "completed" or job.get("source_commit") != config["source_commit"]
            or str(job.get("github_run_id")) != str(config["run_id"])
            or job.get("config", {}).get("experiment") != PurePosixPath(config["destination"]).name):
        raise ValueError("Job record does not match the declared completed experiment")
    if [stage.get("stage") for stage in job.get("completed_stages", [])] != ["generate", "scratch", "warm"]:
        raise ValueError("Training stages are incomplete")
    manifest = read_json(source / "generation/dataset_archive.json")
    if (manifest.get("compressed_file") != "positions.jsonl.gz"
            or manifest.get("uncompressed_file") != "positions.jsonl"):
        raise ValueError("Unexpected archived dataset paths")
    dataset = source / "generation/positions.jsonl"
    compressed = source / "generation/positions.jsonl.gz"
    assert_hash(dataset, manifest.get("uncompressed_sha256"), "dataset")
    assert_hash(compressed, manifest.get("compressed_sha256"), "dataset archive")
    if (dataset.stat().st_size != manifest.get("uncompressed_bytes")
            or compressed.stat().st_size != manifest.get("compressed_bytes")):
        raise ValueError("Dataset archive size mismatch")
    with gzip.open(compressed, "rb") as stream:
        h, count = hashlib.sha256(), 0
        while chunk := stream.read(1024 * 1024):
            count += len(chunk)
            if count > dataset.stat().st_size:
                raise ValueError("Dataset gzip expanded beyond declared size")
            h.update(chunk)
    if count != dataset.stat().st_size or h.hexdigest() != manifest["uncompressed_sha256"]:
        raise ValueError("Dataset gzip does not restore original training labels")
    generation = read_json(source / "generation/positions.jsonl.meta.json")
    if generation.get("datasetSha256") != manifest["uncompressed_sha256"]:
        raise ValueError("Generation manifest points at a different dataset")
    comparison = read_json(source / "comparison.json")
    evaluation = {}
    scenario_ids = {}
    for arm in ("scratch", "warm"):
        directory = source / arm
        provenance = read_json(directory / "provenance.json")
        assert_hash(directory / "positioning.json", provenance.get("checkpoint_sha256"), arm + " checkpoint")
        assert_hash(directory / "fresh_evaluation.jsonl", provenance.get("fresh_evaluation_sha256"), arm + " fresh evaluation")
        assert_hash(directory / "evaluation.json", provenance.get("evaluation_report_sha256"), arm + " evaluation report")
        if provenance.get("dataset_sha256") != manifest["uncompressed_sha256"]:
            raise ValueError("An experiment arm trained on another dataset")
        report = read_json(directory / "evaluation.json")
        assert_hash(directory / "frozen_selections.jsonl", report.get("selections_sha256"), arm + " frozen decisions")
        if (report.get("checkpoint_sha256") != provenance["checkpoint_sha256"]
                or report.get("dataset_sha256") != manifest["uncompressed_sha256"]):
            raise ValueError("Evaluation refers to a different checkpoint or dataset")
        records = {}
        with (directory / "fresh_evaluation.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                identifier = row["scenario_id"]
                if row.get("split") != "test" or identifier in records:
                    raise ValueError("Evaluation is not a unique held-out test split")
                records[identifier] = row
        if not records or len(records) != comparison.get("scenarios"):
            raise ValueError("Comparison scenario count differs from raw evaluations")
        mean = math.fsum(row["scores"]["model"]["score"] for row in records.values()) / len(records)
        recorded = comparison.get("mean_scores", {}).get(arm)
        if type(recorded) not in (int, float) or not math.isfinite(recorded) or abs(mean - recorded) > 1e-12:
            raise ValueError("Comparison mean differs from raw evaluations")
        scenario_ids[arm] = records
        evaluation[arm] = {"scope": report.get("scope"), "evaluated_policy": report.get("evaluated_policy"),
                           "results": {split: {k: v for k, v in result.items() if k != "scenario_details"}
                                       for split, result in report["results"].items()}}
    if set(scenario_ids["scratch"]) != set(scenario_ids["warm"]):
        raise ValueError("Scratch and warm evaluation scenarios differ")
    for identifier, left in scenario_ids["scratch"].items():
        right = scenario_ids["warm"][identifier]
        for key in ("combatSeed", "trials", "adapterVersion", "cardsSha256", "rulesetSha256"):
            if left["metadata"].get(key) != right["metadata"].get(key):
                raise ValueError(f"Paired comparison changed {key}")
    return {"comparison": comparison, "evaluation": evaluation,
            "completed_stages": job["completed_stages"], "elapsed_seconds": job.get("elapsed_seconds"),
            "interpretation": "Completed positioning experiment only; no automatic model promotion or full-game strength claim."}


def reject_symlink_ancestors(path: Path) -> None:
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError("Symlink in output path")


def preserve(source: Path, staged: Path, inventory: dict) -> dict:
    representations = {}
    for name, original in sorted(inventory.items()):
        if name == "generation/positions.jsonl":
            representations[name] = dict(original, stored_path=name + ".gz", encoding="gzip")
            continue  # Verified gzip retains its exact bytes; do not duplicate 200+ MB.
        stored = name + ".gz" if name.endswith(".log") else name
        target = staged / stored
        target.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith(".log"):
            with (source / name).open("rb") as raw, target.open("xb") as output:
                with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0, compresslevel=9) as zipped:
                    shutil.copyfileobj(raw, zipped)
        else:
            shutil.copyfile(source / name, target)
        if target.stat().st_size > MAX_SAVED_FILE_BYTES:
            raise ValueError("A preserved file exceeds the bounded Git blob size")
        representations[name] = dict(original, stored_path=stored,
                                     encoding="gzip" if name.endswith(".log") else "identity")
    return representations


def install_staged(staged: Path, destination: Path) -> None:
    reject_symlink_ancestors(destination)
    if destination.exists():
        if not destination.is_dir():
            raise ValueError("Destination exists and is not a directory")
        expected = {p.relative_to(staged).as_posix(): digest(p) for p in staged.rglob("*") if p.is_file()}
        actual = {}
        for path in destination.rglob("*"):
            if path.is_symlink() or not (path.is_dir() or path.is_file()):
                raise ValueError("Unsafe existing destination contents")
            if path.is_file():
                actual[path.relative_to(destination).as_posix()] = digest(path)
        if actual != expected:
            raise ValueError("Existing experiment differs; refusing to overwrite any evidence")
        return
    staged.rename(destination)


def ingest(config: dict, state_dir: Path, download_dir: Path, root: Path = ROOT) -> dict:
    state = read_json(state_dir / "source.json")
    if state.get("config") != config:
        raise ValueError("Configuration changed since authenticated source validation")
    validate_source(config, state["run"], state["artifact"])
    if download_dir.is_symlink() or not download_dir.is_dir():
        raise ValueError("Download path must be a directory")
    downloaded = list(download_dir.iterdir())
    if len(downloaded) != 1 or not downloaded[0].is_file() or downloaded[0].is_symlink():
        raise ValueError("Download must contain exactly one original artifact ZIP")
    destination = root / config["destination"]
    reject_symlink_ancestors(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="training-artifact-source-") as scratch:
        source = Path(scratch)
        inventory = extract_verified(downloaded[0], config, source)
        summary = validate_payload(source, config, root)
        with tempfile.TemporaryDirectory(prefix=".ingest-stage-", dir=destination.parent) as staging:
            staged = Path(staging) / "experiment"
            staged.mkdir()
            representations = preserve(source, staged, inventory)
            stored_files = {p.relative_to(staged).as_posix(): {"sha256": digest(p), "bytes": p.stat().st_size}
                            for p in sorted(staged.rglob("*")) if p.is_file()}
            manifest = {"schema_version": 1, "source": config,
                        "source_run_url": f"https://github.com/{config['repository']}/actions/runs/{config['run_id']}",
                        "source_run_completed_at": state["run"].get("updated_at"),
                        "checks": ["authenticated successful run identity", "original ZIP SHA256 and size",
                                   "strict regular-file path allowlist and size bounds", "dataset gzip round trip",
                                   "checkpoint and evaluation provenance checksums", "paired raw evaluation means and identities"],
                        "original_files": representations, "stored_files": stored_files,
                        "notes": ["Original dataset gzip is preserved byte-for-byte; raw JSONL restores exactly.",
                                  "Every original log is retained as deterministic gzip.",
                                  "No training rerun, current-patch relabeling, or model promotion occurred."]}
            write_json(staged / "artifact_ingest.json", manifest)
            write_json(staged / "result_summary.json", summary)
            install_staged(staged, destination)
    print("INGEST_RESULT_JSON=" + json.dumps(summary, separators=(",", ":"), allow_nan=False))
    print(json.dumps({"preserved_run": config["run_id"], "destination": config["destination"],
                      "original_files": len(inventory), "artifact_sha256": config["artifact_sha256"]}))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "ingest"))
    parser.add_argument("--config", type=Path, default=ROOT / "config/result-ingest-job.json")
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--download-dir", type=Path)
    args = parser.parse_args()
    config = validate_config(read_json(args.config))
    if args.stage == "prepare":
        prepare(config, args.state_dir)
    else:
        if args.download_dir is None:
            parser.error("ingest requires --download-dir")
        ingest(config, args.state_dir, args.download_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
