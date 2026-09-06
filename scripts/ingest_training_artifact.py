#!/usr/bin/env python3
"""Preserve one declared successful training artifact; never rerun or promote it.

The companion workflow downloads the ZIP with actions/download-artifact v8.
This script checks its exact previously recorded digest before inspecting or
extracting paths. Dataset gzip is retained once; logs become deterministic gzip.
The resulting manifest accounts for every original byte and representation.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
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
    if config["source_workflow"] != ".github/workflows/train.yml":
        raise ValueError("Unexpected source workflow")
    if not isinstance(config["source_commit"], str) or not re.fullmatch(r"[0-9a-f]{40}", config["source_commit"]):
        raise ValueError("Invalid source commit")
    for key, maximum in (("run_id", 10**15), ("artifact_id", 10**15),
                         ("artifact_bytes", MAX_ARCHIVE_BYTES), ("artifact_files", 128)):
        if type(config[key]) is not int or not 1 <= config[key] <= maximum:
            raise ValueError(f"Invalid {key}")
    if not isinstance(config["artifact_sha256"], str) or not HEX64.fullmatch(config["artifact_sha256"]):
        raise ValueError("Invalid artifact SHA256")
    if not isinstance(config["artifact_name"], str) or not re.fullmatch(
            rf"training-{config['run_id']}-[1-9][0-9]*", config["artifact_name"]):
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
        for key in ("run_id", "artifact_id", "destination"):
            stream.write(f"{key}={config[key]}\n")
    print(json.dumps({"validated_source_run": config["run_id"], "artifact": config["artifact_id"],
                      "status": "completed", "conclusion": "success"}))


def allowed_file(name: str) -> bool:
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


def safe_member(info: zipfile.ZipInfo) -> str:
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
    if not allowed_file(name):
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
        names = [safe_member(info) for info in members]
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


def validate_payload(source: Path, config: dict) -> dict:
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
        summary = validate_payload(source, config)
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
