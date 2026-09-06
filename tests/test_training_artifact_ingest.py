import contextlib
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("training_artifact_ingest", ROOT / "scripts/ingest_training_artifact.py")
ingest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ingest)


class TrainingArtifactIngestTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "config/result-ingest-job.json").read_text())

    def metadata(self, config=None):
        config = config or self.config
        run = {"id": config["run_id"], "head_sha": config["source_commit"], "head_branch": "main",
               "path": config["source_workflow"], "status": "completed", "conclusion": "success",
               "repository": {"full_name": config["repository"]},
               "head_repository": {"full_name": config["repository"]}, "updated_at": "2026-09-05T14:22:02Z"}
        artifact = {"id": config["artifact_id"], "name": config["artifact_name"],
                    "size_in_bytes": config["artifact_bytes"], "expired": False,
                    "digest": "sha256:" + config["artifact_sha256"],
                    "workflow_run": {"id": config["run_id"], "head_sha": config["source_commit"],
                                     "head_branch": "main"}}
        return run, artifact

    def fixture(self, root):
        source = root / "original"
        for child in ("generation", "scratch", "warm"):
            (source / child).mkdir(parents=True)
        dataset = b'{"scenario_id":"fixture","split":"train"}\n'
        dataset_hash = hashlib.sha256(dataset).hexdigest()
        raw = source / "generation/positions.jsonl"
        raw.write_bytes(dataset)
        compressed = source / "generation/positions.jsonl.gz"
        compressed.write_bytes(gzip.compress(dataset, mtime=0))
        ingest.write_json(source / "generation/dataset_archive.json", {
            "compressed_file": compressed.name, "uncompressed_file": raw.name,
            "compressed_sha256": ingest.digest(compressed), "uncompressed_sha256": dataset_hash,
            "compressed_bytes": compressed.stat().st_size, "uncompressed_bytes": len(dataset)})
        ingest.write_json(source / "generation/positions.jsonl.meta.json", {"datasetSha256": dataset_hash})
        ingest.write_json(source / "job.json", {
            "status": "completed", "source_commit": self.config["source_commit"],
            "github_run_id": str(self.config["run_id"]),
            "config": {"experiment": Path(self.config["destination"]).name},
            "completed_stages": [{"stage": stage, "seconds": 1} for stage in ("generate", "scratch", "warm")],
            "elapsed_seconds": 3})
        ingest.write_json(source / "comparison.json", {
            "scenarios": 1, "mean_scores": {"scratch": .5, "warm": .75},
            "warm_minus_scratch": {"mean": .25, "ci95": [.25, .25]}})
        for arm, score in (("scratch", .5), ("warm", .75)):
            target = source / arm
            (target / "positioning.json").write_text('{"weights":[1,2,3]}\n')
            (target / "frozen_selections.jsonl").write_text('{"scenario_id":"heldout"}\n')
            row = {"scenario_id": "heldout", "split": "test", "scores": {"model": {"score": score}},
                   "metadata": {"combatSeed": 982451653, "trials": 1024,
                                "adapterVersion": "firestone-combat-v2-tribes",
                                "cardsSha256": "cards", "rulesetSha256": "rules"}}
            (target / "fresh_evaluation.jsonl").write_text(json.dumps(row) + "\n")
            checkpoint_hash = ingest.digest(target / "positioning.json")
            ingest.write_json(target / "evaluation.json", {
                "scope": "positioning only", "evaluated_policy": "neural_proposals_plus_simulation_search",
                "checkpoint_sha256": checkpoint_hash, "dataset_sha256": dataset_hash,
                "selections_sha256": ingest.digest(target / "frozen_selections.jsonl"),
                "results": {"test": {"scenario_count": 1, "mean_scores": {"model": score},
                                     "scenario_details": [row]}}})
            ingest.write_json(target / "provenance.json", {
                "checkpoint_sha256": checkpoint_hash, "dataset_sha256": dataset_hash,
                "fresh_evaluation_sha256": ingest.digest(target / "fresh_evaluation.jsonl"),
                "evaluation_report_sha256": ingest.digest(target / "evaluation.json")})
            (target / "train.log").write_bytes(b"fixture log\nwith exact original bytes\n")
        download = root / "download"
        download.mkdir()
        archive = download / "training-fixture.zip"
        files = sorted(p for p in source.rglob("*") if p.is_file())
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
            for file in files:
                zipped.write(file, file.relative_to(source).as_posix())
        config = dict(self.config, artifact_files=len(files), artifact_bytes=archive.stat().st_size,
                      artifact_sha256=ingest.digest(archive))
        state = root / "state"
        state.mkdir()
        run, artifact = self.metadata(config)
        ingest.write_json(state / "source.json", {"config": config, "run": run, "artifact": artifact})
        return source, download, state, config

    def test_config_has_a_single_bounded_safe_repository_destination(self):
        self.assertEqual(ingest.validate_config(self.config), self.config)
        for change in ({"destination": "runs/../escape"}, {"destination": "/tmp/escape"},
                       {"destination": "runs/20260905-test/child"}, {"run_id": True},
                       {"artifact_bytes": 10**12}, {"source_commit": "HEAD"},
                       {"source_branch": "untrusted"}, {"extra": "field"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                ingest.validate_config(dict(self.config, **change))

    def test_source_must_be_exact_successful_run_and_artifact(self):
        run, artifact = self.metadata()
        ingest.validate_source(self.config, run, artifact)
        for key, value in (("conclusion", "failure"), ("status", "in_progress"),
                           ("head_sha", "0" * 40), ("head_branch", "other"), ("id", 1)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                ingest.validate_source(self.config, dict(run, **{key: value}), artifact)
        for change in ({"expired": True}, {"id": 1}, {"digest": "sha256:" + "0" * 64},
                       {"workflow_run": {"id": 1}}, {"size_in_bytes": 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                ingest.validate_source(self.config, run, dict(artifact, **change))

    def test_archive_rejects_traversal_links_unexpected_files_and_duplicates(self):
        for name in ("../job.json", "/job.json", "scratch/../job.json", "scratch\\train.log",
                     "scratch//train.log", "scratch/./train.log", ".github/workflows/owned.yml"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                ingest.safe_member(zipfile.ZipInfo(name))
        link = zipfile.ZipInfo("scratch/train.log")
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        with self.assertRaisesRegex(ValueError, "symlinks"):
            ingest.safe_member(link)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "duplicate.zip"
            import warnings
            with warnings.catch_warnings(), zipfile.ZipFile(archive, "w") as zipped:
                warnings.simplefilter("ignore", UserWarning)
                zipped.writestr("job.json", "{}")
                zipped.writestr("job.json", "{}")
            config = dict(self.config, artifact_files=2, artifact_bytes=archive.stat().st_size,
                          artifact_sha256=ingest.digest(archive))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                ingest.extract_verified(archive, config, root / "extract")
            self.assertFalse((root / "extract").exists())

    def test_archive_digest_is_checked_before_any_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, download, _, config = self.fixture(root)
            with self.assertRaisesRegex(ValueError, "SHA256"):
                ingest.extract_verified(next(download.iterdir()), dict(config, artifact_sha256="0" * 64), root / "extract")
            self.assertFalse((root / "extract").exists())

    def test_ingest_preserves_all_bytes_once_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, download, state, config = self.fixture(root)
            with contextlib.redirect_stdout(io.StringIO()) as output:
                result = ingest.ingest(config, state, download, root=root)
                ingest.ingest(config, state, download, root=root)
            self.assertIn("INGEST_RESULT_JSON=", output.getvalue())
            self.assertNotIn("scenario_details", result["evaluation"]["scratch"]["results"]["test"])
            destination = root / config["destination"]
            self.assertFalse((destination / "generation/positions.jsonl").exists())
            self.assertFalse((destination / "scratch/train.log").exists())
            manifest = ingest.read_json(destination / "artifact_ingest.json")
            self.assertEqual(len(manifest["original_files"]), config["artifact_files"])
            for name, record in manifest["original_files"].items():
                stored = (destination / record["stored_path"]).read_bytes()
                restored = gzip.decompress(stored) if record["encoding"] == "gzip" else stored
                self.assertEqual(restored, (source / name).read_bytes())
                self.assertEqual(hashlib.sha256(restored).hexdigest(), record["sha256"])
            (destination / "scratch/positioning.json").write_text("changed")
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                ingest.ingest(config, state, download, root=root)
            self.assertEqual((destination / "scratch/positioning.json").read_text(), "changed")

    def test_payload_detects_checkpoint_and_comparison_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            source, _, _, config = self.fixture(Path(directory))
            checkpoint = source / "scratch/positioning.json"
            original = checkpoint.read_bytes()
            checkpoint.write_text('{"weights":[9]}')
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                ingest.validate_payload(source, config)
            checkpoint.write_bytes(original)
            comparison = ingest.read_json(source / "comparison.json")
            comparison["mean_scores"]["warm"] = .9
            ingest.write_json(source / "comparison.json", comparison)
            with self.assertRaisesRegex(ValueError, "mean differs"):
                ingest.validate_payload(source, config)

    def test_destination_symlink_cannot_redirect_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, download, state, config = self.fixture(root)
            (root / "elsewhere").mkdir()
            (root / "runs").symlink_to(root / "elsewhere", target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "Symlink"):
                ingest.ingest(config, state, download, root=root)
            self.assertEqual(list((root / "elsewhere").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
