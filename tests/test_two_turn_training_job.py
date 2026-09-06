import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("two_turn_job", ROOT / "scripts/run_two_turn_job.py")
job = importlib.util.module_from_spec(spec)
spec.loader.exec_module(job)


class TwoTurnJobTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for name in ("data/ruleset.json", "config/turn-budget.json"):
            destination = self.root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
        self.checkpoint = self.root / "model.json"
        self.model = job.Ranker(job.TWO_TURN_FEATURE_NAMES, hidden=16, seed=4)
        self.model.metadata.update(feature_version=job.TWO_TURN_FEATURE_VERSION,
            two_turn_feature_schema_sha256=job.two_turn_schema_id(),
            ruleset_file_sha256=job.file_sha256(self.root / "data/ruleset.json"),
            timing_profile_sha256=job.TimingProfile.load(self.root / "config/turn-budget.json").fingerprint)
        self.model.save(self.checkpoint)
        self.config = dict(experiment="test-current-policy", trajectories=80, label_samples=4,
            evaluation_episodes=32, evaluation_samples=8, seed=202609062, hidden=16,
            timeout_seconds=6600, epochs=[5, 15], policy_checkpoint="model.json",
            policy_checkpoint_sha256=job.file_sha256(self.checkpoint), continuation_policy="practical")

    def test_finite_hash_pinned_policy_and_explicit_continuation(self):
        self.assertEqual(job.validate_config(self.config, self.root), self.config)
        command = job.command(self.config, Path("/tmp/new-two-turn-run"), Path("/tmp/hsbrsim"))
        self.assertEqual(command[command.index("--continuation-policy") + 1], "practical")
        self.assertIn("--policy-checkpoint", command)
        self.assertNotIn("--allow-historical", command)
        for change in ({"trajectories": 10**8}, {"label_samples": True},
                       {"timeout_seconds": 0}, {"epochs": [15, 5]}, {"hidden": 32},
                       {"continuation_policy": "undeclared"}, {"policy_checkpoint": "../outside.json"},
                       {"policy_checkpoint_sha256": "0" * 64}, {"allow_historical": True}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                job.validate_config(dict(self.config, **change), self.root)

    def test_changed_current_rules_or_timing_reject_even_with_matching_file_hash(self):
        for key in ("feature_version", "two_turn_feature_schema_sha256", "ruleset_file_sha256", "timing_profile_sha256"):
            original = self.model.metadata[key]
            self.model.metadata[key] = "different"
            self.model.save(self.checkpoint)
            config = dict(self.config, policy_checkpoint_sha256=job.file_sha256(self.checkpoint))
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "schema, ruleset or action timing"):
                job.validate_config(config, self.root)
            self.model.metadata[key] = original


if __name__ == "__main__":
    unittest.main()
