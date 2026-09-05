import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bounded_training_job", ROOT / "scripts/run_training_job.py")
job = importlib.util.module_from_spec(spec)
spec.loader.exec_module(job)


class TrainingJobTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "config/training-job.json").read_text())

    def test_frozen_checkpoint_and_finite_configuration(self):
        self.assertEqual(job.validate_config(self.config), self.config)
        for change in ({"scenarios": 10**9}, {"workers": True}, {"allow_historical": True},
                       {"adapter_version": "legacy-v1"}, {"warm_start_sha256": "0" * 64}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                job.validate_config(dict(self.config, **change))

    def test_both_arms_reuse_same_labels_and_only_warm_arm_resumes(self):
        plan = job.commands(self.config, Path("/tmp/new-training-test"))
        self.assertEqual([name for name, _ in plan], ["generate", "scratch", "warm"])
        scratch, warm = plan[1][1], plan[2][1]
        self.assertEqual(scratch[scratch.index("--data") + 1], warm[warm.index("--data") + 1])
        self.assertNotIn("--warm-start", scratch)
        self.assertIn("--warm-start", warm)
        self.assertTrue(all("--allow-historical" not in command for _, command in plan))

    def test_comparison_pairs_scenarios_and_rejects_rng_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            for name, score in (("scratch", .5), ("warm", .6)):
                (out / name).mkdir()
                rows = [{"scenario_id": str(i), "split": "test",
                         "metadata": {"combatSeed": i + 10, "trials": 1024,
                                      "adapterVersion": job.CURRENT_ADAPTER_VERSION,
                                      "cardsSha256": "cards", "rulesetSha256": "rules"},
                         "scores": {"model": {"score": score}}} for i in range(30)]
                if name == "warm":
                    rows.reverse()
                (out / name / "fresh_evaluation.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")
            result = job.compare(out)
            self.assertEqual(result["scenarios"], 30)
            self.assertAlmostEqual(result["mean_scores"]["warm"] - result["mean_scores"]["scratch"], .1)
            rows[0]["metadata"]["combatSeed"] += 1
            (out / "warm/fresh_evaluation.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")
            with self.assertRaisesRegex(ValueError, "combatSeed"):
                job.compare(out)


if __name__ == "__main__":
    unittest.main()
