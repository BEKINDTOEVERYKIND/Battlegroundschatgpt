"""Actor-training labels, whole-trajectory support and frozen-test selection."""
from copy import deepcopy
import gzip
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from bg_ai.recruiting import RecruitAction
from bg_ai.learning import Scenario
from bg_ai.verified_trajectory_archive import ArchiveIntegrityError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("recruit_policy_training", ROOT / "scripts/train_recruit_policy.py")
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


def policy_record(score, *, complete=True, violations=0):
    return {"complete": complete, "penalized_score": score if complete else 0.,
            "timing_budget_violations": violations}


class RecruitPolicyTrainingTests(unittest.TestCase):
    def test_teacher_targets_include_every_exact_tied_best_action(self):
        actions = [RecruitAction("play", "a"), RecruitAction("play", "b"), RecruitAction("freeze")]
        with patch.object(trainer, "heuristic_action", return_value=actions[0]), \
             patch.object(trainer, "heuristic_value", side_effect=[900., 900., -10.]):
            np.testing.assert_array_equal(trainer.teacher_targets(object(), actions), [1., 1., 0.])
        with patch.object(trainer, "heuristic_action", return_value=actions[1]):
            with self.assertRaisesRegex(ValueError, "Candidate zero"):
                trainer.teacher_targets(object(), actions)

    def test_failed_episodes_stay_in_denominator_and_cannot_win_selection(self):
        rows = [
            {"policies": {"practical": policy_record(.5), "learner": policy_record(1.)}},
            {"policies": {"practical": policy_record(.5), "learner": policy_record(1., complete=False)}},
        ]
        summary = trainer.evaluation_summary(rows, ["practical", "learner"])
        self.assertEqual(summary["learner"]["failure_penalized_mean"], .5)
        self.assertEqual(summary["learner"]["paired_all_attempts"]["mean_delta"], 0.)
        self.assertEqual(summary["learner"]["paired_supported_intersection"]["mean_delta"], .5)
        self.assertEqual(summary["learner"]["failed_episodes"], 1)
        selected = trainer.select_validation(summary, ["learner"])
        self.assertEqual(selected["deployment_policy"], "practical")
        self.assertEqual(selected["eligible_learners"], [])
        self.assertEqual(selected["experimental_learner"], "learner")

    def test_validation_cannot_promote_and_only_frozen_learner_can_pass_independent_test(self):
        rows = [{"policies": {"practical": policy_record(.4), "chosen": policy_record(.6),
                              "other": policy_record(.5)}} for _ in range(5)]
        validation = trainer.evaluation_summary(rows, ["practical", "chosen", "other"])
        selected = trainer.select_validation(validation, ["chosen", "other"])
        self.assertEqual(selected["experimental_learner"], "chosen")
        self.assertEqual(selected["deployment_policy"], "practical")
        self.assertFalse(selected["learner_promoted_within_fixture"])
        passed = trainer.test_promotion(selected, validation, current_snapshot_verified=True)
        self.assertEqual(passed["deployment_policy"], "chosen")
        self.assertFalse(passed["full_game_ready"])
        self.assertEqual(selected["deployment_policy"], "practical")
        for field, value in (("failed_episodes", 1), ("timing_budget_violations", 1)):
            changed = deepcopy(validation)
            changed["chosen"][field] = value
            self.assertFalse(trainer.test_promotion(selected, changed,
                current_snapshot_verified=True)["learner_promoted_within_fixture"])
        changed = deepcopy(validation)
        changed["chosen"]["paired_all_attempts"]["ci95"][0] = 0.
        self.assertFalse(trainer.test_promotion(selected, changed,
            current_snapshot_verified=True)["learner_promoted_within_fixture"])
        self.assertFalse(trainer.test_promotion(selected, validation,
            current_snapshot_verified=False)["learner_promoted_within_fixture"])

    def test_seed_families_are_separate_and_reproducible(self):
        plan = trainer.seed_plan(202609077, 1024, 32, 128)
        self.assertEqual(plan, trainer.seed_plan(202609077, 1024, 32, 128))
        self.assertEqual(plan["validation"][0] - plan["train"][0], 20000000)
        self.assertEqual(plan["test"][0] - plan["train"][0], 30000000)
        self.assertEqual(len(set(sum(plan.values(), []))), 1024+32+128)
        with self.assertRaises(ValueError):
            trainer.seed_plan(-1, 10, 2, 2)

    def test_only_complete_expert_trajectories_and_train_labels_are_accepted(self):
        decision = {"decision_id": "s", "split": "train", "observation": {"turn": 1},
            "candidates": [{"features": [1., 0.], "preferred": 1.},
                           {"features": [0., 1.], "preferred": 0.}]}
        trajectory = {"seed": 2, "complete": True, "combats": [{}, {}], "decisions": [decision], "score": .123}
        scenarios = trainer.trajectory_scenarios(trajectory)
        np.testing.assert_array_equal(scenarios[0].scores, [1., 0.])
        self.assertNotIn(.123, scenarios[0].scores)
        for fields in ({"complete": False}, {"combats": [{}]}):
            with self.assertRaisesRegex(ValueError, "complete two-combat"):
                trainer.trajectory_scenarios(dict(trajectory, **fields))
        changed = deepcopy(trajectory)
        changed["decisions"][0]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "validation/test"):
            trainer.trajectory_scenarios(changed)

    def test_actor_controls_every_own_decision_and_callback_leaves_opponents_practical(self):
        observations = [SimpleNamespace(player_id=0, turn=1), SimpleNamespace(player_id=1, turn=1),
                        SimpleNamespace(player_id=0, turn=2)]
        receipts = [{"turn": turn, "outcomes": [{"player_a": 0, "player_b": turn, "winner_id": 0}]}
                    for turn in (1, 2)]
        raw = SimpleNamespace(phase="complete", combat_receipts=receipts)
        timed = SimpleNamespace(_visible=observations[0], engine=raw, trace=[])
        pending = iter(observations[1:] + [None])
        timed.step = lambda action: setattr(timed, "_visible", next(pending))
        factory = SimpleNamespace(setup=lambda seed: (timed, {}))
        model, calls = object(), []
        def choose(obs, current):
            calls.append((obs.player_id, obs.turn, current))
            return RecruitAction("end_turn")
        with patch.object(trainer, "choose_action", side_effect=choose), \
             patch.object(trainer, "enrich_two_turn_observation", side_effect=lambda obs: obs), \
             patch.object(trainer, "observation_record", return_value={}):
            result = trainer.play_episode(factory, None, 123, model=model)
        self.assertEqual(calls, [(0, 1, model), (1, 1, None), (0, 2, model)])
        self.assertEqual(result["score"], 1.)
        self.assertEqual(len(result["commands"]), 3)

    def test_expert_archive_finalization_failure_blocks_fitting(self):
        trajectory = {"seed": 13, "complete": True,
            "combats": [{"receipt": {"turn": 1}}, {"receipt": {"turn": 2}}],
            "decisions": [{"decision_id": "expert-13-0", "split": "train", "observation": {"turn": 1},
                "candidates": [{"features": [1., 0.], "preferred": 1.},
                               {"features": [0., 1.], "preferred": 0.}]}]}
        bridge = SimpleNamespace(requests=0, close=lambda: None)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(trainer, "PolicyFactory"), \
             patch.object(trainer, "CombatBridge", return_value=bridge), \
             patch.object(trainer, "play_episode", return_value=trajectory), \
             patch.object(trainer.VerifiedTrajectoryWriter, "finish", side_effect=ArchiveIntegrityError("incomplete archive")), \
             patch.object(trainer, "fit_policy") as fit:
            out = Path(directory) / "run"
            with self.assertRaisesRegex(ArchiveIntegrityError, "incomplete archive"):
                trainer.main(["--engine-root", directory, "--out", str(out), "--trajectories", "1",
                    "--validation-episodes", "2", "--test-episodes", "2", "--hidden", "2", "--epochs", "1",
                    "--seed", "13", "--allow-historical"])
            fit.assert_not_called()
            self.assertFalse((out / "expert_trajectories.jsonl.gz").exists())
            self.assertTrue((out / "failure.json").exists())

    def test_source_snapshot_captures_indirect_local_script_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            hashes = trainer.source_snapshot(out)
            with gzip.open(out / "execution_sources.json.gz", "rt") as stream:
                payload = json.load(stream)
            relative = "scripts/archive_recruit_runs.py"
            self.assertEqual(hashes[relative], trainer.sha(ROOT / relative))
            self.assertEqual(payload[relative]["content"], (ROOT / relative).read_text())
            manifest = json.loads((out / "execution_source_manifest.json").read_text())
            self.assertEqual(manifest["capture_version"], trainer.SOURCE_CAPTURE_VERSION)
            self.assertIn(relative, manifest["loaded_local_python_modules"])
            self.assertEqual(manifest["source_archive_sha256"], trainer.sha(out / "execution_sources.json.gz"))

    def test_reuse_validates_destination_footer_and_manifest_decision_count(self):
        trajectory = {"seed": 9, "complete": True,
            "combats": [{"receipt": {"turn": 1}}, {"receipt": {"turn": 2}}]}
        scenario = Scenario("expert-9-0", "train", np.zeros((2, 2)), np.array([1., 0.]), {})
        with tempfile.TemporaryDirectory() as directory:
            source, destination = Path(directory) / "source", Path(directory) / "copy"
            source.mkdir(); destination.mkdir()
            archive = source / "expert_trajectories.jsonl.gz"
            with gzip.open(archive, "wt") as stream:
                stream.write(json.dumps(trajectory) + "\n")
            expected = {"attempted_seeds": [9]}
            manifest = dict(expected, accepted_trajectories=1, decisions=1, failures=[], archive_sha256=trainer.sha(archive))
            trainer.write_json(source / "expert_data_manifest.json", manifest)
            def truncated_copy(origin, target):
                Path(target).write_bytes(Path(origin).read_bytes()[:-8])
            with patch.object(trainer, "trajectory_scenarios", return_value=[scenario]), \
                 patch.object(trainer.shutil, "copyfile", side_effect=truncated_copy):
                with self.assertRaisesRegex(ArchiveIntegrityError, "Strict trajectory"):
                    trainer.reuse_expert_data(source, expected, destination)
            self.assertFalse((destination / "expert_data_manifest.json").exists())
            manifest["decisions"] = 2
            trainer.write_json(source / "expert_data_manifest.json", manifest)
            with patch.object(trainer, "trajectory_scenarios", return_value=[scenario]):
                with self.assertRaisesRegex(ValueError, "count mismatch"):
                    trainer.reuse_expert_data(source, expected, destination)


if __name__ == "__main__":
    unittest.main()
