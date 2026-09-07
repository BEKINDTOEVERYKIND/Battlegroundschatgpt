"""Actor-training labels, whole-trajectory support and frozen-test selection."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from bg_ai.recruiting import RecruitAction

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


if __name__ == "__main__":
    unittest.main()
