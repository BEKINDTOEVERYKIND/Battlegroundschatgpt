"""Guard the pilot's learning horizon, complete-policy evaluation and split unit."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from bg_ai.recruiting import RecruitAction
from bg_ai.learning import Dataset, Ranker, Scenario
from bg_ai.turn_budget import TimingProfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("two_turn_pilot", ROOT / "scripts/train_two_turn_recruit.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def receipt(turn, winner):
    return {"turn": turn, "outcomes": [{"player_a": 0, "player_b": turn,
                                         "winner_id": winner}]}


class TwoTurnCurriculumTests(unittest.TestCase):
    def test_objective_requires_both_combats_and_rewards_later_improvement(self):
        self.assertEqual(pilot.episode_score([receipt(1, 1), receipt(2, 0)]), .5)
        self.assertEqual(pilot.episode_score([receipt(1, None), receipt(2, 0)]), .75)
        self.assertEqual(pilot.episode_score([receipt(1, 1), receipt(2, 2)]), 0)
        for invalid in ([], [receipt(1, 0)], [receipt(2, 0), receipt(1, 0)]):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "both actual"):
                pilot.episode_score(invalid)

    def test_cross_split_duplicate_rejects_whole_episode(self):
        trajectory = {"split": "validation", "decisions": [{"state_fingerprint": "new"},
                                                            {"state_fingerprint": "duplicate"}]}
        with self.assertRaisesRegex(ValueError, "across trajectory splits"):
            pilot.check_split_overlap(trajectory, {"duplicate": "train"})
        pilot.check_split_overlap(trajectory, {"duplicate": "validation"})
        self.assertEqual([pilot.episode_split(i) for i in range(5)], ["train"]*4+["validation"])

    def test_neural_policy_is_used_for_every_own_decision_across_turns(self):
        observations = [SimpleNamespace(player_id=0, turn=1), SimpleNamespace(player_id=1, turn=1),
                        SimpleNamespace(player_id=0, turn=2)]
        raw = SimpleNamespace(phase="complete", combat_receipts=[receipt(1, 0), receipt(2, 0)],
                              player_board=lambda i: {"player_id": i})
        timed = SimpleNamespace(_visible=observations[0], engine=raw, trace=[])
        pending = iter(observations[1:] + [None])
        timed.step = lambda action: setattr(timed, "_visible", next(pending))
        model = object()
        calls = []
        def choose(obs, policy):
            calls.append((obs.player_id, obs.turn, policy))
            return RecruitAction("end_turn")
        with patch.object(pilot, "policy_action", side_effect=choose):
            result = pilot.rollout(timed, None, 123, model=model)
        self.assertEqual(calls, [(0, 1, model), (1, 1, None), (0, 2, model)])
        self.assertEqual(len(result["commands"]), 3)
        self.assertEqual(result["score"], 1)

    def test_unsupported_counterfactual_aborts_before_behavior_trajectory_survives(self):
        action = RecruitAction("end_turn")
        obs = SimpleNamespace(player_id=0, turn=1)
        timed = SimpleNamespace(_visible=obs, fork=lambda: object())
        factory = SimpleNamespace(setup=lambda seed: (timed, {}))
        with patch.object(pilot, "heuristic_action", return_value=action), \
             patch.object(pilot, "candidates_for", return_value=[action, RecruitAction("upgrade")]), \
             patch.object(pilot, "encode_two_turn_actions", return_value=np.zeros((2, 2))), \
             patch.object(pilot, "observation_record", return_value={}), \
             patch.object(pilot, "rollout", side_effect=RuntimeError("Tier2 next-turn shop")):
            with self.assertRaisesRegex(RuntimeError, "Tier2 next-turn shop"):
                pilot.build_trajectory(factory, None, index=0, seed=123, samples=1)

    def test_collection_and_continuation_policies_are_independently_declared(self):
        behavior, continuation = object(), object()
        for continuation_model in (None, continuation):
            with self.subTest(continuation=continuation_model is not None):
                action = RecruitAction("end_turn")
                obs = SimpleNamespace(player_id=0, turn=1)
                raw = SimpleNamespace(phase="complete", combat_receipts=[receipt(1, 0), receipt(2, 0)])
                timed = SimpleNamespace(_visible=obs, engine=raw, trace=[], fork=lambda: object())
                timed.step = lambda action: setattr(timed, "_visible", None)
                factory = SimpleNamespace(setup=lambda seed: (timed, {}))
                with patch.object(pilot, "policy_action", return_value=action) as choose, \
                     patch.object(pilot, "candidates_for", return_value=[action]), \
                     patch.object(pilot, "encode_two_turn_actions", return_value=np.zeros((1, 2))), \
                     patch.object(pilot, "observation_record", return_value={}), \
                     patch.object(pilot, "rollout", return_value={"score": 1}) as branch:
                    result = pilot.build_trajectory(factory, None, index=0, seed=123, samples=1,
                        behavior_model=behavior, continuation_model=continuation_model)
                choose.assert_called_once_with(obs, behavior)
                self.assertIs(branch.call_args.kwargs["model"], continuation_model)
                self.assertEqual(result["behavior_policy"], "frozen_checkpoint")

    def test_warm_fitting_does_not_modify_behavior_parameters_optimizer_or_rng(self):
        names = pilot.TWO_TURN_FEATURE_NAMES
        source = Ranker(names, hidden=2, seed=7)
        features = np.zeros((2, len(names)))
        features[1, 0] = 1
        dataset = Dataset([Scenario("independent-new-training", "train", features,
                                    np.array([0., 1.]), {})], names)
        original = source.params["w1"].copy()
        rng = source.rng.bit_generator.state
        trained = pilot.training_model(source, 2)
        trained.fit(dataset, epochs=2)
        np.testing.assert_array_equal(source.params["w1"], original)
        self.assertEqual(source.step, 0)
        self.assertEqual(source.rng.bit_generator.state, rng)
        self.assertGreater(trained.step, 0)

    def test_behavior_checkpoint_checks_current_schema_ruleset_and_clock(self):
        profile = TimingProfile.load(ROOT / "config/turn-budget.json")
        model = Ranker(pilot.TWO_TURN_FEATURE_NAMES, hidden=2, seed=7)
        metadata = {"feature_version": pilot.TWO_TURN_FEATURE_VERSION,
            "two_turn_feature_schema_sha256": pilot.two_turn_schema_id(),
            "ruleset_file_sha256": pilot.sha(ROOT / "data/ruleset.json"),
            "timing_profile_sha256": profile.fingerprint}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.json"
            model.metadata.update(metadata)
            model.save(path)
            pilot.load_policy_checkpoint(path, ROOT / "data/ruleset.json", profile)
            for key in metadata:
                model.metadata.update(metadata)
                model.metadata[key] = "stale"
                model.save(path)
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                    pilot.load_policy_checkpoint(path, ROOT / "data/ruleset.json", profile)

    def test_legacy_pilot_sidecar_preserves_reserved_test_seed_exclusion(self):
        model = Ranker(pilot.TWO_TURN_FEATURE_NAMES, hidden=2)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.json"
            model.save(path)
            pilot.write_json(path.parent / "frozen_evaluation_plan.json",
                {"checkpoint_sha256": pilot.sha(path), "seeds": [1, 2, 3]})
            self.assertEqual(pilot.previous_evaluation_seeds(path, model), {1, 2, 3})

    def test_prior_holdout_cannot_become_new_training_validation_or_test(self):
        for generation, evaluation in (([11, 12], [20]), ([10, 11], [12])):
            with self.subTest(generation=generation), self.assertRaisesRegex(ValueError, "inherited reserved"):
                pilot.reserve_evaluation_seeds(generation, evaluation, {12})
        self.assertEqual(pilot.reserve_evaluation_seeds([10, 11], [20, 21], {1, 2}), [1, 2, 20, 21])


if __name__ == "__main__":
    unittest.main()
