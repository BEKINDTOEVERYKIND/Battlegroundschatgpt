"""Offline policy selection, provenance and finite explicit-choice advice."""
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from bg_ai.choice_timing import CHOICE_TIMING_VERSION
from bg_ai.learning import Ranker
from bg_ai.recruiting import RecruitAction
from bg_ai.turn_budget import TimingProfile
from bg_ai.two_turn_features_v2 import TWO_TURN_FEATURE_NAMES, TWO_TURN_FEATURE_VERSION, two_turn_schema_id

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("recommend_policy", ROOT / "scripts/recommend_recruit_policy.py")
adviser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adviser)


class RecommendRecruitPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        self.profile = TimingProfile.load(ROOT / "config/turn-budget.json")
        self.rules = ROOT / "data/ruleset.json"
        self.record = json.loads((ROOT / "examples/recruit-policy-observation.json").read_text())
        self.model = Ranker(TWO_TURN_FEATURE_NAMES, hidden=2)
        for value in self.model.params.values():
            value[:] = 0
        self.model.params["w1"][TWO_TURN_FEATURE_NAMES.index("recruit.action.freeze"), 0] = 1
        self.model.params["w2"][0] = 1
        self.model.metadata.update(feature_version=TWO_TURN_FEATURE_VERSION,
            feature_schema_sha256=two_turn_schema_id(), ruleset_file_sha256=adviser.digest(self.rules),
            timing_profile_sha256=self.profile.fingerprint, choice_timing_version=CHOICE_TIMING_VERSION,
            full_game_ready=False, training_objective="positive_action_set_softmax_v1")
        self.selection = {"deployment_policy": "practical", "experimental_learner": "test-policy",
            "learner_promoted_within_fixture": False, "independent_test_gate_passed": False}
        self.save()

    def save(self):
        self.model.save(self.run / "experimental_model.json")
        self.selection["experimental_checkpoint_sha256"] = adviser.digest(self.run / "experimental_model.json")
        (self.run / "deployment_policy.json").write_text(json.dumps(self.selection))
        (self.run / "frozen_evaluation_plan.json").write_text(json.dumps({
            "selection_complete_before_test": True,
            "experimental_checkpoint_sha256": self.selection["experimental_checkpoint_sha256"]}))

    def recommend(self, record=None, **kwargs):
        return adviser.recommend(record or self.record, self.run, self.rules, self.profile, **kwargs)

    def test_default_honors_practical_and_experimental_is_explicit(self):
        before = deepcopy(self.record)
        default = self.recommend()
        self.assertEqual(default["policy_used"], "practical")
        self.assertEqual(default["chosen_action"]["kind"], "buy")
        self.assertIsNone(default["candidate_policy_logits"])
        experimental = self.recommend(experimental=True)
        self.assertEqual(experimental["chosen_action"]["kind"], "freeze")
        self.assertEqual(experimental["chosen_action_cost_ms"], 1000)
        self.assertEqual(experimental["policy_used"], "test-policy")
        self.assertTrue(experimental["experimental_override"])
        self.assertIn("not combat probabilities", experimental["score_definition"])
        self.assertLessEqual(experimental["candidate_count"], 16)
        self.assertFalse(experimental["action_executed"])
        self.assertEqual(experimental["remaining_budget_after_command"]["actions_used"], 1)
        self.assertEqual(self.record, before)

    def test_declared_passed_actor_is_used_without_override(self):
        self.selection.update(deployment_policy="test-policy", learner_promoted_within_fixture=True,
                              independent_test_gate_passed=True)
        self.save()
        result = self.recommend()
        self.assertEqual(result["policy_used"], "test-policy")
        self.assertFalse(result["experimental_override"])
        self.selection["independent_test_gate_passed"] = False
        self.save()
        with self.assertRaisesRegex(ValueError, "passing frozen-test"):
            self.recommend()

    def test_gradient_run_can_retain_an_exact_previously_promoted_bc_reference(self):
        self.model.save(self.run / "frozen_bc_reference.json")
        reference_hash = adviser.digest(self.run / "frozen_bc_reference.json")
        prior = {"experimental_checkpoint_sha256": reference_hash,
                 "independent_test_gate_passed": True, "learner_promoted_within_fixture": True}
        (self.run / "frozen_bc_prior_selection.json").write_text(json.dumps(prior))
        (self.run / "preregistration.json").write_text(json.dumps({"reference_sha256": reference_hash}))
        self.selection.update(deployment_policy="bc_reference", existing_baseline="bc_reference")
        self.model.params["w1"][:] = 0
        self.model.params["w1"][TWO_TURN_FEATURE_NAMES.index("recruit.action.end_turn"), 0] = 1
        self.save()
        default = self.recommend()
        self.assertEqual(default["policy_used"], "bc_reference")
        self.assertEqual(default["checkpoint_sha256"], reference_hash)
        self.assertEqual(default["chosen_action"]["kind"], "freeze")
        experimental = self.recommend(experimental=True)
        self.assertEqual(experimental["policy_used"], "test-policy")
        self.assertEqual(experimental["chosen_action"]["kind"], "end_turn")
        prior["independent_test_gate_passed"] = False
        (self.run / "frozen_bc_prior_selection.json").write_text(json.dumps(prior))
        with self.assertRaisesRegex(ValueError, "previously passing"):
            self.recommend()

    def test_hash_and_schema_metadata_mismatch_fail_before_prediction(self):
        path = self.run / "experimental_model.json"
        path.write_text(path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.recommend(experimental=True)
        self.save()
        for key in ("feature_version", "feature_schema_sha256", "ruleset_file_sha256",
                    "timing_profile_sha256", "choice_timing_version"):
            saved = self.model.metadata[key]
            self.model.metadata[key] = "stale"
            self.save()
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                self.recommend(experimental=True)
            self.model.metadata[key] = saved
        self.save()
        wrong = deepcopy(self.record)
        wrong["ruleset_sha256"] = "old"
        with self.assertRaisesRegex(ValueError, "different ruleset"):
            self.recommend(wrong)

    def test_timer_can_only_tighten_and_forged_budgets_are_rejected(self):
        original = self.recommend(experimental=True)
        extended = self.recommend(experimental=True, remaining_ms=999999)
        self.assertEqual(original["remaining_budget_before_action"], extended["remaining_budget_before_action"])
        expired = self.recommend(experimental=True, remaining_ms=5000)
        self.assertEqual(expired["chosen_action"]["kind"], "end_turn")
        self.assertEqual(expired["chosen_action_cost_ms"], 0)
        self.assertTrue(expired["turn_closed_by_action"])
        for key, value in (("remaining_ms", 999999), ("remaining_actions", 999),
                           ("charged_ms", 999999), ("reserved_choice_ms", 2000),
                           ("choice_timing_version", "old")):
            changed = deepcopy(self.record)
            changed["action_budget"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.recommend(changed)

    def test_alliance_flag_reserves_both_explicit_choices_but_only_charges_one_command(self):
        record = deepcopy(self.record)
        record["private_state"]["hand"] = [{"entity_id": "flag", "card_id": "BG31_880",
            "card_type": "spell", "text": "Choose One - Give a minion +3/+1; or +1/+3.",
            "cost": 1, "temporary_spellcraft": False, "enchantments": []}]
        record["legal_actions"] = [asdict(RecruitAction("cast_spell", "flag")), asdict(RecruitAction("end_turn"))]
        result = self.recommend(record)
        self.assertEqual(result["chosen_action_cost_ms"], 1500)
        self.assertEqual(result["forced_choice_commands_after_action"], 2)
        self.assertEqual(result["reserved_choice_cost_ms_after_action"], 4000)
        self.assertEqual(result["minimum_complete_sequence_cost_ms"], 5500)
        self.assertEqual(result["remaining_budget_after_command"]["actions_used"], 1)
        self.assertEqual(result["remaining_budget_after_command"]["remaining_ms"], 8500)
        shortened = self.recommend(record, remaining_ms=10000)
        self.assertEqual(shortened["chosen_action"]["kind"], "end_turn")

    def test_shorter_timer_cannot_synthesize_a_pending_choice_resolution(self):
        record = deepcopy(self.record)
        target = record["private_state"]["shop"][0]
        record["private_state"]["pending_choice"] = {"kind": "spell_target", "options": [target]}
        record["legal_actions"] = [asdict(RecruitAction("choose", target_id=target["entity_id"], choice_index=0))]
        record["action_budget"].update(reserved_choice_commands=1, reserved_choice_ms=2000)
        result = self.recommend(record)
        self.assertEqual(result["chosen_action"]["kind"], "choose")
        self.assertEqual(result["chosen_action_cost_ms"], 2000)
        self.assertEqual(result["forced_choice_commands_after_action"], 0)
        with self.assertRaisesRegex(ValueError, "no longer permits"):
            self.recommend(record, remaining_ms=6000)


if __name__ == "__main__":
    unittest.main()
