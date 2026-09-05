"""Regressions for training against an unsupported or stale recruit engine."""

import copy
import unittest

from bg_ai.recruiting import (
    HSBRSIM_REVISION, RecruitCoverageError, assess_hsbrsim,
    require_full_game_ready, ruleset_digest,
)


class RecruitingCoverageTests(unittest.TestCase):
    def setUp(self):
        self.ruleset = {
            "patch": "36.4.2.251332",
            "active": {"minion_ids": ["M"], "hero_ids": ["H"],
                       "tavern_spell_ids": ["S"], "dark_gift_ids": ["G"],
                       "lesser_trinket_ids": ["T"], "greater_trinket_ids": []},
        }
        self.cards = [{"id": x, "name": x} for x in ["M", "H", "S", "G", "T"]]
        self.cards[1]["heroPowerCardId"] = "HP"
        self.inventory = {
            "revision": HSBRSIM_REVISION, "dirty_checkout": False,
            "upstream_patch": "36.2.2.249896",
            "cards": {x: {"id": x} for x in ["M", "H", "S", "G", "T"]},
            "script_ids": ["M", "HP", "S", "T"],
            "dark_gift_handler_ids": ["G"], "deferred_dark_gift_ids": [],
            "trinket_api_present": True,
        }

    def test_current_pool_missing_handler_is_not_silently_removed(self):
        self.inventory["script_ids"].remove("S")
        report = assess_hsbrsim(self.ruleset, self.cards, self.inventory)
        self.assertEqual(report["categories"]["spells"]["required"], 1)
        self.assertIn("spells:missing_handlers:1", report["blockers"])
        with self.assertRaises(RecruitCoverageError):
            require_full_game_ready(report, self.ruleset)

    def test_deferred_handler_cannot_count_as_supported(self):
        self.inventory["deferred_dark_gift_ids"] = ["G"]
        report = assess_hsbrsim(self.ruleset, self.cards, self.inventory)
        gap = report["categories"]["dark_gifts"]["missing_handlers"][0]
        self.assertTrue(gap["deferred"])
        self.assertIn("dark_gifts:missing_handlers:1", report["blockers"])

    def test_same_ids_do_not_hide_patch_mismatch(self):
        report = assess_hsbrsim(self.ruleset, self.cards, self.inventory)
        self.assertIn("patch_mismatch", report["blockers"])
        self.assertFalse(report["full_game_ready"])

    def test_reused_success_report_cannot_validate_next_rotation(self):
        report = {"ruleset_sha256": ruleset_digest(self.ruleset),
                  "blockers": [], "full_game_ready": True,
                  "conformance_validated": True}
        next_rotation = copy.deepcopy(self.ruleset)
        next_rotation["active"]["minion_ids"].append("M2")
        with self.assertRaisesRegex(RecruitCoverageError, "different ruleset"):
            require_full_game_ready(report, next_rotation)

    def test_registration_alone_never_implies_validated_full_game(self):
        self.inventory["upstream_patch"] = self.ruleset["patch"]
        report = assess_hsbrsim(self.ruleset, self.cards, self.inventory)
        self.assertFalse(report["full_game_ready"])
        self.assertIn("current_patch_behavioral_conformance_unvalidated", report["blockers"])

    def test_minion_mana_cost_is_not_recruitment_price(self):
        self.cards[0].update(cost=0, attack=4, health=5, techLevel=3)
        self.inventory["cards"]["M"].update(cost=3, atk=4, health=5, tech_level=3)
        report = assess_hsbrsim(self.ruleset, self.cards, self.inventory)
        self.assertEqual(report["categories"]["minions"]["numeric_differences"], [])

    def test_blocked_trinket_candidates_are_audited_without_promotion(self):
        self.ruleset["components"] = {"trinket": {"status": "blocked"}}
        self.ruleset["candidate"] = {"lesser_trinket_ids": ["T"], "greater_trinket_ids": []}
        self.ruleset["active"]["lesser_trinket_ids"] = []
        report = assess_hsbrsim(self.ruleset, self.cards, self.inventory)
        self.assertIn("ruleset_component_blocked:trinket", report["blockers"])
        self.assertEqual(report["categories"]["candidate_trinkets"]["pool_status"], "unverified_candidate")
        self.assertNotIn("trinkets", report["categories"])
        self.assertEqual(self.ruleset["active"]["lesser_trinket_ids"], [])


if __name__ == "__main__":
    unittest.main()
