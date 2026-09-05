"""The advisory boundary cannot bypass the current fixture or finite clock."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from recommend_recruit import recommend
from bg_ai.hsbrsim_adapter import FIXTURE_SCOPE
from bg_ai.learning import Ranker
from bg_ai.recruit_features import RECRUIT_V2_FEATURE_NAMES
from bg_ai.recruiting import ruleset_digest
from bg_ai.turn_budget import TimingProfile, TurnBudget


class RecruitAdvisorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.profile = TimingProfile(1, 5000, {"buy": 250, "play": 500, "choose": 1000})
        self.ruleset = {"name": "synthetic-advisory-scope-test"}
        self.checkpoint = Path(self.tmp.name) / "model.json"
        model = Ranker(RECRUIT_V2_FEATURE_NAMES, hidden=2)
        model.metadata.update(recruit_feature_version="recruit-visible-action-v2-choice-zone",
                              timing_profile_sha256=self.profile.fingerprint)
        model.save(self.checkpoint)
        self.observation = {"player_id": 0, "turn": 1,
            "public_state": {"scope": FIXTURE_SCOPE, "players": [{"player_id": 0, "hero_id": "TB_BaconShop_HERO_34"}]},
            "private_state": {"gold": 3, "tavern_tier": 1, "board": [], "hand": [],
                "shop": [{"entity_id": "shop0", "card_id": "BGS_119", "card_type": "minion", "cost": 3, "attack": 2, "health": 1}]},
            "legal_actions": [{"kind": "buy", "entity_id": "shop0"}, {"kind": "end_turn"}],
            "ruleset_sha256": ruleset_digest(self.ruleset), "time_remaining_ms": 30000,
            "action_budget": TurnBudget(self.profile, 30000).snapshot()}

    def test_frozen_version_selects_a_legal_action_without_executing(self):
        result = recommend(self.observation, self.checkpoint, self.ruleset, self.profile)
        self.assertIn(result["chosen_action"]["kind"], ("buy", "end_turn"))
        self.assertFalse(result["action_executed"])
        self.assertFalse(result["full_game_policy"])
        self.assertEqual(result["remaining_budget_before_action"]["actions_used"], 0)

    def test_unaffordable_advertised_action_is_rejected(self):
        self.observation["action_budget"].update(remaining_ms=500, remaining_actions=1)
        with self.assertRaisesRegex(ValueError, "unaffordable action"):
            recommend(self.observation, self.checkpoint, self.ruleset, self.profile)

    def test_stale_ruleset_and_untrained_hero_fail(self):
        with self.assertRaisesRegex(ValueError, "another ruleset"):
            recommend(self.observation, self.checkpoint, {"name": "changed"}, self.profile)
        self.observation["public_state"]["players"][0]["hero_id"] = "TB_BaconShop_HERO_39"
        with self.assertRaisesRegex(ValueError, "only with Patchwerk"):
            recommend(self.observation, self.checkpoint, self.ruleset, self.profile)

    def test_foreign_card_cannot_expand_the_fixture(self):
        self.observation["private_state"]["shop"][0]["card_id"] = "unvalidated-card"
        with self.assertRaisesRegex(ValueError, "outside the trained fixture"):
            recommend(self.observation, self.checkpoint, self.ruleset, self.profile)

if __name__ == "__main__":
    unittest.main()
