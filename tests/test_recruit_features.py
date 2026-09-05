import unittest
from dataclasses import replace
import numpy as np
from bg_ai.recruit_features import RECRUIT_FEATURE_NAMES, encode_legal_actions, encode_recruit_action
from bg_ai.recruiting import RecruitAction, RecruitObservation
from bg_ai.turn_budget import TimingProfile, TurnBudget

class RecruitFeatureTests(unittest.TestCase):
    def observation(self):
        profile = TimingProfile(1, 5000, {"buy": 250, "play": 500})
        buy = RecruitAction("buy", entity_id="shop0")
        return RecruitObservation(0, 1, {}, {"gold": 3, "tavern_tier": 1,
            "shop": [{"entity_id": "shop0", "cardId": "old-card", "name": "Old name",
                "attack": 3, "health": 1, "text": "<b>Battlecry:</b> Gain 1 Gold.", "cost": 3}]},
            (buy, RecruitAction("end_turn")), "test", 30000, TurnBudget(profile, 30000).snapshot())

    def test_card_identity_changes_do_not_change_features(self):
        obs = self.observation()
        a = encode_legal_actions(obs)
        card = dict(obs.private_state["shop"][0], entity_id="fresh0", cardId="next-rotation", name="New name")
        fresh = replace(obs, private_state=dict(obs.private_state, shop=[card]),
            legal_actions=(RecruitAction("buy", entity_id="fresh0"), RecruitAction("end_turn")))
        np.testing.assert_array_equal(a, encode_legal_actions(fresh))

    def test_timing_changes_inputs(self):
        obs = self.observation()
        faster = replace(obs, action_budget=dict(obs.action_budget, remaining_ms=3000, remaining_actions=3))
        a, b = encode_legal_actions(obs), encode_legal_actions(faster)
        self.assertEqual(a.shape, (2, len(RECRUIT_FEATURE_NAMES)))
        self.assertFalse(np.array_equal(a, b))
        idx = RECRUIT_FEATURE_NAMES.index("recruit.context.remaining_actions")
        self.assertAlmostEqual(b[0, idx], 3 / 60)

    def test_missing_budget_and_illegal_action_fail(self):
        obs = self.observation()
        with self.assertRaisesRegex(ValueError, "budget"):
            encode_legal_actions(replace(obs, action_budget=None))
        with self.assertRaisesRegex(ValueError, "legal mask"):
            encode_recruit_action(obs, RecruitAction("sell", entity_id="shop0"))

    def test_hidden_or_future_metadata_has_no_effect(self):
        obs = self.observation()
        extra = replace(obs, private_state=dict(obs.private_state, future_shop=[{"attack": 999}], rng_state=35),
                        public_state={"hidden_opponent": [{"attack": 999}], "future_rollout_score": 1.0})
        np.testing.assert_array_equal(encode_legal_actions(obs), encode_legal_actions(extra))

    def test_target_attributes_and_spell_text_are_encoded(self):
        obs = self.observation()
        spell = {"entity_id": "hand0", "type": "SPELL", "text": "Give a minion +2/+2.", "cost": 1}
        body = {"entity_id": "board0", "attack": 1, "health": 3, "text": ""}
        cast = RecruitAction("cast_spell", "hand0", "board0")
        obs = replace(obs, private_state=dict(obs.private_state, hand=[spell], board=[body]), legal_actions=(cast,))
        vector = encode_recruit_action(obs, cast)
        self.assertEqual(vector[RECRUIT_FEATURE_NAMES.index("recruit.context.action_is_spell")], 1)
        self.assertGreater(vector[RECRUIT_FEATURE_NAMES.index("recruit.target_card.health_log")], 0)

    def test_scalar_choice_options_have_distinct_semantics(self):
        obs = self.observation()
        actions = (RecruitAction("choose", choice_index=0), RecruitAction("choose", choice_index=1))
        obs = replace(obs, private_state=dict(obs.private_state, pending_choice={"options": ["atk", "health"]}), legal_actions=actions)
        vectors = encode_legal_actions(obs)
        indices = [i for i, name in enumerate(RECRUIT_FEATURE_NAMES) if ".choice_option." in name]
        self.assertFalse(np.array_equal(vectors[0, indices], vectors[1, indices]))

    def test_equal_stats_owned_and_shop_targets_are_distinguishable(self):
        obs = self.observation()
        board = dict(obs.private_state["shop"][0], entity_id="board0")
        actions = (RecruitAction("choose", target_id="board0", choice_index=0),
                   RecruitAction("choose", target_id="shop0", choice_index=1))
        obs = replace(obs, private_state=dict(obs.private_state, board=[board],
            pending_choice={"options": [board, obs.private_state["shop"][0]]}), legal_actions=actions)
        vectors = encode_legal_actions(obs)
        board_idx = RECRUIT_FEATURE_NAMES.index("recruit.context.target_board")
        self.assertEqual(vectors[0, board_idx], 1)
        self.assertEqual(vectors[1, board_idx], 0)

    def test_board_order_and_spell_scaling_remain_visible(self):
        obs = self.observation()
        first = dict(obs.private_state["shop"][0], entity_id="board0")
        second = dict(first, entity_id="board1", attack=1, health=8, activate_used=True)
        obs = replace(obs, private_state=dict(obs.private_state, board=[first, second]))
        reverse = replace(obs, private_state=dict(obs.private_state, board=[second, first]))
        buffed = replace(obs, private_state=dict(obs.private_state, counters={"TAVERN_SPELL_EXTRA_ATK": 2}))
        self.assertFalse(np.array_equal(encode_legal_actions(obs), encode_legal_actions(reverse)))
        self.assertFalse(np.array_equal(encode_legal_actions(obs), encode_legal_actions(buffed)))

    def test_raw_card_id_choice_requires_semantic_resolution(self):
        obs = self.observation()
        action = RecruitAction("choose", choice_index=0)
        obs = replace(obs, private_state=dict(obs.private_state, pending_choice={"options": ["BG20_100"]}), legal_actions=(action,))
        with self.assertRaisesRegex(ValueError, "semantic card attributes"):
            encode_recruit_action(obs, action)

    def test_buy_needs_time_for_explicit_play(self):
        obs = self.observation()
        card = dict(obs.private_state["shop"][0], card_type="minion")
        obs = replace(obs, private_state=dict(obs.private_state, shop=[card]),
            action_budget=dict(obs.action_budget, remaining_ms=2000, remaining_actions=2))
        vector = encode_recruit_action(obs, obs.legal_actions[0])
        self.assertEqual(vector[RECRUIT_FEATURE_NAMES.index("recruit.context.timing_costs_known")], 1)
        self.assertEqual(vector[RECRUIT_FEATURE_NAMES.index("recruit.context.immediate_card_completion_fits")], 0)

    def test_unknown_legacy_profile_marks_costs_missing(self):
        obs = self.observation()
        budget = dict(obs.action_budget, profile_sha256="unknown")
        budget.pop("action_cost_ms")
        obs = replace(obs, action_budget=budget)
        vector = encode_recruit_action(obs, obs.legal_actions[0])
        self.assertEqual(vector[RECRUIT_FEATURE_NAMES.index("recruit.context.timing_costs_known")], 0)

    def test_unknown_visible_entity_fails(self):
        obs = self.observation()
        fake = RecruitAction("buy", "missing")
        with self.assertRaisesRegex(ValueError, "absent from visible"):
            encode_recruit_action(replace(obs, legal_actions=(fake,)), fake)

if __name__ == "__main__":
    unittest.main()
