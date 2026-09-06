import unittest
from dataclasses import replace

import numpy as np

from bg_ai.learning import Dataset, Ranker, Scenario
from bg_ai.recruit_features import (
    RECRUIT_FEATURE_NAMES, RECRUIT_V2_FEATURE_NAMES,
    encode_legal_actions, encode_legal_actions_v2,
)
from bg_ai.recruiting import RecruitAction, RecruitObservation
from bg_ai.turn_budget import TimingProfile, TurnBudget
from bg_ai.two_turn_features import (
    LOBBY_TRIBES, TWO_TURN_FEATURE_NAMES, encode_two_turn_action,
    encode_two_turn_actions, two_turn_schema_id, warm_start_two_turn_ranker,
)


class TwoTurnFeatureTests(unittest.TestCase):
    def observation(self):
        profile = TimingProfile(1, 5000, {"buy": 250, "play": 500})
        return RecruitObservation(0, 1,
            {"valid_tribes": ["BEAST", "DEMON", "MECH", "NAGA", "PIRATE"]},
            {"gold": 3, "tavern_tier": 1, "frozen": True,
             "deferred_next_turn_gold": 2,
             "shop": [{"entity_id": "shop0", "card_id": "old-card", "name": "Old name",
                       "attack": 3, "health": 1, "cost": 3,
                       "text": "Battlecry: Gain 1 Gold next turn."}]},
            (RecruitAction("buy", "shop0"), RecruitAction("end_turn")),
            "test", 30000, TurnBudget(profile, 30000).snapshot())

    def test_new_context_distinguishes_pool_and_exact_promised_amount(self):
        observation = self.observation()
        baseline = encode_two_turn_actions(observation)
        changed = replace(observation,
            public_state={"valid_tribes": ["BEAST", "DEMON", "MECH", "MURLOC", "PIRATE"]},
            private_state=dict(observation.private_state, deferred_next_turn_gold=4))
        altered = encode_two_turn_actions(changed)
        np.testing.assert_array_equal(baseline[:, :len(RECRUIT_FEATURE_NAMES)],
                                      encode_legal_actions(observation))
        for tribe in LOBBY_TRIBES:
            index = TWO_TURN_FEATURE_NAMES.index("recruit.two_turn.lobby_" + tribe.lower())
            self.assertEqual(baseline[0, index], tribe in observation.public_state["valid_tribes"])
        delta = altered[0] - baseline[0]
        self.assertEqual(np.count_nonzero(delta), 3)
        self.assertAlmostEqual(delta[-1], 0.2)
        self.assertEqual(baseline[0, RECRUIT_FEATURE_NAMES.index("recruit.context.frozen")], 1)

    def test_pool_order_and_hidden_identifiers_do_not_change_inputs(self):
        observation = self.observation()
        changed = replace(observation,
            public_state=dict(observation.public_state,
                valid_tribes=list(reversed(observation.public_state["valid_tribes"])),
                future_rollout_score=1, opponent_hand=[{"attack": 999}]),
            private_state=dict(observation.private_state, rng_state=12345,
                future_shop=[{"attack": 999}], combat_receipt={"winner": 0},
                shop=[dict(observation.private_state["shop"][0],
                           entity_id="new-entity", card_id="next-rotation", name="New name")]),
            legal_actions=(RecruitAction("buy", "new-entity"), RecruitAction("end_turn")))
        np.testing.assert_array_equal(encode_two_turn_actions(observation),
                                      encode_two_turn_actions(changed))

    def test_missing_or_invalid_pool_fails_closed(self):
        observation = self.observation()
        invalid = [None, [], "BEAST", ["BEAST"] * 5,
                   ["BEAST", "DEMON", "MECHANICAL", "NAGA", "PIRATE"],
                   ["BEAST", "DEMON", "MECH", "NAGA", "ALL"],
                   ["BEAST", "DEMON", "MECH", "NAGA", []]]
        for tribes in invalid:
            with self.subTest(tribes=tribes), self.assertRaisesRegex(ValueError, "valid_tribes"):
                encode_two_turn_actions(replace(observation, public_state={"valid_tribes": tribes}))
        with self.assertRaisesRegex(ValueError, "valid_tribes"):
            encode_two_turn_actions(replace(observation, public_state={}))

    def test_missing_or_invalid_deferred_gold_fails_closed(self):
        observation = self.observation()
        for amount in (None, True, -1, 1.5, "2", float("nan"), float("inf"), 10**1000):
            with self.subTest(amount=type(amount)), self.assertRaisesRegex(ValueError, "deferred_next_turn_gold"):
                encode_two_turn_actions(replace(observation,
                    private_state=dict(observation.private_state, deferred_next_turn_gold=amount)))
        state = dict(observation.private_state)
        del state["deferred_next_turn_gold"]
        with self.assertRaisesRegex(ValueError, "deferred_next_turn_gold"):
            encode_two_turn_actions(replace(observation, private_state=state))

    def test_legal_mask_and_single_action_encoding_are_preserved(self):
        observation = self.observation()
        baseline = encode_two_turn_actions(observation)
        np.testing.assert_array_equal(baseline[0], encode_two_turn_action(observation, observation.legal_actions[0]))
        np.testing.assert_array_equal(baseline[[1]], encode_two_turn_actions(observation, [observation.legal_actions[1]]))
        with self.assertRaisesRegex(ValueError, "legal mask"):
            encode_two_turn_actions(observation, [RecruitAction("sell", "shop0")])
        with self.assertRaisesRegex(ValueError, "No affordable legal actions"):
            encode_two_turn_actions(observation, [])
        self.assertEqual(len(two_turn_schema_id()), 64)
        self.assertEqual(len(set(TWO_TURN_FEATURE_NAMES)), len(TWO_TURN_FEATURE_NAMES))

    def test_warm_start_preserves_trained_predictions_optimizer_and_source(self):
        observation = self.observation()
        for names, encoder in ((RECRUIT_V2_FEATURE_NAMES, encode_legal_actions_v2),
                               (RECRUIT_FEATURE_NAMES, encode_legal_actions)):
            with self.subTest(schema=len(names)):
                features = encoder(observation)
                source = Ranker(names, hidden=4, seed=3)
                source.fit(Dataset([Scenario("training-fixture", "train", features,
                                             np.array([0.25, 0.75]), {})], names), epochs=2)
                prediction = source.predict(features).copy()
                state = source.rng.bit_generator.state
                source_weights = source.params["w1"].copy()
                target = warm_start_two_turn_ranker(source)
                np.testing.assert_allclose(target.predict(encode_two_turn_actions(observation)),
                                           prediction, rtol=1e-13, atol=1e-13)
                self.assertEqual(target.step, source.step)
                self.assertEqual(target.rng.bit_generator.state, state)
                self.assertNotIn("feature_schema_transitions", source.metadata)
                old_indices = [TWO_TURN_FEATURE_NAMES.index(name) for name in names]
                for moment in ("m", "v"):
                    np.testing.assert_array_equal(getattr(target, moment)["w1"][old_indices],
                                                  getattr(source, moment)["w1"])
                target.params["w1"][0, 0] += 1
                np.testing.assert_array_equal(source.params["w1"], source_weights)
        with self.assertRaisesRegex(ValueError, "exact frozen"):
            warm_start_two_turn_ranker(Ranker(("unknown",), hidden=2))


if __name__ == "__main__":
    unittest.main()
