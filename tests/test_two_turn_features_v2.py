"""Lifetime schema, checkpoint transfer and actual visible-engine regressions."""
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from bg_ai.learning import Dataset, Ranker, Scenario
from bg_ai.recruiting import RecruitAction, RecruitObservation
from bg_ai.turn_budget import TimingProfile, TurnBudget
from bg_ai.two_turn_features import (
    TWO_TURN_FEATURE_NAMES as V1_NAMES, encode_two_turn_actions as encode_v1,
)
from bg_ai.two_turn_features_v2 import (
    LIFETIME_METADATA_VERSION, TWO_TURN_FEATURE_NAMES, enrich_two_turn_observation,
    encode_two_turn_action, encode_two_turn_actions, two_turn_schema_id,
    warm_start_two_turn_ranker,
)

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = Path(os.environ.get("HSBRSIM_ROOT", str(ROOT.parent / "research/HSBRSIM")))


def observation():
    card = {"entity_id": "owned", "card_id": "BG29_611", "card_type": "minion",
            "attack": 4, "health": 2, "tier": 1, "text": "", "cost": 3,
            "enchantments": [{"attack": 2, "health": 0, "temporary": True,
                              "source_id": "BG23_000t", "blood_gem": False}],
            "temporary_spellcraft": False}
    profile = TimingProfile.load(ROOT / "config/turn-budget.json")
    return RecruitObservation(0, 1,
        {"scope": "current-two-turn-tier1-opening-v1",
         "valid_tribes": ["BEAST", "DEMON", "MECH", "NAGA", "PIRATE"]},
        {"gold": 3, "tavern_tier": 1, "board": [card], "hand": [], "shop": [],
         "deferred_next_turn_gold": 0},
        (RecruitAction("sell", "owned"), RecruitAction("end_turn")),
        "test", 60000, TurnBudget(profile, 60000).snapshot())


class LifetimeFeatureTests(unittest.TestCase):
    def test_exact_prefix_and_single_action_mask(self):
        raw = observation()
        enriched = enrich_two_turn_observation(raw)
        features = encode_two_turn_actions(enriched)
        self.assertEqual(features.shape, (2, 1311))
        self.assertEqual(TWO_TURN_FEATURE_NAMES[:1129], V1_NAMES)
        np.testing.assert_array_equal(features[:, :1129], encode_v1(raw))
        np.testing.assert_array_equal(features[0], encode_two_turn_action(enriched, enriched.legal_actions[0]))
        np.testing.assert_array_equal(features[[1]], encode_two_turn_actions(enriched, [enriched.legal_actions[1]]))
        with self.assertRaisesRegex(ValueError, "legal mask"):
            encode_two_turn_actions(enriched, [RecruitAction("buy", "owned")])
        with self.assertRaisesRegex(ValueError, "No affordable legal actions"):
            encode_two_turn_actions(enriched, [])
        self.assertEqual(len(set(TWO_TURN_FEATURE_NAMES)), len(TWO_TURN_FEATURE_NAMES))
        self.assertEqual(len(two_turn_schema_id()), 64)

    def test_enrichment_is_nonmutating_and_ignores_hidden_data(self):
        raw = observation()
        before = deepcopy(raw)
        enriched = enrich_two_turn_observation(raw)
        enriched.private_state["board"][0]["enchantments"][0]["attack"] = 900
        self.assertEqual(raw, before)
        changed = replace(raw, private_state=dict(raw.private_state, rng_state=999,
            future_shop=[{"attack": 900}], combat_receipt={"winner": 7}),
            public_state=dict(raw.public_state, opponent_hand=[{"attack": 500}]))
        np.testing.assert_array_equal(encode_two_turn_actions(enrich_two_turn_observation(raw)),
                                      encode_two_turn_actions(enrich_two_turn_observation(changed)))
        self.assertIsNone(enrich_two_turn_observation(None))

    def test_missing_unknown_and_stale_metadata_fail_closed(self):
        raw = observation()
        with self.assertRaisesRegex(ValueError, "enrich observation"):
            encode_two_turn_actions(raw)
        for field in ("enchantments", "temporary_spellcraft"):
            changed = deepcopy(raw)
            del changed.private_state["board"][0][field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                enrich_two_turn_observation(changed)
        invalid = [dict(temporary="true"), dict(attack=float("inf")),
                   dict(health=True), dict(attack=10**1000),
                   dict(source_id="unknown-future-enchantment"),
                   dict(keywords=["DIVINE_SHIELD"])]
        for fields in invalid:
            changed = deepcopy(raw)
            changed.private_state["board"][0]["enchantments"][0].update(fields)
            with self.subTest(fields=list(fields)), self.assertRaises(ValueError):
                enrich_two_turn_observation(changed)
        changed = enrich_two_turn_observation(raw)
        changed.private_state["board"][0]["enchantments"][0]["attack"] = 4
        with self.assertRaisesRegex(ValueError, "stale"):
            encode_two_turn_actions(changed)
        for zone in ("board", "hand", "shop"):
            changed = deepcopy(raw)
            del changed.private_state[zone]
            with self.subTest(zone=zone), self.assertRaisesRegex(ValueError, "explicit bounded"):
                enrich_two_turn_observation(changed)

    def test_ordered_hand_shop_slots_and_candidate_lifetimes(self):
        raw = observation()
        changed = deepcopy(raw)
        temporary = changed.private_state["board"][0]
        permanent = deepcopy(temporary)
        permanent.update(entity_id="shop-permanent")
        permanent["enchantments"][0].update(temporary=False, source_id="")
        changed.private_state["shop"] = [permanent]
        changed.private_state["hand"] = [dict(entity_id="spell", card_id="BG23_000t",
            card_type="spell", text="Give a minion +2 Attack until next turn.",
            enchantments=[], temporary_spellcraft=True)]
        changed = replace(changed, legal_actions=(RecruitAction("sell", "owned"),
            RecruitAction("buy", "shop-permanent"),
            RecruitAction("cast_spell", "spell", target_id="owned"), RecruitAction("end_turn")))
        features = encode_two_turn_actions(enrich_two_turn_observation(changed))
        def at(view, field):
            return TWO_TURN_FEATURE_NAMES.index(f"recruit.two_turn.lifetime.{view}.{field}")
        self.assertEqual(features[0, at("board_slot0", "temporary_attack")], 0.1)
        self.assertEqual(features[0, at("board_slot0", "expires_next_recruit")], 1)
        self.assertEqual(features[1, at("shop_slot0", "permanent_attack")], 0.1)
        self.assertEqual(features[1, at("action_card", "expires_next_recruit")], 0)
        self.assertEqual(features[2, at("hand_slot0", "discard_at_recruit_end")], 1)
        self.assertEqual(features[2, at("target_card", "temporary_attack")], 0.1)
        self.assertEqual(features[3, at("action_card", "present")], 0)

    def test_migration_preserves_predictions_all_optimizer_arrays_rng_and_file_roundtrip(self):
        raw = observation()
        features = encode_v1(raw)
        source = Ranker(V1_NAMES, hidden=4, seed=71)
        source.fit(Dataset([Scenario("lifetime-migration", "train", features,
            np.array([0.2, 0.8]), {})], V1_NAMES), epochs=3)
        original = deepcopy(source)
        target = warm_start_two_turn_ranker(source)
        new_features = encode_two_turn_actions(enrich_two_turn_observation(raw))
        np.testing.assert_allclose(target.predict(new_features), source.predict(features),
                                   rtol=1e-13, atol=1e-13)
        self.assertEqual(target.rng.bit_generator.state, source.rng.bit_generator.state)
        self.assertEqual(target.step, source.step)
        for group in ("params", "m", "v"):
            for key in ("w1", "b1", "w2"):
                expected, actual = getattr(source, group)[key], getattr(target, group)[key]
                np.testing.assert_array_equal(actual[:1129] if key == "w1" else actual, expected)
                if key == "w1":
                    self.assertFalse(actual[1129:].any())
        np.testing.assert_array_equal(target.mean[:1129], source.mean)
        np.testing.assert_array_equal(target.scale[:1129], source.scale)
        self.assertFalse(target.mean[1129:].any())
        self.assertTrue((target.scale[1129:] == 1).all())
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "migrated.json"
            target.save(path)
            restored = Ranker.load(path, TWO_TURN_FEATURE_NAMES)
            np.testing.assert_array_equal(restored.predict(new_features), target.predict(new_features))
            self.assertEqual(restored.rng.bit_generator.state, target.rng.bit_generator.state)
        target.params["w1"][0, 0] += 1
        np.testing.assert_array_equal(source.params["w1"], original.params["w1"])
        self.assertEqual(source.metadata, original.metadata)
        for names in (V1_NAMES[::-1], V1_NAMES[:-1], TWO_TURN_FEATURE_NAMES):
            with self.subTest(count=len(names)), self.assertRaisesRegex(ValueError, "exact frozen 1129"):
                warm_start_two_turn_ranker(Ranker(names, hidden=2))


@unittest.skipUnless((EXTERNAL / "hsrl2/game.py").is_file(), "Pinned HSBRSIM checkout required")
class ActualEngineLifetimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from bg_ai.hsbrsim_opening import build_opening_database
        cls.tribes = ("BEAST", "DEMON", "MECH", "NAGA", "PIRATE")
        cls.current = build_opening_database(EXTERNAL, valid_tribes=cls.tribes)
        cls.rules = json.loads((ROOT / "data/ruleset.json").read_text())
        cls.profile = TimingProfile.load(ROOT / "config/turn-budget.json")

    def make(self):
        from bg_ai.hsbrsim_opening import OpeningFixtureSpec
        from bg_ai.opening_transition import TwoTurnOpeningRecruitEngine, TimedTwoTurnOpeningEngine
        fixture = OpeningFixtureSpec(valid_tribes=self.tribes,
            initial_boards=(("BG29_611", "BG23_000"),) + ((),) * 7,
            initial_hands=(("BG28_897", "BG31_880", "BG20_GEM"),) + ((),) * 7)
        engine = TwoTurnOpeningRecruitEngine(self.current.db, self.rules,
            timer_ms=lambda player, turn: 60000, provenance=self.current.provenance, fixture=fixture)
        timed = TimedTwoTurnOpeningEngine.for_fixture(engine, self.profile, self.rules)
        return timed, timed.reset(seed=11)

    def cast(self, timed, obs, cid, target):
        card = next(c for c in obs.private_state["hand"] if c["card_id"] == cid)
        obs = timed.step(next(a for a in obs.legal_actions if a.kind == "cast_spell"
                             and a.entity_id == card["entity_id"]))
        if cid == "BG31_880":
            obs = timed.step(next(a for a in obs.legal_actions if a.kind == "choose" and a.choice_index == 0))
        return timed.step(next(a for a in obs.legal_actions if a.kind == "choose" and a.target_id == target))

    def test_real_casts_have_same_current_stats_but_distinct_lifetime_vectors(self):
        temporary, ot = self.make()
        permanent, op = self.make()
        target = ot.private_state["board"][0]["entity_id"]
        ot = self.cast(temporary, ot, "BG23_000t", target)
        ot = self.cast(temporary, ot, "BG28_897", target)  # +2 temporary/+2 permanent Attack, +2 Health
        op = self.cast(permanent, op, "BG31_880", target)
        op = self.cast(permanent, op, "BG20_GEM", target)  # +4 permanent Attack/+2 permanent Health
        tc, pc = ot.private_state["board"][0], op.private_state["board"][0]
        self.assertEqual((tc["attack"], tc["health"]), (pc["attack"], pc["health"]))
        self.assertTrue(any(e["temporary"] for e in tc["enchantments"]))
        self.assertFalse(any(e["temporary"] for e in pc["enchantments"]))
        # Isolate the actual same-stats card views in identical visible context.
        # Whole trajectories differ in leftover hands, and are not claimed equal.
        projected = replace(ot, private_state=dict(ot.private_state,
            board=[pc, *ot.private_state["board"][1:]]))
        end = RecruitAction("end_turn")
        np.testing.assert_array_equal(encode_v1(ot, [end]), encode_v1(projected, [end]))
        tf = encode_two_turn_actions(enrich_two_turn_observation(ot), [end])
        pf = encode_two_turn_actions(enrich_two_turn_observation(projected), [end])
        self.assertFalse(np.array_equal(tf, pf))
        key = TWO_TURN_FEATURE_NAMES.index("recruit.two_turn.lifetime.board_slot0.temporary_attack")
        self.assertEqual(tf[0, key], 0.1)
        self.assertEqual(pf[0, key], 0)

    def test_replay_fork_preserves_actual_buffs_without_mutating_parent(self):
        timed, obs = self.make()
        target = obs.private_state["board"][0]["entity_id"]
        obs = self.cast(timed, obs, "BG23_000t", target)
        before = deepcopy(obs)
        expected = encode_two_turn_actions(enrich_two_turn_observation(obs))
        branch = timed.fork()
        branched = enrich_two_turn_observation(branch._visible)
        np.testing.assert_array_equal(encode_two_turn_actions(branched), expected)
        branch.engine.game.heroes[0].board[0]._buffs[0].atk += 10
        branched.private_state["board"][0]["enchantments"][0]["attack"] += 20
        self.assertEqual(obs, before)
        np.testing.assert_array_equal(encode_two_turn_actions(enrich_two_turn_observation(timed._visible)), expected)
        self.assertEqual(timed.engine.game.heroes[0].board[0]._buffs[0].atk, 2)


if __name__ == "__main__":
    unittest.main()
