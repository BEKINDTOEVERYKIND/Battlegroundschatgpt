"""Real current-card command graphs and finite-budget conformance."""
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import unittest

from bg_ai.choice_timing import (
    ChoiceTimedTwoTurnOpeningEngine, CHOICE_TIMING_VERSION,
    opening_choice_completion,
)
from bg_ai.hsbrsim_adapter import UnsupportedRecruitTransition
from bg_ai.hsbrsim_opening import OpeningFixtureSpec, TRIBES, TIER1_MINIONS, build_opening_database
from bg_ai.opening_transition import TwoTurnOpeningRecruitEngine, TimedTwoTurnOpeningEngine
from bg_ai.recruiting import RecruitAction, RecruitCoverageError
from bg_ai.turn_budget import TimingProfile

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = Path(os.environ.get("HSBRSIM_ROOT", str(ROOT.parent / "research/HSBRSIM")))


def zones(first):
    return (tuple(first),) + ((),) * 7


@unittest.skipUnless((EXTERNAL / "hsrl2/game.py").is_file(), "Pinned HSBRSIM checkout required")
class ChoiceTimingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = json.loads((ROOT / "data/ruleset.json").read_text())
        cls.cards = {c["id"]: c for c in json.loads((ROOT / "data/reference_cards.json").read_text())}
        cls.profile = TimingProfile.load(ROOT / "config/turn-budget.json")
        cls.databases = {}

    def make(self, *, board=(), hand=(), usable_ms=20000, timer=None, seed=3):
        required = {self.cards[cid]["races"][0] for cid in board + hand if self.cards[cid].get("races")}
        for tribe in sorted(TRIBES):
            if len(required) >= 5:
                break
            required.add(tribe)
        spec = OpeningFixtureSpec(valid_tribes=tuple(sorted(required)),
                                  initial_boards=zones(board), initial_hands=zones(hand))
        if spec.valid_tribes not in self.databases:
            self.databases[spec.valid_tribes] = build_opening_database(EXTERNAL, valid_tribes=spec.valid_tribes)
        current = self.databases[spec.valid_tribes]
        raw = TwoTurnOpeningRecruitEngine(current.db, self.rules,
            timer_ms=timer or (lambda player, turn: usable_ms + self.profile.reserve_ms),
            provenance=current.provenance, fixture=spec)
        timed = ChoiceTimedTwoTurnOpeningEngine.for_fixture(raw, self.profile, self.rules)
        return timed, timed.reset(seed=seed)

    def action(self, observation, kind, cid=None, **attrs):
        ids = {c["entity_id"] for zone in ("board", "hand", "shop")
               for c in observation.private_state[zone] if c["card_id"] == cid}
        return next(a for a in observation.legal_actions if a.kind == kind
                    and (cid is None or a.entity_id in ids)
                    and all(getattr(a, key) == value for key, value in attrs.items()))

    def test_targeted_spells_fit_only_with_explicit_target_command(self):
        for cid in ("BG28_503", "BG28_897", "BG20_GEM", "BG23_000t"):
            with self.subTest(card=cid):
                t, o = self.make(hand=(cid,), usable_ms=3499)
                raw_cast = self.action(t.engine.observe(), "cast_spell", cid)
                self.assertNotIn(raw_cast, o.legal_actions)
                with self.assertRaises(ValueError):
                    t.step(raw_cast)
                self.assertEqual(t._budgets[(0, 1)].spent_ms, 0)
                t, o = self.make(hand=(cid,), usable_ms=3500)
                o = t.step(self.action(o, "cast_spell", cid))
                self.assertEqual(o.action_budget["reserved_choice_commands"], 1)
                self.assertEqual(o.action_budget["charged_ms"], 1500)
                self.assertEqual(o.legal_actions, t.engine.observe().legal_actions)
                self.assertTrue(all(a.kind == "choose" for a in o.legal_actions))
                t.step(o.legal_actions[0])
                budget = t._budgets[(0, 1)]
                self.assertEqual((budget.actions_used, budget.spent_ms, budget.remaining_ms), (2, 3500, 0))
                self.assertFalse(t.engine.game.pending_choices)

    def test_alliance_reserves_both_choices_and_fork_preserves_remaining_sequence(self):
        t, o = self.make(hand=("BG31_880",), usable_ms=5499)
        self.assertNotIn(self.action(t.engine.observe(), "cast_spell", "BG31_880"), o.legal_actions)
        t, o = self.make(hand=("BG31_880",), usable_ms=5500)
        o = t.step(self.action(o, "cast_spell", "BG31_880"))
        self.assertEqual(o.action_budget["reserved_choice_commands"], 2)
        self.assertEqual(o.private_state["pending_choice"]["kind"], "choose_one")
        branch = t.fork()
        self.assertEqual(branch._visible.action_budget, o.action_budget)
        self.assertEqual(branch._visible.private_state, o.private_state)
        o = t.step(o.legal_actions[0])
        self.assertEqual(o.action_budget["reserved_choice_commands"], 1)
        self.assertEqual(o.action_budget["reserved_choice_ms"], 2000)
        self.assertEqual(o.legal_actions, t.engine.observe().legal_actions)
        self.assertEqual(branch._visible.action_budget["reserved_choice_commands"], 2)
        second_branch = t.fork()
        self.assertEqual(second_branch._visible.action_budget, o.action_budget)
        t.step(o.legal_actions[0])
        budget = t._budgets[(0, 1)]
        self.assertEqual((budget.actions_used, budget.spent_ms), (3, 5500))
        self.assertEqual(second_branch._budgets[(0, 1)].spent_ms, 3500)
        self.assertFalse(t.engine.game.pending_choices)

    def test_sprout_discover_cost_and_options_are_preserved(self):
        t, o = self.make(hand=("BG33_101",), usable_ms=3499)
        self.assertNotIn(self.action(t.engine.observe(), "cast_spell", "BG33_101"), o.legal_actions)
        t, o = self.make(hand=("BG33_101",), usable_ms=3500)
        o = t.step(self.action(o, "cast_spell", "BG33_101"))
        self.assertEqual(o.private_state["pending_choice"]["kind"], "discover_minion")
        self.assertEqual(o.legal_actions, t.engine.observe().legal_actions)
        expected = o.private_state["pending_choice"]["options"][0]["card_id"]
        t.step(o.legal_actions[0])
        self.assertEqual(t.engine.game.heroes[0].hand[0].card_id, expected)
        self.assertEqual(t._budgets[(0, 1)].actions_used, 2)
        self.assertFalse(t.engine.game.heroes[0].board)

    def test_visible_plan_does_not_consult_hidden_engine_or_pool(self):
        t, o = self.make(hand=("BG33_101",))
        action = self.action(o, "cast_spell", "BG33_101")
        visible_only = replace(o, public_state={}, private_state={"hand": deepcopy(o.private_state["hand"])})
        plan = opening_choice_completion(visible_only, action)
        self.assertEqual(plan.kinds, ("discover_minion",))
        self.assertTrue(plan.optional)

    def test_zero_option_discover_releases_upper_bound_without_free_or_fake_command(self):
        t, o = self.make(hand=("BG33_101",), usable_ms=3500)
        # Deliberate edge-state conformance fixture: the policy still receives
        # the same observation and mask regardless of hidden pool availability.
        before = o.legal_actions
        for cid in t.engine.allowed_minions:
            t.engine.game.minion_pool._available[cid] = 0
        self.assertEqual(t._accept(t.engine.observe()).legal_actions, before)
        o = t.step(self.action(o, "cast_spell", "BG33_101"))
        self.assertEqual(o.action_budget["reserved_choice_commands"], 0)
        self.assertEqual(o.action_budget["actions_used"], 1)
        self.assertEqual(o.action_budget["charged_ms"], 1500)
        self.assertFalse(t.engine.game.pending_choices)

    def test_activate_reserves_target_and_keeps_all_original_targets(self):
        board = ("BG36_345", "BG36_200", "BG29_611")
        t, o = self.make(board=board, usable_ms=3499)
        self.assertNotIn(self.action(t.engine.observe(), "activate", "BG36_345"), o.legal_actions)
        t, o = self.make(board=board, usable_ms=3500)
        guard = o.private_state["board"][0]["entity_id"]
        o = t.step(self.action(o, "activate", "BG36_345"))
        self.assertEqual(o.legal_actions, t.engine.observe().legal_actions)
        self.assertEqual(len(o.legal_actions), 2)
        self.assertNotIn(guard, {a.target_id for a in o.legal_actions})
        t.step(o.legal_actions[0])
        self.assertEqual(t._budgets[(0, 1)].spent_ms, 3500)

    def test_every_current_tier1_minion_play_needs_no_forced_choice(self):
        for cid in sorted(TIER1_MINIONS):
            with self.subTest(card=cid):
                t, o = self.make(hand=(cid,), usable_ms=1500)
                action = self.action(o, "play", cid, position=0)
                self.assertIsNone(opening_choice_completion(o, action))
                t.step(action)
                self.assertEqual(t._budgets[(0, 1)].spent_ms, 1500)
                self.assertEqual(t.engine.game.heroes[0].board[0].card_id, cid)
                self.assertFalse(t.engine.game.pending_choices)

    def test_battlecry_and_spellcraft_generated_cards_still_need_their_own_plays(self):
        for cid, generated, count in (("BG20_100", "BG20_GEM", 2), ("BG23_000", "BG23_000t", 1)):
            with self.subTest(card=cid):
                t, o = self.make(hand=(cid,), usable_ms=1500)
                t.step(self.action(o, "play", cid, position=0))
                self.assertEqual([c.card_id for c in t.engine.game.heroes[0].hand], [generated] * count)
                self.assertEqual(t._budgets[(0, 1)].actions_used, 1)

    def test_freeze_remains_available_and_finite_after_cast_no_longer_fits(self):
        t, o = self.make(hand=("BG28_503",), usable_ms=4500)
        o = t.step(self.action(o, "freeze"))
        self.assertIn(self.action(t.engine.observe(), "cast_spell", "BG28_503"), o.legal_actions)
        o = t.step(self.action(o, "freeze"))
        self.assertNotIn(self.action(t.engine.observe(), "cast_spell", "BG28_503"), o.legal_actions)
        self.assertIn(RecruitAction("freeze"), o.legal_actions)
        while o is not None and o.player_id == 0:
            o = t.step(self.action(o, "freeze"))
        budget = t._budgets[(0, 1)]
        self.assertEqual((budget.actions_used, budget.spent_ms), (4, 4000))
        self.assertFalse(t.engine.game.pending_choices)

    def test_shorter_observed_timer_during_pending_choice_still_aborts_without_refund(self):
        remaining = [10000]
        t, o = self.make(hand=("BG31_880",), timer=lambda p, turn: remaining[0])
        # Start with exactly enough for two choices under the initial timer.
        remaining[0] = 10500
        t._accept(t.engine.observe())  # observation cannot refill initial 5000ms
        self.assertNotIn(self.action(t.engine.observe(), "cast_spell", "BG31_880"), t._visible.legal_actions)
        remaining[0] = 11000
        t, o = self.make(hand=("BG31_880",), timer=lambda p, turn: remaining[0])
        o = t.step(self.action(o, "cast_spell", "BG31_880"))
        remaining[0] = 6500
        with self.assertRaisesRegex(UnsupportedRecruitTransition, "timer shrank"):
            t.step(o.legal_actions[0])
        self.assertEqual(t._budgets[(0, 1)].spent_ms, 3500)
        self.assertTrue(t.engine.game.pending_choices)
        self.assertIsNone(t._visible)
        with self.assertRaisesRegex(UnsupportedRecruitTransition, "already failed"):
            t.fork()

    def test_unsupported_triples_are_not_removed_from_legal_actions(self):
        t, o = self.make(hand=("BG29_611", "BG29_611"), seed=1)
        # Find a deterministic ordinary duplicate shop among current seeds.
        for seed in range(100):
            o = t.reset(seed=seed)
            if any(c["card_id"] == "BG29_611" for c in o.private_state["shop"]):
                break
        action = self.action(o, "buy", "BG29_611")
        self.assertIn(action, t.engine.observe().legal_actions)
        with self.assertRaisesRegex(UnsupportedRecruitTransition, "Triple"):
            t.step(action)
        self.assertEqual(t._budgets[(0, 1)].spent_ms, 1250)

    def test_data_revision_is_explicit_and_old_wrapper_is_unchanged(self):
        t, o = self.make(hand=("BG28_503",), usable_ms=1500)
        self.assertEqual(t.coverage()["choice_timing_version"], CHOICE_TIMING_VERSION)
        self.assertFalse(t.coverage()["full_game_ready"])
        old = TimedTwoTurnOpeningEngine.for_fixture(t.engine.fork(), self.profile, self.rules)
        old_o = old.reset(seed=3)
        self.assertIn(self.action(old.engine.observe(), "cast_spell", "BG28_503"), old_o.legal_actions)
        raw = t.engine.fork()
        raw.provenance = dict(raw.provenance, reference_cards_sha256="changed")
        with self.assertRaisesRegex(RecruitCoverageError, "current-data audit"):
            ChoiceTimedTwoTurnOpeningEngine.for_fixture(raw, self.profile, self.rules)


if __name__ == "__main__":
    unittest.main()
