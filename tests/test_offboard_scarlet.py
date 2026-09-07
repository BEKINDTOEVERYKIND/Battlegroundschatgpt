from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import unittest

from bg_ai.hsbrsim_opening import build_opening_database
from bg_ai.hsbrsim_adapter import UnsupportedRecruitTransition
from bg_ai.offboard_scarlet import (
    SCARLET, VERSION, ScarletHandFixtureSpec, HandScarletOpeningEngine,
    TimedHandScarletOpeningEngine,
)
from bg_ai.turn_budget import TimingProfile

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = Path(os.environ.get("HSBRSIM_ROOT", ROOT.parent / "research/HSBRSIM"))
TRIBES = ("DRAGON", "NAGA", "BEAST", "MECH", "PIRATE")


@unittest.skipUnless((EXTERNAL / "hsrl2/game.py").is_file(), "Pinned HSBRSIM required")
class HandScarletTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.current = build_opening_database(EXTERNAL, valid_tribes=TRIBES)
        cls.rules = json.loads((ROOT / "data/ruleset.json").read_text())
        cls.profile = TimingProfile.load(ROOT / "config/turn-budget.json")

    def make(self, attack=3, health=0, seed=4, board=(), hand_extra=()):
        spec = ScarletHandFixtureSpec(valid_tribes=TRIBES,
            initial_hands=((SCARLET,) + tuple(hand_extra),) + ((),) * 7,
            initial_boards=(tuple(board),) + ((),) * 7,
            initial_scarlet_hand_buffs=(((0, attack, health),),) + ((),) * 7)
        raw = HandScarletOpeningEngine(self.current.db, self.rules, fixture=spec,
            provenance=self.current.provenance, timer_ms=lambda player, turn: 60000)
        timed = TimedHandScarletOpeningEngine.for_fixture(raw, self.profile, self.rules)
        return timed, timed.reset(seed=seed)

    def act(self, timed, observation, kind, **attrs):
        return timed.step(next(a for a in observation.legal_actions if a.kind == kind
                              and all(getattr(a, k) == v for k, v in attrs.items())))

    def test_hand_crossing_gains_shield_before_play_and_marker_is_visible(self):
        t, o = self.make()
        row = o.private_state["hand"][0]
        self.assertEqual((row["attack"], row["health"]), (6, 3))
        self.assertTrue(row["divine_shield"])
        self.assertTrue(row["scarlet_trigger_consumed"])
        o = self.act(t, o, "play", entity_id=row["entity_id"])
        self.assertTrue(o.private_state["board"][0]["divine_shield"])
        self.assertEqual(o.public_state["scope"], VERSION)

    def test_below_threshold_and_exact_crossing(self):
        from hsrl2.entity import Buff
        t, o = self.make(2)
        card = t.engine.game.heroes[0].hand[0]
        self.assertFalse(card.divine_shield)
        card.add_buff(Buff(1, 0))
        self.assertTrue(card.divine_shield)
        self.assertTrue(card._scarlet_done)

    def test_one_time_marker_survives_health_buffs_and_play(self):
        from hsrl2.entity import Buff
        from hsrl2.tags import GameTag
        t, o = self.make()
        card = t.engine.game.heroes[0].hand[0]
        card.clear(GameTag.DIVINE_SHIELD)
        card.add_buff(Buff(2, 3))
        self.assertFalse(card.divine_shield)
        t.engine.game.play_minion(t.engine.game.heroes[0], card)
        self.assertFalse(card.divine_shield)
        self.assertTrue(card._scarlet_done)

    def test_tavern_threshold_is_rejected_before_any_mutation(self):
        from hsrl2.entity import Buff
        t, o = self.make(0)
        game, hero = t.engine.game, t.engine.game.heroes[0]
        from hsrl2.tags import Zone
        card = game.create_minion(SCARLET, controller=hero)
        card.zone = Zone.TAVERN
        before = (card.atk, card.health, len(card.buffs))
        with self.assertRaisesRegex(UnsupportedRecruitTransition, "Tavern Scarlet"):
            card.add_buff(Buff(3, 2))
        self.assertEqual((card.atk, card.health, len(card.buffs)), before)
        self.assertFalse(card.divine_shield)

    def test_fork_reconstructs_synthetic_handbuff_marker_and_preserves_clock(self):
        t, o = self.make(3, 2)
        o = self.act(t, o, "play", entity_id=o.private_state["hand"][0]["entity_id"])
        branch = t.fork()
        self.assertEqual(branch.engine.observe().private_state, t.engine.observe().private_state)
        self.assertEqual(branch.trace, t.trace)
        self.assertEqual(branch._budgets[(0, 1)].snapshot(), t._budgets[(0, 1)].snapshot())

    def test_actual_new_worker_receipt_advances_but_legacy_marker_is_rejected(self):
        t, o = self.make()
        o = self.act(t, o, "play", entity_id=o.private_state["hand"][0]["entity_id"])
        while o is not None:
            o = self.act(t, o, "end_turn")
        request = t.engine.combat_request(seed=197)
        run = subprocess.run(["node", "simulator/offboard-scarlet-firestone.mjs"], cwd=ROOT,
            input=json.dumps(request) + "\n", text=True, capture_output=True, check=True)
        receipt = json.loads(run.stdout)
        self.assertNotIn("error", receipt)
        legacy = deepcopy(receipt)
        del legacy["effect_transport_version"]
        with self.assertRaisesRegex(ValueError, "Scarlet-aware"):
            t.advance_combat(legacy)
        o = t.advance_combat(receipt)
        self.assertEqual(o.turn, 2)
        self.assertTrue(o.private_state["board"][0]["scarlet_trigger_consumed"])
        branch = t.fork()
        self.assertEqual(branch.engine.combat_receipts, t.engine.combat_receipts)

    def test_current_pool_and_other_guards_remain_closed(self):
        t, o = self.make()
        c = t.coverage()
        self.assertEqual(c["current_tier1_minions"], 22)
        self.assertEqual(c["current_tier1_spells"], 8)
        self.assertFalse(c["full_game_ready"])
        self.assertFalse(c["production_training_enabled"])
        self.assertFalse(c["tavern_scarlet_threshold_supported"])
        self.assertFalse(c["tavern_fugitive_spell_trigger_supported"])
        self.assertIn("tier2_next_turn_shop_after_turn1_upgrade", c["blockers"])
        with self.assertRaisesRegex(UnsupportedRecruitTransition, "Ordinary goldens"):
            t.engine.game.create_minion(SCARLET, controller=t.engine.game.heroes[0], golden=True)

    def test_actual_board_fugitive_cast_works_but_tavern_target_still_aborts(self):
        t, o = self.make(0, board=("BG36_921",), hand_extra=("BG28_897",))
        spell = next(c for c in o.private_state["hand"] if c["card_id"] == "BG28_897")
        o = self.act(t, o, "cast_spell", entity_id=spell["entity_id"])
        choice = t.engine._choice()
        index = next(i for i, c in enumerate(choice.options) if c.card_id == "BG36_921"
                     and c in t.engine.game.heroes[0].board)
        o = self.act(t, o, "choose", choice_index=index)
        fugitive = next(c for c in o.private_state["board"] if c["card_id"] == "BG36_921")
        # Current Banana is +2/+2; Fugitive adds exactly +1 Health.
        self.assertEqual((fugitive["attack"], fugitive["health"]), (7, 5))
        self.assertEqual(len(t.trace), 2)  # Cast and target choice are separate timed commands.
        t, o = self.make(0, seed=13, hand_extra=("BG28_897",))
        spell = next(c for c in o.private_state["hand"] if c["card_id"] == "BG28_897")
        o = self.act(t, o, "cast_spell", entity_id=spell["entity_id"])
        choice = t.engine._choice()
        index = next(i for i, c in enumerate(choice.options) if c.card_id == "BG36_921")
        target = choice.options[index]
        before = (target.atk, target.health, len(t.engine.game.heroes[0].hand))
        with self.assertRaisesRegex(UnsupportedRecruitTransition, "Tavern Fugitive"):
            self.act(t, o, "choose", choice_index=index)
        self.assertEqual((target.atk, target.health, len(t.engine.game.heroes[0].hand)), before)
