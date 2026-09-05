"""Conformance examples run against the real external hsrl2 engine when present."""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
ENGINE = Path(os.environ.get("HSBRSIM_ROOT", ROOT.parent / "research/HSBRSIM"))


@unittest.skipUnless((ENGINE / "hsrl2/game.py").is_file(), "external pinned HSBRSIM checkout unavailable")
class TestCurrentEffects(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ENGINE))
        from bg_ai.hsbrsim_data import build_current_database
        cls.db = build_current_database(ENGINE).db
        from hsrl2.game import Game
        from hsrl2.hero import Hero
        from hsrl2.tags import GameTag, Race, Zone
        from hsrl2.actions.stats import Buff
        from hsrl2.events import Listener
        from bg_ai import hsbrsim_effects
        cls.Game, cls.Hero = Game, Hero
        cls.Tag, cls.Race, cls.Zone = GameTag, Race, Zone
        cls.Buff, cls.Listener = Buff, Listener
        cls.ext = hsbrsim_effects

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(self.ext.installed_current_effects(self.db))
        self.game = self.Game([self.Hero("TEST_A", "A"), self.Hero("TEST_B", "B")], self.db, seed=42)
        self.a, self.b = self.game.heroes
        self.a.health, self.a.armor = 30, 3

    def put(self, card_id, hero=None):
        hero = hero or self.a
        minion = self.game.create_minion(card_id, controller=hero)
        self.game.summon(hero, minion)
        return minion

    def dummy(self, hero=None, *, attack=0, health=1000):
        from hsrl2.minion import Minion
        hero = hero or self.a
        minion = Minion("TEST_DUMMY", "Dummy", atk=attack, health=health)
        self.game.summon(hero, minion)
        return minion

    def test_soul_rewinder_health_armor_snapshot_and_current_values(self):
        normal = self.put("BG26_174")
        golden = self.put("BG26_174_G")
        self.assertEqual((normal.max_health, golden.max_health), (2, 4))
        self.a.take_damage(8)
        self.assertEqual((self.a.health, self.a.armor), (30, 3))
        self.assertEqual((normal.max_health, golden.max_health), (4, 8))
        self.a.take_damage(1)
        self.assertEqual((normal.max_health, golden.max_health), (6, 12))
        self.b.take_damage(2)
        self.assertEqual((normal.max_health, golden.max_health), (6, 12))

    def test_soul_rewinder_missing_snapshot_fails_loudly(self):
        self.put("BG26_174")
        with self.assertRaisesRegex(self.ext.EffectValidationError, "pre-damage"):
            self.game.events.fire(self.game, "hero_damage_taken", hero=self.a, amount=1)

    def test_eredar_accumulation_generates_cards_without_auto_casting(self):
        escapist = self.put("BG36_733")
        other = self.dummy(attack=2, health=3)
        self.a.take_damage(3)
        self.assertEqual(len(self.a.hand), 0)
        self.a.take_damage(6)
        self.assertEqual([c.card_id for c in self.a.hand], ["BG28_607", "BG28_607"])
        self.assertEqual(escapist._escapist_dmg_taken, 1)
        self.assertEqual((other.atk, other.max_health), (2, 3))
        self.assertEqual(self.a.get(self.Tag.TAVERN_SPELLS_CAST_THIS_GAME, 0), 0)
        self.b.take_damage(10)
        self.assertEqual(len(self.a.hand), 2)
        self.a.take_damage(0)
        self.assertEqual(len(self.a.hand), 2)

    def test_golden_eredar_generates_two_copies_per_threshold(self):
        self.put("BG36_733_G")
        self.a.take_damage(5)
        self.assertEqual(len(self.a.hand), 2)
        self.a.take_damage(3)
        self.assertEqual(len(self.a.hand), 4)
        self.assertTrue(all(c.card_id == "BG28_607" for c in self.a.hand))

    def test_eredar_and_rewinder_trigger_in_either_board_order(self):
        for order in [("BG26_174", "BG36_733"), ("BG36_733", "BG26_174")]:
            game = self.Game([self.Hero("TEST_A", "A"), self.Hero("TEST_B", "B")], self.db, seed=4)
            hero = game.heroes[0]
            hero.health, hero.armor = 30, 3
            for cid in order:
                game.summon(hero, game.create_minion(cid, controller=hero))
            hero.take_damage(4)
            self.assertEqual((hero.health, hero.armor), (30, 3))
            self.assertEqual([c.card_id for c in hero.hand], ["BG28_607"])

    def test_eredar_leaving_board_removes_listener(self):
        source = self.put("BG36_733")
        self.assertTrue(self.game.sell_minion(self.a, source))
        self.a.take_damage(4)
        self.assertEqual(len(self.a.hand), 0)

    def test_eredar_full_hand_uses_engine_pending_queue(self):
        self.put("BG36_733")
        for _ in range(10):
            self.a.add_to_hand(self.game.create_spell("BG20_GEM", controller=self.a))
        self.a.take_damage(8)
        self.assertEqual(len(self.a.hand), 10)
        self.assertEqual([card.card_id for hero, card in self.game.pending_hand_queue],
                         ["BG28_607", "BG28_607"])

    def test_jailbird_uses_buffed_stats_and_max_health_not_blood_gems(self):
        source = self.put("BG36_333")
        self.game.run_actions(self.Buff(source, atk=7, health=11))
        source.health -= 2
        source.set(self.Tag.GEMS_PLAYED_ON, 100)
        self.a.set(self.Tag.BLOOD_GEM_BONUS_ATK, 100)
        defender = self.dummy(self.b)
        expected = (source.atk, source.max_health)
        before = defender.health
        self.game.run_script_hook(source, "rally", {"target": defender})
        golem = next(m for m in self.a.board if m.card_id == self.ext.JAILBIRD_GOLEM)
        self.assertEqual((golem.atk, golem.max_health), expected)
        self.assertEqual(defender.health, before - expected[0])
        self.assertEqual(self.a.board.index(golem), self.a.board.index(source) + 1)

    def test_golden_jailbird_doubles_stats_once(self):
        source = self.put("BG36_333_G")
        defender = self.dummy(self.b)
        self.game.run_script_hook(source, "rally", {"target": defender})
        golem = next(m for m in self.a.board if m.card_id == self.ext.JAILBIRD_GOLEM)
        self.assertEqual((golem.atk, golem.max_health), (source.atk * 2, source.max_health * 2))

    def test_jailbird_full_board_has_no_extra_attack(self):
        source = self.put("BG36_333")
        for _ in range(6):
            self.dummy()
        defender = self.dummy(self.b)
        before = defender.health
        self.game.run_script_hook(source, "rally", {"target": defender})
        self.assertEqual(defender.health, before)
        self.assertEqual(len(self.a.board), 7)

    def test_sanguine_each_intrinsic_trigger_uses_current_values(self):
        normal = self.put("BG23_017")
        golden = self.put("BG23_017_G")
        self.game.run_script_hook(normal, "battlecry")
        self.assertEqual((self.a.get(self.Tag.BLOOD_GEM_BONUS_ATK),
                          self.a.get(self.Tag.BLOOD_GEM_BONUS_HEALTH)), (2, 1))
        self.game.run_script_hook(normal, "deathrattle")
        self.assertEqual((self.a.get(self.Tag.BLOOD_GEM_BONUS_ATK),
                          self.a.get(self.Tag.BLOOD_GEM_BONUS_HEALTH)), (4, 2))
        self.game.run_script_hook(golden, "battlecry")
        self.game.run_script_hook(golden, "deathrattle")
        self.assertEqual((self.a.get(self.Tag.BLOOD_GEM_BONUS_ATK),
                          self.a.get(self.Tag.BLOOD_GEM_BONUS_HEALTH)), (12, 6))

    def test_parameter_driven_tichondrius_and_ashen_use_new_values(self):
        demon = self.put("BG26_523")
        self.put("BG32_873")
        tavern_minion = self.game.create_minion("BG26_174", controller=self.a)
        tavern_minion.zone = self.Zone.TAVERN
        self.a.tavern.append(tavern_minion)
        before = (demon.atk, demon.max_health, tavern_minion.atk, tavern_minion.max_health)
        self.a.take_damage(1)
        self.assertEqual((self.a.health, self.a.armor), (30, 3))
        self.assertEqual((demon.atk, demon.max_health, tavern_minion.atk, tavern_minion.max_health),
                         (before[0] + 4, before[1] + 4, before[2] + 2, before[3] + 2))

    def test_parameter_driven_mighty_dragonbreath_stacks_current_values(self):
        dragon = self.put("BG32_822")
        dragon.set(self.Tag.DIVINE_SHIELD, True)
        before = (dragon.atk, dragon.max_health)
        spell = self.game.create_spell("BG36_246", controller=self.a)
        self.game.run_script_hook(spell, "on_play")
        self.assertEqual((dragon.atk, dragon.max_health), (before[0] + 9, before[1] + 6))

    def test_parameter_driven_sanguine_refiner_and_utility_drone(self):
        refiner = self.put("BG33_885")
        self.game.run_script_hook(refiner, "rally")
        self.assertEqual((self.a.get(self.Tag.BLOOD_GEM_BONUS_ATK),
                          self.a.get(self.Tag.BLOOD_GEM_BONUS_HEALTH)), (1, 2))
        drone = self.put("BG26_152")
        before = (drone.atk, drone.max_health)
        self.game._magnetic_stack[drone.uuid] = ["TEST_ATTACHMENT", "TEST_ATTACHMENT_2"]
        self.game.run_script_hook(drone, "end_of_turn")
        self.assertEqual((drone.atk, drone.max_health), (before[0] + 8, before[1] + 10))

    def test_nested_install_restores_and_unimplemented_effects_stay_unregistered(self):
        from hsrl2.scripts import REGISTRY
        before = dict(REGISTRY)
        with self.assertRaisesRegex(RuntimeError, "test abort"):
            with self.ext.installed_current_effects(self.db) as report:
                self.assertEqual(len(report.registered_ids), 8)
                self.assertFalse(report.full_game_ready)
                self.assertFalse(report.golden_battlecry_play_ready)
                raise RuntimeError("test abort")
        self.assertEqual(REGISTRY, before)
        for cid in ("BG33_319", "EBG_Spell_037"):
            self.assertNotIn(cid, REGISTRY)
            self.assertIn(cid, self.ext.DEFERRED_EFFECTS)

    def test_old_database_is_rejected_before_registry_changes(self):
        from hsrl2.db import CardDB
        from hsrl2.scripts import REGISTRY
        old = CardDB.load(ENGINE / "data")
        before = dict(REGISTRY)
        with self.assertRaises(self.ext.EffectValidationError):
            with self.ext.installed_current_effects(old):
                self.fail("should not install old data")
        self.assertEqual(REGISTRY, before)


if __name__ == "__main__":
    unittest.main()
