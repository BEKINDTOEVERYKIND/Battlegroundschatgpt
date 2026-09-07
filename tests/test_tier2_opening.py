"""Exact current Tier-2 inventory and real pinned-engine Forest Rover regressions."""
from dataclasses import replace
import json
import os
from pathlib import Path
import unittest

from bg_ai.tier2_opening import (
    BEETLES, BLOCKERS, Tier2ConformanceError, beetle_bonus,
    build_frontier_manifest, tier2_beetle_game_type, validate_rover_definitions,
)

ROOT = Path(__file__).resolve().parents[1]
ENGINE = Path(os.environ.get("HSBRSIM_ROOT", ROOT.parent / "research/HSBRSIM"))


class Tier2InventoryTests(unittest.TestCase):
    def test_inventory_contains_every_current_tier2_card_without_filters(self):
        rules = json.loads((ROOT / "data/ruleset.json").read_text())
        active = set(rules["active"]["shop_minion_ids"]) | set(rules["active"]["tavern_spell_ids"])
        cards = json.loads((ROOT / "data/reference_cards.json").read_text())
        expected = {c["id"] for c in cards if c["id"] in active and c.get("techLevel") == 2}
        self.assertEqual(expected, set(BLOCKERS))
        self.assertEqual(len(expected), 41)


@unittest.skipUnless((ENGINE / "hsrl2/game.py").is_file(), "Pinned HSBRSIM checkout required")
class ForestRoverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from bg_ai.hsbrsim_data import build_current_database
        cls.db = build_current_database(ENGINE).db
        from hsrl2.hero import Hero
        from hsrl2.tags import GameTag, Race
        cls.Hero, cls.Tag, cls.Race = Hero, GameTag, Race
        cls.Game = tier2_beetle_game_type()

    def setUp(self):
        self.g = self.Game([self.Hero("TEST_A"), self.Hero("TEST_B")], self.db, seed=17)
        self.a, self.b = self.g.heroes

    def put(self, cid, hero=None, *, golden=False):
        hero = hero or self.a
        card = self.g.create_minion(cid, controller=hero, golden=golden)
        self.g.summon(hero, card)
        return card

    def play_rover(self, *, golden=False):
        card = self.g.create_minion("BG31_801", controller=self.a, golden=golden)
        self.a.add_to_hand(card)
        self.assertTrue(self.g.play_minion(self.a, card))
        return card

    def test_beetle_bonus_excludes_other_beasts_and_opponent(self):
        beetle = self.put("BG28_603t")
        other_beast = self.put("BG26_805")
        enemy = self.put("BG28_603t", self.b)
        before = (other_beast.atk, other_beast.max_health, enemy.atk, enemy.max_health)
        rover = self.play_rover()
        self.assertEqual(beetle_bonus(self.a), (2, 1))
        self.assertEqual((beetle.atk, beetle.max_health, beetle.health), (4, 3, 3))
        self.assertEqual((rover.atk, rover.max_health), (1, 1))
        self.assertEqual((other_beast.atk, other_beast.max_health, enemy.atk, enemy.max_health), before)
        self.assertEqual(getattr(self.a, "race_auras", {}), {})

    def test_future_and_hand_beetles_share_player_bonus_without_copying_it(self):
        held = self.g.create_minion("BG28_603t", controller=self.a)
        self.a.add_to_hand(held)
        self.play_rover()
        self.assertEqual((held.atk, held.max_health, held.health), (4, 3, 3))
        future = self.put("BG28_603t")
        golden = self.put("BG28_603t", golden=True)
        self.assertEqual((future.atk, future.max_health), (4, 3))
        self.assertEqual((golden.atk, golden.max_health), (6, 5))
        self.assertEqual(len(future.buffs), 0)
        self.assertEqual(len(golden.buffs), 0)

    def test_golden_battlecry_has_full_intrinsic_amount_once(self):
        source = self.play_rover(golden=True)
        self.assertEqual(beetle_bonus(self.a), (4, 2))
        self.assertEqual(self.a.get(self.Tag.COUNTER_BATTLECRIES), 1)
        self.assertEqual(self.a.get(self.Tag.GOLDEN_MINIONS_PLAYED), 1)
        self.assertTrue(source.is_golden)

    def test_golden_brann_is_strongest_repeat_and_retrigger_matches_play(self):
        from hsrl2.actions.trigger import TriggerBattlecry
        self.put("BG_LOE_077")
        self.put("BG_LOE_077", golden=True)
        source = self.play_rover(golden=True)
        self.assertEqual(beetle_bonus(self.a), (12, 6))
        self.assertEqual(self.a.get(self.Tag.COUNTER_BATTLECRIES), 3)
        self.g.run_actions([TriggerBattlecry(source)])
        self.assertEqual(beetle_bonus(self.a), (24, 12))
        self.assertEqual(self.a.get(self.Tag.COUNTER_BATTLECRIES), 6)

    def test_damage_and_other_auras_are_preserved(self):
        from hsrl2.actions.stats import Buff
        from hsrl2.actions.racefx import ApplyRaceAura
        beetle = self.put("BG28_603t")
        self.g.run_actions([Buff(beetle, atk=3, health=4), ApplyRaceAura(self.a, self.Race.BEAST, 5, 6)])
        beetle.take_damage(2)
        before = (beetle.atk, beetle.max_health, beetle.health)
        self.play_rover()
        self.assertEqual((beetle.atk, beetle.max_health, beetle.health),
                         (before[0] + 2, before[1] + 1, before[2] + 1))
        self.assertEqual(beetle.max_health - beetle.health, 2)

    def test_selling_rover_does_not_remove_player_bonus(self):
        source = self.play_rover()
        self.g.sell_minion(self.a, source)
        future = self.put("BG28_603t")
        self.assertEqual((future.atk, future.max_health), (4, 3))

    def test_deathrattle_spawns_correct_normal_tokens_with_current_bonus(self):
        for golden, count, expected in ((False, 1, (4, 3)), (True, 2, (6, 4))):
            self.setUp()
            rover = self.play_rover(golden=golden)
            rover.take_damage(rover.health)
            self.g.check_deaths()
            tokens = [c for c in self.a.board if c.card_id in BEETLES]
            self.assertEqual(len(tokens), count)
            self.assertTrue(all(c.card_id == "BG28_603t" for c in tokens))
            self.assertTrue(all((c.atk, c.max_health) == expected for c in tokens))

    def test_current_definition_and_full_pool_gates_remain_explicit(self):
        original = self.db.get("BG31_801")
        class BadDB:
            def get(inner, cid):
                return replace(original, script_data_num_3=999) if cid == original.id else self.db.get(cid)
        with self.assertRaises(Tier2ConformanceError):
            validate_rover_definitions(BadDB())
        self.assertFalse(self.g.full_game_ready)
        self.assertFalse(self.g.tier2_pool_ready)
        with self.assertRaisesRegex(Tier2ConformanceError, "persistent Beetle globals"):
            self.g.run_combat(self.a, self.b)

    def test_manifest_has_exact_handler_sources_without_equating_registration_to_support(self):
        report = build_frontier_manifest(ENGINE)
        self.assertEqual(report["pool_counts"], {"minions": 34, "tavern_spells": 7})
        self.assertEqual(len(report["cards"]), 41)
        self.assertFalse(report["tier2_pool_ready"])
        self.assertTrue(all(c["upstream_source_sha256"] for c in report["cards"]))
        self.assertTrue(all(not c["opening_integrated"] for c in report["cards"]))
