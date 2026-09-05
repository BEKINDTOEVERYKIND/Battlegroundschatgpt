"""Real pinned-engine transition tests; skip only if dependency is unavailable."""
from contextlib import ExitStack
from dataclasses import replace
import os
from pathlib import Path
import unittest

from bg_ai.hsbrsim_data import build_current_database
from bg_ai.hsbrsim_heroes import (
    EXTENSION_IDS, FLIGHTPATHS, GALEWING, IRONFORGE, PLAGUELANDS,
    WESTFALL, HeroExtensionError, hero_extension_state,
    installed_current_hero_extensions,
)

ROOT = Path(__file__).resolve().parents[1]
ENGINE = Path(os.environ.get("HSBRSIM_ROOT", ROOT.parent / "research/HSBRSIM"))


@unittest.skipUnless((ENGINE / "hsrl2/game.py").is_file(), "external HSBRSIM checkout unavailable")
class TestGalewingTransitions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.current = build_current_database(ENGINE)
        from hsrl2.game import Game
        from hsrl2.hero import Hero
        from hsrl2.scripts import REGISTRY
        from hsrl2.tags import GameTag
        cls.Game, cls.Hero, cls.REGISTRY, cls.GameTag = Game, Hero, REGISTRY, GameTag

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.report = self.stack.enter_context(installed_current_hero_extensions(self.current.db))
        self.hero = self.Hero("BG20_HERO_283", "Galewing")
        self.other = self.Hero("TB_BaconShop_HERO_11", "Other")
        self.game = self.Game([self.hero, self.other], self.current.db, seed=917)
        self.game.choice_policy = None
        self.game.start_game()

    def choose_path(self, path):
        self.assertTrue(self.game.use_hero_power(self.hero))
        choice = self.game.pending_choices.pop(0)
        self.assertEqual(choice.kind, "galewing_flightpath")
        choice.choose(choice.options.index(path))
        return choice

    def begin_turn(self, turn):
        self.game.turn = turn
        self.game._begin_recruit_for(self.hero)

    def test_initial_choice_is_explicit_free_and_has_three_paths(self):
        gold = self.hero.gold
        self.assertTrue(self.game.use_hero_power(self.hero))
        self.assertEqual(self.hero.gold, gold)
        self.assertEqual(len(self.game.pending_choices), 1)
        self.assertEqual(tuple(self.game.pending_choices[0].options), FLIGHTPATHS)
        self.assertTrue(hero_extension_state(self.hero)["galewing_choice_pending"])
        self.assertEqual(self.game.hero_power_def(self.hero).id, GALEWING)
        self.assertEqual(self.hero.hand, [])
        self.assertFalse(self.game.use_hero_power(self.hero))

    def test_inflight_passive_power_blocks_reuse(self):
        self.choose_path(IRONFORGE)
        self.assertEqual(self.game.hero_power_def(self.hero).id, IRONFORGE)
        self.assertFalse(self.game.use_hero_power(self.hero))
        self.begin_turn(2)
        self.assertFalse(self.game.use_hero_power(self.hero))
        self.assertEqual(hero_extension_state(self.hero)["galewing_due_turn"], 3)

    def test_ironforge_rewards_after_exactly_two_turns_after_income_reset(self):
        self.choose_path(IRONFORGE)
        self.begin_turn(2)
        self.assertEqual(self.hero.gold, 4)
        self.begin_turn(3)
        self.assertEqual(self.hero.gold, 7)  # Turn-3 base income 5 plus reward 2.
        self.assertEqual(self.game.hero_power_def(self.hero).id, GALEWING)
        self.assertEqual(hero_extension_state(self.hero)["galewing_last_path"], IRONFORGE)
        self.assertIsNone(hero_extension_state(self.hero)["galewing_due_turn"])

    def test_no_consecutive_flightpath_and_previous_path_returns_later(self):
        self.choose_path(IRONFORGE)
        self.begin_turn(3)
        self.assertTrue(self.game.use_hero_power(self.hero))
        choice = self.game.pending_choices.pop(0)
        self.assertEqual(choice.options, [WESTFALL, PLAGUELANDS])
        choice.choose(0)
        self.begin_turn(4)
        self.assertTrue(self.game.use_hero_power(self.hero))
        self.assertEqual(self.game.pending_choices[0].options, [IRONFORGE, PLAGUELANDS])

    def test_westfall_gives_a_playable_spell_and_does_not_autocast(self):
        self.choose_path(WESTFALL)
        self.assertEqual(self.hero.hand, [])
        self.begin_turn(2)
        self.assertEqual(len(self.hero.hand), 1)
        spell = self.hero.hand[0]
        definition = self.current.db.get(spell.card_id)
        self.assertTrue(definition.is_pool_spell)
        self.assertEqual(definition.cost, 1)
        self.assertEqual(self.hero.get(self.GameTag.CARDS_PLAYED_THIS_TURN, 0), 0)

    def test_westfall_cost_filter_allows_higher_tier_spells(self):
        self.choose_path(WESTFALL)
        # Fix the available-pool boundary, keeping actual RNG/acquire/hand APIs.
        # Boon of Beetles is a current 1-cost Tier-4 spell.
        self.game.spell_pool._available.clear()
        self.game.spell_pool._available["BG28_603"] = 1
        self.begin_turn(2)
        self.assertEqual(self.hero.tavern_tier, 1)
        self.assertEqual([card.card_id for card in self.hero.hand], ["BG28_603"])
        self.assertEqual(self.game.spell_pool.available("BG28_603"), 1)

    def test_westfall_empty_candidate_pool_fails_explicitly(self):
        self.choose_path(WESTFALL)
        self.game.spell_pool._available.clear()
        with self.assertRaisesRegex(HeroExtensionError, "no available current"):
            self.begin_turn(2)

    def test_westfall_full_hand_uses_pending_hand_queue(self):
        self.choose_path(WESTFALL)
        for _ in range(10):
            self.hero.add_to_hand(self.game.create_spell("BG20_GEM", controller=self.hero))
        self.begin_turn(2)
        self.assertEqual(len(self.hero.hand), 10)
        queued = [(owner, entity) for owner, entity in self.game.pending_hand_queue
                  if owner is self.hero]
        self.assertEqual(len(queued), 1)
        self.assertEqual(self.current.db.get(queued[0][1].card_id).cost, 1)

    def test_plaguelands_uses_tavern_tier_when_flight_completes(self):
        self.choose_path(PLAGUELANDS)
        self.begin_turn(2)
        self.assertEqual(self.game.pending_choices, [])
        self.hero.set(self.GameTag.TAVERN_TIER, 3)
        self.begin_turn(3)
        self.assertEqual(self.game.pending_choices, [])
        self.begin_turn(4)
        self.assertEqual(len(self.game.pending_choices), 1)
        choice = self.game.pending_choices.pop(0)
        self.assertEqual(choice.kind, "discover_galewing_minion")
        self.assertTrue(all(self.current.db.get(cid).tech_level == 3
                            for cid in choice.options))
        picked = choice.options[0]
        available = self.game.minion_pool.available(picked)
        self.assertEqual(self.hero.hand, [])  # Discovery is an explicit action.
        choice.choose(0)
        self.assertEqual(self.game.minion_pool.available(picked), available - 1)
        self.assertEqual(self.hero.hand[0].card_id, picked)

    def test_plaguelands_pool_only_contains_actual_current_minions(self):
        self.choose_path(PLAGUELANDS)
        self.hero.set(self.GameTag.TAVERN_TIER, 5)
        self.begin_turn(4)
        choice = self.game.pending_choices[0]
        for cid in choice.options:
            definition = self.current.db.get(cid)
            self.assertTrue(definition.is_pool_minion)
            self.assertEqual(definition.tech_level, 5)
            self.assertGreater(self.game.minion_pool.available(cid), 0)

    def test_other_players_turn_does_not_complete_a_flight(self):
        self.choose_path(IRONFORGE)
        gold = self.hero.gold
        self.game.turn = 3
        self.game._begin_recruit_for(self.other)
        self.assertEqual(self.hero.gold, gold)
        self.assertEqual(self.game.hero_power_def(self.hero).id, IRONFORGE)

    def test_double_bind_does_not_double_reward(self):
        self.REGISTRY[GALEWING].on_bind(self.hero, self.game)
        self.choose_path(IRONFORGE)
        self.begin_turn(3)
        self.assertEqual(self.hero.gold, 7)
        self.game.events.fire(self.game, "turn_start", hero=self.hero, turn=3)
        self.assertEqual(self.hero.gold, 7)

    def test_choice_cannot_be_replayed_to_reset_flight_timer(self):
        choice = self.choose_path(IRONFORGE)
        with self.assertRaisesRegex(HeroExtensionError, "already resolved"):
            choice.choose(choice.options.index(WESTFALL))
        self.assertEqual(hero_extension_state(self.hero)["galewing_due_turn"], 3)

    def test_replaced_power_does_not_receive_old_flight_reward(self):
        self.choose_path(IRONFORGE)
        self.game.replace_hero_power(self.hero, "TB_BaconShop_HP_035")
        self.begin_turn(3)
        self.assertEqual(self.hero.gold, 5)
        self.assertEqual(self.game.hero_power_def(self.hero).id, "TB_BaconShop_HP_035")
        self.assertIsNone(hero_extension_state(self.hero)["galewing_path"])

    def test_eliminated_hero_does_not_receive_reward(self):
        self.choose_path(IRONFORGE)
        self.hero.health = 0
        self.hero.armor = 0
        self.begin_turn(3)
        self.assertEqual(self.hero.gold, 5)
        self.assertFalse(self.game.use_hero_power(self.hero))

    def test_nested_registry_context_restores_exact_previous_objects_on_error(self):
        previous = {cid: self.REGISTRY[cid] for cid in EXTENSION_IDS}
        with self.assertRaisesRegex(RuntimeError, "probe"):
            with installed_current_hero_extensions(self.current.db):
                self.assertIsNot(self.REGISTRY[GALEWING], previous[GALEWING])
                raise RuntimeError("probe")
        for cid, original in previous.items():
            self.assertIs(self.REGISTRY[cid], original)

    def test_stale_definition_is_rejected_before_registry_mutation(self):
        from hsrl2.db import CardDB
        stale = CardDB()
        for cid in EXTENSION_IDS:
            definition = self.current.db.get(cid)
            stale.register(replace(definition, cost=99) if cid == GALEWING else definition)
        before = dict(self.REGISTRY)
        with self.assertRaisesRegex(HeroExtensionError, "Stale"):
            with installed_current_hero_extensions(stale):
                self.fail("stale database was accepted")
        self.assertEqual(self.REGISTRY, before)

    def test_registry_report_does_not_clear_full_game_gate(self):
        self.assertEqual(self.report["hero_power_ids"], [GALEWING])
        self.assertEqual(self.report["generated_power_ids"], list(FLIGHTPATHS))
        self.assertFalse(self.report["full_game_training_ready"])
        self.assertTrue(self.report["unverified"])


if __name__ == "__main__":
    unittest.main()
