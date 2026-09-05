"""Current-card intrinsic amounts, dispatch counts and real-engine integration."""
from contextlib import ExitStack
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
ENGINE = Path(os.environ.get("HSBRSIM_ROOT", ROOT.parent / "research/HSBRSIM"))


@unittest.skipUnless((ENGINE / "hsrl2/game.py").is_file(), "external pinned HSBRSIM checkout unavailable")
class TestBattlecryMigration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ENGINE))
        from bg_ai.hsbrsim_data import build_current_database
        from bg_ai import hsbrsim_battlecries, hsbrsim_rules
        from hsrl2.game import Game
        from hsrl2.hero import Hero
        from hsrl2.tags import GameTag, Zone
        cls.db = build_current_database(ENGINE).db
        cls.module, cls.rules = hsbrsim_battlecries, hsbrsim_rules
        cls.Game, cls.Hero, cls.Tag, cls.Zone = Game, Hero, GameTag, Zone

    def setUp(self):
        self.game = self.Game([self.Hero("A", "A"), self.Hero("B", "B")], self.db, seed=4)
        self.hero = self.game.heroes[0]
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(self.rules.installed_recruit_rules(self.game))
        self.stack.enter_context(self.module.installed_battlecry_rules(self.game))

    def card(self, cid, *, golden=False, play=False):
        m = self.game.create_minion(cid, controller=self.hero, golden=golden)
        self.hero.add_to_hand(m)
        if play:
            self.assertTrue(self.game.play_minion(self.hero, m))
        return m

    def brann(self, *, golden=False):
        m = self.game.create_minion("BG_LOE_077", controller=self.hero, golden=golden)
        self.game.summon(self.hero, m)
        return m

    def generated(self, cid):
        return sum(c.card_id == cid for c in self.hero.hand) + sum(
            c.card_id == cid for h, c in self.game.pending_hand_queue if h is self.hero)

    def test_current_tier_one_battlecry_coverage_is_complete(self):
        expected = {d.id for d in self.db.pool_minions() if d.tech_level == 1 and "battlecry" in d.keywords}
        self.assertEqual(expected, {"BG20_100", "BG26_135", "BG31_330"})
        self.assertFalse(expected - self.module.BATTLECRY_SCRIPTS.keys())
        for cid in expected:
            self.assertIn(self.db.golden_version(self.db.get(cid)).id, self.module.BATTLECRY_SCRIPTS)

    def test_normal_razorfen_gives_two_gems_and_one_trigger(self):
        self.card("BG20_100", play=True)
        self.assertEqual(self.generated("BG20_GEM"), 2)
        self.assertEqual(self.hero.get(self.Tag.COUNTER_BATTLECRIES), 1)

    def test_golden_razorfen_gives_four_gems_and_one_trigger(self):
        self.card("BG20_100", golden=True, play=True)
        self.assertEqual(self.generated("BG20_GEM"), 4)
        self.assertEqual(self.hero.get(self.Tag.COUNTER_BATTLECRIES), 1)
        self.assertEqual(self.hero.get(self.Tag.GOLDEN_MINIONS_PLAYED), 1)

    def test_multiple_normal_branns_do_not_stack(self):
        self.brann()
        self.brann()
        self.card("BG20_100", play=True)
        self.assertEqual(self.generated("BG20_GEM"), 4)
        self.assertEqual(self.hero.get(self.Tag.COUNTER_BATTLECRIES), 2)

    def test_highest_brann_applies_to_full_golden_intrinsic(self):
        self.brann()
        self.brann(golden=True)
        self.card("BG20_100", golden=True, play=True)
        self.assertEqual(self.generated("BG20_GEM"), 12)
        self.assertEqual(self.hero.get(self.Tag.COUNTER_BATTLECRIES), 3)

    def test_multiple_golden_branns_still_only_three_triggers(self):
        self.brann(golden=True)
        self.brann(golden=True)
        self.card("BG20_100", play=True)
        self.assertEqual(self.generated("BG20_GEM"), 6)
        self.assertEqual(self.hero.get(self.Tag.COUNTER_BATTLECRIES), 3)

    def test_dead_or_silenced_brann_does_not_apply(self):
        first = self.brann(golden=True)
        first.set(self.Tag.SILENCED, True)
        second = self.brann()
        second.set(self.Tag.DEAD, True)
        self.assertEqual(self.module.battlecry_multiplier(self.game, self.hero), 1)

    def test_busker_normal_is_one_deferred_gold_not_immediate(self):
        self.hero.gold = 7
        self.card("BG26_135", play=True)
        self.assertEqual(self.hero.gold, 7)
        self.assertEqual(len(self.game.deferred_actions), 1)
        self.game.turn = 2
        self.game._begin_recruit_for(self.hero)
        self.assertEqual(self.hero.gold, 5)  # Turn two income 4 plus the deferred 1.

    def test_busker_golden_is_two_deferred_gold_from_one_trigger(self):
        self.card("BG26_135", golden=True, play=True)
        self.assertEqual(len(self.game.deferred_actions), 1)
        self.assertEqual(self.hero.get(self.Tag.COUNTER_BATTLECRIES), 1)
        self.game.turn = 2
        self.game._begin_recruit_for(self.hero)
        self.assertEqual(self.hero.gold, 6)

    def test_ominous_normal_reduces_next_purchase_by_one(self):
        self.card("BG31_330", play=True)
        self.assertEqual(self.hero.get(self.Tag.NEXT_SPELL_COST_REDUCTION), 1)
        spell = self.game.create_spell("BG28_607", controller=self.hero)
        spell.zone = self.Zone.TAVERN
        self.hero.tavern.append(spell)
        self.hero.gold = 10
        self.assertTrue(self.game.buy_from_tavern(self.hero, spell))
        self.assertEqual(self.hero.gold, 7)
        self.assertEqual(self.hero.get(self.Tag.NEXT_SPELL_COST_REDUCTION), 0)

    def test_ominous_golden_discount_two_from_one_trigger(self):
        self.card("BG31_330", golden=True, play=True)
        self.assertEqual(self.hero.get(self.Tag.NEXT_SPELL_COST_REDUCTION), 2)
        self.assertEqual(self.hero.get(self.Tag.COUNTER_BATTLECRIES), 1)

    def test_shell_normal_coin_generation_does_not_occupy_spell_pool_or_play(self):
        before = self.game.spell_pool.available("BG28_810")
        self.hero.gold = 0
        self.card("BG23_002", play=True)
        self.assertEqual(self.generated("BG28_810"), 1)
        self.assertEqual(self.game.spell_pool.available("BG28_810"), before)
        self.assertEqual(self.hero.gold, 0)

    def test_shell_golden_generates_two_coins_from_one_trigger(self):
        self.card("BG23_002", golden=True, play=True)
        self.assertEqual(self.generated("BG28_810"), 2)
        self.assertEqual(self.hero.get(self.Tag.COUNTER_BATTLECRIES), 1)

    def test_sanguine_normal_current_amount(self):
        self.card("BG23_017", play=True)
        self.assertEqual((self.hero.get(self.Tag.BLOOD_GEM_BONUS_ATK),
                          self.hero.get(self.Tag.BLOOD_GEM_BONUS_HEALTH)), (2, 1))
        self.assertEqual(self.hero.get(self.Tag.COUNTER_BATTLECRIES), 1)

    def test_sanguine_golden_current_amount_then_independent_deathrattle(self):
        m = self.card("BG23_017", golden=True, play=True)
        self.assertEqual((self.hero.get(self.Tag.BLOOD_GEM_BONUS_ATK),
                          self.hero.get(self.Tag.BLOOD_GEM_BONUS_HEALTH)), (4, 2))
        self.game.run_script_hook(m, "deathrattle")
        self.assertEqual((self.hero.get(self.Tag.BLOOD_GEM_BONUS_ATK),
                          self.hero.get(self.Tag.BLOOD_GEM_BONUS_HEALTH)), (8, 4))
        self.assertEqual(self.hero.get(self.Tag.COUNTER_BATTLECRIES), 1)

    def test_normal_choose_one_is_manual_and_not_a_battlecry(self):
        self.brann(golden=True)
        self.card("BG32_237", play=True)
        self.assertEqual(self.hero.get(self.Tag.TAVERN_SPELL_EXTRA_ATK), 0)
        choice = self.game.pending_choices.pop(0)
        self.assertEqual(choice.kind, "choose_one")
        choice.choose(0)
        self.assertEqual(self.hero.get(self.Tag.TAVERN_SPELL_EXTRA_ATK), 1)
        self.assertEqual(self.hero.get(self.Tag.COUNTER_BATTLECRIES), 0)

    def test_golden_choose_one_intrinsic_is_two_and_ignores_brann(self):
        self.brann(golden=True)
        self.card("BG32_237", golden=True, play=True)
        choice = self.game.pending_choices.pop(0)
        choice.choose(1)
        self.assertEqual(self.hero.get(self.Tag.TAVERN_SPELL_EXTRA_HEALTH), 2)
        self.assertEqual(self.hero.get(self.Tag.COUNTER_BATTLECRIES), 0)
        self.assertEqual(self.hero.get(self.Tag.GOLDEN_MINIONS_PLAYED), 1)

    def test_choose_both_resolves_each_branch_once(self):
        m = self.card("BG32_237", golden=True)
        m.set(self.Tag.CHOOSE_BOTH, True)
        self.brann()
        self.game.play_minion(self.hero, m)
        self.assertEqual(self.game.pending_choices, [])
        self.assertEqual((self.hero.get(self.Tag.TAVERN_SPELL_EXTRA_ATK),
                          self.hero.get(self.Tag.TAVERN_SPELL_EXTRA_HEALTH)), (2, 2))
        self.assertEqual(self.hero.get(self.Tag.COUNTER_BATTLECRIES), 0)

    def test_native_trigger_battlecry_action_uses_migrated_dispatcher(self):
        from hsrl2.actions.trigger import TriggerBattlecry
        self.brann(golden=True)
        source = self.game.create_minion("BG20_100", controller=self.hero, golden=True)
        self.game.summon(self.hero, source)
        self.game.run_actions(TriggerBattlecry(source))
        self.assertEqual(self.generated("BG20_GEM"), 12)
        self.assertEqual(self.hero.get(self.Tag.COUNTER_BATTLECRIES), 3)

    def test_choose_one_cannot_be_retriggered_as_battlecry(self):
        from hsrl2.actions.trigger import TriggerBattlecry
        source = self.game.create_minion("BG32_237", controller=self.hero, golden=True)
        self.game.summon(self.hero, source)
        self.game.run_actions(TriggerBattlecry(source))
        self.assertEqual(self.hero.get(self.Tag.TAVERN_SPELL_EXTRA_ATK), 0)
        self.assertEqual(self.hero.get(self.Tag.COUNTER_BATTLECRIES), 0)
        self.assertEqual(self.game.pending_choices, [])

    def test_non_battlecry_play_has_no_spurious_battlecry_events(self):
        from hsrl2.events import Listener
        calls = []
        self.game.events.register(Listener("battlecry_trigger", self.hero, lambda g, **kw: calls.append(kw)))
        self.brann(golden=True)
        self.card("BG25_001", golden=True, play=True)
        self.assertEqual(calls, [])
        self.assertEqual(self.hero.get(self.Tag.COUNTER_BATTLECRIES), 0)

    def test_unknown_battlecry_rejected_before_hand_or_board_changes(self):
        source = self.card("BG25_011", golden=True)
        with self.assertRaisesRegex(self.module.UnmigratedBattlecry, "not been migrated"):
            self.game.play_minion(self.hero, source)
        self.assertEqual(self.hero.hand, [source])
        self.assertEqual(self.hero.board, [])

    def test_missing_choose_one_handler_cannot_become_a_no_op(self):
        source = self.card("BG31_320", golden=True)
        source.scripts = None
        with self.assertRaises(self.module.UnmigratedBattlecry):
            self.game.play_minion(self.hero, source)
        self.assertEqual(self.hero.hand, [source])
        self.assertEqual(self.hero.board, [])

    def test_unknown_repeat_source_fails_before_battlecry_play(self):
        aura = self.card("BG25_001", play=True)
        aura.set(self.Tag.BATTLECRY_DOUBLER, True)
        source = self.card("BG20_100")
        with self.assertRaisesRegex(self.module.UnmigratedBattlecry, "repeat source"):
            self.game.play_minion(self.hero, source)
        self.assertEqual(self.hero.hand, [source])
        self.assertEqual(self.hero.board, [aura])

    def test_triple_golden_razorfen_can_play_then_reward_remains_separate(self):
        ms = []
        for _ in range(3):
            self.assertTrue(self.game.minion_pool.acquire("BG20_100"))
            ms.append(self.card("BG20_100"))
        self.game.check_for_triple(self.hero, ms[-1])
        golden, = self.hero.hand
        self.game.play_minion(self.hero, golden)
        self.assertEqual(self.generated("BG20_GEM"), 4)
        self.assertEqual(self.generated(self.rules.TRIPLE_REWARD), 1)
        self.assertEqual(self.game.pending_choices, [])
        self.assertEqual(self.hero.get(self.Tag.COUNTER_BATTLECRIES), 1)

    def test_installation_is_instance_local_and_restores_registry_unchanged(self):
        from hsrl2.scripts import REGISTRY
        before = dict(REGISTRY)
        other = self.Game([self.Hero("C", "C"), self.Hero("D", "D")], self.db, seed=3)
        old = other._run_play_effects.__func__
        with self.assertRaisesRegex(ValueError, "abort"):
            with self.module.installed_battlecry_rules(other) as report:
                self.assertEqual(len(report.migrated_ids), 12)
                self.assertFalse(report.full_game_ready)
                raise ValueError("abort")
        self.assertIs(other._run_play_effects.__func__, old)
        self.assertEqual(REGISTRY, before)
        self.assertNotIn("_bg_ai_migrated_battlecry_ids", other.__dict__)


if __name__ == "__main__":
    unittest.main()
