"""Real-engine tests for ordinary triples and the separate reward/play/choice steps."""
from contextlib import ExitStack
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
ENGINE = Path(os.environ.get("HSBRSIM_ROOT", ROOT.parent / "research/HSBRSIM"))


@unittest.skipUnless((ENGINE / "hsrl2/game.py").is_file(), "external pinned HSBRSIM checkout unavailable")
class TestRecruitRules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ENGINE))
        from bg_ai.hsbrsim_data import build_current_database
        from bg_ai import hsbrsim_rules
        from hsrl2.game import Game
        from hsrl2.hero import Hero
        from hsrl2.tags import GameTag, Zone, Race
        from hsrl2.entity import Buff
        from hsrl2.events import Listener
        cls.db = build_current_database(ENGINE).db
        cls.rules = hsbrsim_rules
        cls.Game, cls.Hero, cls.Tag, cls.Zone, cls.Race = Game, Hero, GameTag, Zone, Race
        cls.Buff, cls.Listener = Buff, Listener

    def setUp(self):
        self.game = self.Game([self.Hero("TEST_A", "A"), self.Hero("TEST_B", "B")], self.db, seed=10)
        self.a, self.b = self.game.heroes
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(self.rules.installed_recruit_rules(self.game))

    def obtain(self, cid="BG25_001", *, board=False):
        self.assertTrue(self.game.minion_pool.acquire(cid))
        m = self.game.create_minion(cid, controller=self.a)
        if board:
            self.game.summon(self.a, m)
        else:
            self.a.add_to_hand(m)
        return m

    def triple(self, cid="BG25_001", *, boards=2):
        ms = [self.obtain(cid, board=i < boards) for i in range(3)]
        self.game.check_for_triple(self.a, ms[-1])
        return ms, next(m for m in self.a.hand if m.is_golden)

    def test_mixed_zone_triple_returns_golden_to_hand_without_reward(self):
        ms, golden = self.triple()
        self.assertEqual(self.a.board, [])
        self.assertEqual(self.a.hand, [golden])
        self.assertTrue(golden.has(self.Tag.TRIPLE_REWARD_PENDING))
        self.assertEqual(self.game.pending_choices, [])
        self.assertTrue(all(m.zone == self.Zone.REMOVED for m in ms))
        self.assertEqual((golden.atk, golden.max_health), (4, 2))

    def test_reward_granted_only_when_golden_played_and_uses_tavern_tier(self):
        _, golden = self.triple()
        self.a.set(self.Tag.TAVERN_TIER, 5)
        self.assertTrue(self.game.play_minion(self.a, golden))
        reward, = self.a.hand
        self.assertEqual(reward.card_id, self.rules.TRIPLE_REWARD)
        self.assertEqual(reward._bg_ai_triple_reward_tier, 6)
        self.assertEqual(self.game.pending_choices, [])
        self.assertFalse(golden.has(self.Tag.TRIPLE_REWARD_PENDING))
        self.game._grant_triple_reward(self.a, golden)
        self.assertEqual(self.a.hand, [reward])

    def test_reward_tier_is_frozen_when_granted_not_when_cast(self):
        self.a.set(self.Tag.TAVERN_TIER, 3)
        reward = self.rules.grant_triple_reward(self.game, self.a)
        self.a.set(self.Tag.TAVERN_TIER, 5)
        self.assertTrue(self.game.play_spell(self.a, reward))
        choice, = self.game.pending_choices
        self.assertEqual(choice.kind, "triple_reward")
        self.assertTrue(all(self.db.get(cid).tech_level == 4 for cid in choice.options))
        self.assertEqual(len(choice.options), 3)
        self.assertEqual(self.a.hand, [])

    def test_tier_six_reward_caps_at_six(self):
        self.a.set(self.Tag.TAVERN_TIER, 6)
        reward = self.rules.grant_triple_reward(self.game, self.a)
        self.assertEqual(reward._bg_ai_triple_reward_tier, 6)

    def test_reward_choice_manual_and_acquires_once(self):
        reward = self.rules.grant_triple_reward(self.game, self.a)
        before = dict(self.game.minion_pool._available)
        self.game.play_spell(self.a, reward)
        choice, = self.game.pending_choices
        self.assertEqual(dict(self.game.minion_pool._available), before)
        pick = choice.options[1]
        choice.choose(1)
        self.game.pending_choices.remove(choice)
        self.assertEqual(self.a.hand[0].card_id, pick)
        self.assertEqual(self.game.minion_pool.available(pick), before[pick] - 1)
        with self.assertRaisesRegex(self.rules.UnsupportedTripleTransition, "already resolved"):
            choice.choose(1)

    def test_empty_reward_pool_grants_pouch_to_hand_without_free_gold(self):
        for cid in self.game.minion_pool.candidates(max_tier=4, min_tier=4):
            while self.game.minion_pool.acquire(cid):
                pass
        self.a.gold = 1
        reward = self.rules.grant_triple_reward(self.game, self.a, reward_tier=4)
        self.game.play_spell(self.a, reward)
        self.assertEqual(self.game.pending_choices, [])
        pouch, = self.a.hand
        self.assertEqual(pouch.card_id, self.rules.CONSOLATION_POUCH)
        self.assertEqual(self.a.gold, 1)
        self.game.play_spell(self.a, pouch)
        self.assertEqual(self.a.gold, 4)
        self.assertEqual(self.a.hand, [])

    def test_fewer_than_three_choices_and_lobby_tribes_respected(self):
        self.game.minion_pool.active_races = {self.Race.UNDEAD}
        keep = next(d.id for d in self.db.pool_minions() if d.tech_level == 2 and d.race == self.Race.UNDEAD)
        for cid in self.game.minion_pool.candidates(max_tier=2, min_tier=2):
            if cid != keep:
                while self.game.minion_pool.acquire(cid):
                    pass
        reward = self.rules.grant_triple_reward(self.game, self.a, reward_tier=2)
        self.game.play_spell(self.a, reward)
        self.assertEqual(self.game.pending_choices[0].options, [keep])

    def test_triple_preserves_buffs_without_reapplying_gain_events(self):
        ms = [self.obtain(board=i < 2) for i in range(3)]
        ms[0].add_buff(self.Buff(2, 3, source_id="test-buff"))
        ms[1].add_buff(self.Buff(4, 5, gem=True))
        ms[1].set(self.Tag.GEMS_PLAYED_ON, 1)
        ms[2].set(self.Tag.DIVINE_SHIELD, True)
        # A permanent race aura affects the new entity once, not once per source.
        self.a.race_auras = {self.Race.UNDEAD: (7, 11)}
        gained = []
        self.game.events.register(self.Listener("buff_applied", self.a,
                                                lambda g, **kw: gained.append(kw)))
        self.game.check_for_triple(self.a, ms[-1])
        golden, = self.a.hand
        self.assertEqual((golden.atk, golden.max_health), (4 + 2 + 4 + 7, 2 + 3 + 5 + 11))
        self.assertEqual(golden.get(self.Tag.GEMS_PLAYED_ON), 1)
        self.assertTrue(golden.divine_shield)
        self.assertTrue(golden.taunt)
        self.assertEqual(gained, [])
        self.assertIsNot(golden._buffs[0], ms[0]._buffs[0])

    def test_consumed_participant_listeners_are_removed(self):
        ms = [self.obtain(board=i < 2) for i in range(3)]
        calls = []
        self.game.events.register(self.Listener("test_event", ms[0], lambda g, **kw: calls.append(1)))
        self.game.check_for_triple(self.a, ms[-1])
        self.game.events.fire(self.game, "test_event")
        self.assertEqual(calls, [])

    def test_normal_triple_then_sell_returns_exact_three_pool_copies(self):
        initial = self.game.minion_pool.available("BG25_001")
        _, golden = self.triple()
        self.assertEqual(self.game.minion_pool.available("BG25_001"), initial - 3)
        self.game.play_minion(self.a, golden)
        self.assertTrue(self.game.sell_minion(self.a, golden))
        self.assertEqual(self.game.minion_pool.available("BG25_001"), initial)

    def test_reward_pick_can_trigger_new_triple(self):
        first = self.obtain()
        self.obtain()
        reward = self.rules.grant_triple_reward(self.game, self.a, reward_tier=1)
        for cid in self.game.minion_pool.candidates(max_tier=1, min_tier=1):
            if cid != first.card_id:
                while self.game.minion_pool.acquire(cid):
                    pass
        self.game.play_spell(self.a, reward)
        choice = self.game.pending_choices.pop(0)
        choice.choose(0)
        golden, = self.a.hand
        self.assertTrue(golden.is_golden)
        self.assertTrue(golden.has(self.Tag.TRIPLE_REWARD_PENDING))
        self.assertEqual(self.game.pending_choices, [])

    def test_unsupported_merge_is_rejected_before_removing_participants(self):
        for kind in ("magnetic", "dark_gift", "counter", "temporary", "set_stats"):
            with self.subTest(kind=kind):
                game = self.Game([self.Hero("A", "A"), self.Hero("B", "B")], self.db, seed=9)
                hero = game.heroes[0]
                ms = []
                for _ in range(3):
                    m = game.create_minion("BG25_001", controller=hero)
                    hero.add_to_hand(m)
                    ms.append(m)
                if kind == "magnetic": game._magnetic_stack[ms[0].uuid] = ["BG25_001"]
                if kind == "dark_gift": ms[0].set(self.Tag.DARK_GIFT, "TEST")
                if kind == "counter": ms[0]._unaudited_counter = 4
                if kind == "temporary": ms[0].add_buff(self.Buff(1, 1, temporary=True))
                if kind == "set_stats": ms[0].set(self.Tag.BASE_ATK, 20)
                with self.rules.installed_recruit_rules(game):
                    with self.assertRaises(self.rules.UnsupportedTripleTransition):
                        game.check_for_triple(hero, ms[-1])
                self.assertEqual(hero.hand, ms)
                self.assertEqual(hero.board, [])

    def test_golden_battlecry_play_stays_blocked_before_mutation(self):
        _, golden = self.triple("BG20_100", boards=0)
        before = list(self.a.hand)
        with self.assertRaisesRegex(self.rules.UnsupportedTripleTransition, "Golden Battlecry"):
            self.game.play_minion(self.a, golden)
        self.assertEqual(self.a.hand, before)
        self.assertEqual(self.a.board, [])

    def test_golden_choose_one_also_stays_blocked(self):
        golden = self.game.create_minion("BG32_237", controller=self.a, golden=True)
        self.a.add_to_hand(golden)
        with self.assertRaisesRegex(self.rules.UnsupportedTripleTransition, "Choose One"):
            self.game.play_minion(self.a, golden)
        self.assertEqual(self.a.hand, [golden])
        self.assertEqual(self.a.board, [])

    def test_reward_without_fixed_tier_is_rejected(self):
        raw = self.game.create_spell(self.rules.TRIPLE_REWARD, controller=self.a)
        raw.scripts = self.rules.TripleRewardScript
        with self.assertRaisesRegex(self.rules.UnsupportedTripleTransition, "frozen reward tier"):
            self.game.run_script_hook(raw, "on_play")

    def test_instance_rules_restore_without_modifying_other_games(self):
        untouched = self.Game([self.Hero("A", "A"), self.Hero("B", "B")], self.db, seed=9)
        original = untouched.check_for_triple.__func__
        with self.assertRaisesRegex(ValueError, "abort"):
            with self.rules.installed_recruit_rules(untouched):
                self.assertIsNot(untouched.check_for_triple.__func__, original)
                raise ValueError("abort")
        self.assertIs(untouched.check_for_triple.__func__, original)
        self.assertNotIn("check_for_triple", untouched.__dict__)


if __name__ == "__main__":
    unittest.main()
