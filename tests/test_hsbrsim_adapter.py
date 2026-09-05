"""Behavioral contracts exercised against the pinned external hsrl2 engine."""
import json
import os
from pathlib import Path
import unittest

from bg_ai.hsbrsim_adapter import (EarlyRecruitFixtureEngine, EarlyRecruitFixtureSpec,
                                  SUPPORTED_MINIONS, SUPPORTED_SPELLS,
                                  UnsupportedRecruitTransition)
from bg_ai.hsbrsim_data import build_current_database
from bg_ai.recruiting import RecruitAction, RecruitCoverageError
from bg_ai.timed_recruiting import TimedRecruitEngine
from bg_ai.turn_budget import TimingProfile

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = Path(os.environ.get("HSBRSIM_ROOT", str(ROOT.parent / "research/HSBRSIM")))


def zones(first):
    return (tuple(first),) + ((),) * 7


@unittest.skipUnless((EXTERNAL / 'hsrl2/game.py').is_file(), 'Explicit pinned external HSBRSIM checkout required')
class ActualRecruitAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.built = build_current_database(EXTERNAL,
            fixture_minion_ids=SUPPORTED_MINIONS, fixture_spell_ids=SUPPORTED_SPELLS)
        cls.rules = json.loads((ROOT / 'data/ruleset.json').read_text())
        cls.profile = TimingProfile.load(ROOT / 'config/turn-budget.json')

    def make(self, *, board=(), hand=(), ms=60000, max_turns=2, hero=None, seed=2):
        spec = EarlyRecruitFixtureSpec(max_turns=max_turns,
             initial_boards=zones(board), initial_hands=zones(hand),
             hero_ids=((hero,) + ('TB_BaconShop_HERO_34',) * 7) if hero else ('TB_BaconShop_HERO_34',) * 8)
        engine = EarlyRecruitFixtureEngine(self.built.db, self.rules,
                timer_ms=lambda p, t: ms, provenance=self.built.provenance, fixture=spec)
        timed = TimedRecruitEngine.for_fixture(engine, self.profile, self.rules)
        return timed, timed.reset(seed=seed)

    def action(self, observation, kind, entity_id=None, **kw):
        return next(a for a in observation.legal_actions if a.kind == kind
                    and (entity_id is None or a.entity_id == entity_id)
                    and all(getattr(a, k) == v for k,v in kw.items()))

    def test_buy_does_not_play_and_play_generates_explicit_gem_choices(self):
        t, o = self.make()
        geom = next(c for c in o.private_state['shop'] if c['card_id'] == 'BG20_100')
        o = t.step(self.action(o, 'buy', geom['entity_id']))
        self.assertEqual(o.private_state['board'], [])
        self.assertEqual(len(o.private_state['hand']), 1)
        self.assertEqual(o.action_budget['actions_used'], 1)
        o = t.step(self.action(o, 'play', geom['entity_id'], position=0))
        self.assertEqual([c['card_id'] for c in o.private_state['hand']], ['BG20_GEM']*2)
        self.assertEqual(o.private_state['board'][0]['attack'], 2)
        gem = o.private_state['hand'][0]
        o = t.step(self.action(o, 'cast_spell', gem['entity_id']))
        self.assertTrue(all(a.kind == 'choose' for a in o.legal_actions))
        target = geom['entity_id']
        o = t.step(next(a for a in o.legal_actions if a.target_id == target))
        self.assertEqual(o.private_state['board'][0]['attack'], 3)
        self.assertEqual(o.private_state['board'][0]['health'], 2)
        self.assertEqual(o.action_budget['actions_used'], 4)
        self.assertEqual(len(o.private_state['hand']), 1)

    def test_activate_target_is_explicit_and_charged_separately(self):
        t, o = self.make(board=('BG36_345', 'BGS_119'))
        guard, target = o.private_state['board']
        o = t.step(self.action(o, 'activate', guard['entity_id']))
        self.assertEqual(o.private_state['gold'], 2)
        self.assertEqual(o.private_state['board'][1]['attack'], target['attack'])
        self.assertEqual([a.target_id for a in o.legal_actions], [target['entity_id']])
        o = t.step(o.legal_actions[0])
        definition = self.built.db.get('BG36_345')
        self.assertEqual(o.private_state['board'][1]['attack'], target['attack'] + definition.num(0))
        self.assertEqual(o.action_budget['actions_used'], 2)
        self.assertFalse(any(a.kind == 'activate' for a in o.legal_actions))

    def test_choose_one_has_no_automatic_branch(self):
        t, o = self.make(hand=('BG32_237',))
        o = t.step(self.action(o, 'play', position=0))
        self.assertEqual(o.private_state['counters']['TAVERN_SPELL_EXTRA_ATK'], 0)
        self.assertEqual(len(o.legal_actions), 2)
        o = t.step(next(a for a in o.legal_actions if a.choice_index == 1))
        self.assertEqual(o.private_state['counters']['TAVERN_SPELL_EXTRA_HEALTH'], 1)
        self.assertEqual(o.private_state['counters']['TAVERN_SPELL_EXTRA_ATK'], 0)

    def test_tavern_spell_can_target_a_tavern_minion(self):
        t, o = self.make(hand=('BG28_897',))
        target = o.private_state['shop'][0]
        o = t.step(self.action(o, 'cast_spell'))
        o = t.step(next(a for a in o.legal_actions if a.target_id == target['entity_id']))
        changed = next(c for c in o.private_state['shop'] if c['entity_id'] == target['entity_id'])
        d = self.built.db.get('BG28_897')
        self.assertEqual(changed['attack'], target['attack'] + d.num(0))
        self.assertEqual(changed['health'], target['health'] + d.num(1))

    def test_spell_reenters_pool_at_purchase_and_is_not_returned_twice(self):
        t, o = self.make()
        coin = next(c for c in o.private_state['shop'] if c['card_id'] == 'BG28_810')
        pool = t.engine.game.spell_pool
        before = pool.available('BG28_810')
        o = t.step(self.action(o, 'buy', coin['entity_id']))
        self.assertEqual(pool.available('BG28_810'), before + 1)
        o = t.step(self.action(o, 'cast_spell', coin['entity_id']))
        self.assertEqual(pool.available('BG28_810'), before + 1)

    def test_end_turn_never_plays_hand_and_global_round_waits_for_eight(self):
        t, o = self.make(hand=('BG20_100',))
        raw = t.engine
        for i in range(8):
            self.assertEqual(o.player_id, i)
            self.assertEqual(o.turn, 1)
            o = t.step(self.action(o, 'end_turn'))
            if i < 7:
                self.assertEqual(raw.game.turn, 1)
                self.assertEqual(raw.finished_players, set(range(i+1)))
        self.assertEqual(o.turn, 2)
        self.assertEqual(o.player_id, 0)
        self.assertEqual(len(raw.game.heroes[0].hand), 1)
        self.assertEqual(raw.game.heroes[0].board, [])
        self.assertEqual(raw.finished_players, set())
        self.assertEqual(raw.game.heroes[0].gold, 4)
        for i in range(8):
            o = t.step(self.action(o, 'end_turn'))
        self.assertIsNone(o)
        self.assertEqual(raw.game.turn, 2)
        self.assertEqual(len(raw.terminal_boards()), 8)
        with self.assertRaises(RecruitCoverageError): raw.final_placements()

    def test_freeze_toggles_and_manual_refresh_replaces_frozen_shop(self):
        t, o = self.make()
        before = {c['entity_id'] for c in o.private_state['shop']}
        o = t.step(self.action(o, 'freeze'))
        self.assertTrue(o.private_state['frozen'])
        o = t.step(self.action(o, 'freeze'))
        self.assertFalse(o.private_state['frozen'])
        o = t.step(self.action(o, 'freeze'))
        o = t.step(self.action(o, 'refresh'))
        self.assertFalse(o.private_state['frozen'])
        self.assertFalse(before & {c['entity_id'] for c in o.private_state['shop']})
        self.assertEqual(o.private_state['gold'], 2)

    def test_frozen_shop_with_spell_refills_missing_minion(self):
        t, o = self.make()
        buy = next(a for a in o.legal_actions if a.kind == 'buy' and a.entity_id != o.private_state['shop'][-1]['entity_id'])
        o = t.step(buy)
        preserved = {c['entity_id'] for c in o.private_state['shop']}
        o = t.step(self.action(o, 'freeze'))
        for i in range(8):
            o = t.step(self.action(o, 'end_turn'))
        self.assertEqual(o.turn, 2)
        self.assertEqual(sum(c['card_type']=='minion' for c in o.private_state['shop']), 3)
        self.assertEqual(sum(c['card_type']=='spell' for c in o.private_state['shop']), 1)
        self.assertTrue(preserved <= {c['entity_id'] for c in o.private_state['shop']})

    def test_only_board_minions_sell_and_moves_are_single_commands(self):
        t, o = self.make(board=('BGS_119','BG25_001'), hand=('BG20_100',))
        board = o.private_state['board']
        self.assertEqual({a.entity_id for a in o.legal_actions if a.kind=='sell'}, {c['entity_id'] for c in board})
        o = t.step(self.action(o, 'move', board[0]['entity_id'], position=1))
        self.assertEqual(o.private_state['board'][1]['entity_id'], board[0]['entity_id'])
        o = t.step(self.action(o, 'sell', board[0]['entity_id']))
        self.assertEqual(o.private_state['gold'], 4)
        self.assertEqual(o.action_budget['actions_used'], 2)

    def test_pending_choice_timeout_errors_instead_of_selecting(self):
        t, o = self.make(board=('BGS_119',), hand=('BG28_897',), ms=7000)
        before = t.engine.game.heroes[0].board[0].atk
        with self.assertRaisesRegex(UnsupportedRecruitTransition, 'Pending-choice timeout'):
            t.step(self.action(o, 'cast_spell'))
        self.assertEqual(t.engine.game.heroes[0].board[0].atk, before)
        self.assertEqual(len(t.engine.game.pending_choices), 1)

    def test_budget_exhaustion_ends_current_player_without_refund(self):
        t, o = self.make(ms=8000)
        for _ in range(3):
            o = t.step(self.action(o, 'freeze'))
        self.assertEqual(o.player_id, 1)
        self.assertIn((0,1), t._closed)
        self.assertEqual(t._budgets[(0,1)].remaining_actions, 0)
        self.assertEqual(t.engine.finished_players, {0})

    def test_fork_replays_callbacks_without_cross_mutation_or_budget_refill(self):
        t, o = self.make(hand=('BG32_237',))
        o = t.step(self.action(o, 'play', position=0))
        branch = t.fork()
        self.assertEqual(branch._visible, o)
        bo = branch.step(next(a for a in branch._visible.legal_actions if a.choice_index==0))
        self.assertEqual(bo.action_budget['actions_used'], 2)
        self.assertEqual(bo.private_state['counters']['TAVERN_SPELL_EXTRA_ATK'], 1)
        self.assertEqual(t.engine.game.heroes[0].get(t.engine._T.TAVERN_SPELL_EXTRA_ATK,0), 0)
        self.assertEqual(len(t.engine.game.pending_choices), 1)

    def test_policy_observation_omits_opponent_hidden_state(self):
        t, o = self.make()
        before = json.dumps(o.public_state, sort_keys=True)
        t.engine.game.heroes[7].gold = 99
        t.engine.game.heroes[7].hand.append(t.engine.game.create_spell('BG28_810',controller=t.engine.game.heroes[7]))
        after = json.dumps(t.engine.observe().public_state, sort_keys=True)
        self.assertEqual(before, after)
        for player in o.public_state['players']:
            self.assertNotIn('board',player)
            self.assertNotIn('hand',player)
            self.assertNotIn('shop',player)
            self.assertNotIn('gold',player)
        self.assertNotIn('rng', str(o.private_state).lower())

    def test_unsupported_card_and_full_game_gate_fail_closed(self):
        t, o = self.make()
        with self.assertRaises(RecruitCoverageError):
            TimedRecruitEngine(t.engine,self.profile,self.rules)
        rogue=t.engine.game.create_minion('BG36_201',controller=t.engine.game.heroes[0])
        t.engine.game.heroes[0].hand.append(rogue)
        with self.assertRaisesRegex(UnsupportedRecruitTransition,'Unsupported generated card'):
            t.engine.observe()

    def test_pyramad_power_steals_to_hand_and_charges_action(self):
        t, o = self.make(hero='TB_BaconShop_HERO_39', hand=('BG28_810','BG28_810'))
        for _ in range(2):
            o=t.step(self.action(o,'cast_spell'))
        d=t.engine.game.hero_power_def(t.engine.game.heroes[0])
        self.assertGreaterEqual(t.engine.game.heroes[0].gold,d.cost)
        o=t.step(self.action(o,'hero_power'))
        self.assertEqual(len(o.private_state['hand']),1)
        self.assertEqual(o.private_state['board'],[])
        m=o.private_state['hand'][0]
        self.assertEqual(m['health'],2*self.built.db.get(m['card_id']).health)
        self.assertEqual(o.action_budget['actions_used'],3)

if __name__=='__main__': unittest.main()
