"""Actual-engine behavior for the complete current Tier1 opening pool."""
import json
import os
from pathlib import Path
import unittest

from bg_ai.hsbrsim_opening import *
from bg_ai.turn_budget import TimingProfile
from bg_ai.recruiting import RecruitCoverageError, RecruitAction

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = Path(os.environ.get('HSBRSIM_ROOT', str(ROOT.parent / 'research/HSBRSIM')))

def zones(first): return (tuple(first),) + ((),)*7

@unittest.skipUnless((EXTERNAL/'hsrl2/game.py').is_file(), 'Pinned external HSBRSIM checkout required')
class CurrentOpeningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = json.loads((ROOT/'data/ruleset.json').read_text())
        cls.cards = {c['id']:c for c in json.loads((ROOT/'data/reference_cards.json').read_text())}
        cls.profile = TimingProfile.load(ROOT/'config/turn-budget.json')
        cls.databases = {}

    def make(self, *, board=(), hand=(), tribes=None, ms=60000, seed=3):
        if tribes is None:
            required = set()
            for cid in board+hand:
                races=self.cards[cid].get('races',[])
                if races: required.add(races[0])
            for tribe in sorted(TRIBES):
                if len(required)>=5: break
                required.add(tribe)
            tribes=tuple(sorted(required))
        spec=OpeningFixtureSpec(valid_tribes=tuple(tribes), initial_boards=zones(board), initial_hands=zones(hand))
        key=spec.valid_tribes
        if key not in self.databases:
            self.databases[key]=build_opening_database(EXTERNAL,valid_tribes=key)
        current=self.databases[key]
        raw=OpeningRecruitEngine(current.db,self.rules,timer_ms=lambda p,t:ms,provenance=current.provenance,fixture=spec)
        timed=TimedOpeningEngine.for_fixture(raw,self.profile,self.rules)
        return timed,timed.reset(seed=seed)

    def action(self,o,kind,eid=None,**attrs):
        return next(a for a in o.legal_actions if a.kind==kind and (eid is None or a.entity_id==eid)
                    and all(getattr(a,k)==v for k,v in attrs.items()))

    def end(self,t,o):
        while o is not None:
            o=t.step(self.action(o,'end_turn'))
        return t.engine.combat_snapshot(0)

    def test_exact_current_pool_and_all_tribes_without_missing_mechs(self):
        expected={cid for cid in self.rules['active']['minion_ids'] if self.cards[cid].get('techLevel')==1}
        spells={cid for cid in self.rules['active']['tavern_spell_ids'] if self.cards[cid].get('techLevel')==1}
        self.assertEqual(expected,TIER1_MINIONS)
        self.assertEqual(len(expected),22)
        self.assertEqual(spells,TIER1_SPELLS)
        self.assertEqual(len(spells),8)
        t,o=self.make(tribes=('MECHANICAL','BEAST','DEMON','NAGA','PIRATE'))
        self.assertIn('MECH',t.engine.fixture.valid_tribes)
        self.assertIn('BG26_146',t.engine.allowed_minions)
        self.assertIn('BG29_611',t.engine.allowed_minions)

    def test_dual_tribe_seer_offered_if_either_tribe_is_in(self):
        for tribe in ('DEMON','NAGA'):
            others=tuple(t for t in sorted(TRIBES) if t not in ('DEMON','NAGA'))[:4]
            t,o=self.make(tribes=(tribe,)+others)
            self.assertIn('BG31_330',t.engine.allowed_minions)
        t,o=self.make(tribes=('BEAST','DRAGON','ELEMENTAL','MECH','MURLOC'))
        self.assertNotIn('BG31_330',t.engine.allowed_minions)

    def test_all_twenty_two_current_minions_have_explicit_play_transitions(self):
        exercised=set()
        for cid in sorted(TIER1_MINIONS):
            with self.subTest(card_id=cid):
                t,o=self.make(hand=(cid,))
                eid=o.private_state['hand'][0]['entity_id']
                if cid in UNVERIFIED_SELF_PLAY:
                    before=list(t.engine.game.heroes[0].hand)
                    with self.assertRaisesRegex(UnsupportedRecruitTransition,'Self-play trigger'):
                        t.step(self.action(o,'play',eid,position=0))
                    self.assertEqual(t.engine.game.heroes[0].hand,before)
                    self.assertEqual(t.engine.game.heroes[0].board,[])
                    continue
                o=t.step(self.action(o,'play',eid,position=0))
                self.assertEqual(o.private_state['board'][0]['card_id'],cid)
                self.assertEqual(o.action_budget['actions_used'],1)
                exercised.add(cid)
        self.assertEqual(exercised,TIER1_MINIONS-UNVERIFIED_SELF_PLAY)

    def test_myrmidon_gives_spell_that_requires_a_separate_target(self):
        t,o=self.make(hand=('BG23_000',))
        o=t.step(self.action(o,'play',position=0))
        self.assertEqual([x['card_id'] for x in o.private_state['hand']],['BG23_000t'])
        target=o.private_state['board'][0]
        o=t.step(self.action(o,'cast_spell'))
        self.assertEqual(o.private_state['board'][0]['attack'],target['attack'])
        self.assertTrue(all(a.kind=='choose' for a in o.legal_actions))
        o=t.step(next(a for a in o.legal_actions if a.target_id==target['entity_id']))
        self.assertEqual(o.private_state['board'][0]['attack'],target['attack']+2)
        self.assertEqual(o.action_budget['actions_used'],3)
        snap=self.end(t,o)
        self.assertEqual(snap['board'][0]['attack'],3)
        self.assertTrue(any(e['temporary'] for e in snap['board'][0]['enchantments']))

    def test_unused_spellcraft_disappears_and_flighty_hand_is_exported(self):
        t,o=self.make(hand=('BG23_000','BG32_330'))
        myrm=next(c for c in o.private_state['hand'] if c['card_id']=='BG23_000')
        o=t.step(self.action(o,'play',myrm['entity_id'],position=0))
        snap=self.end(t,o)
        self.assertEqual([x['card_id'] for x in snap['hand']],['BG32_330'])
        self.assertFalse(snap['recruit_return_state']['next_turn_state_available'])

    def test_alliance_branch_then_target_then_buff_and_spell_event(self):
        t,o=self.make(board=('BG36_921',),hand=('BG31_880',))
        before=o.private_state['board'][0]
        o=t.step(self.action(o,'cast_spell'))
        self.assertEqual(o.private_state['pending_choice']['kind'],'choose_one')
        self.assertIsInstance(o.private_state['pending_choice']['options'][0],dict)
        self.assertIn('text',o.private_state['pending_choice']['options'][0])
        o=t.step(self.action(o,'choose',choice_index=0))
        self.assertEqual(o.private_state['pending_choice']['kind'],'spell_target')
        self.assertEqual(o.private_state['board'][0]['health'],before['health'])
        self.assertEqual(len(o.private_state['hand']),1)
        o=t.step(next(a for a in o.legal_actions if a.target_id==before['entity_id']))
        self.assertEqual(o.private_state['board'][0]['attack'],before['attack']+3)
        self.assertEqual(o.private_state['board'][0]['health'],before['health']+1+1)
        self.assertEqual(o.action_budget['actions_used'],3)
        self.assertEqual(o.private_state['hand'],[])

    def test_aureate_is_golden_in_shop_hand_and_board_without_reward(self):
        t,o=self.make(hand=('BG32_236','BG32_236','BG32_236'))
        self.assertTrue(all(c['golden'] for c in o.private_state['hand']))
        for _ in range(3):
            o=t.step(self.action(o,'play',position=0))
        self.assertEqual(len(o.private_state['board']),3)
        self.assertTrue(all(c['golden'] for c in o.private_state['board']))
        self.assertEqual(o.private_state['hand'],[])
        before=t.engine.game.minion_pool.available('BG32_236')
        o=t.step(self.action(o,'sell'))
        self.assertEqual(t.engine.game.minion_pool.available('BG32_236'),before+1)
        for hero in t.engine.game.heroes:
            for card in hero.tavern:
                if card.card_id=='BG32_236':self.assertTrue(card.is_golden)

    def test_lullabot_magnetic_preserves_buffs_keywords_and_end_effect(self):
        t,o=self.make(board=('BG29_611',),hand=('BG26_146','BG28_503'))
        host=o.private_state['board'][0]
        # Give Fortify to a Tavern Lullabot is not guaranteed; directly put buff
        # through the real engine effect onto our test hand's Magnetic entity.
        from hsrl2.entity import Buff
        mag=t.engine.game.heroes[0].hand[0]
        mag.add_buff(Buff(atk=2,health=3,source_id='test'))
        mag.set(t.engine._T.TAUNT,True)
        o=t.engine.observe(); t._accept(o); o=t._visible
        magrow=next(c for c in o.private_state['hand'] if c['card_id']=='BG26_146')
        o=t.step(self.action(o,'play',magrow['entity_id'],target_id=host['entity_id']))
        self.assertEqual(len(o.private_state['board']),1)
        self.assertEqual(o.private_state['board'][0]['attack'],host['attack']+4)
        self.assertEqual(o.private_state['board'][0]['health'],host['health']+5)
        self.assertTrue(o.private_state['board'][0]['taunt'])
        snap=self.end(t,o)
        self.assertEqual(snap['board'][0]['health'],host['health']+6)
        self.assertEqual(snap['board'][0]['magnetic_attachments'],['BG26_146'])

    def test_sell_magnetic_host_returns_attachment_to_pool(self):
        t,o=self.make(board=('BG29_611',),hand=('BG26_146',))
        host=o.private_state['board'][0]['entity_id']
        o=t.step(self.action(o,'play',target_id=host))
        before=t.engine.game.minion_pool.available('BG26_146')
        o=t.step(self.action(o,'sell',host))
        self.assertEqual(t.engine.game.minion_pool.available('BG26_146'),before+1)

    def test_all_eight_spells_are_played_with_explicit_decisions(self):
        exercised=set()
        for cid in sorted(TIER1_SPELLS):
            with self.subTest(spell=cid):
                t,o=self.make(board=('BG36_345',),hand=(cid,))
                eid=o.private_state['hand'][0]['entity_id']
                o=t.step(self.action(o,'cast_spell',eid))
                while o.private_state.get('pending_choice'):
                    o=t.step(o.legal_actions[0])
                self.assertNotIn(eid,[x['entity_id'] for x in o.private_state['hand']])
                exercised.add(cid)
        self.assertEqual(exercised,TIER1_SPELLS)

    def test_discover_options_are_current_lobby_minions_and_manually_chosen(self):
        t,o=self.make(hand=('BG33_101',))
        o=t.step(self.action(o,'cast_spell'))
        options=o.private_state['pending_choice']['options']
        self.assertTrue({x['card_id'] for x in options}<=t.engine.allowed_minions)
        self.assertEqual(o.private_state['hand'],[])
        o=t.step(o.legal_actions[0])
        self.assertEqual(len(o.private_state['hand']),1)
        self.assertEqual(o.private_state['board'],[])
        self.assertEqual(o.action_budget['actions_used'],2)

    def test_fork_and_timing_keep_choice_branches_isolated(self):
        t,o=self.make(board=('BG36_345',),hand=('BG31_880',))
        o=t.step(self.action(o,'cast_spell'))
        b=t.fork()
        self.assertEqual(b._visible,o)
        bo=b.step(b._visible.legal_actions[0])
        self.assertEqual(bo.action_budget['actions_used'],2)
        self.assertEqual(t._visible.action_budget['actions_used'],1)
        self.assertEqual(t._visible.private_state['pending_choice']['kind'],'choose_one')

    def test_refresh_after_upgrade_aborts_without_incomplete_pool(self):
        t,o=self.make(hand=('BG28_810',)*3)
        for _ in range(3):o=t.step(self.action(o,'cast_spell'))
        o=t.step(self.action(o,'upgrade'))
        self.assertEqual(o.private_state['tavern_tier'],2)
        with self.assertRaisesRegex(UnsupportedRecruitTransition,'beyond Tier1'):
            t.step(self.action(o,'refresh'))

    def test_generated_third_copy_aborts_before_automatic_triple(self):
        t,o=self.make(board=('BG36_345','BG36_345'))
        hero=t.engine.game.heroes[0]
        third=t.engine.game.create_minion('BG36_345',controller=hero)
        before=list(hero.board)
        with self.assertRaisesRegex(UnsupportedRecruitTransition,'Generated triple'):
            t.engine.game.pending_hand_add(hero,third)
        self.assertEqual(hero.board,before)
        self.assertEqual(hero.hand,[])
        self.assertEqual(t.engine.game.pending_choices,[])

    def test_offboard_scarlet_threshold_fails_closed(self):
        t,o=self.make(hand=('BG35_814',))
        from hsrl2.entity import Buff
        scarlet=t.engine.game.heroes[0].hand[0]
        before=scarlet.atk
        with self.assertRaisesRegex(UnsupportedRecruitTransition,'Off-board Scarlet'):
            scarlet.add_buff(Buff(atk=3))
        self.assertEqual(scarlet.atk,before)

    def test_molten_rock_does_not_trigger_itself_but_observes_next_elemental(self):
        t,o=self.make(hand=('BGS_127','BGS_119'))
        rock=next(c for c in o.private_state['hand'] if c['card_id']=='BGS_127')
        o=t.step(self.action(o,'play',rock['entity_id'],position=0))
        self.assertEqual(o.private_state['board'][0]['health'],3)
        other=next(c for c in o.private_state['hand'] if c['card_id']=='BGS_119')
        o=t.step(self.action(o,'play',other['entity_id'],position=1))
        self.assertEqual(o.private_state['board'][0]['health'],4)

    def test_weaver_matches_post_demon_change_client_replay_order(self):
        t,o=self.make(hand=('BGS_004','BGS_004'))
        hp=o.private_state['health']
        first=o.private_state['hand'][0]['entity_id']
        o=t.step(self.action(o,'play',first,position=0))
        self.assertEqual(o.private_state['health'],hp)
        self.assertEqual(o.private_state['board'][0]['attack'],1)
        self.assertEqual(o.private_state['board'][0]['health'],3)
        second=o.private_state['hand'][0]['entity_id']
        o=t.step(self.action(o,'play',second,position=1))
        self.assertEqual(o.private_state['health'],hp-1)
        self.assertEqual(o.private_state['board'][0]['attack'],3)
        self.assertEqual(o.private_state['board'][0]['health'],5)
        self.assertEqual(o.private_state['board'][1]['attack'],1)
        self.assertEqual(o.private_state['board'][1]['health'],3)

    def test_offboard_fugitive_target_does_not_silently_skip_trigger(self):
        t,o=self.make(hand=('BG28_897',),tribes=('BEAST','DEMON','MECH','NAGA','PIRATE'))
        target=t.engine.game.create_minion('BG36_921',controller=t.engine.game.heroes[0])
        target.zone=t.engine._Zone.TAVERN
        t.engine.game.heroes[0].tavern.append(target)
        t._accept(t.engine.observe());o=t._visible
        o=t.step(self.action(o,'cast_spell'))
        choose=next(a for a in o.legal_actions if a.target_id==t.engine._id(target))
        health=target.health
        with self.assertRaisesRegex(UnsupportedRecruitTransition,'Tavern Fugitive'):
            t.step(choose)
        self.assertEqual(target.health,health)

    def test_full_game_and_choice_timeout_still_fail_closed(self):
        t,o=self.make(board=('BG36_345',),hand=('BG31_880',),ms=9000)
        with self.assertRaises(RecruitCoverageError):TimedRecruitEngine(t.engine,self.profile,self.rules)
        o=t.step(self.action(o,'cast_spell'))
        with self.assertRaisesRegex(UnsupportedRecruitTransition,'Pending-choice timeout'):
            t.step(o.legal_actions[0])

if __name__=='__main__':unittest.main()
