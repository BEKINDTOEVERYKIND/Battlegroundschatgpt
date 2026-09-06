"""Real engine/Firestone persistence, legal commands, privacy and clock tests."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import unittest

from bg_ai.hsbrsim_opening import (
    OpeningFixtureSpec, build_opening_database, TIER1_MINIONS, TRIBES,
)
from bg_ai.opening_transition import (
    TwoTurnOpeningRecruitEngine, TimedTwoTurnOpeningEngine, canonical_hash,
)
from bg_ai.hsbrsim_adapter import UnsupportedRecruitTransition
from bg_ai.recruiting import RecruitAction
from bg_ai.turn_budget import TimingProfile

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = Path(os.environ.get("HSBRSIM_ROOT", str(ROOT.parent / "research/HSBRSIM")))


def zones(first):
    return (tuple(first),) + ((),) * 7


@unittest.skipUnless((EXTERNAL / "hsrl2/game.py").is_file(), "Pinned HSBRSIM checkout required")
class OpeningTransitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = json.loads((ROOT / "data/ruleset.json").read_text())
        cls.cards = {c["id"]: c for c in json.loads((ROOT / "data/reference_cards.json").read_text())}
        cls.profile = TimingProfile.load(ROOT / "config/turn-budget.json")
        cls.databases = {}
        cls.worker = subprocess.Popen(["node", "simulator/opening-transition-firestone.mjs"],
            cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    @classmethod
    def tearDownClass(cls):
        cls.worker.stdin.close()
        cls.worker.wait(timeout=15)
        cls.worker.stdout.close()
        stderr = cls.worker.stderr.read()
        cls.worker.stderr.close()
        if cls.worker.returncode:
            raise AssertionError(stderr)

    def make(self, *, board=(), hand=(), boards=None, tribes=None, seed=3, timer=None):
        if tribes is None:
            required = set()
            for cid in board + hand + tuple(c for b in (boards or ()) for c in b):
                races = self.cards[cid].get("races", [])
                if races:
                    required.add(races[0])
            for tribe in sorted(TRIBES):
                if len(required) >= 5:
                    break
                required.add(tribe)
            tribes = tuple(sorted(required))
        spec = OpeningFixtureSpec(valid_tribes=tuple(tribes), initial_boards=boards or zones(board),
                                  initial_hands=zones(hand))
        if spec.valid_tribes not in self.databases:
            self.databases[spec.valid_tribes] = build_opening_database(EXTERNAL, valid_tribes=spec.valid_tribes)
        current = self.databases[spec.valid_tribes]
        raw = TwoTurnOpeningRecruitEngine(current.db, self.rules,
            timer_ms=timer or (lambda p, t: 60000), provenance=current.provenance, fixture=spec)
        timed = TimedTwoTurnOpeningEngine.for_fixture(raw, self.profile, self.rules)
        return timed, timed.reset(seed=seed)

    def act(self, timed, observation, kind, card_id=None, **attrs):
        entities = observation.private_state.get("hand", []) + observation.private_state.get("board", []) + observation.private_state.get("shop", [])
        ids = {x["entity_id"] for x in entities if x["card_id"] == card_id}
        action = next(a for a in observation.legal_actions if a.kind == kind
                      and (card_id is None or a.entity_id in ids)
                      and all(getattr(a, k) == v for k, v in attrs.items()))
        return timed.step(action)

    def end(self, timed, observation):
        while observation is not None:
            observation = self.act(timed, observation, "end_turn")
        self.assertEqual(timed.engine.phase, "awaiting_combat")

    def receipt(self, timed, seed=91):
        request = timed.engine.combat_request(seed=seed)
        self.worker.stdin.write(json.dumps(request) + "\n")
        self.worker.stdin.flush()
        result = json.loads(self.worker.stdout.readline())
        self.assertNotIn("error", result)
        self.assertEqual(result["request_sha256"], canonical_hash(request))
        return result

    def advance(self, timed, observation):
        self.end(timed, observation)
        receipt = self.receipt(timed)
        return timed.advance_combat(receipt), receipt

    def test_complete_two_combats_stops_before_dark_gifts_or_native_combat(self):
        t, o = self.make()
        with self.assertRaisesRegex(UnsupportedRecruitTransition, "Native combat"):
            t.engine.game.run_combat(None, None)
        with self.assertRaisesRegex(UnsupportedRecruitTransition, "sampled Firestone"):
            t.engine.game.end_recruit_phase()
        o, first = self.advance(t, o)
        self.assertEqual(o.turn, 2)
        self.assertEqual(o.private_state["gold"], 4)
        self.assertEqual(o.private_state["upgrade_cost"], 4)
        o, second = self.advance(t, o)
        self.assertIsNone(o)
        self.assertEqual(t.engine.phase, "complete")
        self.assertEqual(t.engine.game.turn, 2)
        self.assertEqual(len(t.engine.combat_receipts), 2)
        self.assertEqual(t.engine.game.dark_gift_state, {})
        self.assertEqual([r["turn"] for r in (first, second)], [1, 2])
        with self.assertRaises(ValueError):
            t.advance_combat(second)

    def test_freeze_preserves_real_entities_and_fills_separate_spell_slot(self):
        t, o = self.make()
        original = {c["entity_id"] for c in o.private_state["shop"]}
        buy = next(c for c in o.private_state["shop"] if c["card_type"] == "minion")
        o = self.act(t, o, "buy", entity_id=buy["entity_id"])
        retained = original - {buy["entity_id"]}
        o = self.act(t, o, "freeze")
        o, _ = self.advance(t, o)
        now = o.private_state["shop"]
        self.assertTrue(retained.issubset({c["entity_id"] for c in now}))
        self.assertEqual(sum(c["card_type"] == "minion" for c in now), 3)
        self.assertEqual(sum(c["card_type"] == "spell" for c in now), 1)
        self.assertFalse(o.private_state["frozen"])
        self.assertEqual(o.private_state["hand"][0]["entity_id"], buy["entity_id"])

    def test_unfrozen_shop_is_replaced_and_pool_copies_are_conserved(self):
        t, o = self.make()
        before_ids = {c["entity_id"] for c in o.private_state["shop"]}
        pool = t.engine.game.minion_pool
        def inventory():
            return {cid: pool.available(cid) + sum(c.card_id == cid for h in t.engine.game.heroes
                    for c in h.board + h.hand + h.tavern) for cid in t.engine.allowed_minions}
        before = inventory()
        o, _ = self.advance(t, o)
        self.assertTrue(before_ids.isdisjoint({c["entity_id"] for c in o.private_state["shop"]}))
        self.assertEqual(inventory(), before)

    def test_spellcraft_expires_on_frozen_shop_target_and_regenerates_in_hand(self):
        t, o = self.make(hand=("BG23_000", "BG28_810"))
        o = self.act(t, o, "play", "BG23_000", position=0)
        target = next(c for c in o.private_state["shop"] if c["card_type"] == "minion")
        o = self.act(t, o, "cast_spell", "BG23_000t")
        o = self.act(t, o, "choose", target_id=target["entity_id"])
        o = self.act(t, o, "freeze")
        self.assertEqual(next(c for c in o.private_state["shop"] if c["entity_id"] == target["entity_id"])["attack"], target["attack"] + 2)
        o, _ = self.advance(t, o)
        retained = next(c for c in o.private_state["shop"] if c["entity_id"] == target["entity_id"])
        self.assertEqual(retained["attack"], target["attack"])
        self.assertEqual(sorted(c["card_id"] for c in o.private_state["hand"]), ["BG23_000t", "BG28_810"])
        self.assertTrue(next(c for c in o.private_state["hand"] if c["card_id"] == "BG23_000t")["temporary_spellcraft"])

    def test_combat_damage_is_sampled_capped_and_applied_to_armor_then_health(self):
        # Distinct current minions produce more damage than the opening cap.
        boards = ((), ("BG29_611", "BG26_146", "BG31_803", "BG36_200", "BG36_345")) + ((),) * 6
        t, o = self.make(boards=boards)
        hero = t.engine.game.heroes[0]
        hero.armor = 2
        before = hero.health
        original_board = tuple(t.engine.game.heroes[1].board)
        o, receipt = self.advance(t, o)
        self.assertEqual(receipt["outcomes"][0]["uncapped_damage"], 6)
        self.assertEqual(receipt["outcomes"][0]["damage_to_a"], 5)
        self.assertEqual(hero.armor, 0)
        self.assertEqual(hero.health, before - 3)
        self.assertEqual(tuple(t.engine.game.heroes[1].board), original_board)
        self.assertEqual(len(hero.board), 0)

    def test_combat_summoned_scout_and_tokens_do_not_change_original_hand_or_board(self):
        t, o = self.make(hand=("BG32_330",), board=("BG36_200",))
        hero = t.engine.game.heroes[0]
        hand, board = tuple(hero.hand), tuple(hero.board)
        o, receipt = self.advance(t, o)
        self.assertEqual(receipt["outcomes"][0]["winner_id"], 0)
        self.assertEqual(tuple(hero.hand), hand)
        self.assertEqual(tuple(hero.board), board)

    def test_busker_income_is_exact_and_unspent_gold_is_not_banked(self):
        t, o = self.make(hand=("BG26_135", "BG26_135"))
        o = self.act(t, o, "play", "BG26_135", position=0)
        o = self.act(t, o, "play", "BG26_135", position=0)
        self.assertEqual(o.private_state["deferred_next_turn_gold"], 2)
        self.assertEqual(o.private_state["gold"], 3)
        o, _ = self.advance(t, o)
        self.assertEqual(o.private_state["gold"], 6)
        self.assertEqual(o.private_state["deferred_next_turn_gold"], 0)

    def test_lullabot_growth_persists_and_only_ticks_once_each_end(self):
        t, o = self.make(board=("BG26_146",))
        health = o.private_state["board"][0]["health"]
        o, _ = self.advance(t, o)
        self.assertEqual(o.private_state["board"][0]["health"], health + 1)
        self.end(t, o)
        self.assertEqual(t.engine.combat_snapshot(0)["board"][0]["health"], health + 2)

    def test_temporary_spellcraft_magnetic_transfer_aborts_before_mutating_zones(self):
        t, o = self.make(board=("BG29_611", "BG23_000"), seed=1)
        lullabot = next(c for c in o.private_state["shop"] if c["card_id"] == "BG26_146")
        host = next(c for c in o.private_state["board"] if c["card_id"] == "BG29_611")
        o = self.act(t, o, "cast_spell", "BG23_000t")
        o = self.act(t, o, "choose", target_id=lullabot["entity_id"])
        o = self.act(t, o, "buy", entity_id=lullabot["entity_id"])
        legal = next(a for a in o.legal_actions if a.kind == "play"
                     and a.entity_id == lullabot["entity_id"] and a.target_id == host["entity_id"])
        hero = t.engine.game.heroes[0]
        before = (tuple(hero.hand), tuple(hero.board), hero.board[0].atk)
        with self.assertRaisesRegex(UnsupportedRecruitTransition, "Temporary Spellcraft magnetic"):
            t.step(legal)
        self.assertEqual((tuple(hero.hand), tuple(hero.board), hero.board[0].atk), before)
        self.assertTrue(t.engine._invalid)

    def test_every_current_tier1_minion_can_cross_actual_combat_and_recruit_boundary(self):
        for cid in sorted(TIER1_MINIONS):
            with self.subTest(card=cid):
                t, o = self.make(board=(cid,))
                before = t.engine.game.heroes[0].board[0]
                o, receipt = self.advance(t, o)
                self.assertIs(t.engine.game.heroes[0].board[0], before)
                self.assertEqual(o.private_state["board"][0]["card_id"], cid)
                self.assertEqual(receipt["outcomes"][0]["winner_id"], 0)

    def test_new_turn_gets_new_clock_but_fork_preserves_spent_second_turn_time(self):
        t, o = self.make(timer=lambda p, turn: 60000 if turn == 1 else 12000)
        o = self.act(t, o, "freeze")
        old = o.action_budget
        o, _ = self.advance(t, o)
        self.assertEqual(o.action_budget["actions_used"], 0)
        self.assertLess(o.action_budget["remaining_ms"], old["remaining_ms"])
        o = self.act(t, o, "freeze")
        f = t.fork()
        self.assertEqual(f._visible.action_budget, o.action_budget)
        self.assertEqual(f._visible.private_state, o.private_state)
        self.assertEqual(f._visible.public_state, o.public_state)
        self.assertEqual(f.engine.combat_receipts, t.engine.combat_receipts)
        self.assertIsNot(f.engine.game.heroes[0], t.engine.game.heroes[0])

    def test_fork_replays_expiring_callbacks_and_new_spellcraft_without_cross_branch_mutation(self):
        t, o = self.make(board=("BG23_000",), hand=("BG26_135",))
        o = self.act(t, o, "play", "BG26_135", position=1)
        target = o.private_state["board"][0]
        o = self.act(t, o, "cast_spell", "BG23_000t")
        o = self.act(t, o, "choose", target_id=target["entity_id"])
        o = self.act(t, o, "freeze")
        o, _ = self.advance(t, o)
        self.assertEqual(o.private_state["gold"], 5)
        self.assertEqual(o.private_state["board"][0]["attack"], target["attack"])
        branch = t.fork()
        b = branch._visible
        self.assertEqual(b.private_state, o.private_state)
        b = self.act(branch, b, "cast_spell", "BG23_000t")
        b = self.act(branch, b, "choose", target_id=target["entity_id"])
        self.assertEqual(b.private_state["board"][0]["attack"], target["attack"] + 2)
        self.assertEqual(t.engine.observe().private_state["board"][0]["attack"], target["attack"])
        self.assertEqual(len(t.engine.game.heroes[0].hand), len(branch.engine.game.heroes[0].hand) + 1)

    def test_activate_and_turn_counters_reset_without_erasing_permanent_buffs(self):
        t, o = self.make(board=("BG36_345", "BG36_200"))
        o = self.act(t, o, "activate", "BG36_345")
        o = self.act(t, o, "choose")
        self.assertTrue(o.private_state["board"][0]["activate_used"])
        board = deepcopy(o.private_state["board"])
        o, _ = self.advance(t, o)
        self.assertFalse(o.private_state["board"][0]["activate_used"])
        self.assertEqual([(c["attack"], c["health"]) for c in o.private_state["board"]],
                         [(c["attack"], c["health"]) for c in board])
        self.assertTrue(any(a.kind == "activate" for a in o.legal_actions))
        self.assertEqual(o.private_state["counters"]["GOLD_SPENT_THIS_TURN"], 0)

    def test_receipts_cannot_be_replayed_swapped_averaged_or_applied_mid_recruit(self):
        t, o = self.make()
        with self.assertRaises(ValueError):
            t.advance_combat({})
        self.end(t, o)
        r = self.receipt(t)
        before = [(h.health, h.armor, h.gold) for h in t.engine.game.heroes]
        for field, value in (("request_sha256", "stale"), ("turn", 2)):
            bad = deepcopy(r); bad[field] = value
            with self.assertRaises(ValueError): t.advance_combat(bad)
        for field, value in (("samples", 128), ("damage_to_a", 0.1), ("winner_id", 5), ("player_b", 3)):
            bad = deepcopy(r); bad["outcomes"][0][field] = value
            with self.assertRaises(ValueError): t.advance_combat(bad)
        self.assertEqual([(h.health, h.armor, h.gold) for h in t.engine.game.heroes], before)
        t.advance_combat(r)
        with self.assertRaises(ValueError): t.advance_combat(r)

    def test_turn_two_upgrade_is_legal_but_turn_one_tier2_next_shop_is_explicit_frontier(self):
        t, o = self.make()
        o, _ = self.advance(t, o)
        o = self.act(t, o, "upgrade")
        self.assertEqual(o.private_state["tavern_tier"], 2)
        self.assertEqual(o.private_state["gold"], 0)
        o, _ = self.advance(t, o)
        self.assertIsNone(o)
        t, o = self.make(hand=("BG28_810", "BG28_810"))
        o = self.act(t, o, "cast_spell", "BG28_810")
        o = self.act(t, o, "cast_spell", "BG28_810")
        o = self.act(t, o, "upgrade")
        self.end(t, o)
        r = self.receipt(t)
        before = t.engine.game.heroes[0].health
        with self.assertRaisesRegex(UnsupportedRecruitTransition, "Tier2 next-turn shop"):
            t.advance_combat(r)
        self.assertEqual(t.engine.game.heroes[0].health, before)
        self.assertEqual(t.engine.game.turn, 1)

    def test_observation_excludes_opponent_zones_requests_receipts_and_rng(self):
        t, o = self.make()
        o, _ = self.advance(t, o)
        for player in o.public_state["players"]:
            self.assertEqual(set(player), {"player_id", "hero_id", "health", "armor", "tavern_tier", "alive"})
        public_private = json.dumps({"public": o.public_state, "private": o.private_state})
        for hidden in ("rng", "pool", "snapshots", "request_sha256", "receipt", "outcomes"):
            self.assertNotIn(hidden, public_private)
        self.assertEqual(len(o.public_state["valid_tribes"]), 5)
        self.assertIsInstance(o.private_state["deferred_next_turn_gold"], int)


if __name__ == "__main__":
    unittest.main()
