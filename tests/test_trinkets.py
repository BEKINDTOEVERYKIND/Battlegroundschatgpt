"""Seasonal state, finite choices, offering constraints and real host mutations."""
import dataclasses
import json
from pathlib import Path
import random
import os
import sys
import unittest

from bg_ai.trinkets import (CORE_EFFECTS, HSRLTrinketHost, MinionView,
    OfferContext, OfferingAssumptions, OwnedTrinket, SPHERE_ID,
    TrinketController, TrinketError, TrinketOffer, TrinketSpec,
    catalog_from_snapshot, distinct_minion_type_count, offer_trinkets, validate_offering)

ROOT = Path(__file__).resolve().parents[1]


def context(turn=6, board=(), owned=()):
    return OfferContext(turn, 4, "TB_BaconShop_HERO_34", tuple(board), (),
        frozenset({"BEAST", "DRAGON", "NAGA", "MURLOC", "ELEMENTAL"}), frozenset(owned))


class MemoryHost:
    def __init__(self):
        self.gold = 10
        self.tavern_tier = 4
        self.income_cap = 10
        self.upgrade_cost = 7
        self.undead_attack = 0
        self.hand = []
        self.queue = []
        self.discovered = []
        self.minions = {"m1": [1, 2], "m2": [3, 4]}
        self.shields = set()
        self.hand_minion = "m2"
        self.views = {"m1": MinionView("m1", "minion1"), "m2": MinionView("m2", "minion2")}
        self.gem_bonus = [0, 0]
        self.free_refreshes = 0
        self.tavern_buffs = []
        self.pool = {"elemental": 4, "battlecry": 3}
        self.generated_minions = []

    def spend_gold(self, amount, *, source_id=None): self.gold -= amount
    def gain_gold(self, amount): self.gold += amount
    def increase_income_cap(self, amount): self.income_cap += amount
    def reduce_upgrade_cost(self, amount): self.upgrade_cost = max(0, self.upgrade_cost - amount)
    def add_undead_attack(self, amount): self.undead_attack += amount
    def discover_spell_options(self): return ("spell1", "spell2", "spell3", "spell4")
    def card_discovered(self, card_id): self.discovered.append(card_id)
    def give_card(self, card_id, source_id):
        (self.hand if len(self.hand) < 10 else self.queue).append(card_id)
    def buff_minion(self, entity_id, attack, health, source_id):
        self.minions[entity_id][0] += attack
        self.minions[entity_id][1] += health
    def give_divine_shield(self, entity_id): self.shields.add(entity_id)
    def leftmost_hand_minion(self): return self.hand_minion
    def minion_view(self, entity_id): return self.views[entity_id]
    def board_minions(self): return list(self.views.values())
    def increase_gem_bonus(self, attack, health):
        self.gem_bonus[0] += attack
        self.gem_bonus[1] += health
    def gain_free_refresh(self, amount): self.free_refreshes += amount
    def buff_tavern(self, attack, health, source_id, tribe): self.tavern_buffs.append((attack, health, source_id, tribe))
    def minion_generation_options(self, *, tribe=None, keyword=None):
        key = "elemental" if tribe == "ELEMENTAL" else "battlecry"
        return [key] if self.pool[key] > 0 else []
    def give_pool_minion(self, card_id, source_id):
        self.pool[card_id] -= 1
        self.generated_minions.append(card_id)
        self.give_card(card_id, source_id)


class TestTrinkets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cards = json.loads((ROOT / "data/reference_cards.json").read_text())
        cls.rules = json.loads((ROOT / "data/ruleset.json").read_text())
        cls.catalog = catalog_from_snapshot(cls.cards, cls.rules,
            resolve_sphere_from_official_history=True)

    def controller(self, turn=6):
        host = MemoryHost()
        ctrl = TrinketController(host, self.catalog, allowed_ids=list(CORE_EFFECTS), seed=19)
        ctrl.start_turn(turn, event_id=f"start-{turn}")
        return ctrl, host

    def own(self, ctrl, *ids):
        # Arrange prior acquisitions directly to isolate later hook behavior.
        ctrl.owned.extend(OwnedTrinket(cid, ctrl.turn) for cid in ids)

    def normal_lesser_offer(self, ctrl):
        ids = ("BG30_MagicItem_705", "BG30_MagicItem_847", "BG30_MagicItem_420", "BG35_MagicItem_814")
        offers = tuple(TrinketOffer(cid, self.catalog[cid].cost) for cid in ids)
        ctrl.install_offering(offers, context(ctrl.turn), slot="normal_lesser", stage="lesser")

    def test_catalog_does_not_silently_promote_sphere(self):
        original = catalog_from_snapshot(self.cards, self.rules)
        self.assertNotIn(SPHERE_ID, original)
        self.assertIn(SPHERE_ID, self.catalog)
        self.assertNotIn("BG32_MagicItem_894", self.catalog)
        self.assertEqual(self.catalog[SPHERE_ID].cost, 2)
        self.assertEqual(self.catalog[SPHERE_ID].parameters[0], 3)
        self.assertEqual(self.catalog["BG30_MagicItem_847"].cost, 1)
        self.assertEqual(self.catalog["BG30_MagicItem_996"].cost, 0)

    def test_rotated_snapshot_is_rejected(self):
        with self.assertRaises(TrinketError):
            catalog_from_snapshot(self.cards, dict(self.rules, build=251333))

    def test_unimplemented_effect_rejected_before_payment(self):
        with self.assertRaises(TrinketError):
            TrinketController(MemoryHost(), self.catalog, allowed_ids=["BG30_MagicItem_703"], seed=1)

    def test_purchase_once_and_pay_exact_current_price(self):
        ctrl, host = self.controller()
        self.normal_lesser_offer(ctrl)
        ctrl.buy_trinket("BG30_MagicItem_705")
        self.assertEqual((host.gold, host.upgrade_cost), (8, 4))
        with self.assertRaises(TrinketError): ctrl.buy_trinket("BG30_MagicItem_705")
        self.assertEqual(host.gold, 8)
        ctrl.end_turn(event_id="end6")
        ctrl.start_turn(7, event_id="start7")
        self.assertEqual(host.upgrade_cost, 1)

    def test_insufficient_gold_leaves_offer_and_gold_unchanged(self):
        ctrl, host = self.controller()
        self.normal_lesser_offer(ctrl)
        host.gold = 0
        with self.assertRaises(TrinketError): ctrl.buy_trinket("BG30_MagicItem_705")
        self.assertEqual(host.gold, 0)
        self.assertEqual(len(ctrl.offering), 4)

    def test_normal_timing_and_slot_cannot_be_repeated(self):
        ctrl, _ = self.controller(turn=5)
        with self.assertRaises(TrinketError): self.normal_lesser_offer(ctrl)
        ctrl.end_turn(event_id="end5")
        ctrl.start_turn(6, event_id="start6")
        self.normal_lesser_offer(ctrl)
        ctrl.buy_trinket("BG30_MagicItem_847")
        with self.assertRaises(TrinketError): self.normal_lesser_offer(ctrl)

    def test_wallet_does_not_grant_current_gold_or_repeat_end(self):
        ctrl, host = self.controller()
        self.own(ctrl, "BG30_MagicItem_847")
        ctrl.end_turn(event_id="end")
        self.assertEqual((host.gold, host.income_cap), (10, 11))
        with self.assertRaises(TrinketError): ctrl.end_turn(event_id="different-end-id")
        with self.assertRaises(TrinketError): ctrl.on_refresh(event_id="late-action")
        self.assertEqual(host.income_cap, 11)

    def test_tip_jar_increases_cap_and_current_gold(self):
        ctrl, host = self.controller(turn=9)
        ids = ("BG30_MagicItem_996", "BG30_MagicItem_420t", SPHERE_ID, "BG30_MagicItem_914t")
        offers = tuple(TrinketOffer(cid, max(0, self.catalog[cid].cost - (2 if self.catalog[cid].affiliated_types else 0))) for cid in ids)
        ctrl.install_offering(offers, context(9), slot="normal_greater", stage="greater")
        ctrl.buy_trinket("BG30_MagicItem_996")
        self.assertEqual((host.gold, host.income_cap), (14, 14))

    def test_discover_is_explicit_and_blocks_new_turn(self):
        ctrl, host = self.controller()
        self.normal_lesser_offer(ctrl)
        ctrl.buy_trinket("BG30_MagicItem_420")
        self.assertEqual(host.hand, [])
        self.assertEqual(len(set(ctrl.pending_choice.options)), 3)
        with self.assertRaises(TrinketError): ctrl.end_turn(event_id="end")
        with self.assertRaises(TrinketError): ctrl.on_refresh(event_id="refresh")
        chosen = ctrl.choose(1)
        self.assertEqual(host.hand, [chosen])
        self.assertEqual(host.discovered, [chosen])
        with self.assertRaises(TrinketError): ctrl.choose(0)

    def test_greater_book_preserves_two_separate_choices(self):
        ctrl, host = self.controller()
        item = OwnedTrinket("BG30_MagicItem_420t", 6)
        ctrl.owned.append(item)
        ctrl._apply(item, "acquire")
        ctrl.choose(0)
        self.assertIsNotNone(ctrl.pending_choice)
        ctrl.choose(0)
        self.assertIsNone(ctrl.pending_choice)
        self.assertEqual(len(host.hand), 2)

    def test_sphere_uses_last_tavern_spell_across_turns_and_queues(self):
        ctrl, host = self.controller()
        self.own(ctrl, SPHERE_ID)
        ctrl.on_spell_cast("spell1", tavern_spell=True, target_minion_id=None, event_id="spell1")
        ctrl.on_spell_cast("bloodgem", tavern_spell=False, target_minion_id="m1", event_id="gem")
        host.hand = ["existing"] * 9
        ctrl.end_turn(event_id="end6")
        self.assertEqual(host.hand[-1:], ["spell1"])
        self.assertEqual(host.queue, ["spell1", "spell1"])
        ctrl.start_turn(7, event_id="start7")
        ctrl.end_turn(event_id="end7")
        self.assertEqual(len(host.queue), 5)

    def test_sphere_without_spell_does_not_generate(self):
        ctrl, host = self.controller()
        self.own(ctrl, SPHERE_ID)
        ctrl.end_turn(event_id="end")
        self.assertEqual(host.hand, [])

    def test_wand_counter_carries_between_turns(self):
        ctrl, host = self.controller()
        self.own(ctrl, "BG36_MagicItem_307")
        for i in range(2): ctrl.on_spell_cast("gem", tavern_spell=False, target_minion_id="m1", event_id=f"spell{i}")
        ctrl.end_turn(event_id="end6")
        ctrl.start_turn(7, event_id="start7")
        ctrl.on_spell_cast("spell", tavern_spell=True, target_minion_id=None, event_id="untargeted")
        self.assertEqual(host.gold, 10)
        ctrl.on_spell_cast("gem", tavern_spell=False, target_minion_id="m1", event_id="third")
        self.assertEqual(host.gold, 11)
        self.assertEqual(ctrl.owned[0].counter, 0)

    def test_lorewalker_uses_current_parameters_and_buffs_tavern_target(self):
        ctrl, host = self.controller()
        self.own(ctrl, "BG30_MagicItem_422", "BG30_MagicItem_422t")
        ctrl.on_spell_cast("gem", tavern_spell=False, target_minion_id="m1", event_id="cast")
        self.assertEqual(host.minions["m1"], [15, 16])

    def test_phrasebook_and_shell_receive_distinct_play_events(self):
        ctrl, host = self.controller()
        self.own(ctrl, "BG30_MagicItem_914", "BG36_MagicItem_811")
        ctrl.on_minion_played("m1", event_id="play1")
        ctrl.on_minion_played("m2", event_id="play2")
        self.assertEqual(host.minions["m2"], [9, 10])
        self.assertEqual(host.shields, {"m1"})
        with self.assertRaises(TrinketError): ctrl.on_minion_played("m1", event_id="play1")

    def test_coffins_and_headstone_apply_permanent_undead_attack(self):
        ctrl, host = self.controller()
        self.own(ctrl, "BG30_MagicItem_547", "BG30_MagicItem_547t", "BG32_MagicItem_276")
        ctrl.on_spell_cast("gem", tavern_spell=False, target_minion_id="m1", event_id="gem")
        self.assertEqual(host.undead_attack, 0)
        ctrl.on_spell_cast("spell", tavern_spell=True, target_minion_id=None, event_id="spell")
        ctrl.end_turn(event_id="end")
        self.assertEqual(host.undead_attack, 5)

    def test_goldenizer_counts_acquisition_turn_and_correct_token(self):
        ctrl, host = self.controller()
        self.own(ctrl, "BG30_MagicItem_435")
        for turn in range(6, 9):
            if turn > 6: ctrl.start_turn(turn, event_id=f"start{turn}")
            ctrl.end_turn(event_id=f"end{turn}")
        self.assertEqual(host.hand, ["BG26_813t"])

    def test_coin_purse_only_once(self):
        ctrl, host = self.controller()
        self.own(ctrl, "BG35_MagicItem_814")
        host.tavern_tier = 6
        ctrl.on_tier_changed(event_id="upgrade")
        ctrl.on_tier_changed(event_id="another-tier-event")
        self.assertEqual(host.gold, 22)

    def test_mysterious_orb_changes_next_stage(self):
        ctrl, host = self.controller()
        item = OwnedTrinket("BG35_MagicItem_818", 6)
        ctrl.owned.append(item)
        ctrl._apply(item, "acquire")
        self.assertEqual((ctrl.next_stage_override, host.gold), ("lesser", 20))

    def test_great_boar_grants_current_gems_and_additive_bonus(self):
        ctrl, host = self.controller()
        host.hand = ["existing"] * 8
        for cid in ("BG30_MagicItem_988", "BG30_MagicItem_988t"):
            item = OwnedTrinket(cid, 6)
            ctrl.owned.append(item)
            ctrl._apply(item, "acquire")
        self.assertEqual(host.gem_bonus, [5, 4])
        self.assertEqual(host.hand[-2:], ["BG20_GEM"] * 2)
        self.assertEqual(host.queue, ["BG20_GEM"] * 6)

    def test_fixed_token_generators_repeat_only_on_subsequent_starts(self):
        ctrl, host = self.controller()
        for cid in ("BG30_MagicItem_406", "BG31_MagicItem_903"):
            item = OwnedTrinket(cid, 6)
            ctrl.owned.append(item)
            ctrl._apply(item, "acquire")
        self.assertEqual(host.hand, ["BG28_604", "BG30_802"])
        ctrl.end_turn(event_id="end6")
        self.assertEqual(len(host.hand), 2)
        ctrl.start_turn(7, event_id="start7")
        self.assertEqual(host.hand, ["BG28_604", "BG30_802"] * 2)

    def test_nomi_and_water_wheel_handle_dual_types_and_turn_limits(self):
        ctrl, host = self.controller()
        self.own(ctrl, "BG30_MagicItem_544", "BG30_MagicItem_544t", "BG35_MagicItem_851", "BG32_MagicItem_888")
        dual = MinionView("m1", "dual", frozenset({"ELEMENTAL", "MURLOC"}))
        for i in range(3): ctrl.on_minion_played("m1", minion=dual, event_id=f"play{i}")
        self.assertEqual(len(host.hand), 2)
        self.assertEqual(host.free_refreshes, 3)
        self.assertEqual([(a,h,t) for a,h,_,t in host.tavern_buffs], [(3,2,"ELEMENTAL"),(5,5,"ELEMENTAL")] * 3)
        ctrl.end_turn(event_id="end6")
        ctrl.start_turn(7, event_id="start7")
        ctrl.on_minion_played("m1", minion=dual, event_id="play7")
        self.assertEqual(len(host.hand), 3)
        self.assertEqual(host.free_refreshes, 4)

    def test_cookie_rod_counter_survives_turn_and_queues_two_spells(self):
        ctrl, host = self.controller()
        self.own(ctrl, "BG36_MagicItem_850")
        murloc = MinionView("m1", "murloc", frozenset({"MURLOC"}))
        for i in range(4): ctrl.on_minion_played("m1", minion=murloc, event_id=f"play{i}")
        ctrl.end_turn(event_id="end6")
        ctrl.start_turn(7, event_id="start7")
        host.hand = ["full"] * 10
        ctrl.on_minion_played("m1", minion=murloc, event_id="fifth")
        self.assertEqual(len(host.queue), 2)
        self.assertEqual(ctrl.owned[0].counter, 0)

    def test_demonic_bloodletter_only_tavern_spell_casts(self):
        ctrl, host = self.controller()
        self.own(ctrl, "BG36_MagicItem_800")
        ctrl.on_spell_cast("gem", tavern_spell=False, target_minion_id="m1", event_id="gem")
        ctrl.on_spell_cast("spell", tavern_spell=True, target_minion_id=None, event_id="tavern")
        self.assertEqual(host.tavern_buffs, [(1,1,"BG36_MagicItem_800",None)])

    def test_secret_schematic_counts_free_mech_buys(self):
        ctrl, host = self.controller()
        self.own(ctrl, "BG36_MagicItem_840")
        mech = MinionView("m1", "mech", frozenset({"MECH"}))
        ctrl.on_minion_bought(mech, base_price=0, paid_gold=0, event_id="free")
        ctrl.on_minion_bought(MinionView("m2", "beast", frozenset({"BEAST"})), base_price=3, paid_gold=3, event_id="beast")
        self.assertEqual(len(host.hand), 1)

    def test_warcry_quotes_are_pure_and_only_two_successful_buys_free(self):
        ctrl, host = self.controller()
        self.own(ctrl, "BG36_MagicItem_202")
        minion = MinionView("m1", "battlecry", keywords=frozenset({"BATTLECRY"}))
        for _ in range(10): self.assertEqual(ctrl.quote_minion_buy(minion, 3), 0)
        self.assertEqual(ctrl.owned[0].turn_counter, 0)
        for i in range(2): ctrl.on_minion_bought(minion, base_price=3, paid_gold=0, event_id=f"buy{i}")
        self.assertEqual(ctrl.quote_minion_buy(minion, 3), 3)
        with self.assertRaises(TrinketError): ctrl.on_minion_bought(minion, base_price=3, paid_gold=0, event_id="thirdfree")
        ctrl.on_minion_bought(minion, base_price=3, paid_gold=3, event_id="thirdpaid")
        ctrl.end_turn(event_id="end6")
        ctrl.start_turn(7, event_id="start7")
        self.assertEqual(ctrl.quote_minion_buy(minion, 3), 0)
        with self.assertRaises(TrinketError): ctrl.quote_minion_buy(minion, 3, payment_resource="health")

    def test_booty_bay_uses_gold_units_and_two_distinct_pirates(self):
        ctrl, host = self.controller()
        self.own(ctrl, "BG30_MagicItem_924", "BG30_MagicItem_924t")
        host.views = {i: MinionView(i,i,frozenset({"PIRATE"})) for i in ("m1","m2")}
        ctrl.on_gold_spent(3, event_id="pay3")
        self.assertEqual(host.minions, {"m1":[28,29], "m2":[30,31]})
        ctrl.on_gold_spent(0, event_id="free")
        self.assertEqual(host.minions["m1"], [28,29])
        with self.assertRaises(TrinketError): ctrl.on_gold_spent(3, event_id="pay3")

    def test_trinket_payment_triggers_existing_brew_once(self):
        ctrl, host = self.controller()
        self.own(ctrl, "BG30_MagicItem_924t")
        host.views = {"m1": MinionView("m1","pirate",frozenset({"PIRATE"}))}
        ids=("BG30_MagicItem_705","BG30_MagicItem_847","BG30_MagicItem_420","BG35_MagicItem_814")
        offers=tuple(TrinketOffer(cid,self.catalog[cid].cost) for cid in ids)
        ctrl.install_offering(offers,context(6,owned=["BG30_MagicItem_924t"]),slot="normal_lesser",stage="lesser")
        ctrl.buy_trinket("BG30_MagicItem_705")
        ctrl.on_gold_spent(2,event_id="engine-event-echo",source_kind="trinket_purchase")
        self.assertEqual(host.minions["m1"],[13,14])
        self.assertEqual(host.gold,8)

    def test_booty_bay_does_not_buff_single_pirate_twice_per_gold(self):
        ctrl, host = self.controller()
        self.own(ctrl, "BG30_MagicItem_924t")
        host.views = {"m1": MinionView("m1", "pirate", frozenset({"PIRATE"}))}
        ctrl.on_gold_spent(1, event_id="pay")
        self.assertEqual(host.minions["m1"], [7,8])

    def test_dragonwing_glider_only_explicit_hand_plays(self):
        ctrl, host = self.controller()
        self.own(ctrl, "BG30_MagicItem_900")
        host.views = {"m1": MinionView("m1", "dragon", frozenset({"DRAGON"}))}
        ctrl.on_spell_cast("spell", tavern_spell=True, target_minion_id=None, event_id="autocast")
        self.assertEqual(host.minions["m1"], [1,2])
        ctrl.on_card_played(event_id="hand_play")
        self.assertEqual(host.minions["m1"], [5,6])

    def test_glowing_crystal_counts_representatives_at_start_only(self):
        ctrl, host = self.controller()
        item = OwnedTrinket("BG36_MagicItem_220", 6)
        ctrl.owned.append(item)
        host.views = {"m1": MinionView("m1", "dual", frozenset({"DRAGON","BEAST"})),
                      "m2": MinionView("m2", "all", frozenset({"ALL"}))}
        ctrl._apply(item, "acquire")
        self.assertEqual(host.gold, 10)
        ctrl.end_turn(event_id="end6")
        ctrl.start_turn(7, event_id="start7")
        self.assertEqual(host.gold, 12)
        self.assertEqual(distinct_minion_type_count([MinionView("x","dual",frozenset({"BEAST","DEMON"}))]), 1)

    def test_worn_map_pays_after_two_starts_once(self):
        ctrl, host = self.controller()
        self.own(ctrl, "BG32_MagicItem_428")
        for turn in (7, 8, 9):
            ctrl.end_turn(event_id=f"end{turn-1}")
            ctrl.start_turn(turn, event_id=f"start{turn}")
            self.assertEqual(host.gold, 10 if turn == 7 else 20)

    def test_ornate_clock_moves_due_slot_and_cancels_turn_nine_offer(self):
        ctrl, host = self.controller()
        item = OwnedTrinket("BG32_MagicItem_271", 6)
        ctrl.owned.append(item)
        ctrl._offered_slots.add("normal_lesser")
        ctrl._apply(item, "acquire")
        self.assertEqual((ctrl.greater_offer_turn, host.gold), (7, 12))
        ctrl.end_turn(event_id="end6")
        ctrl.start_turn(7, event_id="start7")
        self.assertEqual(ctrl.scheduled_offering(), ("normal_greater", "greater"))
        ids=("BG30_MagicItem_996","BG30_MagicItem_420t",SPHERE_ID,"BG30_MagicItem_914t")
        offers=tuple(TrinketOffer(cid,max(0,self.catalog[cid].cost-(2 if self.catalog[cid].affiliated_types else 0))) for cid in ids)
        ctrl.install_offering(offers, context(7,owned=[item.card_id]),slot="normal_greater",stage="greater")
        ctrl.buy_trinket("BG30_MagicItem_996")
        for turn in (8,9):
            ctrl.end_turn(event_id=f"end{turn-1}")
            ctrl.start_turn(turn,event_id=f"start{turn}")
        self.assertIsNone(ctrl.scheduled_offering())

    def test_random_minion_generation_consumes_pool_and_reports_exhaustion(self):
        ctrl, host = self.controller()
        item = OwnedTrinket("BG30_MagicItem_430",6)
        ctrl.owned.append(item)
        ctrl._apply(item,"acquire")
        self.assertEqual(host.pool["battlecry"],2)
        host.pool["battlecry"]=0
        ctrl.end_turn(event_id="end6")
        ctrl.start_turn(7,event_id="start7")
        self.assertEqual(host.generated_minions,["battlecry"])
        self.assertEqual(ctrl.trace[-1]["event"],"generation_pool_exhausted")

    def test_lava_lamp_uses_current_seven_sales_counter_across_turns(self):
        ctrl, host = self.controller()
        self.own(ctrl,"BG30_MagicItem_951")
        for i in range(6): ctrl.on_minion_sold(MinionView("m1","sold"),event_id=f"sell{i}")
        self.assertEqual(host.generated_minions,[])
        ctrl.end_turn(event_id="end6")
        ctrl.start_turn(7,event_id="start7")
        ctrl.on_minion_sold(MinionView("m1","sold"),event_id="sell7")
        self.assertEqual(host.generated_minions,["elemental"])
        self.assertEqual(ctrl.owned[0].counter,0)

    def test_turns_cannot_be_skipped_to_hide_scheduled_rewards(self):
        ctrl, host = self.controller()
        self.own(ctrl, "BG30_MagicItem_847")
        ctrl.end_turn(event_id="end6")
        with self.assertRaises(TrinketError): ctrl.start_turn(9, event_id="skip")
        self.assertEqual(host.income_cap, 11)

    def test_unvalidated_combat_event_is_rejected(self):
        ctrl, _ = self.controller()
        with self.assertRaises(TrinketError):
            ctrl.on_spell_cast("spell", tavern_spell=True, target_minion_id=None, event_id="combat", in_combat=True)


class TestOfferingConstraints(unittest.TestCase):
    def setUp(self):
        self.catalog = {
            "a": TrinketSpec("a", "Neutral A", "lesser", 1, frozenset(), False),
            "b": TrinketSpec("b", "Neutral B", "lesser", 5, frozenset(), False),
            "c": TrinketSpec("c", "Beast A", "lesser", 3, frozenset({"BEAST"}), True),
            "d": TrinketSpec("d", "Beast B", "lesser", 3, frozenset({"BEAST"}), True),
            "e": TrinketSpec("e", "Naga Pivot", "lesser", 3, frozenset({"NAGA"}), False),
            "f": TrinketSpec("f", "Naga Pivot 2", "lesser", 3, frozenset({"NAGA"}), False),
        }
        board = [MinionView(str(i), "beast", frozenset({"BEAST"})) for i in range(2)]
        self.ctx = context(board=board)

    def test_invalid_solo_lobby_is_rejected(self):
        with self.assertRaises(ValueError):
            dataclasses.replace(self.ctx, lobby_types=frozenset({"BEAST"}))

    def test_thresholds_change_by_stage(self):
        self.assertIn("BEAST", self.ctx.in_types("lesser"))
        self.assertNotIn("BEAST", self.ctx.in_types("greater"))

    def test_dual_types_no_type_and_hero_affinity(self):
        board = (MinionView("dual", "dual", frozenset({"BEAST", "DEMON"})),) * 2
        ctx = dataclasses.replace(self.ctx, board=board, hero_affinity=frozenset({"NAGA"}))
        self.assertEqual(ctx.counts()["BEAST"], 2)
        self.assertEqual(ctx.counts()["DEMON"], 2)
        self.assertIn("NAGA", ctx.in_types("greater"))

    def test_guarantees_and_pivot_discount_across_seeds(self):
        assumptions = OfferingAssumptions({cid: 1.0 for cid in self.catalog}, True)
        for seed in range(100):
            offers = offer_trinkets(self.catalog, self.ctx, stage="lesser",
                candidate_ids=list(self.catalog), rng=random.Random(seed), assumptions=assumptions)
            validate_offering(offers, self.catalog, self.ctx, "lesser")
            self.assertTrue(any(o.card_id in {"c", "d"} for o in offers))
            for o in offers:
                if o.card_id in {"e", "f"}: self.assertEqual(o.price, 1)
            self.assertLessEqual(sum(o.card_id in {"e", "f"} for o in offers), 1)

    def test_missing_weights_and_empty_constrained_pool_fail(self):
        with self.assertRaises(TrinketError):
            offer_trinkets(self.catalog, self.ctx, stage="lesser", candidate_ids=list(self.catalog),
                rng=random.Random(1), assumptions=OfferingAssumptions({"a": 1}, True))
        ctx = dataclasses.replace(self.ctx, owned_ids=frozenset(self.catalog))
        with self.assertRaises(TrinketError):
            offer_trinkets(self.catalog, ctx, stage="lesser", candidate_ids=list(self.catalog),
                rng=random.Random(1), assumptions=OfferingAssumptions({c: 1 for c in self.catalog}, True))

    def test_owned_id_cannot_be_offered(self):
        ctx = dataclasses.replace(self.ctx, owned_ids=frozenset({"a"}))
        with self.assertRaises(TrinketError):
            validate_offering([TrinketOffer("a", 1), TrinketOffer("b", 5), TrinketOffer("c", 3), TrinketOffer("e", 1)], self.catalog, ctx, "lesser")

    def test_illegal_pivot_price_fails(self):
        with self.assertRaises(TrinketError):
            validate_offering([TrinketOffer("a", 1), TrinketOffer("b", 5), TrinketOffer("c", 3), TrinketOffer("e", 3)], self.catalog, self.ctx, "lesser")

    def test_official_hero_exception(self):
        wallet = TrinketSpec("wallet", "Goblin Wallet", "lesser", 1, frozenset(), False)
        ctx = dataclasses.replace(self.ctx, history={"hero_power_owner": "Gallywix"})
        catalog = dict(self.catalog, wallet=wallet)
        with self.assertRaises(TrinketError):
            validate_offering([TrinketOffer("wallet", 1), TrinketOffer("b", 5), TrinketOffer("c", 3), TrinketOffer("e", 1)], catalog, ctx, "lesser")


EXTERNAL_ROOT = Path(os.environ.get("HSBRSIM_ROOT", ROOT.parent / "research/HSBRSIM"))


@unittest.skipUnless((EXTERNAL_ROOT / "hsrl2/game.py").exists(), "Pinned external hsrl2 checkout unavailable")
class TestHSRLTrinketHost(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from bg_ai.hsbrsim_data import build_current_database
        cls.current = build_current_database(EXTERNAL_ROOT)
        cls.rules = json.loads((ROOT / "data/ruleset.json").read_text())
        cls.cards = json.loads((ROOT / "data/reference_cards.json").read_text())
        cls.catalog = catalog_from_snapshot(cls.cards, cls.rules, resolve_sphere_from_official_history=True)

    def setup_host(self):
        from hsrl2.game import Game
        from hsrl2.hero import Hero
        from hsrl2.tags import GameTag
        hero = Hero("TB_BaconShop_HERO_34")
        hero.set(GameTag.TAVERN_TIER, 4)
        game = Game([hero, Hero("opponent")], self.current.db, seed=1)
        host = HSRLTrinketHost(game, hero, spell_ids=self.rules["active"]["tavern_spell_ids"],
                               minion_ids=self.rules["active"]["shop_minion_ids"])
        return game, hero, host

    def test_real_warcry_buy_quote_pays_zero_zero_then_three(self):
        game,hero,host=self.setup_host()
        ctrl=TrinketController(host,self.catalog,allowed_ids=list(CORE_EFFECTS),seed=9)
        ctrl.start_turn(6,event_id="start6")
        ctrl.owned.append(OwnedTrinket("BG36_MagicItem_202",6))
        hero.gold=10
        ids=list(host.minion_generation_options(keyword="BATTLECRY"))[:3]
        paid=[]
        for i,cid in enumerate(ids):
            self.assertTrue(game.minion_pool.acquire(cid))
            entity=game.create_minion(cid,controller=hero)
            hero.tavern.append(entity)
            view=host.minion_view(entity.uuid)
            quote=ctrl.quote_minion_buy(view,3)
            entity.set(host.tags.COST,quote)
            before=hero.gold
            self.assertTrue(game.buy_from_tavern(hero,entity))
            entity.clear(host.tags.COST)
            delta=before-hero.gold
            paid.append(delta)
            ctrl.on_minion_bought(view,base_price=3,paid_gold=delta,event_id=f"buy{i}")
        self.assertEqual(paid,[0,0,3])
        self.assertEqual(hero.gold,7)

    def test_real_great_boar_gems_apply_bonus_and_do_not_auto_play(self):
        from hsrl2.tags import Zone
        game, hero, host = self.setup_host()
        ctrl=TrinketController(host,self.catalog,allowed_ids=list(CORE_EFFECTS),seed=8)
        ctrl.start_turn(6,event_id="start6")
        item=OwnedTrinket("BG30_MagicItem_988",6)
        ctrl.owned.append(item)
        ctrl._apply(item,"acquire")
        self.assertEqual(len(hero.hand),3)
        self.assertEqual(hero.board,[])
        cid=host.minion_ids[0]
        minion=game.create_minion(cid,controller=hero)
        minion.zone=Zone.PLAY
        hero.board.append(minion)
        before=(minion.atk,minion.health)
        self.assertTrue(game.play_spell(hero,hero.hand[0],target=minion))
        self.assertEqual((minion.atk,minion.health),(before[0]+3,before[1]+2))

    def test_real_persistent_tavern_buff_applies_once_to_retained_and_new_minions(self):
        from hsrl2.minion import Minion
        from hsrl2.tags import Zone
        game, hero, host = self.setup_host()
        elemental=next(c for c in host.minion_ids if "ELEMENTAL" in game.db.get(c).raw["races"])
        dragon=next(c for c in host.minion_ids if "DRAGON" in game.db.get(c).raw["races"])
        for cid in (elemental,dragon):
            self.assertTrue(game.minion_pool.acquire(cid))
            minion=game.create_minion(cid,controller=hero)
            minion.zone=Zone.TAVERN
            hero.tavern.append(minion)
        saved=list(hero.tavern)
        host.buff_tavern(3,2,"BG30_MagicItem_544","ELEMENTAL")
        self.assertEqual(saved[0].atk,game.db.get(elemental).atk+3)
        self.assertEqual(saved[1].atk,game.db.get(dragon).atk)
        hero.set(host.tags.FROZEN,True)
        game.refresh_tavern(hero,free=True)
        self.assertEqual(saved[0].atk,game.db.get(elemental).atk+3)
        game.refresh_tavern(hero,free=True)
        for minion in hero.tavern:
            if isinstance(minion,Minion):
                d=game.db.get(minion.card_id)
                expected=3 if "ELEMENTAL" in d.raw["races"] or "ALL" in d.raw["races"] else 0
                self.assertEqual(minion.atk,d.atk+expected)

    def test_queued_third_copy_forms_triple_without_second_pool_charge(self):
        game,hero,host=self.setup_host()
        cid=next(c for c in host.minion_ids if game.db.golden_version(game.db.get(c)) is not None)
        before=game.minion_pool.available(cid)
        for _ in range(2):host.give_pool_minion(cid,"fixture")
        for _ in range(8):host.give_card("BG20_GEM","fixture")
        host.give_pool_minion(cid,"BG30_MagicItem_430")
        host.give_card("BG26_813t","BG30_MagicItem_435")
        hero.hand.pop()
        host.flush_generated_cards()
        goldens=[c for c in hero.hand if c.has(host.tags.GOLDEN)]
        self.assertEqual(len(goldens),1)
        self.assertEqual(game.minion_pool.available(cid),before-3)
        self.assertEqual(game.pending_hand_queue,[])
        self.assertEqual(hero.hand[-1].card_id,"BG26_813t")
        self.assertEqual(hero.board,[])

    def test_real_generated_minion_reserves_pool_while_waiting(self):
        game, hero, host = self.setup_host()
        for _ in range(10):host.give_card("BG20_GEM","test")
        cid=host.minion_generation_options(keyword="BATTLECRY")[0]
        before=game.minion_pool.available(cid)
        host.give_pool_minion(cid,"BG30_MagicItem_430")
        self.assertEqual(game.minion_pool.available(cid),before-1)
        self.assertEqual(len(game.pending_hand_queue),1)
        hero.hand.pop(0)
        host.flush_generated_cards()
        self.assertEqual(hero.hand[-1].card_id,cid)
        self.assertEqual(game.minion_pool.available(cid),before-1)
        self.assertEqual(hero.board,[])

    def test_real_recycling_refresh_is_consumed_by_existing_engine(self):
        game, hero, host = self.setup_host()
        hero.gold=0
        host.gain_free_refresh(1)
        game.refresh_tavern(hero)
        self.assertEqual(hero.gold,0)
        self.assertEqual(hero.get(host.tags.FREE_REFRESH_REMAINING),0)

    def test_real_spell_grant_queue_drains_when_space_opens(self):
        game, hero, host = self.setup_host()
        for _ in range(10): host.give_card("BG20_GEM", "test")
        host.give_card("BG26_813t", "BG30_MagicItem_435")
        self.assertEqual(len(hero.hand), 10)
        self.assertEqual(len(game.pending_hand_queue), 1)
        hero.hand.pop(0)
        host.flush_generated_cards()
        self.assertEqual(hero.hand[-1].card_id, "BG26_813t")
        self.assertEqual(game.pending_hand_queue, [])
        self.assertEqual(hero.board, [])

    def test_mixed_source_queue_fails_before_partial_delivery(self):
        game, hero, host = self.setup_host()
        for _ in range(10): host.give_card("BG20_GEM", "test")
        host.give_card("BG26_813t", "BG30_MagicItem_435")
        other = game.create_spell("BG20_GEM", controller=hero)
        game.pending_hand_queue.append((hero, other))
        hero.hand.pop(0)
        with self.assertRaises(TrinketError): host.flush_generated_cards()
        self.assertEqual(len(hero.hand), 9)
        self.assertEqual(len(game.pending_hand_queue), 2)

    def test_real_buff_and_undead_aura_survive_in_recruit_state(self):
        from hsrl2.minion import Minion
        from hsrl2.tags import Race
        game, hero, host = self.setup_host()
        minion = Minion("test", "test", atk=2, health=3, race=Race.UNDEAD)
        minion.controller = hero
        minion.game = game
        hero.board.append(minion)
        host.buff_minion(minion.uuid, 4, 4, "BG30_MagicItem_422")
        host.add_undead_attack(2)
        host.give_divine_shield(minion.uuid)
        self.assertEqual((minion.atk, minion.health), (8, 7))
        self.assertTrue(minion.has(host.tags.DIVINE_SHIELD))
        self.assertTrue(all(not buff.temporary for buff in minion.buffs))

    def test_real_current_spell_discover_and_sphere_controller(self):
        game, hero, host = self.setup_host()
        ctrl = TrinketController(host, self.catalog, allowed_ids=list(CORE_EFFECTS), seed=5)
        ctrl.start_turn(9, event_id="start9")
        ctrl.owned.append(OwnedTrinket(SPHERE_ID, 9))
        spell = host.discover_spell_options()[0]
        ctrl.on_spell_cast(spell, tavern_spell=True, target_minion_id=None, event_id="cast")
        ctrl.end_turn(event_id="end9")
        self.assertEqual([card.card_id for card in hero.hand], [spell] * 3)
        self.assertEqual(hero.board, [])
        self.assertTrue(set(host.discover_spell_options()) <= set(self.rules["active"]["tavern_spell_ids"]))


if __name__ == "__main__":
    unittest.main()
