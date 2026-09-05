"""Current-source consistency and external data projection regressions."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bg_ai.hsbrsim_data import (
    CurrentDataError, ROOT, export_current_definitions, verify_engine_checkout,
)


class CurrentDefinitionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exported = export_current_definitions()
        cls.entries = cls.exported.by_id
        cls.ruleset = json.loads((ROOT / "data/ruleset.json").read_text())
        cls.cards = json.loads((ROOT / "data/reference_cards.json").read_text())

    def test_all_current_pools_match_exact_manifest_ids(self):
        r = self.exported.provenance
        self.assertEqual(r["active_counts"]["minion_ids"], 246)
        self.assertEqual(r["active_counts"]["hero_ids"], 116)
        self.assertEqual(r["active_counts"]["hero_power_ids"], 116)
        self.assertEqual(r["active_counts"]["tavern_spell_ids"], 71)
        self.assertEqual(r["active_counts"]["dark_gift_ids"], 43)
        for flag, pool in (("is_pool_minion", "shop_minion_ids"),
                           ("is_pool_spell", "tavern_spell_ids"),
                           ("dark_gift", "dark_gift_ids"),
                           ("hero_draftable", "hero_ids")):
            self.assertEqual({e["id"] for e in self.entries.values() if e[flag]},
                             set(self.ruleset["active"][pool]))

    def test_current_stats_replace_old_numbers_and_keep_zero(self):
        goldrinn = self.entries["BGS_018"]
        self.assertEqual((goldrinn["atk"], goldrinn["health"], goldrinn["tech_level"]), (7, 7, 5))
        investigator = self.entries["BG36_509"]
        self.assertEqual((investigator["atk"], investigator["health"]), (5, 6))
        for card in self.cards:
            row = self.entries[card["id"]]
            for number, tid in enumerate((2, 3, 2889, 2919, 2920, 2921), 1):
                self.assertEqual(row.get(f"script_data_num_{number}"),
                                 card["rawTags"].get(str(tid)))
        self.assertEqual(self.entries["BG31_843"]["script_data_num_1"], 4)
        self.assertEqual(self.entries["BG31_809"]["script_data_num_4"], 5)

    def test_source_keywords_are_not_referenced_keywords(self):
        # Fortify grants Taunt to its target; it does not itself have Taunt.
        self.assertFalse(self.entries["BG28_503"]["taunt"])
        self.assertTrue(self.entries["BG25_001"]["taunt"])
        self.assertTrue(self.entries["BG25_001"]["reborn"])
        self.assertFalse(self.entries["BGS_126"]["cleave"])
        self.assertEqual(self.entries["BG36_503"]["activate_cost"], 2)
        # Onyxia's Avenge threshold is 4; script parameter 1 is Whelp stats (1).
        self.assertEqual(self.entries["BG22_HERO_305p"]["avenge"], 4)
        self.assertEqual(self.entries["BG22_HERO_305p"]["script_data_num_1"], 1)

    def test_generated_goldens_bans_and_trinkets_never_enter_shop(self):
        self.assertIn("BG20_GEM", self.entries)
        self.assertFalse(self.entries["BG20_GEM"]["is_pool_spell"])
        self.assertFalse(self.entries["BGS_121"]["is_pool_minion"])
        for cid in self.ruleset["active"]["tier7_minion_ids"]:
            self.assertFalse(self.entries[cid]["is_pool_minion"])
        for row in self.entries.values():
            if row.get("triple_base_id"):
                self.assertFalse(row["is_pool_minion"])
            self.assertFalse(row["is_current_trinket"])
        self.assertEqual(self.exported.provenance["candidate_trinket_count"], 228)
        # An always-golden shop card is not an ordinary triple-golden definition.
        self.assertTrue(self.entries["BG32_236"]["is_pool_minion"])

    def test_hero_armor_power_and_spell_types_project_explicitly(self):
        hero = self.entries["BG20_HERO_101"]
        self.assertEqual(hero["armor"], 14)
        self.assertEqual(hero["hero_power_id"], "BG20_HERO_101p")
        self.assertEqual(self.entries["TB_BaconShop_HERO_34"]["armor"], 0)
        for cid in self.ruleset["active"]["tavern_spell_ids"]:
            self.assertEqual(self.entries[cid]["source_card_type"], 42)
            self.assertEqual(self.entries[cid]["card_type"], 5)
        self.assertEqual(self.entries["BG28_507"]["cost"], 0)
        self.assertEqual(self.entries["BG28_507"]["cost_source"], "absent_client_tag_zero")

    def test_every_active_structural_reference_resolves(self):
        p = self.exported.provenance
        self.assertEqual(p["reference_dependency_closure_count"], 1099)
        self.assertEqual(p["missing_reachable_references"], [])
        self.assertEqual(p["active_golden_links_validated"], 246)
        self.assertEqual(p["hero_power_links_validated"], 116)
        self.assertFalse(p["full_game_ready"])
        self.assertEqual(len(p["dual_tribe_minion_ids"]), 10)
        self.assertEqual(self.entries["BG_DEEP_015"]["races"], ["MECH", "UNDEAD"])
        for link in p["reference_dependency_edges"]:
            self.assertIn(link["to"], self.entries)

    def _altered_sources(self, change):
        """Re-hash edited JSON to test semantic checks independently of hashing."""
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name)
        rules = copy.deepcopy(self.ruleset)
        cards = copy.deepcopy(self.cards)
        change(rules, cards)
        path = root / "cards.json"
        path.write_text(json.dumps(cards))
        rules["checksums"]["reference_cards.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
        rp = root / "ruleset.json"
        rp.write_text(json.dumps(rules))
        return temp, dict(ruleset_path=rp, cards_path=path)

    def test_flat_numeric_tampering_fails_even_after_rehash(self):
        def change(rules, cards):
            next(c for c in cards if c["id"] == "BG36_509")["attack"] = 2
        temp, kwargs = self._altered_sources(change)
        with temp, self.assertRaisesRegex(CurrentDataError, "Flattened field"):
            export_current_definitions(**kwargs)

    def test_missing_or_changed_raw_client_data_fails(self):
        for mutate in (lambda c: c["rawTags"].pop("47"),
                       lambda c: c["rawTags"].__setitem__("47", 2)):
            temp, kwargs = self._altered_sources(
                lambda rules, cards: mutate(next(c for c in cards if c["id"] == "BG36_509")))
            with temp, self.assertRaisesRegex(CurrentDataError, "numeric tag mismatch"):
                export_current_definitions(**kwargs)

    def test_retired_minion_cannot_be_smuggled_into_partition(self):
        def change(rules, cards):
            rules["active"]["shop_minion_ids"].append("BGS_121")
        temp, kwargs = self._altered_sources(change)
        with temp, self.assertRaisesRegex(CurrentDataError, "partition"):
            export_current_definitions(**kwargs)

    def test_blocked_candidate_cannot_be_activated(self):
        def change(rules, cards):
            rules["active"]["lesser_trinket_ids"].append(rules["candidate"]["lesser_trinket_ids"][0])
        temp, kwargs = self._altered_sources(change)
        with temp, self.assertRaisesRegex(CurrentDataError, "Blocked candidate"):
            export_current_definitions(**kwargs)

    def test_wrong_or_dirty_revision_rejected_before_import(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)
            (path / "hsrl2").mkdir()
            (path / "hsrl2/db.py").write_text("raise AssertionError('never import')")
            with patch("bg_ai.hsbrsim_data.subprocess.check_output", side_effect=["bad\n", ""]):
                with self.assertRaisesRegex(CurrentDataError, "Unreviewed"):
                    verify_engine_checkout(path)
            from bg_ai.recruiting import HSBRSIM_REVISION
            with patch("bg_ai.hsbrsim_data.subprocess.check_output", side_effect=[HSBRSIM_REVISION, " M hsrl2/db.py"]):
                with self.assertRaisesRegex(CurrentDataError, "dirty"):
                    verify_engine_checkout(path)


class ExternalCardDBIntegrationTests(unittest.TestCase):
    def test_external_empty_database_registers_current_values_once(self):
        engine = Path(os.environ.get("HSBRSIM_ROOT", ROOT.parent / "research/HSBRSIM"))
        if not (engine / "hsrl2/db.py").is_file():
            self.skipTest("Set HSBRSIM_ROOT to the clean pinned external checkout")
        source = r'''
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from bg_ai.hsbrsim_data import build_current_database, CurrentDataError
r = build_current_database(Path(sys.argv[2]))
cards = json.loads(Path(sys.argv[3]).read_text())
assert len(r.db) == len(cards) == 5613
for c in cards:
    d = r.db.get(c['id']); t = c['rawTags']
    assert d.atk == t.get('47', 0), c['id']
    assert d.health == t.get('45', 0), c['id']
    assert d.tech_level == t.get('1440', 0), c['id']
    assert d.armor == t.get('292', 0), c['id']
    assert d.cost == t.get('48', 0), c['id']
    for i, tag in enumerate(('2', '3', '2889', '2919')):
        assert d.num(i) == t.get(tag), (c['id'], i)
    assert d.raw['raw_tags'] == t
assert len(r.db.pool_minions()) == 234
assert len(r.db.pool_spells()) == 71
assert len(r.db.dark_gifts()) == 43
assert r.db.get('BG28_503').card_type.value == 5
assert r.db.get('BG20_HERO_101').hero_power_id == 'BG20_HERO_101p'
assert r.db.get('BG22_HERO_305p').avenge_target == 4
f = build_current_database(Path(sys.argv[2]), fixture_minion_ids=['BG25_001'], fixture_spell_ids=[])
assert [d.id for d in f.db.pool_minions()] == ['BG25_001']
assert f.db.pool_spells() == []
assert f.provenance['fixture_only'] and not f.provenance['full_game_ready']
try:
    build_current_database(Path(sys.argv[2]), fixture_minion_ids=['BGS_121'])
except CurrentDataError:
    pass
else:
    raise AssertionError('inactive fixture accepted')
print('5613 real CardDB definitions checked; current and restricted pools checked')
'''
        run = subprocess.run([sys.executable, "-I", "-c", source, str(ROOT / "python"),
                              str(engine), str(ROOT / "data/reference_cards.json")],
                             text=True, capture_output=True, timeout=120)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("5613 real CardDB definitions checked", run.stdout)


if __name__ == "__main__":
    unittest.main()
