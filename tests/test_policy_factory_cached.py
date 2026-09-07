"""Source invalidation, mutable-data isolation, and complete trajectory equality."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from bg_ai.hsbrsim_data import CurrentDataError, export_current_definitions
from bg_ai.hsbrsim_opening import build_opening_database
from bg_ai.policy_factory_cached import CachedPolicyFactory, IsolatedExportCache
from bg_ai.turn_budget import TimingProfile

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = Path(os.environ.get("HSBRSIM_ROOT", str(ROOT.parent / "research/HSBRSIM")))
TRIBES = ("BEAST", "DEMON", "MECH", "NAGA", "PIRATE")


@unittest.skipUnless((EXTERNAL / "hsrl2/game.py").is_file(), "Pinned HSBRSIM checkout required")
class CachedPolicyFactoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache = IsolatedExportCache(EXTERNAL)
        cls.ruleset = json.loads((ROOT / "data/ruleset.json").read_text())
        cls.profile = TimingProfile.load(ROOT / "config/turn-budget.json")

    def copy_sources(self, temp):
        destination = Path(temp)
        for relative in ("ruleset.json", "reference_cards.json",
                         "source/CardDefs.Bacon.xml.gz", "source/firestone_enums.json"):
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / "data" / relative, target)
        return dict(ruleset_path=destination / "ruleset.json",
                    cards_path=destination / "reference_cards.json",
                    client_xml_path=destination / "source/CardDefs.Bacon.xml.gz")

    def test_ordered_definitions_and_provenance_match_uncached_builder(self):
        baseline = build_opening_database(EXTERNAL, valid_tribes=TRIBES)
        cached = self.cache.build(TRIBES)
        self.assertEqual(baseline.definitions, cached.definitions)
        self.assertEqual(baseline.provenance, cached.provenance)
        self.assertEqual(list(baseline.db._by_id), list(cached.db._by_id))
        self.assertEqual([d.id for d in baseline.db.pool_minions()],
                         [d.id for d in cached.db.pool_minions()])
        self.assertEqual([d.id for d in baseline.db.pool_spells()],
                         [d.id for d in cached.db.pool_spells()])

    def test_views_and_future_builds_isolate_nested_mutations(self):
        one, two = self.cache.build(TRIBES), self.cache.build(TRIBES)
        cid = "BG29_611"
        untouched = deepcopy(two.db.get(cid).raw)
        one.db.get(cid).raw["raw_tags"]["47"] = -777
        one.db.get(cid).raw["races"].append("INVENTED")
        one.provenance["fixture_minion_ids"].clear()
        self.assertEqual(two.db.get(cid).raw, untouched)
        self.assertTrue(two.provenance["fixture_minion_ids"])
        self.assertIsNot(one.db.get(cid), two.db.get(cid))
        self.assertIsNot(one.db.get(cid).raw["raw_tags"], two.db.get(cid).raw["raw_tags"])
        third = self.cache.build(TRIBES)
        self.assertEqual(third.db.get(cid).raw, untouched)
        self.assertEqual(third.provenance, two.provenance)
        self.assertIsInstance(self.cache._source_json, bytes)
        self.assertIsInstance(self.cache._provenance_json, bytes)

    def test_each_frozen_source_change_invalidates_before_view_construction(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = self.copy_sources(temp)
            cache = IsolatedExportCache(EXTERNAL, **paths)
            sources = list(paths.values()) + [paths["client_xml_path"].parent / "firestone_enums.json"]
            for source in sources:
                with self.subTest(source=source.name):
                    original = source.read_bytes()
                    source.write_bytes(original + b"\n")
                    with patch.object(cache, "_CardDB", side_effect=AssertionError("View created before verification")):
                        with self.assertRaisesRegex(CurrentDataError, "changed source"):
                            cache.build(TRIBES)
                    source.write_bytes(original)
            self.assertTrue(cache.build(TRIBES).db.pool_minions())

    def test_changed_source_after_export_is_not_adopted_as_expected_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = self.copy_sources(temp)

            def export_then_change(**kwargs):
                exported = export_current_definitions(**kwargs)
                path = paths["ruleset_path"]
                path.write_bytes(path.read_bytes() + b"\n")
                return exported

            with patch("bg_ai.policy_factory_cached.export_current_definitions", side_effect=export_then_change):
                with self.assertRaisesRegex(CurrentDataError, "changed source"):
                    IsolatedExportCache(EXTERNAL, **paths)

    def test_warm_factory_revalidates_sources(self):
        factory = CachedPolicyFactory(EXTERNAL, self.ruleset, self.profile)
        factory.setup(710041)
        with patch.object(factory._export_cache, "validate", side_effect=CurrentDataError("changed source")):
            with self.assertRaisesRegex(CurrentDataError, "changed source"):
                factory.setup(710041)

    def test_view_limit_requires_exact_positive_integer(self):
        for value in (0, -1, True, False, 1.5, "16", None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    CachedPolicyFactory(EXTERNAL, self.ruleset, self.profile, max_cached_views=value)

    def test_warm_factory_rejects_caller_ruleset_mutation(self):
        ruleset = deepcopy(self.ruleset)
        factory = CachedPolicyFactory(EXTERNAL, ruleset, self.profile)
        factory.setup(710041)
        ruleset["patch"] = "changed"
        with self.assertRaisesRegex(CurrentDataError, "ruleset differs"):
            factory.setup(710041)

    def test_warm_access_refreshes_least_recently_used_order(self):
        factory = CachedPolicyFactory(EXTERNAL, self.ruleset, self.profile, max_cached_views=2)
        _, first = factory.setup(710041)
        _, second = factory.setup(710042)
        factory.setup(710041)
        self.assertEqual(list(factory.databases), [second["valid_tribes"], first["valid_tribes"]])
        _, third = factory.setup(710043)
        self.assertEqual(list(factory.databases), [first["valid_tribes"], third["valid_tribes"]])

    def test_three_complete_seeded_combat_trajectories_equal_baseline(self):
        # Existing trainer functions are a read-only oracle; the production
        # cache module has no imports from trainers or the audit prototype.
        sys.path.insert(0, str(ROOT / "scripts"))
        try:
            from train_recruit_policy import PolicyFactory, play_episode
            from train_two_turn_recruit import CombatBridge
        finally:
            sys.path.remove(str(ROOT / "scripts"))
        baseline = PolicyFactory(EXTERNAL, self.ruleset, self.profile)
        cached = CachedPolicyFactory(EXTERNAL, self.ruleset, self.profile)
        with tempfile.TemporaryDirectory() as temp:
            bridge = CombatBridge(Path(temp) / "firestone.log")
            try:
                for seed in (710041, 710042, 710043):
                    with self.subTest(seed=seed):
                        expected = play_episode(baseline, bridge, seed, collect=True)
                        actual = play_episode(cached, bridge, seed, collect=True)
                        repeated = play_episode(cached, bridge, seed, collect=True)
                        self.assertTrue(expected["complete"])
                        self.assertEqual(len(expected["combats"]), 2)
                        self.assertEqual(actual, expected)
                        self.assertEqual(repeated, actual)
            finally:
                bridge.close()
                bridge.process.stdout.close()

    def test_eviction_preserves_active_game_and_complete_repeated_trajectory(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        try:
            from train_recruit_policy import PolicyFactory, play_episode
            from train_two_turn_recruit import CombatBridge
        finally:
            sys.path.remove(str(ROOT / "scripts"))
        baseline = PolicyFactory(EXTERNAL, self.ruleset, self.profile)
        cached = CachedPolicyFactory(EXTERNAL, self.ruleset, self.profile, max_cached_views=1)
        held_game, held_setup = cached.setup(710041)
        held_db = held_game.engine.db
        held_observation = deepcopy(held_game._visible)
        cached.setup(710042)
        self.assertEqual(len(cached.databases), 1)
        self.assertNotIn(held_setup["valid_tribes"], cached.databases)
        self.assertIs(held_game.engine.db, held_db)
        self.assertEqual(held_game._visible, held_observation)
        with tempfile.TemporaryDirectory() as temp:
            bridge = CombatBridge(Path(temp) / "firestone.log")
            try:
                expected = play_episode(baseline, bridge, 710041, collect=True)
                first = play_episode(cached, bridge, 710041, collect=True)
                first_db = next(iter(cached.databases.values())).db
                middle = play_episode(cached, bridge, 710042, collect=True)
                repeated = play_episode(cached, bridge, 710041, collect=True)
                self.assertTrue(middle["complete"])
                self.assertEqual(first, expected)
                self.assertEqual(repeated, first)
                self.assertEqual(len(cached.databases), 1)
                self.assertIsNot(next(iter(cached.databases.values())).db, first_db)
            finally:
                bridge.close()
                bridge.process.stdout.close()


if __name__ == "__main__":
    unittest.main()
