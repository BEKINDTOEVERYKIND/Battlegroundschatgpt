#!/usr/bin/env python3
"""Standalone deterministic audit of an isolated verified-export cache.

Does not modify or monkeypatch the trainer. Run separately from an executing
experiment; results concern setup speed and equality, never playing strength.
"""
from __future__ import annotations

import argparse
import cProfile
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import random
import statistics
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from bg_ai.hsbrsim_data import (
    CurrentDataError, CurrentDatabase, export_current_definitions,
    verify_engine_checkout,
)
from bg_ai.hsbrsim_opening import (
    OpeningFixtureSpec, TIER1_MINIONS, TIER1_SPELLS, TRIBES,
    build_opening_database, lobby_minion_ids,
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


class IsolatedExportCache:
    """Prototype: reuse verified parsing, retain independent mutable CardDBs.

    Every build rechecks exact source bytes and clean pinned engine status.
    A cached immutable JSON string is decoded into fresh nested dictionaries;
    no CardDef/raw dictionary is shared between two returned databases.
    """

    def __init__(self, engine_root):
        self.engine_root = verify_engine_checkout(engine_root)
        exported = export_current_definitions()
        self._source_json = canonical(exported.definitions)
        self._provenance_json = canonical(exported.provenance)
        paths = (ROOT / "data/ruleset.json", ROOT / "data/reference_cards.json",
                 ROOT / "data/source/CardDefs.Bacon.xml.gz", ROOT / "data/source/firestone_enums.json")
        self._checksums = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        self._allowed = {"minions": frozenset(d["id"] for d in exported.definitions
            if d["is_pool_minion"] and d["tech_level"] == 1),
            "spells": frozenset(d["id"] for d in exported.definitions
            if d["is_pool_spell"] and d["tech_level"] == 1)}
        if self._allowed != {"minions": TIER1_MINIONS, "spells": TIER1_SPELLS}:
            raise CurrentDataError("Current opening pool needs a new conformance audit")
        sys.path.insert(0, str(self.engine_root))
        try:
            self._CardDB = importlib.import_module("hsrl2.db").CardDB
            self._CardDef = importlib.import_module("hsrl2.defs").CardDef
        finally:
            sys.path.remove(str(self.engine_root))

    def build(self, tribes):
        spec = OpeningFixtureSpec(valid_tribes=tuple(tribes))
        verify_engine_checkout(self.engine_root)
        for path, expected in self._checksums.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise CurrentDataError("Verified-export cache invalidated by changed source: " + str(path))
        definitions = json.loads(self._source_json)
        selected = lobby_minion_ids(definitions, spec.valid_tribes)
        db = self._CardDB()
        for entry in definitions:
            entry["is_pool_minion"] = entry["id"] in selected
            entry["is_pool_spell"] = entry["id"] in TIER1_SPELLS
            if db.get(entry["id"]) is not None:
                raise CurrentDataError("Duplicate cached definition")
            db.register(self._CardDef.from_json(entry))
        provenance = json.loads(self._provenance_json)
        provenance.update(engine_checkout_clean=True, fixture_only=True,
            constructed_definitions_sha256=digest(definitions),
            fixture_minion_ids=sorted(d.id for d in db.pool_minions()),
            fixture_spell_ids=sorted(d.id for d in db.pool_spells()))
        return CurrentDatabase(db, provenance, tuple(definitions))


def audit(engine_root, progress_path=None):
    from bg_ai.choice_timing import ChoiceTimedTwoTurnOpeningEngine
    from bg_ai.opening_transition import TwoTurnOpeningRecruitEngine
    from bg_ai.turn_budget import TimingProfile
    from train_recruit_policy import PolicyFactory, play_episode
    from train_two_turn_recruit import CombatBridge

    ruleset = json.loads((ROOT / "data/ruleset.json").read_text())
    profile = TimingProfile.load(ROOT / "config/turn-budget.json")
    # Performance-only seeds, separated from every main experiment seed family.
    seeds = (710041, 710042, 710043)
    tuples = [tuple(sorted(random.Random(seed).sample(sorted(TRIBES), 5))) for seed in seeds]
    profiler = cProfile.Profile()
    profiler.enable()
    profiled = build_opening_database(engine_root, valid_tribes=tuples[0])
    profiler.disable()
    profile_rows = []
    for entry in profiler.getstats():
        name = entry.code.co_name if hasattr(entry.code, "co_name") else str(entry.code)
        if name in ("build_opening_database", "build_current_database", "export_current_definitions", "_read_xml"):
            profile_rows.append({"function": name, "calls": entry.callcount, "cumulative_seconds": entry.totaltime})
    start = time.perf_counter()
    cache = IsolatedExportCache(engine_root)
    initialization = time.perf_counter() - start
    builds = []
    for tribes in tuples:
        start = time.perf_counter()
        original = build_opening_database(engine_root, valid_tribes=tribes)
        original_seconds = time.perf_counter() - start
        start = time.perf_counter()
        cached = cache.build(tribes)
        cached_seconds = time.perf_counter() - start
        assert original.definitions == cached.definitions
        assert original.provenance == cached.provenance
        assert list(original.db._by_id) == list(cached.db._by_id)
        assert [d.id for d in original.db.pool_minions()] == [d.id for d in cached.db.pool_minions()]
        assert [d.id for d in original.db.pool_spells()] == [d.id for d in cached.db.pool_spells()]
        assert all(original.db.get(cid) == cached.db.get(cid) for cid in original.db._by_id)
        builds.append({"tribes": tribes, "uncached_seconds": original_seconds,
            "cached_seconds": cached_seconds, "all_definitions_and_provenance_equal": True,
            "ordered_pool_ids_equal": True,
            "constructed_definitions_sha256": cached.provenance["constructed_definitions_sha256"]})
    # Test writable dictionaries cannot leak from one cached view to another.
    one, two = cache.build(tuples[0]), cache.build(tuples[0])
    cid = next(iter(TIER1_MINIONS))
    before = deepcopy(two.db.get(cid).raw)
    one.db.get(cid).raw["raw_tags"]["47"] = -777
    assert two.db.get(cid).raw == before
    assert one.db.get(cid) is not two.db.get(cid)
    assert one.db.get(cid).raw is not two.db.get(cid).raw
    assert one.db.get(cid).raw["raw_tags"] is not two.db.get(cid).raw["raw_tags"]

    class CachedPolicyFactory(PolicyFactory):
        def setup(self, seed):
            rng = random.Random(seed)
            tribes = tuple(sorted(rng.sample(sorted(TRIBES), 5)))
            timers = (rng.choice((15000, 20000, 30000)), rng.choice((15000, 20000, 30000)))
            if tribes not in self.databases:
                self.databases[tribes] = cache.build(tribes)
            database = self.databases[tribes]
            raw = TwoTurnOpeningRecruitEngine(database.db, self.ruleset,
                timer_ms=lambda player, turn: timers[turn - 1] if player == 0 else 60000,
                provenance=database.provenance, fixture=OpeningFixtureSpec(valid_tribes=tribes))
            timed = ChoiceTimedTwoTurnOpeningEngine.for_fixture(raw, self.profile, self.ruleset)
            timed.reset(seed=seed)
            return timed, {"seed": seed, "valid_tribes": tribes, "player_zero_timer_ms": timers,
                "other_player_timer_ms": 60000, "provenance": database.provenance}

    original_factory = PolicyFactory(engine_root, ruleset, profile)
    cached_factory = CachedPolicyFactory(engine_root, ruleset, profile)
    episodes = []
    with tempfile.TemporaryDirectory(prefix="opening-cache-audit-") as temp:
        bridge = CombatBridge(Path(temp) / "firestone.log")
        try:
            for seed in seeds:
                start = time.perf_counter()
                original = play_episode(original_factory, bridge, seed, collect=True)
                original_seconds = time.perf_counter() - start
                start = time.perf_counter()
                cached = play_episode(cached_factory, bridge, seed, collect=True)
                cached_seconds = time.perf_counter() - start
                assert original == cached, "Cached factory changed a complete seeded trajectory"
                start = time.perf_counter()
                repeated = play_episode(cached_factory, bridge, seed, collect=True)
                warm_seconds = time.perf_counter() - start
                assert cached == repeated, "Reusing a scoped DB changed a repeated trajectory"
                episodes.append({"seed": seed, "uncached_cold_seconds": original_seconds,
                    "cached_cold_seconds": cached_seconds, "existing_tuple_warm_seconds": warm_seconds,
                    "commands": len(cached["commands"]), "complete_trajectory_equal": True,
                    "repeat_equal": True, "trajectory_sha256": digest(cached)})
        finally:
            bridge.close()
    current = json.loads(progress_path.read_text()) if progress_path and progress_path.exists() else None
    means = {"uncached_build_seconds": statistics.mean(row["uncached_seconds"] for row in builds),
             "cached_build_seconds": statistics.mean(row["cached_seconds"] for row in builds)}
    return {"schema_version": 1, "scope": "setup performance and deterministic equality; no policy selection",
        "prototype": "verified export cached; independently decoded CardDefs/raw dictionaries per view",
        "executing_training_sources_modified": False, "main_test_outcomes_read": False,
        "profile": profile_rows, "cache_initialization_seconds": initialization,
        "builds": builds, "means": means,
        "cold_build_speedup_excluding_one_time_export": means["uncached_build_seconds"] / means["cached_build_seconds"],
        "nested_mutation_isolation_passed": True, "episodes": episodes,
        "main_progress_only": current,
        "limitations": ["Timing overlaps a live training process and is hardware/load dependent",
            "Prototype is not wired into an executing or frozen experiment",
            "Fresh scoped CardDefs retain existing memory use; sharing raw objects needs a read-only contract",
            "Only three performance seeds and current two-turn supported practical trajectories compared"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--progress", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Use a fresh audit report path")
    result = audit(args.engine_root, args.progress)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "build_speedup": result["cold_build_speedup_excluding_one_time_export"],
        "trajectory_equality": all(row["complete_trajectory_equal"] for row in result["episodes"]),
        "means": result["means"], "episodes": result["episodes"]}, indent=2))


if __name__ == "__main__":
    main()
