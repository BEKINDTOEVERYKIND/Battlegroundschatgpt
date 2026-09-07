"""Reuse a verified export while keeping each scoped database independent.

Only serialized immutable bytes are cached across database views. Every view
gets fresh CardDefs and nested raw dictionaries in the original source order.
This changes setup cost, not the opening curriculum or its random draws.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import importlib
import json
from pathlib import Path
import random
import sys

from .choice_timing import ChoiceTimedTwoTurnOpeningEngine
from .hsbrsim_data import (
    ROOT, CurrentDataError, CurrentDatabase, export_current_definitions,
    verify_engine_checkout,
)
from .hsbrsim_opening import (
    OpeningFixtureSpec, TIER1_MINIONS, TIER1_SPELLS, TRIBES, lobby_minion_ids,
)
from .opening_transition import TwoTurnOpeningRecruitEngine
from .recruiting import ruleset_digest


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode()


class IsolatedExportCache:
    """Cache verified JSON bytes; never share mutable data across views.

    Source expectations come from the validated export's provenance, not from
    a later filesystem read that could silently bless a changed source.
    """

    def __init__(self, engine_root, *, ruleset_path=ROOT / "data/ruleset.json",
                 cards_path=ROOT / "data/reference_cards.json",
                 client_xml_path=ROOT / "data/source/CardDefs.Bacon.xml.gz"):
        self.engine_root = verify_engine_checkout(engine_root)
        ruleset_path, cards_path, client_xml_path = map(
            Path, (ruleset_path, cards_path, client_xml_path))
        exported = export_current_definitions(ruleset_path=ruleset_path,
            cards_path=cards_path, client_xml_path=client_xml_path)
        provenance = exported.provenance
        self._checksums = (
            (ruleset_path, provenance["ruleset_file_sha256"]),
            (cards_path, provenance["reference_cards_sha256"]),
            (client_xml_path, provenance["client_xml_sha256"]),
            (client_xml_path.parent / "firestone_enums.json", provenance["source_enums_sha256"]),
        )
        self.validate()
        self._source_json = _canonical(exported.definitions)
        self._provenance_json = _canonical(provenance)
        self.ruleset_sha256 = provenance["ruleset_sha256"]
        minions = frozenset(d["id"] for d in exported.definitions
            if d["is_pool_minion"] and d["tech_level"] == 1)
        spells = frozenset(d["id"] for d in exported.definitions
            if d["is_pool_spell"] and d["tech_level"] == 1)
        if minions != TIER1_MINIONS or spells != TIER1_SPELLS:
            raise CurrentDataError("Current opening pool needs a new conformance audit")
        sys.path.insert(0, str(self.engine_root))
        try:
            self._CardDB = importlib.import_module("hsrl2.db").CardDB
            self._CardDef = importlib.import_module("hsrl2.defs").CardDef
        finally:
            sys.path.remove(str(self.engine_root))

    def validate(self):
        """Reject changed frozen sources or an unreviewed external checkout."""
        verify_engine_checkout(self.engine_root)
        for path, expected in self._checksums:
            try:
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError as exc:
                raise CurrentDataError("Verified-export source unavailable: " + str(path)) from exc
            if actual != expected:
                raise CurrentDataError("Verified-export cache invalidated by changed source: " + str(path))

    def build(self, tribes):
        spec = OpeningFixtureSpec(valid_tribes=tuple(tribes))
        self.validate()
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
            constructed_definitions_sha256=hashlib.sha256(_canonical(definitions)).hexdigest(),
            fixture_minion_ids=sorted(d.id for d in db.pool_minions()),
            fixture_spell_ids=sorted(d.id for d in db.pool_spells()))
        return CurrentDatabase(db, provenance, tuple(definitions))


class CachedPolicyFactory:
    """Identical seeded setup API with cached parsing and isolated tribe views."""

    def __init__(self, engine_root, ruleset, profile, *, max_cached_views=16):
        if type(max_cached_views) is not int or max_cached_views <= 0:
            raise ValueError("max_cached_views must be a positive integer")
        self.engine_root, self.ruleset, self.profile = engine_root, ruleset, profile
        self.max_cached_views = max_cached_views
        self.databases = OrderedDict()
        self._export_cache = IsolatedExportCache(engine_root)
        if ruleset_digest(ruleset) != self._export_cache.ruleset_sha256:
            raise CurrentDataError("Policy factory ruleset differs from the verified export")

    def setup(self, seed):
        if ruleset_digest(self.ruleset) != self._export_cache.ruleset_sha256:
            raise CurrentDataError("Policy factory ruleset differs from the verified export")
        rng = random.Random(seed)
        tribes = tuple(sorted(rng.sample(sorted(TRIBES), 5)))
        timers = (rng.choice((15000, 20000, 30000)), rng.choice((15000, 20000, 30000)))
        if tribes not in self.databases:
            self.databases[tribes] = self._export_cache.build(tribes)
            while len(self.databases) > self.max_cached_views:
                # Active games retain their own references to an evicted DB.
                self.databases.popitem(last=False)
        else:
            # A warm tribe view must not make changed inputs invisible.
            self._export_cache.validate()
            self.databases.move_to_end(tribes)
        database = self.databases[tribes]
        raw = TwoTurnOpeningRecruitEngine(database.db, self.ruleset,
            timer_ms=lambda player, turn: timers[turn - 1] if player == 0 else 60000,
            provenance=database.provenance, fixture=OpeningFixtureSpec(valid_tribes=tribes))
        timed = ChoiceTimedTwoTurnOpeningEngine.for_fixture(raw, self.profile, self.ruleset)
        timed.reset(seed=seed)
        return timed, {"seed": seed, "valid_tribes": tribes, "player_zero_timer_ms": timers,
            "other_player_timer_ms": 60000, "provenance": database.provenance}
