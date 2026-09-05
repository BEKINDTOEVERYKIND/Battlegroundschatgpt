"""Build a fresh external hsrl2 CardDB from the pinned current client snapshot.

Reference definitions are not offer permissions. Only explicit manifest pools
become pool flags; structural links merely establish that referenced definitions
exist. This bridge does not authorize full-game training or validate scripts.
"""
from __future__ import annotations

import gzip
import hashlib
import importlib
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .recruiting import HSBRSIM_REVISION, ruleset_digest

ROOT = Path(__file__).resolve().parents[2]
NUM_TAGS = (2, 3, 2889, 2919, 2920, 2921)
FIELD_TAGS = {
    "atk": 47, "health": 45, "cost": 48, "card_race": 200,
    "card_type": 202, "card_set": 183, "tech_level": 1440,
    "rarity": 203, "armor": 292, "triple_upgrade_id": 1429,
    "triple_base_id": 1471, "activate_cost": 4090, "spellcraft_id": 2359,
    "evolution_card_id": 2519, "evolution_card_id_2": 3778,
    "companion_id": 2130, "hero_power_dbf": 380, "sell_value": 1587,
    "spell_school": 1635,
}
KEYWORD_TAGS = {
    "windfury": 189, "taunt": 190, "divine_shield": 194,
    "deathrattle": 217, "battlecry": 218, "poisonous": 363,
    "magnetic": 849, "reborn": 1085, "venomous": 2853,
    "start_of_combat": 1531, "rally": 4204, "activate": 4089,
    "stealth": 19, "choose_one": 443,
}
SUBSET_TAGS = {
    "dragon": 1591, "murloc": 1592, "demon": 1593, "beast": 1594,
    "mech": 1595, "pirate": 1596, "elemental": 1688,
    "quilboar": 1845, "naga": 2272, "undead": 2347,
}
# Named structural references only. Arbitrary large script numbers are NOT DBF
# IDs. A linked buddy/cosmetic is a dependency, never general offer eligibility.
LINK_TAGS = {
    1429: "golden", 1471: "normal", 380: "hero_power", 2130: "buddy",
    2359: "spellcraft", 2519: "evolution", 3778: "evolution_2",
    4314: "guide_related", 2376: "hero_power_base_hero",
}
POOL_TYPES = {
    "minion_ids": 4, "shop_minion_ids": 4, "tier7_minion_ids": 4,
    "hero_ids": 3, "hero_power_ids": 10, "tavern_spell_ids": 42,
    "dark_gift_ids": 5, "lesser_trinket_ids": 44, "greater_trinket_ids": 44,
}
FLAT_TAGS = {
    "attack": 47, "health": 45, "cost": 48, "techLevel": 1440,
    "armor": 292, "heroPowerDbfId": 380,
    "battlegroundsPremiumDbfId": 1429, "battlegroundsNormalDbfId": 1471,
    "scriptDataNum1": 2, "scriptDataNum2": 3,
}


class CurrentDataError(ValueError):
    """An exact source, definition, linkage or external checkout check failed."""


@dataclass(frozen=True)
class CurrentDataExport:
    definitions: tuple[dict[str, Any], ...]
    provenance: dict[str, Any]

    @property
    def by_id(self) -> dict[str, dict[str, Any]]:
        return {entry["id"]: entry for entry in self.definitions}


@dataclass(frozen=True)
class CurrentDatabase:
    db: Any
    provenance: dict[str, Any]
    definitions: tuple[dict[str, Any], ...]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_engine_checkout(engine_root: Path) -> Path:
    """Check HEAD and tracked/untracked state before executing external Python."""
    root = Path(engine_root).resolve()
    if not (root / "hsrl2/db.py").is_file():
        raise CurrentDataError(f"Missing external hsrl2 checkout: {root}")
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
            text=True).strip()
    except subprocess.CalledProcessError as exc:
        raise CurrentDataError("Could not verify external Git checkout") from exc
    if revision != HSBRSIM_REVISION:
        raise CurrentDataError(f"Unreviewed external revision: {revision}")
    if status:
        raise CurrentDataError("External checkout is dirty; refusing imports")
    # Do not mix an already imported hsrl2 from a different installation.
    for name, module in tuple(sys.modules.items()):
        if name == "hsrl2" or name.startswith("hsrl2."):
            source = getattr(module, "__file__", None)
            if source and not Path(source).resolve().is_relative_to(root / "hsrl2"):
                raise CurrentDataError(f"Foreign external module already loaded: {name}")
    return root


def _read_xml(path: Path) -> tuple[int, dict[str, dict[str, Any]]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        xml = ET.parse(handle).getroot()
    if xml.tag != "CardDefs" or not xml.get("build"):
        raise CurrentDataError("Client XML has no CardDefs build")
    result = {}
    dbfs = set()
    for entity in xml.findall("Entity"):
        cid, dbf = entity.get("CardID"), int(entity.get("ID", "0"))
        if not cid or not dbf or cid in result or dbf in dbfs:
            raise CurrentDataError("Missing or duplicate client identity")
        dbfs.add(dbf)
        numbers, names, strings = {}, {}, {}
        for tag in entity.findall("Tag"):
            tid = str(int(tag.attrib["enumID"]))
            kind = tag.get("type")
            if kind in ("Int", "Card"):
                if tid in numbers:
                    raise CurrentDataError(f"Duplicate tag {cid}:{tid}")
                numbers[tid] = int(tag.attrib["value"])
                names[tid] = tag.get("name", tid)
            elif kind == "LocString":
                strings[tid] = tag.findtext("enUS", "")
        result[cid] = {"dbf": dbf, "numbers": numbers,
                       "tag_names": names, "strings": strings,
                       "referenced_tags": [(t.get("enumID"), t.get("name"))
                                           for t in entity.findall("ReferencedTag")]}
    return int(xml.attrib["build"]), result


def _assert_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise CurrentDataError(f"Expected integer {label}, received {value!r}")
    return value


def _definition(card: Mapping[str, Any], pools: Mapping[str, set[str]],
                candidates: set[str]) -> dict[str, Any]:
    cid = card["id"]
    raw = card["rawTags"]
    def tag(tid: int, default=None):
        return raw.get(str(tid), default)
    out = {"id": cid, "dbf_id": card["dbfId"], "name": card["name"],
           "text": card["text"], "raw_tags": dict(raw),
           "source_tags": dict(card["tags"]), "races": list(card["races"]),
           "source_mechanics": list(card["mechanics"]),
           "referenced_keywords": list(card["referencedTags"]),
           "source_snapshot_build": card["snapshotBuild"]}
    for key, tid in FIELD_TAGS.items():
        if str(tid) in raw:
            out[key] = tag(tid)
    out["source_card_type"] = out["card_type"]
    # hsrl2 creates tavern spells as generic Spell entities and lacks enum 42.
    # This is an explicit protocol projection; the source type remains intact.
    if out["card_type"] == 42:
        out["card_type"] = 5
    # Missing numeric client tags represent zero. Distinguish a minion's mana
    # cost (this field) from the engine's separate 3-gold purchase operation.
    out.setdefault("cost", 0)
    out.setdefault("armor", 0)
    out["cost_source"] = "client_tag_48" if "48" in raw else "absent_client_tag_zero"
    out["armor_source"] = "client_tag_292" if "292" in raw else "absent_client_tag_zero"
    for index, tid in enumerate(NUM_TAGS, 1):
        if str(tid) in raw:
            out[f"script_data_num_{index}"] = tag(tid)
    for key, tid in KEYWORD_TAGS.items():
        out[key] = bool(tag(tid, 0))
    for key, tid in SUBSET_TAGS.items():
        out[f"subset_{key}"] = bool(tag(tid, 0))
    # Some client text has an explicit Avenge(4) independent of script parameter
    # 1 (Onyxia's parameter is Whelp stats). Never equate the generic flag to 1.
    if tag(2129, 0):
        match = re.search(r"Avenge\s*\(\s*(\d+|\{[0-5]\})\s*\)", card["text"])
        if not match:
            raise CurrentDataError(f"Avenge threshold unavailable for {cid}")
        token = match.group(1)
        value = tag(NUM_TAGS[int(token[1])]) if token.startswith("{") else int(token)
        if value is None or value <= 0:
            raise CurrentDataError(f"Invalid Avenge threshold for {cid}")
        out["avenge"] = value
    # Cleave has no native client tag. Restrict this projection to the exact
    # explicit text, not any mention of adjacent damage (e.g. Wildfire Elemental).
    normalized = " ".join(card["text"].split())
    out["cleave"] = "Also damages adjacent minions." in normalized
    out["unplayable"] = bool(tag(1020, 0))
    out["health_cost"] = bool(tag(2911, 0))
    out["dark_gift"] = cid in pools["dark_gift_ids"]
    out["is_pool_minion"] = cid in pools["shop_minion_ids"]
    out["is_pool_spell"] = cid in pools["tavern_spell_ids"]
    out["hero_draftable"] = cid in pools["hero_ids"]
    roles = [key for key, ids in pools.items() if cid in ids]
    out["current_roles"] = roles or (["candidate_trinket"] if cid in candidates else ["reference_only"])
    out["is_current_trinket"] = (cid in pools["lesser_trinket_ids"] or
                                   cid in pools["greater_trinket_ids"])
    return out


def export_current_definitions(*, ruleset_path: Path = ROOT / "data/ruleset.json",
                               cards_path: Path = ROOT / "data/reference_cards.json",
                               client_xml_path: Path = ROOT / "data/source/CardDefs.Bacon.xml.gz") -> CurrentDataExport:
    """Cross-check both frozen sources and emit fresh, independent definitions."""
    ruleset_path, cards_path, client_xml_path = map(Path, (ruleset_path, cards_path, client_xml_path))
    ruleset = json.loads(ruleset_path.read_text())
    checksums = ruleset.get("checksums", {})
    for path, key in ((cards_path, "reference_cards.json"),
                      (client_xml_path, "source/CardDefs.Bacon.xml.gz")):
        if not checksums.get(key) or _sha(path) != checksums[key]:
            raise CurrentDataError(f"Frozen source checksum mismatch: {key}")
    enums_path = client_xml_path.parent / "firestone_enums.json"
    if not checksums.get("source/firestone_enums.json") or _sha(enums_path) != checksums["source/firestone_enums.json"]:
        raise CurrentDataError("Frozen source checksum mismatch: source/firestone_enums.json")
    enums = json.loads(enums_path.read_text())
    xml_build, xml = _read_xml(client_xml_path)
    if xml_build != ruleset.get("build"):
        raise CurrentDataError("Client build differs from ruleset")
    cards = json.loads(cards_path.read_text())
    if not isinstance(cards, list):
        raise CurrentDataError("Reference cards must be an array")
    by_id, by_dbf = {}, {}
    for card in cards:
        for key in ("id", "dbfId", "name", "text", "rawTags", "tags", "races",
                    "mechanics", "referencedTags", "snapshotBuild"):
            if key not in card:
                raise CurrentDataError(f"Missing reference field {key}")
        cid, dbf = card["id"], _assert_int(card["dbfId"], "dbfId")
        if cid in by_id or dbf in by_dbf or cid not in xml:
            raise CurrentDataError(f"Duplicate or unknown reference identity: {cid}")
        source = xml[cid]
        if card["snapshotBuild"] != xml_build or dbf != source["dbf"]:
            raise CurrentDataError(f"Client/reference identity mismatch: {cid}")
        for key, value in card["rawTags"].items():
            _assert_int(value, f"{cid}:tag:{key}")
        if card["rawTags"] != source["numbers"]:
            raise CurrentDataError(f"Client/reference numeric tag mismatch: {cid}")
        if card["name"] != source["strings"].get("185", "") or card["text"] != source["strings"].get("184", ""):
            raise CurrentDataError(f"Client/reference text mismatch: {cid}")
        raw = card["rawTags"]
        races = {raw["200"]} if raw.get("200") else set()
        races.update(race for tid, race in enums["secondary-race-tags"].items() if raw.get(tid))
        expected_races = sorted({enums["race"].get(str(race), f"UNKNOWN_RACE_{race}") for race in races})
        if card["races"] != expected_races:
            raise CurrentDataError(f"Client/reference tribe mismatch: {cid}")
        expected_tags = {}
        for tid, value in raw.items():
            xml_name = source["tag_names"][tid]
            name = enums["game-tags"].get(tid) or xml_name
            if not name or name.isdecimal():
                name = "UNMAPPED_" + tid
            expected_tags[name] = value
            if xml_name and not xml_name.isdecimal() and xml_name != name:
                expected_tags[xml_name] = value
        if card["tags"] != expected_tags:
            raise CurrentDataError(f"Client/reference named tag mismatch: {cid}")
        expected_referenced = sorted({enums["game-tags"].get(tid, name)
                                      for tid, name in source["referenced_tags"]})
        if card["referencedTags"] != expected_referenced:
            raise CurrentDataError(f"Client/reference referenced tag mismatch: {cid}")
        if "202" not in card["rawTags"]:
            raise CurrentDataError(f"Missing card type: {cid}")
        for key, tid in FLAT_TAGS.items():
            if key in card:
                _assert_int(card[key], f"{cid}:{key}")
            if key in card and card[key] != card["rawTags"].get(str(tid)):
                raise CurrentDataError(f"Flattened field contradicts client: {cid}:{key}")
        by_id[cid], by_dbf[dbf] = card, cid
    if set(by_id) != set(xml):
        raise CurrentDataError("Reference catalog differs from frozen client entities")
    pools = {}
    for key, expected_type in POOL_TYPES.items():
        values = ruleset.get("active", {}).get(key)
        if not isinstance(values, list) or len(values) != len(set(values)):
            raise CurrentDataError(f"Missing or duplicate explicit pool: {key}")
        pools[key] = set(values)
        for cid in values:
            if cid not in by_id or by_id[cid]["rawTags"]["202"] != expected_type:
                raise CurrentDataError(f"Wrong or missing type in pool {key}: {cid}")
            if by_id[cid].get("duosOnly") or by_id[cid]["rawTags"].get("1471"):
                raise CurrentDataError(f"Duos/golden definition in active pool: {cid}")
    if pools["shop_minion_ids"] & pools["tier7_minion_ids"] or (
            pools["shop_minion_ids"] | pools["tier7_minion_ids"] != pools["minion_ids"]):
        raise CurrentDataError("Normal-shop and special tier-7 pools must partition minions")
    for cid in pools["minion_ids"]:
        raw = by_id[cid]["rawTags"]
        if not all(k in raw for k in ("45", "47", "1440", "1429")):
            raise CurrentDataError(f"Incomplete minion stats/tier/golden link: {cid}")
        if ((raw["1440"] != 7) if cid in pools["tier7_minion_ids"] else
                not 1 <= raw["1440"] <= 6):
            raise CurrentDataError(f"Wrong minion pool tier: {cid}")
    linked_powers = set()
    for cid in pools["hero_ids"]:
        raw = by_id[cid]["rawTags"]
        if "45" not in raw or "380" not in raw or raw["380"] not in by_dbf:
            raise CurrentDataError(f"Incomplete hero health/power definition: {cid}")
        linked_powers.add(by_dbf[raw["380"]])
    if linked_powers != pools["hero_power_ids"]:
        raise CurrentDataError("Active heroes do not map exactly to active hero powers")
    for cid in pools["tavern_spell_ids"]:
        if "1440" not in by_id[cid]["rawTags"]:
            raise CurrentDataError(f"Missing tavern spell tier: {cid}")
    candidates = set(ruleset.get("candidate", {}).get("lesser_trinket_ids", [])) | set(
        ruleset.get("candidate", {}).get("greater_trinket_ids", []))
    if ruleset.get("components", {}).get("trinket", {}).get("status") == "blocked" and (
            pools["lesser_trinket_ids"] or pools["greater_trinket_ids"]):
        raise CurrentDataError("Blocked candidate trinkets cannot enter active pools")
    if pools["minion_ids"] & set(ruleset.get("known_banned_minion_ids", [])):
        raise CurrentDataError("Banned minion in active pool")
    definitions = tuple(_definition(c, pools, candidates) for c in cards)
    entries = {entry["id"]: entry for entry in definitions}
    for entry in definitions:
        power = entry.get("hero_power_dbf")
        if power in by_dbf:
            entry["hero_power_id"] = by_dbf[power]
    active_roots = set().union(*pools.values())
    pending, reachable, links, missing = list(sorted(active_roots)), set(), [], []
    while pending:
        cid = pending.pop()
        if cid in reachable:
            continue
        reachable.add(cid)
        for tid, kind in LINK_TAGS.items():
            target = by_id[cid]["rawTags"].get(str(tid))
            if not target:
                continue
            if target not in by_dbf:
                missing.append({"id": cid, "tag": tid, "target_dbf": target})
                continue
            other = by_dbf[target]
            links.append({"from": cid, "kind": kind, "to": other, "tag": tid})
            pending.append(other)
    if missing:
        raise CurrentDataError(f"Missing reachable reference dependencies: {missing}")
    # Every active normal minion's golden must link back to the same base.
    for cid in pools["minion_ids"]:
        base = entries[cid]
        golden = entries[by_dbf[base["triple_upgrade_id"]]]
        if golden.get("triple_base_id") != base["dbf_id"]:
            raise CurrentDataError(f"Nonreciprocal active golden link: {cid}")
    duals = sorted(cid for cid in pools["minion_ids"] if len(by_id[cid]["races"]) > 1)
    later_params = sorted(cid for cid in active_roots if any(
        str(tid) in by_id[cid]["rawTags"] for tid in NUM_TAGS[4:]))
    provenance = {
        "schema_version": 1, "engine_revision": HSBRSIM_REVISION,
        "ruleset_sha256": ruleset_digest(ruleset), "ruleset_file_sha256": _sha(ruleset_path),
        "reference_cards_sha256": _sha(cards_path), "client_xml_sha256": _sha(client_xml_path),
        "client_build": xml_build, "patch": ruleset.get("patch"),
        "source_enums_sha256": _sha(enums_path),
        "definitions_sha256": hashlib.sha256(json.dumps(definitions, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest(),
        "source_client_commit": ruleset.get("source_client_commit"),
        "definition_count": len(definitions),
        "active_counts": {key: len(ids) for key, ids in pools.items()},
        "candidate_trinket_count": len(candidates),
        "all_numeric_tags_and_text_match_client": True,
        "active_golden_links_validated": len(pools["minion_ids"]),
        "hero_power_links_validated": len(linked_powers),
        "reference_dependency_root_count": len(active_roots),
        "reference_dependency_closure_count": len(reachable),
        "reference_dependency_ids": sorted(reachable),
        "reference_dependency_edges": sorted(links, key=lambda x: (x["from"], x["tag"], x["to"])),
        "missing_reachable_references": [],
        "dual_tribe_minion_ids": duals,
        "active_cards_with_script_parameters_5_or_6": later_params,
        "full_game_ready": False,
        "blockers": ["behavioral_conformance_unvalidated", "generated_script_dependencies_not_complete",
                     "external_engine_single_race_representation", "current_trinket_component_blocked"],
        "scope": "Current definition and explicit pool bridge; structural links are not generation permissions",
    }
    return CurrentDataExport(definitions, provenance)


def build_current_database(engine_root: Path, *,
                           ruleset_path: Path = ROOT / "data/ruleset.json",
                           cards_path: Path = ROOT / "data/reference_cards.json",
                           client_xml_path: Path = ROOT / "data/source/CardDefs.Bacon.xml.gz",
                           fixture_minion_ids: Sequence[str] | None = None,
                           fixture_spell_ids: Sequence[str] | None = None) -> CurrentDatabase:
    """Construct an empty external CardDB, registering each current source once.

    Fixture subsets are explicit experiments and can never be labeled current
    full-game environments. This function does not mutate the external checkout.
    """
    exported = export_current_definitions(ruleset_path=ruleset_path, cards_path=cards_path,
                                           client_xml_path=client_xml_path)
    root = verify_engine_checkout(engine_root)
    for selected, field in ((fixture_minion_ids, "is_pool_minion"),
                            (fixture_spell_ids, "is_pool_spell")):
        if selected is not None:
            allowed = {entry["id"] for entry in exported.definitions if entry[field]}
            if len(selected) != len(set(selected)) or not set(selected) <= allowed:
                raise CurrentDataError(f"Fixture includes duplicate or inactive {field} IDs")
    sys.path.insert(0, str(root))
    try:
        CardDB = importlib.import_module("hsrl2.db").CardDB
        CardDef = importlib.import_module("hsrl2.defs").CardDef
    finally:
        sys.path.remove(str(root))
    db = CardDB()
    definitions = []
    for original in exported.definitions:
        entry = dict(original)
        if fixture_minion_ids is not None:
            entry["is_pool_minion"] = entry["id"] in fixture_minion_ids
        if fixture_spell_ids is not None:
            entry["is_pool_spell"] = entry["id"] in fixture_spell_ids
        if db.get(entry["id"]) is not None:
            raise CurrentDataError("Duplicate registration would trigger stale-data merge")
        db.register(CardDef.from_json(entry))
        definitions.append(entry)
    provenance = dict(exported.provenance)
    provenance["engine_checkout_clean"] = True
    provenance["constructed_definitions_sha256"] = hashlib.sha256(
        json.dumps(definitions, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    provenance["fixture_only"] = fixture_minion_ids is not None or fixture_spell_ids is not None
    if provenance["fixture_only"]:
        provenance["fixture_minion_ids"] = sorted(d.id for d in db.pool_minions())
        provenance["fixture_spell_ids"] = sorted(d.id for d in db.pool_spells())
    return CurrentDatabase(db, provenance, tuple(definitions))
