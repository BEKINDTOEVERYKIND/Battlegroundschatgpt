#!/usr/bin/env python3
"""Freeze and verify the live Solo pool against Blizzard and a pinned game client.

This is a DATA verifier, not a claim that a simulator implements the full game.
Unknown rotations, mismatched pool flags and incomplete pages fail closed.
Run --offline to reproduce from checked-in sources, --check to check integrity.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
BUILD = 251332
PATCH = "36.4.2"
HSDATA_COMMIT = "dee8a641ef8427cf853ca707c2e427e752a1e11f"
RULESET_ID = "battlegrounds-solo-36.4.2.251332-20260905"
DATE = "2026-09-05"
KINDS = ("minion", "hero", "spell", "trinket", "darkGift")
GALLERY = "https://hearthstone.blizzard.com/en-us/api/cards"
PATCH_URL = "https://hearthstone.blizzard.com/en-us/news/24296231/3642-patch-notes"
BANS_URL = "https://us.forums.blizzard.com/en/hearthstone/t/364-highlights-known-issues/153567"
XML_URL = f"https://raw.githubusercontent.com/HearthSim/hsdata/{HSDATA_COMMIT}/CardDefs.Bacon.xml"
MECHANICS = {
    "dark_gifts": True, "trinkets": True, "tavern_spells": True,
    "activate": True, "rally": True, "fishbait": True, "lockbox": True,
    "quests": False, "buddies": False, "anomalies": False, "timewarped_tavern": False,
}
# Global systems are disabled above. Hero/card-specific access is a separate
# reachability problem: e.g. E.T.C. can still generate a Buddy and Sire a Quest.
KNOWN_MINION_BANS = {"BGS_121": "Gentle Djinni"}
CLASSES = {2: "DRUID", 3: "HUNTER", 4: "MAGE", 5: "PALADIN", 6: "PRIEST",
           7: "ROGUE", 8: "SHAMAN", 9: "WARLOCK", 10: "WARRIOR", 12: "NEUTRAL",
           14: "DEMONHUNTER", 1: "DEATHKNIGHT"}


def sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def dump(path: Path, obj: object, *, pretty: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      indent=2 if pretty else None,
                      separators=None if pretty else (",", ":")) + "\n"
    path.write_text(data, encoding="utf-8")


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={
        "User-Agent": "Battlegroundschatgpt/0.1 (read-only ruleset verification)",
        "Accept": "application/json,text/xml,text/html",
    })
    with urllib.request.urlopen(request, timeout=40) as response:
        return response.read()


def gallery_url(kind: str, page: int = 1) -> str:
    return GALLERY + "?" + urllib.parse.urlencode({
        "gameMode": "battlegrounds", "bgGameMode": "solos",
        "bgCardType": kind, "pageSize": 450, "page": page, "locale": "en_US",
    })


def download_gallery(kind: str) -> dict:
    first = json.loads(fetch(gallery_url(kind)))
    cards = list(first.get("cards", []))
    pages = first.get("pageCount", 0)
    if not isinstance(pages, int) or pages < 1 or pages > 25:
        raise ValueError(f"Invalid page count for {kind}: {pages}")
    for page in range(2, pages + 1):
        result = json.loads(fetch(gallery_url(kind, page)))
        if result.get("page") != page or result.get("cardCount") != first.get("cardCount"):
            raise ValueError(f"Gallery changed while paginating {kind}; retry snapshot")
        cards.extend(result.get("cards", []))
    if len(cards) != first.get("cardCount") or len({c["id"] for c in cards}) != len(cards):
        raise ValueError(f"Incomplete/duplicate {kind} pages")
    return {"cards": cards, "cardCount": len(cards), "pageCount": pages, "page": 1}


def extract_cards(source: Path) -> tuple[list[dict], dict]:
    enums = json.loads((source / "firestone_enums.json").read_text())
    tree = ET.fromstring(gzip.decompress((source / "CardDefs.Bacon.xml.gz").read_bytes()))
    if int(tree.get("build", -1)) != BUILD:
        raise ValueError(f"Client build {tree.get('build')} does not match audited build {BUILD}")
    result = []
    for entity in tree:
        raw = {int(t.get("enumID")): int(t.get("value")) for t in entity.findall("Tag")
               if t.get("value") is not None and t.get("type") in ("Int", "Card")}
        tags = {}
        for tag in entity.findall("Tag"):
            number = tag.get("enumID")
            if number is None or int(number) not in raw:
                continue
            # Some newly extracted XML names are literal numeric VALUES. Never
            # use these as keys or unrelated tags would overwrite one another.
            name = enums["game-tags"].get(number) or tag.get("name", "")
            if not name or name.isdecimal():
                name = "UNMAPPED_" + number
            tags[name] = raw[int(number)]
            xml_name = tag.get("name", "")
            if xml_name and not xml_name.isdecimal() and xml_name != name:
                tags[xml_name] = raw[int(number)]
        loc = {t.get("name"): t.findtext("enUS", "") for t in entity.findall("Tag")
               if t.get("type") == "LocString"}
        race_ids = {raw[200]} if raw.get(200) else set()
        for number, race_id in enums["secondary-race-tags"].items():
            if raw.get(int(number)):
                race_ids.add(race_id)
        races = sorted({enums["race"].get(str(r), f"UNKNOWN_RACE_{r}") for r in race_ids})
        premium = bool(raw.get(12) or raw.get(1471))
        # mechanics are booleans/numeric mechanic fields, not ReferencedTags.
        mechanics = sorted(k for k, v in tags.items() if v and k in {
            "TAUNT", "DIVINE_SHIELD", "WINDFURY", "STEALTH", "REBORN", "POISONOUS",
            "VENOMOUS", "DEATHRATTLE", "BATTLECRY", "MAGNETIC", "CHOOSE_ONE", "AURA",
            "AVENGE", "BACON_RALLY", "INTERACTABLE_OBJECT", "CANT_ATTACK", "IMMUNE",
            "IMMUNE_WHILE_ATTACKING", "DISCOVER", "BATTLECRY_REPEAT", "SPELLCRAFT",
            "BACON_SPELLCRAFT", "MEGA_WINDFURY", "CANT_BE_ATTACKED", "CANT_BE_DESTROYED",
        })
        card = {
            "id": entity.get("CardID"), "dbfId": int(entity.get("ID")),
            "name": loc.get("CARDNAME", ""), "text": loc.get("CARDTEXT", ""),
            "collectionText": loc.get("CARDTEXT", ""), "set": "Battlegrounds",
            "cardClass": CLASSES.get(raw.get(199), "NEUTRAL"),
            "playerClass": CLASSES.get(raw.get(199), "NEUTRAL"),
            "classes": [CLASSES.get(raw.get(199), "NEUTRAL")],
            "type": enums["card-type"].get(str(raw.get(202)), "UNKNOWN"),
            "races": races, "mechanics": mechanics, "tags": tags,
            "rawTags": {str(k): v for k, v in sorted(raw.items())},
            "referencedTags": sorted({enums["game-tags"].get(t.get("enumID"), t.get("name"))
                                      for t in entity.findall("ReferencedTag")}),
            "premium": premium, "isBaconPool": False, "isBaconPoolSpell": False,
            "battlegroundsHero": bool(raw.get(1491)),
            "battlegroundsHeroSkin": bool(tags.get("BACON_SKIN")),
            "duosOnly": bool(tags.get("IS_BACON_DUOS_EXCLUSIVE")),
            "snapshotBuild": BUILD, "referenceOnly": True,
            "otherTags": [],
        }
        for tag, field in [(47, "attack"), (45, "health"), (48, "cost"), (292, "armor"),
                           (1440, "techLevel"), (1429, "battlegroundsPremiumDbfId"),
                           (1471, "battlegroundsNormalDbfId"), (380, "heroPowerDbfId"),
                           (2, "scriptDataNum1"), (3, "scriptDataNum2")]:
            if tag in raw:
                card[field] = raw[tag]
        if races:
            card["race"] = races[0] if len(races) == 1 else enums["race"].get(str(raw.get(200)), races[0])
        if raw.get(1635):
            card["spellSchool"] = enums["spell-school"].get(str(raw[1635]), str(raw[1635]))
        related = set()
        for tag in entity.findall("Tag"):
            name = tag.get("name", "")
            if tag.get("type") == "Card" or any(s in name for s in ("CARD_ID", "MINION_ID", "COMPANION_ID", "SPELLCRAFT_ID")):
                if tag.get("value"):
                    related.add(int(tag.get("value")))
        card["relatedCardDbfIds"] = sorted(related)
        result.append(card)
    return sorted(result, key=lambda c: c["id"]), enums


def verify_sources(cards: list[dict], source: Path) -> tuple[dict, dict, list[dict]]:
    by_id = {c["id"]: c for c in cards}
    by_dbf = {c["dbfId"]: c for c in cards}
    active = {}
    components = {}
    discrepancies = []
    mapping = {"minion": "minion_ids", "hero": "hero_ids", "spell": "tavern_spell_ids",
               "darkGift": "dark_gift_ids"}
    flags = {"minion": 1456, "hero": 1491, "spell": 3081, "darkGift": 3567}
    gallery = {}
    for kind in KINDS:
        payload = json.loads((source / f"gallery_{kind}.json").read_text())
        entries = payload.get("cards", [])
        if not entries or len(entries) != payload.get("cardCount"):
            raise ValueError(f"Empty or truncated {kind} gallery snapshot")
        if len({c["id"] for c in entries}) != len(entries):
            raise ValueError(f"Duplicate dbf IDs in {kind} gallery")
        ids = set()
        for item in entries:
            if item["id"] not in by_dbf:
                raise ValueError(f"Unknown gallery dbf ID {item['id']}, wrong client snapshot")
            c = by_dbf[item["id"]]
            if item.get("battlegrounds", {}).get("duosOnly"):
                raise ValueError(f"Duos-only card leaked into Solo query: {c['id']}")
            ids.add(c["id"])
            for gf, cf in [("attack", "attack"), ("health", "health"), ("manaCost", "cost"), ("armor", "armor")]:
                if cf in c and gf in item and c[cf] != item[gf]:
                    raise ValueError(f"Gallery/client stats disagree: {c['id']} {cf} {c[cf]} != {item[gf]}")
            if item.get("battlegrounds", {}).get("tier") is not None and c.get("techLevel") != item["battlegrounds"]["tier"]:
                raise ValueError(f"Gallery/client tier disagree: {c['id']}")
            c["galleryDbfId"] = item["id"]
            c["subsetTribes"] = item.get("battlegrounds", {}).get("subsetTribes", [])
        gallery[kind] = ids
        if kind in flags:
            expected = {c["id"] for c in cards if c["rawTags"].get(str(flags[kind])) and not c["duosOnly"]}
            # Nightmare includes templates/enchants; only actual offered spells.
            if kind == "darkGift":
                expected = {i for i in expected if by_id[i]["type"] == "SPELL"}
            if ids != expected:
                raise ValueError(f"{kind} gallery/client pool mismatch: gallery-only={sorted(ids-expected)}, client-only={sorted(expected-ids)}")
            active[mapping[kind]] = sorted(ids)
            components[kind] = {"status": "verified", "count": len(ids), "evidence": "complete official Solo gallery equals client pool tags"}

    # Date-specific patch sentinels must all match, not simply filename/version.
    sentinels = {"BG36_210": (4, 4, 6), "BGS_018": (5, 7, 7),
                 "BG36_509": (3, 5, 6), "BG36_508": (4, 5, 3),
                 "BG32_846": (6, 6, 9), "BG36_202": (3, 2, 1)}
    for cid, expected in sentinels.items():
        actual = tuple(by_id[cid].get(k) for k in ("techLevel", "attack", "health"))
        if actual != expected:
            raise ValueError(f"Patch sentinel {cid}: {actual} != {expected}")
    for cid in KNOWN_MINION_BANS:
        if cid in active["minion_ids"]:
            raise ValueError(f"Currently banned minion {cid} appears in active pool")
    if by_id["BG28_845"]["cost"] != 2:
        raise ValueError("Natural Blessing patch sentinel is outdated")
    sanctify = [c["id"] for c in cards if c["name"] == "Sanctify"]
    if set(sanctify) & set(active["tavern_spell_ids"]):
        raise ValueError("Sanctify remains in pool despite 36.4.2 removal")

    trinket_ids = gallery["trinket"]
    # Resolve removed names against canonical client records, including school.
    removed_names = {"Blessing Portrait": 11, "Copper Coil": 11, "Cowrie Necklace": 11,
                     "Deathwhisper Sticker": 11, "Coral Spear": 12}
    removed = {c["id"] for c in cards if c["name"] in removed_names
               and c["rawTags"].get("1635") == removed_names[c["name"]]}
    for cid in sorted(trinket_ids & removed):
        discrepancies.append({"component": "trinket", "card_id": cid,
                              "reason": "official gallery/client still show card removed by latest official patch", "resolution": "excluded", "source": PATCH_URL})
    trinket_ids -= removed
    client_trinkets = {c["id"] for c in cards if c["type"] == "BATTLEGROUND_TRINKET"
                       and c["rawTags"].get("3407") and not c["duosOnly"]} - removed
    unexplained = client_trinkets ^ trinket_ids
    for cid in sorted(unexplained):
        discrepancies.append({"component": "trinket", "card_id": cid,
                              "reason": "client/gallery eligibility disagreement", "resolution": "unresolved; component unavailable to trainer"})
    candidate = {
        "lesser_trinket_ids": sorted(i for i in trinket_ids if by_id[i]["rawTags"].get("1635") == 11),
        "greater_trinket_ids": sorted(i for i in trinket_ids if by_id[i]["rawTags"].get("1635") == 12),
        "unresolved_trinket_ids": sorted(unexplained),
    }
    if len(candidate["lesser_trinket_ids"]) + len(candidate["greater_trinket_ids"]) != len(trinket_ids):
        raise ValueError("Unknown trinket school; cannot classify Lesser/Greater")
    components["trinket"] = {"status": "blocked" if unexplained else "verified",
                             "candidate_count": len(trinket_ids), "unresolved_ids": sorted(unexplained)}
    active["lesser_trinket_ids"] = [] if unexplained else candidate["lesser_trinket_ids"]
    active["greater_trinket_ids"] = [] if unexplained else candidate["greater_trinket_ids"]
    active["shop_minion_ids"] = [i for i in active["minion_ids"] if 1 <= by_id[i].get("techLevel", 0) <= 6]
    active["tier7_minion_ids"] = [i for i in active["minion_ids"] if by_id[i].get("techLevel") == 7]
    active["hero_power_ids"] = sorted({by_dbf[by_id[i]["heroPowerDbfId"]]["id"] for i in active["hero_ids"]})
    active_ids = set().union(*(set(v) for v in active.values()))
    for c in cards:
        c["isBaconPool"] = c["id"] in active["minion_ids"]
        c["isBaconPoolSpell"] = c["id"] in active["tavern_spell_ids"]
        c["referenceOnly"] = c["id"] not in active_ids
    # Keep dormant entities solely as effect definitions. Never derive a draw
    # pool from reference presence, premium, text, race or set membership.
    return {"active": active, "candidate": candidate, "components": components}, by_id, discrepancies


def check(data_dir: Path) -> dict:
    manifest = json.loads((data_dir / "ruleset.json").read_text())
    for name, expected in manifest["checksums"].items():
        p = (data_dir / name).resolve()
        if not p.is_relative_to(data_dir.resolve()) or sha256(p.read_bytes()) != expected:
            raise ValueError(f"Snapshot checksum failed: {name}")
    cards = json.loads((data_dir / manifest["reference_cards_file"]).read_text())
    by_id = {c["id"]: c for c in cards}
    if len(by_id) != len(cards):
        raise ValueError("Duplicate CardIDs")
    for kind, ids in manifest["active"].items():
        if len(ids) != len(set(ids)) or not set(ids) <= by_id.keys():
            raise ValueError(f"Invalid active {kind}")
    if {c["id"] for c in cards if c["isBaconPool"]} != set(manifest["active"]["minion_ids"]):
        raise ValueError("Minion pool flag/whitelist disagreement")
    for cid in manifest["active"]["minion_ids"]:
        c = by_id[cid]
        # Aureate Laureate is a verified shop minion that is always Golden.
        # Actual upgraded variants point back to a normal DbF ID (tag 1471).
        if c.get("battlegroundsNormalDbfId") or c["duosOnly"] or cid in KNOWN_MINION_BANS:
            raise ValueError(f"Invalid pool entity {cid}")
        if c.get("battlegroundsPremiumDbfId") not in {r["dbfId"] for r in cards}:
            raise ValueError(f"Missing golden reference {cid}")
    return {"ruleset_id": manifest["ruleset_id"], "integrity": "verified",
            "combat_pool_verified": manifest["combat_pool_verified"],
            "counts": {k: len(v) for k, v in manifest["active"].items()},
            "fullgame_supported": False, "blockers": manifest["blockers"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="rebuild from frozen evidence without changing the observation date")
    parser.add_argument("--check", action="store_true", help="verify checksums and eligibility without network or writes")
    parser.add_argument("--refresh", action="store_true", help="download into an isolated candidate requiring patch/hotfix review; never change the frozen trainer snapshot")
    parser.add_argument("--output", type=Path, help="output directory; --refresh cannot overwrite the frozen data directory")
    args = parser.parse_args()
    if sum((args.offline, args.check, args.refresh)) > 1:
        parser.error("Use only one of --offline, --check or --refresh")
    candidate_time = dt.datetime.now(dt.timezone.utc)
    default_dir = ROOT / "data"
    if args.refresh:
        default_dir = ROOT / "data" / "candidates" / candidate_time.strftime("%Y%m%dT%H%M%SZ")
    data_dir = (args.output or default_dir).resolve()
    if args.refresh and data_dir == (ROOT / "data").resolve():
        parser.error("--refresh writes a review candidate; it cannot overwrite the frozen training snapshot")
    source = data_dir / "source"
    if args.check or not (args.offline or args.refresh):
        print(json.dumps(check(data_dir), indent=2))
        return
    if args.refresh:
        # A new client requires explicit review of patch-specific assertions.
        latest = fetch("https://raw.githubusercontent.com/HearthSim/hsdata/master/README.md").decode()
        version = re.search(r"Version:\s*([\d.]+)", latest)
        if not version or version.group(1) != f"{PATCH}.{BUILD}":
            dump(data_dir / "ruleset.json", {"status": "blocked", "snapshot_verified": False,
                 "combat_pool_verified": False, "observed_at": candidate_time.isoformat(),
                 "candidate_live_version": version.group(1) if version else None,
                 "blockers": ["Live version changed; update pinned commit and review patch/hotfix rules before training."]}, pretty=True)
            raise ValueError("Live game version changed: audit the new ruleset before training")
        source.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "data/source/firestone_enums.json", source / "firestone_enums.json")
        xml = fetch(XML_URL)
        (source / "CardDefs.Bacon.xml.gz").write_bytes(gzip.compress(xml, mtime=0))
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            results = list(pool.map(download_gallery, KINDS))
        for kind, result in zip(KINDS, results):
            dump(source / f"gallery_{kind}.json", result)
    cards, _ = extract_cards(source)
    pools, _, discrepancies = verify_sources(cards, source)
    dump(data_dir / "reference_cards.json", cards)
    dump(data_dir / "source_discrepancies.json", discrepancies, pretty=True)
    by_id = {c["id"]: c for c in cards}
    gift_rules = {
        "status": "partial", "build": BUILD,
        "window_source": "Client numeric tags 4853 and 4854, cross-checked with Blizzard's 36.2.2 update; missing maximum means unbounded.",
        "button": {"unlock_turn": 3, "gold_cost": 3, "max_uses_per_turn": 1, "max_uses_per_game": 3, "options": 3},
        "minion_tiers_by_turn": {"3": [2], "4": [2, 3], "5": [3], "6": [3, 4], "7": [4], "8": [4, 5], "9": [4, 5, 6], "10": [5, 6]},
        "minion_tiers_after_turn10": {"value": [5, 6], "status": "inferred; verify against client logs before full-game training"},
        "offerings": [{"id": i, "name": by_id[i]["name"], "min_turn": by_id[i]["rawTags"].get("4853"), "max_turn": by_id[i]["rawTags"].get("4854")} for i in pools["active"]["dark_gift_ids"]],
        "unresolved": ["Joint minion/gift sampling weights, including rare gifts and minion preferences.", "All cumulative eligibility and hero-exception restrictions require an executable audited implementation."]
    }
    dump(data_dir / "dark_gift_offering_rules.json", gift_rules, pretty=True)
    blockers = [
        "Trinket pool has an unresolved client/gallery disagreement; candidate trinkets are excluded from active lists.",
        "Dark Gift joint-offering probabilities and all selection restrictions are not fully specified by public card metadata.",
        "Global seasonal systems being inactive does not disable hero-specific generation of Buddies/Quests.",
        "Data verification does not establish full recruit-phase or full-game simulator support.",
    ]
    manifest = {
        "schema_version": 1, "ruleset_id": RULESET_ID, "patch": PATCH, "build": BUILD,
        "season": 14, "season_name": "Dark Gifts of Dalaran", "mode": "solo",
        "status": "blocked" if pools["components"]["trinket"]["status"] == "blocked" else "verified",
        "snapshot_verified": pools["components"]["trinket"]["status"] == "verified",
        "combat_pool_verified": True, "fullgame_supported": False,
        "verified_at": DATE + "T00:00:00Z", "observed_date": DATE,
        "date_semantics": "Day of source retrieval; midnight is a date marker, not a claimed retrieval second.",
        "source_client_commit": HSDATA_COMMIT, "reference_cards_file": "reference_cards.json",
        "mechanics": MECHANICS, **pools, "blockers": blockers,
        "known_banned_minion_ids": list(KNOWN_MINION_BANS),
        "tier7_policy": "Special acquisition only; ordinary shop/refresh uses shop_minion_ids, not all minion_ids.",
        "reference_policy": "Complete pinned Battlegrounds client definitions preserve tokens/goldens and hero-generated exceptions. Reference-only cards cannot be sampled, bought or offered without an explicitly audited generation rule.",
        "source_evidence": [
            {"kind": "blizzard_patch", "url": PATCH_URL, "patch": PATCH, "published_date": "2026-09-03"},
            {"kind": "blizzard_current_bans", "url": BANS_URL, "observed_date": DATE, "banned_minion_ids": list(KNOWN_MINION_BANS)},
            {"kind": "client_data", "url": XML_URL, "commit": HSDATA_COMMIT, "build": BUILD},
            *[{"kind": "blizzard_solo_gallery", "category": k, "url": gallery_url(k), "file": f"source/gallery_{k}.json"} for k in KINDS],
            {"kind": "season", "url": "https://hearthstone.blizzard.com/en-us/news/24290432"},
            {"kind": "dark_gift_offering_rules", "url": "https://us.forums.blizzard.com/en/hearthstone/t/battlegrounds-developer-insight-dark-gifts/163606"},
        ],
    }
    if args.refresh:
        manifest.update(status="blocked", snapshot_verified=False, combat_pool_verified=False,
                        observed_date=candidate_time.date().isoformat(), observed_at=candidate_time.isoformat(),
                        verified_at=None, candidate_requires_human_or_agent_review=True)
        manifest["blockers"].insert(0, "Refresh candidate: review current official patch notes and bans before promoting eligibility; not approved for training.")
    frozen_files = [data_dir / "reference_cards.json", data_dir / "source_discrepancies.json",
                    data_dir / "dark_gift_offering_rules.json", *sorted(source.iterdir())]
    manifest["checksums"] = {str(p.relative_to(data_dir)): sha256(p.read_bytes())
                             for p in frozen_files if p.is_file()}
    dump(data_dir / "ruleset.json", manifest, pretty=True)
    print(json.dumps(check(data_dir), indent=2))


if __name__ == "__main__":
    main()
