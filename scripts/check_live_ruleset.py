#!/usr/bin/env python3
"""Read-only live-source preflight for a frozen Battlegrounds training snapshot.

No baseline is updated automatically. Network failure or changed semantic data
means training cannot be described as current. --allow-historical is explicit
offline reproduction, and its report always says current=false.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import hashlib
import html
import json
from pathlib import Path
import re
import sys
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
GALLERY_KINDS = ("minion", "hero", "spell", "darkGift")
SEMANTIC_VERSION = "gallery-semantic-v1"
SCALARS = ("id", "name", "classId", "minionTypeId", "cardTypeId", "cardSetId",
           "spellSchoolId", "manaCost", "attack", "health", "armor", "parentId")
LISTS = ("multiClassIds", "multiTypeIds", "childIds", "keywordIds")
BG_SCALARS = ("tier", "hero", "quest", "reward", "duosOnly", "solosOnly",
              "upgradeId", "heroPowerId", "companionId")


class LiveRulesetError(RuntimeError):
    """A fail-closed preflight error with a serializable evidence report."""
    def __init__(self, report: dict):
        self.report = report
        super().__init__("Current ruleset preflight failed: " + "; ".join(report["failures"]))


def digest(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def json_digest(value: object) -> str:
    return digest(json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":")).encode("utf-8"))


def canonical_card(card: dict) -> dict:
    """Only gameplay/identity fields; avoid changing CDN images and presentation."""
    result = {key: card.get(key) for key in SCALARS}
    for key in LISTS:
        result[key] = sorted(card.get(key) or [])
    text = html.unescape(re.sub(r"<[^>]*>", " ", card.get("text") or ""))
    result["text"] = " ".join(text.split())
    bg = card.get("battlegrounds") or {}
    result["battlegrounds"] = {key: bg.get(key) for key in BG_SCALARS}
    result["battlegrounds"]["subsetTribes"] = sorted(bg.get("subsetTribes") or [])
    return result


def gallery_observation(payload: dict) -> dict:
    cards = payload.get("cards")
    if not isinstance(cards, list) or not cards:
        raise ValueError("Missing or empty card list")
    ids = [c.get("id") for c in cards]
    if any(not isinstance(i, int) for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Missing or duplicate DBF IDs")
    if payload.get("cardCount") != len(cards):
        raise ValueError("Incomplete gallery pagination")
    if any(c.get("battlegrounds", {}).get("duosOnly") for c in cards):
        raise ValueError("Solo source contains a Duos-only card")
    semantic = sorted((canonical_card(c) for c in cards), key=lambda c: c["id"])
    return {"card_count": len(cards), "semantic_sha256": json_digest(semantic),
            "semantic_version": SEMANTIC_VERSION}


def known_issues_observation(payload: dict) -> dict:
    """Read only the maintained first post, regardless of comment activity."""
    posts = payload.get("post_stream", {}).get("posts", [])
    first = next((p for p in posts if p.get("post_number") == 1), None)
    if first is None or not first.get("cooked"):
        raise ValueError("Maintained first official post is missing")
    return {
        "topic_id": payload.get("id"), "topic_title": payload.get("title"),
        "post_id": first.get("id"), "post_number": first["post_number"],
        "username": first.get("username"), "updated_at": first.get("updated_at"),
        "cooked_sha256": digest(first["cooked"].encode("utf-8")),
    }


def fetch(url: str, timeout: float = 40) -> bytes:
    request = urllib.request.Request(url, headers={
        "User-Agent": "Battlegroundschatgpt/0.1 (read-only current-ruleset check)",
        "Accept": "application/json,text/plain,text/html",
    })
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def fetch_gallery(url: str, timeout: float) -> dict:
    first = json.loads(fetch(url, timeout))
    pages = first.get("pageCount")
    if not isinstance(pages, int) or not 1 <= pages <= 25 or first.get("page") != 1:
        raise ValueError("Invalid gallery pagination response")
    cards = list(first.get("cards", []))
    for page in range(2, pages + 1):
        parts = urllib.parse.urlsplit(url)
        query = dict(urllib.parse.parse_qsl(parts.query))
        query["page"] = str(page)
        next_url = urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(query)))
        item = json.loads(fetch(next_url, timeout))
        if item.get("page") != page or item.get("cardCount") != first.get("cardCount"):
            raise ValueError("Gallery changed during pagination")
        cards.extend(item.get("cards", []))
    return gallery_observation({"cards": cards, "cardCount": first.get("cardCount")})


def read_live_source(name: str, settings: dict, timeout: float) -> dict:
    url = settings["url"]
    if name == "client_version":
        readme = fetch(url, timeout).decode("utf-8")
        match = re.search(r"Version:\s*([\d.]+)", readme)
        if not match:
            raise ValueError("Client source has no declared version")
        return {"version": match.group(1)}
    if name == "known_issues":
        return known_issues_observation(json.loads(fetch(url, timeout)))
    if name.startswith("gallery_"):
        return fetch_gallery(url, timeout)
    raise ValueError(f"Unrecognized live source: {name}")


def compare_observation(expected: dict, actual: dict) -> list[str]:
    return [f"{key}: expected {value!r}, observed {actual.get(key)!r}"
            for key, value in expected.items() if actual.get(key) != value]


def check_live_ruleset(root: Path = ROOT, *, allow_historical: bool = False,
                       timeout: float = 40, guard_path: Path | None = None) -> dict:
    """Return an auditable report; raise LiveRulesetError on any live uncertainty."""
    started = time.monotonic()
    root = Path(root).resolve()
    if allow_historical:
        return {"schema_version": 1, "status": "historical", "current": False,
                "checked_at": None, "network_checked": False, "sources": {},
                "warning": "Live freshness was explicitly skipped for historical/offline reproduction; do not claim this run used the current game."}
    report = {"schema_version": 1, "status": "blocked", "current": False,
              "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "network_checked": True, "sources": {}, "failures": []}
    try:
        baseline_path = guard_path or root / "config/live-source-guard.json"
        baseline_bytes = baseline_path.read_bytes()
        baseline = json.loads(baseline_bytes)
        report["guard_sha256"] = digest(baseline_bytes)
        report["baseline_reviewed_at"] = baseline["reviewed_at"]
        report["ruleset_id"] = baseline["ruleset_id"]
        if baseline.get("semantic_version") != SEMANTIC_VERSION:
            raise ValueError("Guard semantic normalization version mismatch")
        required = {"client_version", "known_issues", *(f"gallery_{k}" for k in GALLERY_KINDS)}
        if set(baseline["sources"]) != required:
            raise ValueError("Guard baseline does not contain all required live sources")
        for filename, expected in baseline["snapshot_sha256"].items():
            path = (root / filename).resolve()
            if not path.is_relative_to(root) or digest(path.read_bytes()) != expected:
                raise ValueError(f"Frozen snapshot differs from reviewed live guard: {filename}")
        # Baseline card semantics must still equal the frozen input evidence.
        for kind in GALLERY_KINDS:
            frozen = json.loads((root / f"data/source/gallery_{kind}.json").read_text())
            actual = gallery_observation(frozen)
            if compare_observation(baseline["sources"][f"gallery_{kind}"]["expected"], actual):
                raise ValueError(f"Baseline does not match frozen gallery evidence: {kind}")
        with ThreadPoolExecutor(max_workers=len(required)) as pool:
            futures = {pool.submit(read_live_source, name, settings, timeout): name
                       for name, settings in baseline["sources"].items()}
            for future in as_completed(futures):
                name = futures[future]
                settings = baseline["sources"][name]
                item = {"url": settings["url"], "expected": settings["expected"]}
                try:
                    actual = future.result()
                    differences = compare_observation(settings["expected"], actual)
                    item.update(observed=actual, status="changed" if differences else "matched")
                    if differences:
                        item["differences"] = differences
                        report["failures"].append(f"{name} changed; review live patch/hotfix data before training")
                except Exception as exc:
                    item.update(status="unavailable", error=f"{type(exc).__name__}: {exc}")
                    report["failures"].append(f"{name} unavailable; current eligibility cannot be established")
                report["sources"][name] = item
    except Exception as exc:
        report["failures"].append(f"Guard integrity error: {type(exc).__name__}: {exc}")
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    if report["failures"]:
        raise LiveRulesetError(report)
    report.update(status="current_sources_match", current=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-historical", "--offline", action="store_true", dest="historical")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--timeout", type=float, default=40)
    parser.add_argument("--report", type=Path, help="optional evidence report file; frozen inputs are never changed")
    args = parser.parse_args()
    try:
        report = check_live_ruleset(args.root, allow_historical=args.historical, timeout=args.timeout)
        code = 0
    except LiveRulesetError as exc:
        report = exc.report
        code = 1
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(serialized)
    print(serialized, end="")
    return code


if __name__ == "__main__":
    sys.exit(main())
