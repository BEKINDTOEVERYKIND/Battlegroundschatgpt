"""Recruit-engine contract and fail-closed audit of an external HSBRSIM checkout.

The upstream engine is never imported into the training process. Inspection runs
in a separate Python process and produces a coverage report, not an assertion of
rules accuracy. A registered script is evidence of code presence only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

HSBRSIM_REVISION = "4e0525a198352557ef816bd3a53566d6ffd4c164"
HSBRSIM_URL = "https://github.com/Gallo13th/HSBRSIM.git"


@dataclass(frozen=True)
class RecruitAction:
    """An action selected from the engine's actual legal action list.

    Entity identifiers refer to this observation, not absolute card-table rows.
    Targets and choice indices must be enumerated by the engine, never guessed
    from English card text. The payload also supports seasonal actions.
    """

    kind: str
    entity_id: str | None = None
    target_id: str | None = None
    position: int | None = None
    choice_index: int | None = None


@dataclass(frozen=True)
class RecruitObservation:
    """Player-visible information only; no opposing shops, hands, or RNG state."""

    player_id: int
    turn: int
    public_state: Mapping[str, Any]
    private_state: Mapping[str, Any]
    legal_actions: Sequence[RecruitAction]
    ruleset_sha256: str
    # Actual remaining recruit window for this player, supplied by the adapter.
    # Missing timing must block a timed rollout; do not guess from round number.
    time_remaining_ms: int | None = None
    action_budget: Mapping[str, Any] | None = None


class RecruitEngine(Protocol):
    """Contract for a future validated eight-player Solo engine integration.

    All pending choices must be surfaced before the next normal action. Combat
    must return persistent recruitment effects, not just damage/win estimates.
    """

    def coverage(self) -> Mapping[str, Any]: ...

    def reset(self, *, seed: int) -> RecruitObservation: ...

    def step(self, action: RecruitAction) -> RecruitObservation | None: ...

    def expire_turn(self) -> RecruitObservation | None:
        """End this recruit turn, resolving pending choices as the client does.

        Automatic resolution creates no extra policy actions. The returned
        observation must belong to a different player/turn, or be terminal.
        """
        ...

    def final_placements(self) -> Mapping[int, float]: ...


class RecruitCoverageError(RuntimeError):
    """Full-game training is not authorized by the engine's conformance report."""


def ruleset_digest(ruleset: Mapping[str, Any]) -> str:
    encoded = json.dumps(ruleset, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def require_full_game_ready(report: Mapping[str, Any], ruleset: Mapping[str, Any]) -> None:
    """Reject stale reports, missing handlers, deferred effects and untested rules."""
    if report.get("ruleset_sha256") != ruleset_digest(ruleset):
        raise RecruitCoverageError("Coverage report belongs to a different ruleset")
    blockers = report.get("blockers")
    if not isinstance(blockers, list) or blockers or report.get("full_game_ready") is not True:
        raise RecruitCoverageError("Full-game engine is not ready: " + "; ".join(
            str(x) for x in (blockers or ["missing successful conformance report"])))
    if report.get("conformance_validated") is not True:
        raise RecruitCoverageError("Registered handlers are not proof of rules conformance")


# These effects are implemented by keyword tags or explicit game.py paths in
# the pinned revision. This list is NOT automatically carried to other engines.
_CORE_MINIONS = frozenset({
    "BG_BOT_911", "BGS_119", "BGS_131", "BG26_175", "BG_DEEP_015", "BG25_001",
})

_INSPECT_SCRIPT = r'''
import json, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
from hsrl2.db import CardDB
from hsrl2.scripts import REGISTRY
from hsrl2 import darkgifts as DG
from hsrl2 import game
db = CardDB.load(root / "data")
fields = ("id", "dbf_id", "name", "text", "atk", "health", "cost", "armor",
          "tech_level", "hero_power_id", "triple_upgrade_id", "triple_base_id",
          "script_data_num_1", "script_data_num_2", "script_data_num_3",
          "script_data_num_4", "is_pool_minion", "is_pool_spell")
cards = {cid: {key: getattr(d, key) for key in fields}
         for cid, d in db._by_id.items()}
dispatch = getattr(game, "_DARK_GIFT_EFFECTS", {})
print(json.dumps({
    "cards": cards,
    "script_ids": sorted(REGISTRY),
    "dark_gift_handler_ids": sorted(dispatch),
    "deferred_dark_gift_ids": sorted(DG.DEFERRED_GIFTS),
    "trinket_api_present": all(hasattr(game.Game, x)
                               for x in ("offer_trinkets", "buy_trinket")),
    "upstream_patch": json.loads((root / "data/bg_summary.json").read_text())["patch"],
}))
'''


def inspect_hsbrsim(engine_root: Path) -> dict[str, Any]:
    """Read an explicit local checkout. Does not download, install or mutate it."""
    engine_root = engine_root.resolve()
    if not (engine_root / "hsrl2/game.py").is_file():
        raise ValueError(f"No HSBRSIM hsrl2 engine at {engine_root}")
    revision = subprocess.check_output(
        ["git", "-C", str(engine_root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(engine_root), "status", "--porcelain", "--untracked-files=all"],
        text=True).strip()
    proc = subprocess.run(
        [sys.executable, "-I", "-c", _INSPECT_SCRIPT, str(engine_root)],
        capture_output=True, text=True, check=False, timeout=120)
    if proc.returncode:
        raise RuntimeError(f"HSBRSIM inspection failed: {proc.stderr[-4000:]}")
    inventory = json.loads(proc.stdout)
    inventory.update(revision=revision, dirty_checkout=bool(dirty))
    return inventory


def _card_index(cards: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result = {}
    for card in cards:
        cid = card.get("id") or card.get("cardId")
        if not isinstance(cid, str):
            raise ValueError("Reference cards must contain canonical string card IDs")
        if cid in result:
            raise ValueError(f"Duplicate card ID: {cid}")
        result[cid] = card
    return result


def assess_hsbrsim(ruleset: Mapping[str, Any], cards: Sequence[Mapping[str, Any]],
                    inventory: Mapping[str, Any]) -> dict[str, Any]:
    """Compare current explicit pools to upstream without silently reducing pools."""
    active = ruleset.get("active")
    if not isinstance(active, dict):
        raise ValueError("Ruleset must contain an explicit active pool mapping")
    current = _card_index(cards)
    by_dbf = {card.get("dbfId"): cid for cid, card in current.items()
              if card.get("dbfId") is not None}
    engine_cards = inventory["cards"]
    scripted = set(inventory["script_ids"])
    deferred = set(inventory["deferred_dark_gift_ids"])
    gifts = set(inventory["dark_gift_handler_ids"]) - deferred
    categories = {"minions": "minion_ids", "heroes": "hero_ids",
                  "spells": "tavern_spell_ids", "dark_gifts": "dark_gift_ids"}
    pools = {category: active.get(key) for category, key in categories.items()}
    trinkets = active.get("lesser_trinket_ids", []) + active.get("greater_trinket_ids", [])
    trinket_status = ruleset.get("components", {}).get("trinket", {}).get("status")
    candidate = ruleset.get("candidate", {})
    # Candidate trinkets are inspected to measure work remaining. They never
    # become an active pool and cannot be fed to a trainer by this path.
    if trinket_status == "blocked":
        pools["candidate_trinkets"] = (candidate.get("lesser_trinket_ids", []) +
                                         candidate.get("greater_trinket_ids", []))
    else:
        pools["trinkets"] = trinkets
    report: dict[str, Any] = {
        "schema_version": 1, "engine": "HSBRSIM/hsrl2", "engine_url": HSBRSIM_URL,
        "engine_revision": inventory["revision"], "upstream_patch": inventory["upstream_patch"],
        "current_patch": ruleset.get("patch"), "current_build": ruleset.get("build"),
        "ruleset_sha256": ruleset_digest(ruleset), "full_game_ready": False,
        "ruleset_digest_format": "canonical_json_utf8_sort_keys_no_extra_whitespace",
        "conformance_validated": False, "categories": {}, "blockers": [],
        "turn_timing": {
            "actual_player_window_validated": False,
            "pending_choice_timeout_validated": False,
            "required_wrapper": "bg_ai.timed_recruiting.TimedRecruitEngine",
        },
        "scope": "Static presence and numeric metadata audit; not behavioral validation",
    }
    blockers = report["blockers"]
    if trinket_status == "blocked":
        blockers.append("ruleset_component_blocked:trinket")
    if inventory["revision"] != HSBRSIM_REVISION:
        blockers.append("unreviewed_engine_revision")
    if inventory.get("dirty_checkout"):
        blockers.append("dirty_engine_checkout")
    patch = str(ruleset.get("patch", ruleset.get("build", "")))
    if patch != str(inventory["upstream_patch"]):
        blockers.append("patch_mismatch")
    for category, ids in pools.items():
        if not isinstance(ids, list) or (not ids and category != "candidate_trinkets"):
            raise ValueError(f"Ruleset requires a nonempty {category} pool list")
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate IDs in {category}")
        missing_defs, missing_handlers, deltas = [], [], []
        for cid in ids:
            if cid not in current:
                raise ValueError(f"Current reference card missing: {cid}")
            card = current[cid]
            old = engine_cards.get(cid)
            row = {"id": cid, "name": card.get("name", cid)}
            if old is None:
                missing_defs.append(row)
                continue
            handler = cid
            if category == "heroes":
                handler = card.get("heroPowerCardId", card.get("heroPowerDbfId"))
                if isinstance(handler, int):
                    handler = by_dbf.get(handler)
                if not handler:
                    # Missing current linkage is a blocker, not a license to use
                    # the previous rotation's hero power implicitly.
                    handler = None
            has_handler = handler in scripted
            if category == "minions" and inventory["revision"] == HSBRSIM_REVISION:
                has_handler = has_handler or cid in _CORE_MINIONS
            if category == "dark_gifts":
                has_handler = cid in gifts
            if not has_handler:
                missing_handlers.append(dict(row, handler_id=handler,
                                             deferred=cid in deferred))
            # Minion mana cost is NOT its 3-gold Battlegrounds purchase cost.
            fields = [("attack", "atk"), ("health", "health"), ("techLevel", "tech_level")]
            if category in ("spells", "trinkets", "candidate_trinkets"):
                fields += [("cost", "cost")]
            if category == "heroes":
                fields += [("armor", "armor")]
            changes = []
            for newkey, oldkey in fields:
                if newkey in card and card[newkey] is not None and card[newkey] != old.get(oldkey):
                    changes.append({"field": newkey, "upstream": old.get(oldkey), "current": card[newkey]})
            if changes:
                deltas.append(dict(row, changes=changes))
        report["categories"][category] = {
            "required": len(ids), "missing_definitions": missing_defs,
            "missing_handlers": missing_handlers, "numeric_differences": deltas,
            "pool_status": "unverified_candidate" if category == "candidate_trinkets" else "active",
        }
        if missing_defs:
            blockers.append(f"{category}:missing_definitions:{len(missing_defs)}")
        if missing_handlers:
            blockers.append(f"{category}:missing_handlers:{len(missing_handlers)}")
        if deltas:
            blockers.append(f"{category}:numeric_differences:{len(deltas)}")
    if not inventory.get("trinket_api_present"):
        blockers.append("missing_trinket_offer_and_purchase_system")
    # Audited source contains a guessed rare-gift weight, omitted mechanic
    # validation and no test corpus from the current client. Even zero presence
    # gaps is insufficient to permit full-game learning.
    blockers.extend([
        "dark_gift_offering_probabilities_unverified",
        "generated_cards_and_golden_dependency_closure_unvalidated",
        "current_patch_behavioral_conformance_unvalidated",
        "actual_player_recruit_window_unvalidated",
        "pending_choice_timeout_behavior_unvalidated",
        "timed_recruit_adapter_not_integrated",
    ])
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-root", required=True, type=Path)
    parser.add_argument("--ruleset", type=Path, default=Path("data/ruleset.json"))
    parser.add_argument("--cards", type=Path, default=Path("data/reference_cards.json"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        report = assess_hsbrsim(json.loads(args.ruleset.read_text()),
                                json.loads(args.cards.read_text()),
                                inspect_hsbrsim(args.engine_root))
        report["ruleset_file_sha256"] = hashlib.sha256(args.ruleset.read_bytes()).hexdigest()
        report["reference_cards_sha256"] = hashlib.sha256(args.cards.read_bytes()).hexdigest()
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"Recruit audit failed closed: {exc}", file=sys.stderr)
        return 2
    body = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(body)
    print(json.dumps({"full_game_ready": report["full_game_ready"],
                      "blockers": report["blockers"]}, indent=2))
    return 0 if report["full_game_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
