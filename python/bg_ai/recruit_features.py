"""Rotation-stable features for finite-budget, visible-state recruit decisions.

Only observation fields and the candidate legal action are encoded. Neither card
IDs, candidate rollout boards, future shop draws, nor engine RNG enter inputs.
The action entity IDs are used solely to locate public/owned card attributes.
"""
from __future__ import annotations

from dataclasses import asdict
import math
from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from bg_ai.features import CARD_FEATURE_NAMES, encode_card
from bg_ai.recruiting import RecruitAction, RecruitObservation
from bg_ai.turn_budget import TimingProfile

RECRUIT_FEATURE_VERSION = "recruit-visible-action-v3-affordability"
COUNTER_NAMES = ("GOLD_SPENT_THIS_TURN", "CARDS_PLAYED_THIS_TURN",
                 "TAVERN_SPELLS_CAST_THIS_GAME", "TAVERN_SPELL_EXTRA_ATK",
                 "TAVERN_SPELL_EXTRA_HEALTH", "HERO_POWER_USED_THIS_TURN")
ACTION_KINDS = ("buy", "play", "sell", "refresh", "freeze", "upgrade", "hero_power",
                "cast_spell", "choose", "buy_trinket", "activate", "move", "end_turn")
CONTEXT_NAMES = (
    "turn", "gold", "tavern_tier", "upgrade_cost", "health", "armor", "frozen",
    "board_count", "hand_count", "shop_count", "remaining_ms", "remaining_actions",
    "actions_used", "action_limit", "initial_available_ms", "reserve_ms", "aps",
    "action_position", "has_target", "action_card_cost", "target_card_cost",
    "action_is_spell", "target_is_spell", "action_legal", "buy_room", "play_room",
    "source_board", "source_hand", "source_shop", "target_board", "target_hand", "target_shop",
    "choice_index", "choice_number", "source_activate_used", "target_activate_used",
    "timing_costs_known", "action_cost_ms", "remaining_ms_after_action",
    "remaining_actions_after_action", "minimum_followup_ms", "immediate_card_completion_fits",
) + tuple("counter_" + key.lower() for key in COUNTER_NAMES) + tuple("board_slot%d_activate_used" % i for i in range(7))
CARD_VIEWS = ("board_mean", "board_max", "hand_mean", "shop_mean", "shop_max", "action_card", "target_card", "choice_option") + tuple("board_slot%d" % i for i in range(7))
RECRUIT_FEATURE_NAMES = tuple("recruit.context." + x for x in CONTEXT_NAMES) + tuple(
    "recruit.action." + k for k in ACTION_KINDS
) + tuple("recruit." + view + "." + name for view in CARD_VIEWS for name in CARD_FEATURE_NAMES)


def _number(value: Any, key: str, default: float = 0) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{key} must be a finite number")
    return float(value)


@lru_cache(maxsize=32768)
def _cached_card(encoded: str) -> np.ndarray:
    return encode_card(json.loads(encoded))


@lru_cache(maxsize=1)
def _default_profile():
    return TimingProfile.load(Path(__file__).resolve().parents[2] / "config/turn-budget.json")


def _costs(budget):
    costs = budget.get("action_cost_ms")
    if costs is None:
        profile = _default_profile()
        if profile.fingerprint == budget.get("profile_sha256"):
            return {kind: profile.action_ms(kind) for kind in profile.extra_ms}
        return {}
    if not isinstance(costs, Mapping) or any(type(v) is not int or v <= 0 for v in costs.values()):
        raise ValueError("Timing costs must be positive integer milliseconds")
    return costs


def _completion_costs(action, card, costs):
    if action.kind == "end_turn":
        return 0, 0, 0, bool(costs)
    kinds = []
    text = re.sub(r"<[^>]+>", " ", str(card.get("text", ""))) if card else ""
    is_minion = bool(card and str(card.get("card_type", card.get("type", ""))).lower() == "minion")
    if action.kind == "buy":
        kinds.append("play" if is_minion else "cast_spell")
    if card and action.kind in ("buy", "play", "cast_spell", "activate"):
        target_text = re.search(r"give (?:a|another) minion", text, re.I)
        needs_choice = (re.search(r"choose one", text, re.I) or
                        (target_text and re.search(r"battlecry", text, re.I))) if is_minion and action.kind in ("buy", "play") else target_text
        if needs_choice:
            kinds.append("choose")
    known = action.kind in costs and all(kind in costs for kind in kinds)
    return costs.get(action.kind, 0), sum(costs.get(kind, 0) for kind in kinds), len(kinds), known


def card_features(card: Mapping[str, Any] | None) -> np.ndarray:
    """Encode spells with zero minion stats, retaining their public effect text."""
    if card is None:
        return encode_card(None)
    item = dict(card)
    item.setdefault("attack", item.get("atk", 0))
    item.setdefault("health", 0)
    item.setdefault("tavernTier", item.get("tech_level", item.get("techLevel", item.get("tier", 0))))
    item.setdefault("text", "")
    if "race" in item and not any(k in item for k in ("races", "tribes", "tribe")):
        item["races"] = [item["race"]]
    # Cache only semantic fields; identifiers and names never enter the key or weights.
    keys = ("attack", "health", "tavernTier", "tier", "golden", "isGolden", "text", "races", "tribes", "tribe",
            "mechanics", "keywords", "taunt", "divine_shield", "divineShield", "windfury", "megaWindfury",
            "reborn", "venomous", "poisonous", "cleave", "deathrattle", "avenge", "frenzy")
    semantic = {key: item[key] for key in keys if key in item}
    return _cached_card(json.dumps(semantic, sort_keys=True, separators=(",", ":")))


def _cards(value: Any, field: str) -> list[Mapping[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, (tuple, list)) or any(not isinstance(c, Mapping) for c in value):
        raise ValueError(f"{field} must contain visible card dictionaries")
    return list(value)


def visible_cards(observation: RecruitObservation) -> dict[str, list[Mapping[str, Any]]]:
    return {key: _cards(observation.private_state.get(key, []), key) for key in ("board", "hand", "shop")}


def action_card(observation: RecruitObservation, action: RecruitAction, *, target: bool = False) -> Mapping[str, Any] | None:
    identity = action.target_id if target else action.entity_id
    if identity is None:
        return None
    for rows in visible_cards(observation).values():
        for card in rows:
            entity = card.get("entity_id", card.get("entityId"))
            if str(entity) == str(identity):
                return card
    raise ValueError(f"Action entity {identity!r} is absent from visible cards")


def _is_spell(card: Mapping[str, Any] | None) -> float:
    return float(bool(card and (card.get("is_spell") or str(card.get("type", card.get("card_type", ""))).upper() == "SPELL")))


def encode_recruit_action(observation: RecruitObservation, action: RecruitAction) -> np.ndarray:
    if action not in observation.legal_actions:
        raise ValueError("Cannot score an action outside the time-constrained legal mask")
    if action.kind not in ACTION_KINDS:
        raise ValueError(f"Feature schema lacks action kind {action.kind}")
    budget = observation.action_budget
    if not isinstance(budget, Mapping):
        raise ValueError("Recruit policy requires an explicit finite action/time budget")
    for field in ("remaining_ms", "remaining_actions", "actions_used", "action_limit", "initial_available_ms", "reserve_ms", "actions_per_second"):
        if field not in budget or _number(budget[field], field) < 0:
            raise ValueError(f"Missing or negative timing feature: {field}")
    private, public = observation.private_state, observation.public_state
    cards = visible_cards(observation)
    source = action_card(observation, action)
    target = action_card(observation, action, target=True)
    zones = {str(c.get("entity_id", c.get("entityId"))): zone for zone, rows in cards.items() for c in rows}
    choice = None
    if action.kind == "choose":
        options = private.get("pending_choice", {}).get("options", [])
        if action.choice_index is None or not 0 <= action.choice_index < len(options):
            raise ValueError("Choice action lacks a visible option")
        choice = options[action.choice_index]
        if isinstance(choice, str) and re.match(r"^(?:BG|BGS_|TB_|HERO_)", choice):
            raise ValueError("Card-ID choice options must resolve to visible semantic card attributes")
    costs = _costs(budget)
    action_ms, followup_ms, followups, costs_known = _completion_costs(action, source, costs)
    context = [
        observation.turn / 15, _number(private.get("gold"), "gold") / 10,
        _number(private.get("tavern_tier", private.get("tier")), "tavern_tier") / 6,
        _number(private.get("upgrade_cost"), "upgrade_cost") / 10,
        _number(private.get("health", public.get("health")), "health") / 60,
        _number(private.get("armor", public.get("armor")), "armor") / 20,
        float(bool(private.get("frozen", False))), len(cards["board"]) / 7,
        len(cards["hand"]) / 10, len(cards["shop"]) / 7,
        budget["remaining_ms"] / 60000, budget["remaining_actions"] / 60,
        budget["actions_used"] / 60, budget["action_limit"] / 60,
        budget["initial_available_ms"] / 60000, budget["reserve_ms"] / 60000,
        budget["actions_per_second"] / 2,
        (action.position + 1) / 8 if action.position is not None else 0,
        float(target is not None), _number(source.get("cost"), "action cost") / 10 if source else 0,
        _number(target.get("cost"), "target cost") / 10 if target else 0,
        _is_spell(source), _is_spell(target), 1.0,
        (10 - len(cards["hand"])) / 10, (7 - len(cards["board"])) / 7,
        *(float(zones.get(str(identity)) == zone) for identity in (action.entity_id, action.target_id) for zone in ("board", "hand", "shop")),
        (action.choice_index + 1) / 10 if action.choice_index is not None else 0,
        math.copysign(math.log1p(abs(choice)), choice) / 4 if type(choice) in (int, float) else 0,
        float(bool(source and source.get("activate_used"))), float(bool(target and target.get("activate_used"))),
        float(costs_known), action_ms / 60000 if costs_known else 0,
        max(0, budget["remaining_ms"] - action_ms) / 60000 if costs_known else 0,
        max(0, budget["remaining_actions"] - (action.kind != "end_turn")) / 60,
        followup_ms / 60000 if costs_known else 0,
        float(costs_known and action_ms + followup_ms <= budget["remaining_ms"] and
              followups + (action.kind != "end_turn") <= budget["remaining_actions"]),
        *(math.log1p(_number(private.get("counters", {}).get(key), key)) / 4 for key in COUNTER_NAMES),
        *(float(bool(cards["board"][i].get("activate_used"))) if i < len(cards["board"]) else 0 for i in range(7)),
    ]
    vectors = {}
    zeros = np.zeros(len(CARD_FEATURE_NAMES))
    for key, rows in cards.items():
        encoded = np.stack([card_features(c) for c in rows]) if rows else zeros[None, :]
        vectors[key + "_mean"] = encoded.mean(axis=0)
        vectors[key + "_max"] = encoded.max(axis=0)
    for i in range(7):
        vectors["board_slot%d" % i] = card_features(cards["board"][i] if i < len(cards["board"]) else None)
    vectors["action_card"] = card_features(source)
    vectors["target_card"] = card_features(target)
    vectors["choice_option"] = card_features(choice if isinstance(choice, Mapping) else
        {"text": "choice " + str(choice)} if choice is not None else None)
    result = np.concatenate((np.asarray(context), np.asarray([float(action.kind == k) for k in ACTION_KINDS]),
                             *(vectors[name] for name in CARD_VIEWS)))
    if result.shape != (len(RECRUIT_FEATURE_NAMES),) or not np.isfinite(result).all():
        raise ValueError("Invalid recruit feature shape/values")
    return result


def encode_legal_actions(observation: RecruitObservation, actions: Sequence[RecruitAction] | None = None) -> np.ndarray:
    candidates = observation.legal_actions if actions is None else actions
    if not candidates:
        raise ValueError("No affordable legal actions")
    return np.stack([encode_recruit_action(observation, action) for action in candidates])


def observation_record(observation: RecruitObservation) -> dict[str, Any]:
    """Readable training trace, retaining IDs for audit but never as model inputs."""
    return asdict(observation)


# Exact semantic schema retained for frozen v2 holdout/transfer experiments.
_V3_ONLY_CONTEXT = frozenset({"timing_costs_known", "action_cost_ms", "remaining_ms_after_action",
    "remaining_actions_after_action", "minimum_followup_ms", "immediate_card_completion_fits"})
RECRUIT_V2_FEATURE_NAMES = tuple(name for name in RECRUIT_FEATURE_NAMES
    if name.removeprefix("recruit.context.") not in _V3_ONLY_CONTEXT)
_V2_INDICES = tuple(RECRUIT_FEATURE_NAMES.index(name) for name in RECRUIT_V2_FEATURE_NAMES)

def encode_legal_actions_v2(observation: RecruitObservation, actions: Sequence[RecruitAction] | None = None) -> np.ndarray:
    """Frozen v2 projection: exact names/order, excluding later affordability inputs."""
    return encode_legal_actions(observation, actions)[:, _V2_INDICES]
