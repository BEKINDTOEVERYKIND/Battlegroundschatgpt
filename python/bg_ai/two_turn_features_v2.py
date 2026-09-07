"""Visible enchantment lifetimes, appended to the frozen two-turn v1 schema.

Enrichment is a pure observation transform. It never reads engine state, future
draws or combat receipts. The audited opening exporter supplies the enchantment
records; missing records and unknown temporary sources are errors, not zeros.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
from typing import Mapping, Sequence

import numpy as np

from .hsbrsim_opening import GENERATED_RECRUIT, TIER1_MINIONS, TIER1_SPELLS
from .recruit_features import action_card
from .recruiting import RecruitAction, RecruitObservation
from .two_turn_features import (
    TWO_TURN_FEATURE_NAMES as V1_FEATURE_NAMES,
    encode_two_turn_action as encode_v1_action,
    encode_two_turn_actions as encode_v1_actions,
)

TWO_TURN_FEATURE_VERSION = "recruit-two-turn-visible-action-v2-lifetimes"
LIFETIME_METADATA_VERSION = "opening-visible-enchantment-lifetimes-v1"
_METADATA_KEY = "two_turn_lifetime"
_SCOPE = "current-two-turn-tier1-opening-v1"
ZONE_LIMITS = {"board": 7, "hand": 10, "shop": 7}
LIFETIME_FIELDS = (
    "present", "temporary_attack", "temporary_health", "permanent_attack",
    "permanent_health", "expires_next_recruit", "discard_at_recruit_end",
)
LIFETIME_VIEWS = tuple(f"{zone}_slot{i}" for zone, size in ZONE_LIMITS.items()
                       for i in range(size)) + ("action_card", "target_card")
TWO_TURN_FEATURE_NAMES = V1_FEATURE_NAMES + tuple(
    f"recruit.two_turn.lifetime.{view}.{field}"
    for view in LIFETIME_VIEWS for field in LIFETIME_FIELDS)

# Only Mini-Trident can create temporary recruit enchantments in this pool.
# Its lifetime is explicitly next recruit, not an inferred generic "temporary".
_TEMPORARY_SOURCES = frozenset({"BG23_000t"})
# The empty source is the audited hsrl2.actions.stats.Buff primitive, which
# deliberately leaves source_id blank for permanent Tavern spells/Activate.
_PERMANENT_SOURCES = TIER1_MINIONS | TIER1_SPELLS | GENERATED_RECRUIT | {""}


def two_turn_schema_id() -> str:
    payload = json.dumps({"version": TWO_TURN_FEATURE_VERSION,
                          "names": list(TWO_TURN_FEATURE_NAMES)}, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _integer(value, name):
    if type(value) is not int:
        raise ValueError(f"{name} must be an explicit finite integer")
    try:
        valid = math.isfinite(value / 20)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be an explicit finite integer")
    return value


def _zones(observation):
    result = {}
    identities = set()
    for zone, limit in ZONE_LIMITS.items():
        rows = observation.private_state.get(zone)
        if (not isinstance(rows, (tuple, list)) or len(rows) > limit
                or any(not isinstance(row, Mapping) for row in rows)):
            raise ValueError(f"Lifetime features require an explicit bounded {zone} list")
        for row in rows:
            identity = row.get("entity_id")
            if not isinstance(identity, str) or not identity or identity in identities:
                raise ValueError("Lifetime features require unique visible entity_id values")
            identities.add(identity)
        result[zone] = rows
    return result


def _lifetime(card):
    buffs = card.get("enchantments")
    if not isinstance(buffs, (tuple, list)):
        raise ValueError("Visible enchantments must be explicit, including an empty list")
    spellcraft = card.get("temporary_spellcraft")
    if type(spellcraft) is not bool:
        raise ValueError("temporary_spellcraft must be an explicit boolean")
    if spellcraft and (card.get("card_type") != "spell"
                       or card.get("card_id") not in _TEMPORARY_SOURCES):
        raise ValueError("Unknown temporary Spellcraft card lifetime")
    totals = {key: 0 for key in ("temporary_attack", "temporary_health",
                                "permanent_attack", "permanent_health")}
    expires = False
    for buff in buffs:
        if not isinstance(buff, Mapping) or type(buff.get("temporary")) is not bool:
            raise ValueError("Each visible enchantment requires an explicit temporary flag")
        temporary, source = buff["temporary"], buff.get("source_id")
        if not isinstance(source, str) or source not in (
                _TEMPORARY_SOURCES if temporary else _PERMANENT_SOURCES):
            raise ValueError("Unknown visible enchantment source/lifetime")
        # No supported opening recruit effect grants a temporary keyword. The
        # original exporter has no keyword lifetime provenance, so an extension
        # cannot silently insert such a grant under this version.
        if buff.get("keywords") or buff.get("temporary_keywords"):
            raise ValueError("Temporary keyword lifetime needs a separately audited exporter")
        prefix = "temporary" if temporary else "permanent"
        for stat in ("attack", "health"):
            totals[f"{prefix}_{stat}"] += _integer(buff.get(stat), f"enchantment {stat}")
        expires |= temporary
    for key, value in totals.items():
        _integer(value, key)
    return {"version": LIFETIME_METADATA_VERSION, **totals,
            "expires_next_recruit": expires,
            "discard_at_recruit_end": spellcraft,
            "temporary_keywords": []}


def enrich_two_turn_observation(observation: RecruitObservation | None):
    """Copy an audited raw/timed observation and add required visible lifetimes.

    Call this after the timing wrapper, before encoding or recording features.
    It preserves the legal mask, timing state and all frozen v1 feature inputs.
    Scope is deliberately restricted; extending the engine requires an explicit
    review of newly possible buff sources and keyword expiration semantics.
    """
    if observation is None:
        return None
    if observation.public_state.get("scope") != _SCOPE:
        raise ValueError("Lifetime enrichment requires the audited two-turn opening scope")
    zones = _zones(observation)
    private = deepcopy(dict(observation.private_state))
    for zone, rows in zones.items():
        private[zone] = [dict(deepcopy(row), **{_METADATA_KEY: _lifetime(row)}) for row in rows]
    private["two_turn_lifetime_schema"] = LIFETIME_METADATA_VERSION
    return replace(observation, private_state=private)


def _validated_zones(observation):
    if observation.private_state.get("two_turn_lifetime_schema") != LIFETIME_METADATA_VERSION:
        raise ValueError("Missing or incompatible two-turn lifetime metadata; enrich observation first")
    zones = _zones(observation)
    for rows in zones.values():
        for card in rows:
            metadata = card.get(_METADATA_KEY)
            if not isinstance(metadata, Mapping) or metadata != _lifetime(card):
                raise ValueError("Missing, stale or incompatible visible card lifetime metadata")
    return zones


def _vector(card):
    if card is None:
        return np.zeros(len(LIFETIME_FIELDS))
    lifetime = card[_METADATA_KEY]
    return np.asarray([1.0, *(lifetime[key] / 20 for key in LIFETIME_FIELDS[1:5]),
                       float(lifetime["expires_next_recruit"]),
                       float(lifetime["discard_at_recruit_end"])])


def _suffix(observation, action, zones):
    vectors = [_vector(rows[i] if i < len(rows) else None)
               for zone, rows in zones.items() for i in range(ZONE_LIMITS[zone])]
    vectors.extend((_vector(action_card(observation, action)),
                    _vector(action_card(observation, action, target=True))))
    return np.concatenate(vectors)


def encode_two_turn_action(observation: RecruitObservation, action: RecruitAction) -> np.ndarray:
    zones = _validated_zones(observation)
    return np.concatenate((encode_v1_action(observation, action), _suffix(observation, action, zones)))


def encode_two_turn_actions(observation: RecruitObservation,
                            actions: Sequence[RecruitAction] | None = None) -> np.ndarray:
    zones = _validated_zones(observation)
    candidates = observation.legal_actions if actions is None else actions
    base = encode_v1_actions(observation, candidates)
    return np.concatenate((base, np.stack([_suffix(observation, action, zones)
                                           for action in candidates])), axis=1)


def warm_start_two_turn_ranker(source):
    """Migrate only an exact frozen 1129-input v1 Ranker, preserving predictions."""
    from .learning import Ranker

    if (not isinstance(source, Ranker) or source.feature_names != V1_FEATURE_NAMES
            or len(source.feature_names) != 1129):
        raise ValueError("Warm start requires the exact frozen 1129-feature two-turn v1 Ranker")
    target = deepcopy(source)
    target.feature_names = TWO_TURN_FEATURE_NAMES
    old_size, size = len(V1_FEATURE_NAMES), len(TWO_TURN_FEATURE_NAMES)
    hidden = source.params["w1"].shape[1]
    target.mean = np.concatenate((source.mean, np.zeros(size - old_size)))
    target.scale = np.concatenate((source.scale, np.ones(size - old_size)))
    for group in ("params", "m", "v"):
        getattr(target, group)["w1"] = np.concatenate((
            getattr(source, group)["w1"], np.zeros((size - old_size, hidden))), axis=0)
    target.metadata.setdefault("feature_schema_transitions", []).append({
        "from_feature_count": old_size, "to_version": TWO_TURN_FEATURE_VERSION,
        "to_schema_sha256": two_turn_schema_id(), "at_optimizer_step": source.step,
        "new_inputs": "zero weights and Adam moments; zero mean and unit scale",
    })
    return target
