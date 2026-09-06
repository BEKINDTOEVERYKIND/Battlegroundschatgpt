"""Visible semantic features for the verified two-turn opening fixture.

The frozen recruit v3 vector remains an exact prefix. The appended context
distinguishes the public lobby pool and owned, explicitly promised future gold.
These fields are required: absence is not evidence of an empty pool or no debt.
Card names/IDs, future draws, combat receipts and RNG never enter the vector.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Sequence

import numpy as np

from .recruit_features import (
    RECRUIT_FEATURE_NAMES, RECRUIT_V2_FEATURE_NAMES,
    encode_legal_actions, encode_recruit_action,
)
from .recruiting import RecruitAction, RecruitObservation

TWO_TURN_FEATURE_VERSION = "recruit-two-turn-visible-action-v1"
LOBBY_TRIBES = ("BEAST", "DEMON", "DRAGON", "ELEMENTAL", "MECH", "MURLOC",
               "NAGA", "PIRATE", "QUILBOAR", "UNDEAD")
TWO_TURN_CONTEXT_NAMES = tuple("recruit.two_turn.lobby_" + tribe.lower()
                               for tribe in LOBBY_TRIBES) + (
    "recruit.two_turn.deferred_next_turn_gold",
)
TWO_TURN_FEATURE_NAMES = RECRUIT_FEATURE_NAMES + TWO_TURN_CONTEXT_NAMES


def two_turn_schema_id() -> str:
    """Identify this explicit schema independently of the combat feature version."""
    payload = json.dumps({"version": TWO_TURN_FEATURE_VERSION,
                          "names": list(TWO_TURN_FEATURE_NAMES)},
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _context(observation: RecruitObservation) -> np.ndarray:
    tribes = observation.public_state.get("valid_tribes")
    if (not isinstance(tribes, (tuple, list)) or len(tribes) != 5
            or any(not isinstance(tribe, str) or tribe not in LOBBY_TRIBES
                   for tribe in tribes) or len(set(tribes)) != 5):
        raise ValueError("valid_tribes must contain exactly five unique canonical lobby tribes")
    amount = observation.private_state.get("deferred_next_turn_gold")
    if type(amount) is not int or amount < 0:
        raise ValueError("deferred_next_turn_gold must be an explicit nonnegative integer")
    try:
        scaled_gold = amount / 10
    except OverflowError as exc:
        raise ValueError("deferred_next_turn_gold is outside the finite feature range") from exc
    if not math.isfinite(scaled_gold):
        raise ValueError("deferred_next_turn_gold is outside the finite feature range")
    return np.asarray([*(float(tribe in tribes) for tribe in LOBBY_TRIBES), scaled_gold])


def encode_two_turn_action(observation: RecruitObservation,
                           action: RecruitAction) -> np.ndarray:
    context = _context(observation)
    return np.concatenate((encode_recruit_action(observation, action), context))


def encode_two_turn_actions(observation: RecruitObservation,
                            actions: Sequence[RecruitAction] | None = None) -> np.ndarray:
    context = _context(observation)
    base = encode_legal_actions(observation, actions)
    return np.concatenate((base, np.broadcast_to(context, (len(base), len(context)))), axis=1)


def warm_start_two_turn_ranker(source):
    """Expand an exact frozen v2/v3 Ranker while preserving its predictions.

    Shared normalization, weights, Adam moments, step and RNG are copied by
    semantic feature name. New input weights/moments start at zero; their mean
    and scale are zero/one. Future fit calls can learn those rows. The source
    checkpoint is untouched, and unknown/partial schemas fail closed.
    """
    from .learning import Ranker

    if not isinstance(source, Ranker) or source.feature_names not in (
            RECRUIT_V2_FEATURE_NAMES, RECRUIT_FEATURE_NAMES):
        raise ValueError("Warm start requires an exact frozen recruit v2 or v3 Ranker schema")
    target = deepcopy(source)
    target.feature_names = TWO_TURN_FEATURE_NAMES
    indices = np.asarray([TWO_TURN_FEATURE_NAMES.index(name) for name in source.feature_names])
    size, hidden = len(TWO_TURN_FEATURE_NAMES), source.params["w1"].shape[1]
    target.mean = np.zeros(size)
    target.scale = np.ones(size)
    target.mean[indices], target.scale[indices] = source.mean, source.scale
    for group in ("params", "m", "v"):
        matrix = np.zeros((size, hidden))
        matrix[indices] = getattr(source, group)["w1"]
        getattr(target, group)["w1"] = matrix
    target.metadata.setdefault("feature_schema_transitions", []).append({
        "from_feature_count": len(source.feature_names),
        "to_version": TWO_TURN_FEATURE_VERSION,
        "to_schema_sha256": two_turn_schema_id(),
        "at_optimizer_step": source.step,
        "new_inputs": "zero weights and Adam moments; zero mean and unit scale",
    })
    return target
