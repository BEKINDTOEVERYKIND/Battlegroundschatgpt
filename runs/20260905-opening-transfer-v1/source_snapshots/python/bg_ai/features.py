"""Stable, card-ID-free features for the combat-positioning subpolicy.

Card dictionaries use the Firestone-style stat/keyword names documented in
``encode_card``. The caller must resolve text-based mechanics and numeric race
enums using the *active* card registry before encoding; we deliberately do not
infer mechanics from card names. Unsupported mechanics remain a model limitation,
not a claim that these minions have no special behaviour in the game engine.

Recruiting and hero selection can reuse ``encode_card`` as their item encoder.
Those policies need their own legal-action/environment implementations; this
module does not pretend a board-positioning learner is a complete game agent.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np


FEATURE_VERSION = "combat-position-v2-text"
MAX_BOARD_SIZE = 7
TEXT_HASH_BUCKETS = 32
TRIBES = ("beast", "demon", "dragon", "elemental", "mech", "murloc", "naga",
          "pirate", "quilboar", "undead", "all")
KEYWORDS = ("taunt", "divine_shield", "windfury", "mega_windfury", "reborn",
            "venomous", "poisonous", "cleave", "deathrattle", "avenge", "frenzy")
_EFFECT_PATTERNS = {
    "summon": r"\b(?:summon|resummon)",
    "stat_buff": r"\b(?:give|gain|gains|get|gets|grant)\b.*(?:\+\d|attack|health)",
    "adjacent": r"\b(?:adjacent|next to)\b",
    "rally": r"\brally\b",
    "start_combat": r"\bstart of combat\b",
    "attack_trigger": r"\b(?:after|whenever|when)\b.*\battacks?\b",
    "death_trigger": r"\b(?:after|whenever|when)\b.*\b(?:dies|die|death)\b",
    "aura": r"\byour (?:other )?minions have\b",
}
EFFECT_FEATURE_NAMES = ("text_present", "effect_number_mean", "effect_number_max") + tuple(
    f"effect_{name}" for name in _EFFECT_PATTERNS
) + tuple(f"effect_text_hash_{i:02d}" for i in range(TEXT_HASH_BUCKETS))
CARD_FEATURE_NAMES = ("present", "attack_log", "health_log", "tier", "golden") + tuple(
    f"tribe_{x}" for x in TRIBES
) + tuple(f"keyword_{x}" for x in KEYWORDS) + EFFECT_FEATURE_NAMES
_IDX = {name: i for i, name in enumerate(CARD_FEATURE_NAMES)}
_PAIR_NAMES = ("attack_gap", "health_gap", "attack_before_taunt",
               "deathrattle_before_attack", "reborn_before_attack",
               "venomous_before_shield")


def _names() -> tuple[str, ...]:
    names = []
    for side in ("friendly", "opponent"):
        for slot in range(MAX_BOARD_SIZE):
            names.extend(f"{side}.slot{slot}.{name}" for name in CARD_FEATURE_NAMES)
        names.extend(f"{side}.mean.{name}" for name in CARD_FEATURE_NAMES)
    for left in range(MAX_BOARD_SIZE):
        for right in range(left + 1, MAX_BOARD_SIZE):
            names.extend(f"friendly.pair{left}_{right}.{name}" for name in _PAIR_NAMES)
    names.extend(("friendly.count", "opponent.count", "friendly.attacks_first",
                  "opponent.attacks_first", "equal_board_size"))
    return tuple(names)


BOARD_FEATURE_NAMES = _names()


def feature_schema_id(names: Sequence[str] = BOARD_FEATURE_NAMES) -> str:
    payload = json.dumps({"version": FEATURE_VERSION, "names": list(names)},
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _normal(value: str) -> str:
    return "".join(c for c in value.lower() if c.isalnum())


def _enabled(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in ("", "0", "false", "none", "no")
    return bool(value)


def _number(card: Mapping[str, Any], *keys: str, default: float = 0) -> float:
    value = next((card[k] for k in keys if k in card and card[k] is not None), default)
    if isinstance(value, bool):
        raise ValueError(f"{keys[0]} must be a number, not a boolean")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{keys[0]} must be finite and nonnegative")
    return value


def _encode_effect_text(text: Any) -> np.ndarray:
    """Fixed signed feature hashing of shared effect words and word pairs.

    This is a compact lexical representation, not a rules interpreter. Hash
    collisions are expected; unique effects can still need more expressive
    features. Numbers use shared magnitude features instead of a growing token
    vocabulary. Neither card IDs nor card names are used to index parameters.
    """
    if text is None:
        text = ""
    if not isinstance(text, str):
        raise ValueError("Card effect text must be a string")
    normalized = re.sub(r"<[^>]*>", " ", html.unescape(text)).lower()
    normalized = re.sub(r"\[[a-z]+\]", " ", normalized)
    normalized = " ".join(normalized.split())
    numbers = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", normalized)]
    prefix = [float(bool(normalized)),
              math.log1p(sum(numbers) / len(numbers)) / 4 if numbers else 0,
              math.log1p(max(numbers)) / 4 if numbers else 0]
    prefix.extend(float(bool(re.search(pattern, normalized))) for pattern in _EFFECT_PATTERNS.values())
    tokens = re.findall(r"[a-z]+|\d+(?:\.\d+)?", normalized)
    tokens = ["number" if t[0].isdigit() else t for t in tokens]
    grams = [(token, 1.0) for token in tokens]
    grams.extend((left + " " + right, 0.7) for left, right in zip(tokens, tokens[1:]))
    hashed = np.zeros(TEXT_HASH_BUCKETS)
    for gram, weight in grams:
        digest = hashlib.sha256(gram.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "little") % TEXT_HASH_BUCKETS
        hashed[bucket] += weight if digest[4] & 1 else -weight
    norm = np.linalg.norm(hashed)
    if norm:
        hashed /= norm
    return np.concatenate((np.asarray(prefix), hashed))


def encode_card(card: Mapping[str, Any] | None) -> np.ndarray:
    """Encode public minion attributes; cardId/name/entity IDs are never inputs.

    Required for a real minion: ``attack`` and ``health``. Optional: tavernTier
    (or tier), golden/isGolden, races/tribes (string or list of string tribe
    names), effect ``text``, and boolean keyword fields in camelCase or snake_case. A mechanics
    (or keywords) list of keyword strings is also accepted. ALL enables every
    tribe; MECHANICAL maps to MECH. Missing optional data encodes as zero.
    """
    out = np.zeros(len(CARD_FEATURE_NAMES), dtype=np.float64)
    if card is None:
        return out
    if "attack" not in card or "health" not in card:
        raise ValueError("Every minion needs attack and health")
    out[:5] = (1, math.log1p(_number(card, "attack")) / 4,
               math.log1p(_number(card, "health")) / 4,
               _number(card, "tavernTier", "tier") / 6,
               float(_enabled(card.get("golden", card.get("isGolden", False)))))
    tribes = card.get("races", card.get("tribes", card.get("tribe", []))) or []
    if isinstance(tribes, str):
        tribes = [tribes]
    if not isinstance(tribes, (list, tuple, set)) or any(not isinstance(t, str) for t in tribes):
        raise ValueError("races/tribes must be string names; resolve numeric enums first")
    normalized = {_normal(t) for t in tribes}
    if "mechanical" in normalized:
        normalized.add("mech")
    if "all" in normalized:
        normalized.update(TRIBES)
    for tribe in TRIBES:
        out[_IDX[f"tribe_{tribe}"]] = float(tribe in normalized)
    mechanics = card.get("mechanics", card.get("keywords", [])) or []
    if isinstance(mechanics, str):
        mechanics = [mechanics]
    if not isinstance(mechanics, (list, tuple, set)):
        raise ValueError("mechanics/keywords must be a list of string names")
    enabled = {_normal(k) for k, value in card.items() if _enabled(value)}
    enabled.update(_normal(m) for m in mechanics if isinstance(m, str))
    for keyword in KEYWORDS:
        out[_IDX[f"keyword_{keyword}"]] = float(_normal(keyword) in enabled)
    out[-len(EFFECT_FEATURE_NAMES):] = _encode_effect_text(card.get("text", ""))
    return out


def encode_board(board: Sequence[Mapping[str, Any]],
                 opponent: Sequence[Mapping[str, Any]] = ()) -> np.ndarray:
    """Encode candidate order and visible opposing order without hidden state.

    Both board slot encodings preserve order. Friendly directed pair features
    express ordering interactions and remain useful after card rotations.
    The representation cannot distinguish unique card effects absent from its
    shared mechanics; add a versioned feature before training such a mechanism.
    """
    if len(board) > MAX_BOARD_SIZE or len(opponent) > MAX_BOARD_SIZE:
        raise ValueError("Battlegrounds boards have at most seven minions")
    arrays = []
    pieces = []
    for cards in (board, opponent):
        encoded = np.stack([encode_card(cards[i] if i < len(cards) else None)
                            for i in range(MAX_BOARD_SIZE)])
        arrays.append(encoded)
        pieces.extend((encoded.reshape(-1), encoded.mean(axis=0)))
    ours = arrays[0]
    attack = _IDX["attack_log"]
    health = _IDX["health_log"]
    for i in range(MAX_BOARD_SIZE):
        for j in range(i + 1, MAX_BOARD_SIZE):
            a, b = ours[i], ours[j]
            present = a[0] * b[0]
            pieces.append(np.array([
                (a[attack] - b[attack]) * present,
                (a[health] - b[health]) * present,
                a[attack] * b[_IDX["keyword_taunt"]],
                a[_IDX["keyword_deathrattle"]] * b[attack],
                a[_IDX["keyword_reborn"]] * b[attack],
                a[_IDX["keyword_venomous"]] * b[_IDX["keyword_divine_shield"]],
            ]))
    pieces.append(np.array([len(board) / 7, len(opponent) / 7,
                            float(len(board) > len(opponent)),
                            float(len(board) < len(opponent)),
                            float(len(board) == len(opponent))]))
    result = np.concatenate(pieces)
    assert result.shape == (len(BOARD_FEATURE_NAMES),)
    return result
