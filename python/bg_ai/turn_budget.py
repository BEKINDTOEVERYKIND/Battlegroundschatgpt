"""Finite virtual turn clocks; CPU speed cannot buy extra game actions.

The rate and per-command delays are training assumptions. Actual observed time
can tighten a budget, never refill it within one player's recruit turn.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


def _milliseconds(value: int, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer number of milliseconds")
    return value


@dataclass(frozen=True)
class TimingProfile:
    actions_per_second: float
    reserve_ms: int
    extra_ms: Mapping[str, int]

    def __post_init__(self):
        rate = self.actions_per_second
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
            raise ValueError("actions_per_second must be finite and positive")
        # Millisecond resolution must still give every command a positive cost.
        if rate > 1000:
            raise ValueError("actions_per_second exceeds the clock's millisecond resolution")
        _milliseconds(self.reserve_ms, "reserve_ms")
        copied = dict(self.extra_ms)
        if not copied or any(not isinstance(k, str) or not k or k == "end_turn" for k in copied):
            raise ValueError("extra_ms must enumerate supported action kinds; end_turn is separate")
        for kind, cost in copied.items():
            _milliseconds(cost, f"extra_ms[{kind}]")
        object.__setattr__(self, "extra_ms", MappingProxyType(copied))

    @classmethod
    def load(cls, path: str | Path) -> "TimingProfile":
        data = json.loads(Path(path).read_text())
        if data.get("schema_version") != 1:
            raise ValueError("Unsupported timing profile schema")
        return cls(data["actions_per_second"], data["reserve_ms"], data["extra_ms"])

    @property
    def base_ms(self) -> int:
        return math.ceil(1000 / self.actions_per_second)

    def action_ms(self, kind: str) -> int:
        if kind not in self.extra_ms:
            raise ValueError(f"Unmapped action kind: {kind}; add an explicit timing cost")
        return self.base_ms + self.extra_ms[kind]

    @property
    def fingerprint(self) -> str:
        body = {"schema_version": 1, "actions_per_second": float(self.actions_per_second),
                "reserve_ms": self.reserve_ms, "extra_ms": dict(self.extra_ms)}
        return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class TurnBudget:
    def __init__(self, profile: TimingProfile, available_ms: int):
        self.profile = profile
        self.available_ms = _milliseconds(available_ms, "available_ms")
        self.usable_ms = max(0, self.available_ms - profile.reserve_ms)
        self.action_limit = math.floor(self.usable_ms * profile.actions_per_second / 1000)
        self.actions_used = 0
        self.spent_ms = 0
        self._remaining_ms = self.usable_ms

    @property
    def remaining_ms(self) -> int:
        return self._remaining_ms

    @property
    def remaining_actions(self) -> int:
        return min(self.action_limit - self.actions_used, self.remaining_ms // self.profile.base_ms)

    def can_afford(self, kind: str) -> bool:
        cost = self.profile.action_ms(kind)
        return self.remaining_actions > 0 and cost <= self.remaining_ms

    def charge(self, kind: str) -> int:
        if not self.can_afford(kind):
            raise ValueError("Action does not fit within this turn's time/action budget")
        cost = self.profile.action_ms(kind)
        self.actions_used += 1
        self.spent_ms += cost
        self._remaining_ms -= cost
        return cost

    def observe_remaining(self, available_ms: int) -> None:
        """Account for a shorter timer/latency; a new observation cannot refill."""
        observed = max(0, _milliseconds(available_ms, "available_ms") - self.profile.reserve_ms)
        self._remaining_ms = min(self._remaining_ms, observed)

    def fork(self) -> "TurnBudget":
        """Planning branch with the same spent budget, independent future charges."""
        branch = TurnBudget(self.profile, self.available_ms)
        branch.actions_used = self.actions_used
        branch.spent_ms = self.spent_ms
        branch._remaining_ms = self._remaining_ms
        return branch

    def snapshot(self) -> dict:
        return {"profile_sha256": self.profile.fingerprint,
                "actions_per_second": self.profile.actions_per_second,
                "initial_available_ms": self.available_ms, "reserve_ms": self.profile.reserve_ms,
                "action_limit": self.action_limit, "actions_used": self.actions_used,
                "remaining_actions": self.remaining_actions, "remaining_ms": self.remaining_ms,
                "charged_ms": self.spent_ms}
