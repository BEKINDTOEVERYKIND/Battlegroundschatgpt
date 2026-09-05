"""Mandatory bounded-action wrapper for a future validated recruit adapter.

The wrapper never sleeps: elapsed game time is a finite resource in each
transition. It supports interleaved players without resetting their clocks.
The existing combat-only positioning learner is not full recruit self-play.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from bg_ai.recruiting import (RecruitAction, RecruitEngine, RecruitObservation,
                             RecruitCoverageError, require_full_game_ready)
from bg_ai.turn_budget import TimingProfile, TurnBudget


class TimedRecruitEngine:
    def __init__(self, engine: RecruitEngine, profile: TimingProfile,
                 ruleset: Mapping[str, Any]):
        report = engine.coverage()
        require_full_game_ready(report, ruleset)
        timing = report.get("turn_timing", {})
        if (timing.get("actual_player_window_validated") is not True or
                timing.get("pending_choice_timeout_validated") is not True):
            raise RecruitCoverageError("Actual player turn windows and pending-choice timeout behavior require validation")
        self.engine = engine
        self.profile = profile
        self._budgets: dict[tuple[int, int], TurnBudget] = {}
        self._closed: set[tuple[int, int]] = set()
        self._last_turn: dict[int, int] = {}
        self._raw: RecruitObservation | None = None
        self._visible: RecruitObservation | None = None
        self.trace: list[dict[str, Any]] = []

    def coverage(self) -> Mapping[str, Any]:
        return dict(self.engine.coverage(), timing_profile_sha256=self.profile.fingerprint)

    def final_placements(self) -> Mapping[int, float]:
        return self.engine.final_placements()

    def reset(self, *, seed: int) -> RecruitObservation | None:
        self._budgets.clear()
        self._closed.clear()
        self._last_turn.clear()
        self.trace.clear()
        self._raw = self._visible = None
        return self._accept(self.engine.reset(seed=seed))

    def _accept(self, observation: RecruitObservation | None) -> RecruitObservation | None:
        # An invalid adapter must not create an endless chain of zero-time turns.
        for _ in range(64):
            if observation is None:
                self._raw = self._visible = None
                return None
            key = observation.player_id, observation.turn
            if key in self._closed or observation.turn < self._last_turn.get(observation.player_id, observation.turn):
                raise RuntimeError("Adapter returned an already closed or earlier recruit turn")
            if observation.time_remaining_ms is None:
                raise ValueError("Adapter must supply actual time_remaining_ms for every recruit observation")
            previous = self._last_turn.get(observation.player_id)
            if previous is not None and observation.turn > previous:
                self._closed.add((observation.player_id, previous))
            self._last_turn[observation.player_id] = observation.turn
            if key not in self._budgets:
                self._budgets[key] = TurnBudget(self.profile, observation.time_remaining_ms)
            budget = self._budgets[key]
            budget.observe_remaining(observation.time_remaining_ms)
            affordable = tuple(a for a in observation.legal_actions
                               if a.kind != "end_turn" and budget.can_afford(a.kind))
            if not affordable:
                self._closed.add(key)
                self.trace.append({"player_id": key[0], "turn": key[1],
                                   "event": "timeout_or_no_affordable_actions", **budget.snapshot()})
                observation = self.engine.expire_turn()
                continue
            end_actions = tuple(a for a in observation.legal_actions if a.kind == "end_turn")
            self._raw = observation
            self._visible = replace(observation, legal_actions=affordable + end_actions,
                                    action_budget=budget.snapshot())
            return self._visible
        raise RuntimeError("Adapter exceeded 64 consecutive automatic recruit-turn expirations")

    def step(self, action: RecruitAction) -> RecruitObservation | None:
        if self._visible is None or action not in self._visible.legal_actions:
            raise ValueError("Action is not in the current time-constrained legal action list")
        assert self._raw is not None
        key = self._raw.player_id, self._raw.turn
        budget = self._budgets[key]
        before = budget.snapshot()
        if action.kind == "end_turn":
            self._closed.add(key)
        else:
            budget.charge(action.kind)
        self.trace.append({"player_id": key[0], "turn": key[1], "event": "action",
                           "kind": action.kind, "before": before, "after": budget.snapshot()})
        # Failure in the engine aborts this rollout; do not retry a partly applied
        # action with a refunded clock or stale policy observation.
        self._visible = self._raw = None
        return self._accept(self.engine.step(action))

    def expire_turn(self) -> RecruitObservation | None:
        if self._raw is None:
            raise ValueError("No active recruit turn")
        self._closed.add((self._raw.player_id, self._raw.turn))
        self._visible = self._raw = None
        return self._accept(self.engine.expire_turn())
