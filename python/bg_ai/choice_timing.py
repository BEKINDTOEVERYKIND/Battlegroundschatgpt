"""Versioned visible-information reservation of explicit opening choice costs.

This is a training-time feasibility rule, not a claim about client rope expiry.
The command which opens a choice and every subsequent choice remain separate
policy decisions, separately charged to the same finite TurnBudget.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from .hsbrsim_adapter import UnsupportedRecruitTransition
from .opening_transition import TimedTwoTurnOpeningEngine
from .recruiting import RecruitCoverageError
from .turn_budget import TurnBudget

CHOICE_TIMING_VERSION = "visible-opening-choice-reservation-v1"
# Command graphs were audited against this current reference snapshot and the
# already pinned hsrl2 engine. A new data export requires another audit.
AUDITED_REFERENCE_SHA256 = "88dd032aa56ddb8a176ace6d5bafa48b2d78b4049fd2b97c0e6a00e663f1ad14"


@dataclass(frozen=True)
class ChoiceCompletion:
    card_id: str
    kinds: tuple[str, ...]
    optional: bool = False


def opening_choice_completion(observation, action) -> ChoiceCompletion | None:
    """Upper bound on forced follow-up commands, from the visible source only.

    The existing engine still owns all legal targets and effects. Discover may
    fizzle in an exhausted pool: reserving its possible one command avoids
    looking at hidden availability. Generated cards need later explicit plays;
    those optional plays are never folded into the initiating command.
    """
    if action.kind not in ("cast_spell", "activate"):
        return None
    zone = "hand" if action.kind == "cast_spell" else "board"
    source = next((card for card in observation.private_state.get(zone, ())
                   if card["entity_id"] == action.entity_id), None)
    if source is None:
        raise RecruitCoverageError("Choice initiator must have a visible source")
    cid = source["card_id"]
    if action.kind == "activate":
        if cid != "BG36_345":
            raise RecruitCoverageError("Unaudited current opening Activate source")
        return ChoiceCompletion(cid, ("activate_target",))
    if cid == "BG31_880":
        return ChoiceCompletion(cid, ("choose_one", "spell_target"))
    if cid in {"BG28_503", "BG28_897", "BG20_GEM", "BG23_000t"}:
        return ChoiceCompletion(cid, ("spell_target",))
    if cid == "BG33_101":
        return ChoiceCompletion(cid, ("discover_minion",), optional=True)
    return None


def _fits(budget, kind: str, following_choices: int) -> bool:
    cost = budget.profile.action_ms(kind)
    if following_choices:
        cost += following_choices * budget.profile.action_ms("choose")
    return (budget.action_limit - budget.actions_used >= 1 + following_choices
            and budget.remaining_ms >= cost)


class ChoiceTimedTwoTurnOpeningEngine(TimedTwoTurnOpeningEngine):
    """Time-mask known command sequences without free choices or timeout guesses."""

    @classmethod
    def for_fixture(cls, engine, profile, ruleset):
        if engine.provenance.get("reference_cards_sha256") != AUDITED_REFERENCE_SHA256:
            raise RecruitCoverageError("Opening choice command graph needs a current-data audit")
        profile.action_ms("choose")
        return super().for_fixture(engine, profile, ruleset)

    def _initialize(self, engine, profile):
        super()._initialize(engine, profile)
        self._choice_reservations: dict[tuple[int, int], ChoiceCompletion] = {}
        self._timing_failed = False

    def reset(self, *, seed):
        self._choice_reservations.clear()
        self._timing_failed = False
        return super().reset(seed=seed)

    def coverage(self):
        return dict(super().coverage(), choice_timing_version=CHOICE_TIMING_VERSION,
                    choice_reservations="visible-source-command-graph-upper-bound",
                    choice_commands_charged_individually=True,
                    pending_choice_timeout="raise_and_abort")

    def fork(self):
        # Parent fork accepts an observation before a subclass can copy its
        # reservation. Copy it first so an in-progress sequence remains valid.
        if self._timing_failed:
            raise UnsupportedRecruitTransition("Timing rollout already failed; reset is required")
        branch = type(self).for_fixture(self.engine.fork(), self.profile, self._fixture_ruleset)
        branch._budgets = {key: budget.fork() for key, budget in self._budgets.items()}
        branch._closed = set(self._closed)
        branch._last_turn = dict(self._last_turn)
        branch._choice_reservations = dict(self._choice_reservations)
        branch.trace = list(self.trace)
        branch._accept(branch.engine.observe())
        return branch

    def _accept(self, observation):
        # Preserve the shared wrapper's per-player clocks and bounded expiry
        # loop; version only the test for affordable complete command sequences.
        if self._timing_failed:
            raise UnsupportedRecruitTransition("Timing rollout already failed; reset is required")
        for _ in range(64):
            if observation is None:
                if self._choice_reservations:
                    raise UnsupportedRecruitTransition("Recruit phase ended with a reserved pending choice")
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
            pending = observation.private_state.get("pending_choice")
            reservation = self._choice_reservations.get(key)
            if reservation and not pending and reservation.optional:
                # A zero-option Discover consumes no imaginary choose command.
                self._choice_reservations.pop(key)
                reservation = None
            if pending:
                if (reservation is None or pending.get("kind") != reservation.kinds[0]
                        or not observation.legal_actions
                        or any(a.kind != "choose" for a in observation.legal_actions)):
                    raise UnsupportedRecruitTransition("Pending choice differs from audited command sequence")
                if not _fits(budget, "choose", len(reservation.kinds) - 1):
                    self._visible = self._raw = None
                    raise UnsupportedRecruitTransition(
                        "Observed timer shrank below reserved explicit choices; pending-choice timeout is unverified")
            elif reservation is not None:
                raise UnsupportedRecruitTransition("Required pending choice missing from audited command sequence")
            affordable = []
            for action in observation.legal_actions:
                if action.kind == "end_turn":
                    continue
                if action.kind == "choose":
                    following = len(reservation.kinds) - 1
                else:
                    plan = opening_choice_completion(observation, action)
                    following = len(plan.kinds) if plan else 0
                if _fits(budget, action.kind, following):
                    affordable.append(action)
            if not affordable:
                self._closed.add(key)
                self.trace.append({"player_id": key[0], "turn": key[1],
                    "event": "timeout_or_no_affordable_actions", **budget.snapshot(),
                    "choice_timing_version": CHOICE_TIMING_VERSION})
                observation = self.engine.expire_turn()
                continue
            self._raw = observation
            count = len(reservation.kinds) if reservation else 0
            self._visible = replace(observation,
                legal_actions=tuple(affordable) + tuple(a for a in observation.legal_actions if a.kind == "end_turn"),
                action_budget=dict(budget.snapshot(), choice_timing_version=CHOICE_TIMING_VERSION,
                    reserved_choice_commands=count,
                    reserved_choice_ms=count * self.profile.action_ms("choose")))
            return self._visible
        raise RuntimeError("Adapter exceeded 64 consecutive automatic recruit-turn expirations")

    def step(self, action):
        if self._visible is None or action not in self._visible.legal_actions:
            raise ValueError("Action is not in the current time-constrained legal action list")
        key = self._raw.player_id, self._raw.turn
        if action.kind == "choose":
            previous = self._choice_reservations[key]
            plan = replace(previous, kinds=previous.kinds[1:], optional=False)
            plan = plan if plan.kinds else None
        else:
            plan = opening_choice_completion(self._raw, action)
        if plan:
            self._choice_reservations[key] = plan
        else:
            self._choice_reservations.pop(key, None)
        # Charges exactly one command, then checks the resulting visible choice.
        # Engine failures retain the spent clock and invalidate that rollout.
        try:
            return super().step(action)
        except Exception:
            self._timing_failed = True
            self._visible = self._raw = None
            raise
