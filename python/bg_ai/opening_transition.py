"""Two-recruit-turn fixture with sampled external combat and real persistence.

Combat requests/receipts are trusted evaluator data, never policy observations.
The old first-combat fixture and its trained artifacts remain unchanged.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from typing import Any, Mapping

from .hsbrsim_opening import (
    OpeningRecruitEngine, OpeningFixtureSpec, TIER1_SPELLS, GENERATED_RECRUIT,
    ALWAYS_GOLDEN, _opening_game_type,
)
from .hsbrsim_adapter import UnsupportedRecruitTransition
from .recruiting import RecruitCoverageError, ruleset_digest
from .timed_recruiting import TimedRecruitEngine

TRANSITION_SCOPE = "current-two-turn-tier1-opening-v1"
RECEIPT_VERSION = "firestone-opening-outcome-v1"
DEFAULT_PAIRINGS = ((0, 1), (2, 3), (4, 5), (6, 7))


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _transition_game_type():
    from hsrl2.constants import TAVERN_OFFERS
    from hsrl2.minion import Minion
    from hsrl2.tags import Zone

    class TransitionGame(_opening_game_type()):
        def attach_magnetic(self, hero, magnetic_minion, host):
            if any(buff.temporary for buff in magnetic_minion._buffs):
                raise UnsupportedRecruitTransition(
                    "Temporary Spellcraft magnetic transfer requires lifetime-preserving attachment")
            return super().attach_magnetic(hero, magnetic_minion, host)

        def refresh_tavern(self, hero, *, auto=False, free=False, spells_only=False):
            frozen = auto and hero.frozen_tavern
            # The parent checks Tier1 before mutating gold, cards or pool.
            result = super().refresh_tavern(hero, auto=auto, free=free,
                                            spells_only=spells_only)
            if frozen:
                # The upstream count subtracts the frozen spell from the minion
                # quota. Current Tavern spells have their own dedicated slot.
                need = TAVERN_OFFERS[hero.tavern_tier] - sum(
                    isinstance(card, Minion) for card in hero.tavern)
                for cid in self.minion_pool.draw_tavern(hero.tavern_tier, max(0, need)):
                    card = self.create_minion(cid, controller=hero)
                    card.zone = Zone.TAVERN
                    spell_slot = next((i for i, c in enumerate(hero.tavern)
                                       if not isinstance(c, Minion)), len(hero.tavern))
                    hero.tavern.insert(spell_slot, card)
            return result

        def run_combat(self, *args, **kwargs):
            raise UnsupportedRecruitTransition("Native combat is disabled in the external-combat fixture")

        def end_recruit_phase(self, *args, **kwargs):
            raise UnsupportedRecruitTransition("Use a sampled Firestone receipt to advance combat")

    return TransitionGame


class TwoTurnOpeningRecruitEngine(OpeningRecruitEngine):
    """Preserve actual shop/hand/board state through two sampled combats.

    Setup uses OpeningFixtureSpec, whose max_turns=1 denotes its original setup
    version. This class explicitly owns a separate two-turn horizon and scope.
    All eight players use Patchwerk; pairings are declared fixture assumptions.
    """

    def __init__(self, db, ruleset: Mapping[str, Any], *, timer_ms, provenance,
                 fixture: OpeningFixtureSpec | None = None):
        self.phase = "uninitialized"
        self._timeline: list[tuple[str, Any]] = []
        self._pending_request = None
        self.combat_receipts: list[dict[str, Any]] = []
        super().__init__(db, ruleset, timer_ms=timer_ms, provenance=provenance, fixture=fixture)
        self._Game = _transition_game_type()

    def reset(self, *, seed):
        self.phase = "recruit"
        self._timeline.clear()
        self.combat_receipts.clear()
        self._pending_request = None
        return super().reset(seed=seed)

    def coverage(self):
        report = super().coverage()
        report.update(scope=TRANSITION_SCOPE, max_recruit_turns=2,
            combat_transition="one-sampled-Firestone-outcome-per-pair",
            native_combat_disabled=True,
            pairings="explicit-synthetic-eight-player-pairings",
            combat_persistence="unchanged-original-entities-plus-sampled-hero-damage",
            opening_pool_complete=True)
        report["blockers"] = [b for b in report["blockers"]
                              if b != "later_turns_and_seasonal_effects"] + [
            "turn3_and_seasonal_effects", "eliminations_and_ghosts",
            "tier2_next_turn_shop_after_turn1_upgrade", "temporary_spellcraft_magnetic_transfer"]
        return report

    def deferred_gold(self, hero) -> int:
        """Exact visible promised gold; unknown deferred actions fail closed."""
        from hsrl2.actions.economy import GainGold
        amount = 0
        for owner, action in self.game.deferred_actions:
            if (owner not in self.game.heroes or not isinstance(action, GainGold)
                    or action.hero is not owner or type(action.amount) is not int
                    or action.amount < 0):
                raise UnsupportedRecruitTransition("Unverified deferred opening action")
            if owner is hero:
                amount += action.amount
        return amount

    def observe(self):
        observation = super().observe()
        if observation is None:
            return None
        self._last_observation = replace(observation,
            public_state=dict(observation.public_state, scope=TRANSITION_SCOPE),
            private_state=dict(observation.private_state,
                deferred_next_turn_gold=self.deferred_gold(self.game.heroes[self.player_id])))
        return self._last_observation

    def step(self, action):
        result = super().step(action)
        self._timeline.append(("action", action))
        return result

    def _apply(self, action):
        # Reject the unsupported transfer before upstream play removes the
        # source from hand or fires any other side effect. Keep the command
        # legal: evaluators reject this branch, never hide the real action.
        if action.kind == "play" and action.target_id is not None:
            source = self._entities.get(action.entity_id)
            if source is not None and any(b.temporary for b in source._buffs):
                raise UnsupportedRecruitTransition(
                    "Temporary Spellcraft magnetic transfer requires lifetime-preserving attachment")
        return super()._apply(action)

    def _end_player(self):
        if self._choice() is not None:
            raise UnsupportedRecruitTransition("Pending-choice timeout is unverified; abort rollout")
        self.finished_players.add(self.player_id)
        remaining = [i for i in range(8) if i not in self.finished_players]
        if remaining:
            self.player_id = remaining[0]
            return self.observe()
        self.game.events.fire(self.game, "turn_end", turn=self.game.turn)
        for hero in self.game.heroes:
            for minion in list(hero.board):
                self.game.run_script_hook(minion, "end_of_turn")
                self.game.check_deaths()
            for card in list(hero.hand):
                if isinstance(card, self._Spell) and card.has(self._T.SPELLCRAFT):
                    hero.hand.remove(card)
                    card.zone = self._Zone.REMOVED
        self._check_scope()
        self.terminal = True  # No recruit observation while awaiting a receipt.
        self.phase = "awaiting_combat"
        self._last_observation = None
        return None

    def _check_scope(self):
        if self.game is None:
            return
        if self._invalid:
            raise UnsupportedRecruitTransition("Two-turn opening rollout already failed")
        if self.game.turn not in (1, 2) or self.game.dark_gift_state or self.game.dark_gift_audit_log:
            raise UnsupportedRecruitTransition("Turn3/seasonal frontier reached")
        if len(self.game.heroes) != 8 or not all(hero.is_alive for hero in self.game.heroes):
            raise UnsupportedRecruitTransition("Eliminations/ghosts exceed the two-turn fixture")
        allowed = self.allowed_minions | TIER1_SPELLS | GENERATED_RECRUIT
        for hero in self.game.heroes:
            if hero.trinkets or getattr(hero, "quest_rewards", []):
                raise UnsupportedRecruitTransition("Seasonal effects exceed the two-turn fixture")
            self.deferred_gold(hero)
            for card in hero.board + hero.hand + hero.tavern:
                if card.card_id not in allowed:
                    raise UnsupportedRecruitTransition(f"Unsupported opening card: {card.card_id}")
                if isinstance(card, self._Minion) and card.is_golden and card.card_id != ALWAYS_GOLDEN:
                    raise UnsupportedRecruitTransition("Ordinary goldens exceed the opening frontier")
                if card.card_id == "BG35_814" and card not in hero.board:
                    if card.atk >= self.db.get(card.card_id).num(0):
                        raise UnsupportedRecruitTransition("Off-board Scarlet threshold conformance is unverified")
        if self.game.pending_hand_queue:
            raise UnsupportedRecruitTransition("Opening full-hand generated queue requires conformance")

    def combat_snapshot(self, player_id):
        if self.phase != "awaiting_combat":
            raise ValueError("All eight players must finish before exporting this combat")
        snapshot = super().combat_snapshot(player_id)
        snapshot.update(scope=TRANSITION_SCOPE, turn=self.game.turn)
        snapshot["recruit_return_state"] = {
            "next_turn_state_available": False,
            "requires_sampled_combat_receipt": True,
            "combat_changes_applied_to_recruitment": False,
            "persistent_enchantments_preserved_in_export": True}
        return snapshot

    def combat_request(self, *, pairings=DEFAULT_PAIRINGS, seed: int):
        self._check_scope()
        if self.phase != "awaiting_combat" or self.game.pending_choices:
            raise ValueError("A completed recruit phase without pending choices is required")
        if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF:
            raise ValueError("Combat seed must be an unsigned 32-bit integer")
        pairs = [list(pair) for pair in pairings]
        if (len(pairs) != 4 or any(len(p) != 2 for p in pairs)
                or any(type(i) is not int for p in pairs for i in p)
                or sorted(i for p in pairs for i in p) != list(range(8))):
            raise ValueError("Exactly four disjoint pairings must cover all eight players")
        request = {"scope": TRANSITION_SCOPE, "receipt_version": RECEIPT_VERSION,
                   "turn": self.game.turn, "seed": seed, "pairings": pairs,
                   "snapshots": [self.combat_snapshot(i) for i in range(8)]}
        self._pending_request = deepcopy(request)
        return request

    def advance_combat(self, receipt):
        """Apply one complete sampled phase; never apply Monte Carlo means.

        Receipt checks finish before state mutation. A later engine frontier
        marks the rollout invalid; callers must discard it rather than retry.
        """
        self._check_scope()
        request = self._pending_request
        if self.phase != "awaiting_combat" or request is None:
            raise ValueError("Create a combat request before submitting its receipt")
        if (receipt.get("scope") != TRANSITION_SCOPE or
                receipt.get("receipt_version") != RECEIPT_VERSION or
                receipt.get("request_sha256") != canonical_hash(request) or
                receipt.get("turn") != self.game.turn or receipt.get("seed") != request["seed"]):
            raise ValueError("Stale or mismatched sampled combat receipt")
        outcomes = receipt.get("outcomes")
        if not isinstance(outcomes, list) or len(outcomes) != 4:
            raise ValueError("Exactly four sampled combat outcomes required")
        for outcome, (a, b) in zip(outcomes, request["pairings"]):
            if (outcome.get("player_a") != a or outcome.get("player_b") != b
                    or outcome.get("samples") != 1):
                raise ValueError("Outcome must match its pair and contain exactly one sample")
            winner = outcome.get("winner_id")
            da, db = outcome.get("damage_to_a"), outcome.get("damage_to_b")
            raw = outcome.get("uncapped_damage")
            if (type(da) is not int or type(db) is not int or type(raw) is not int
                    or not 0 <= da <= 5 or not 0 <= db <= 5 or not 0 <= raw <= 13):
                raise ValueError("Sampled opening damage must be exact integers with the Solo cap")
            if winner is None:
                valid = da == db == raw == 0
            elif type(winner) is int and winner == a:
                valid = da == 0 and db == min(raw, 5) and raw >= 2
            elif type(winner) is int and winner == b:
                valid = db == 0 and da == min(raw, 5) and raw >= 2
            else:
                valid = False
            if not valid:
                raise ValueError("Winner and damage disagree")
            for i, damage in ((a, da), (b, db)):
                h = self.game.heroes[i]
                if h.health + h.armor <= damage:
                    raise UnsupportedRecruitTransition("Sample would cross the elimination frontier")
        if self.game.turn == 1 and any(h.tavern_tier != 1 for h in self.game.heroes):
            raise UnsupportedRecruitTransition("Tier2 next-turn shop requires the complete next-tier pool")
        try:
            for outcome in outcomes:
                a, b = outcome["player_a"], outcome["player_b"]
                winner = outcome["winner_id"]
                for i, other, key in ((a, b, "damage_to_a"), (b, a, "damage_to_b")):
                    hero = self.game.heroes[i]
                    hero._last_combat_result = ("draw" if winner is None else
                                                "win" if winner == i else "loss")
                    hero.take_damage(outcome[key])
                    self.game.combat_memory.setdefault(hero, []).append(self.game.heroes[other])
            saved = deepcopy(dict(receipt))
            self.combat_receipts.append(saved)
            self._timeline.append(("combat", {"request": deepcopy(request), "receipt": saved}))
            self._pending_request = None
            if self.game.turn == 2:
                self.phase = "complete"
                return None
            self.game.turn = 2
            for hero in self.game.heroes:
                self.game._begin_recruit_for(hero)
            self.finished_players.clear()
            self.player_id = 0
            self.terminal = False
            self.phase = "recruit"
            self._check_scope()
            return self.observe()
        except Exception:
            self._invalid = True
            raise

    def fork(self):
        if self._seed is None or self._invalid:
            raise ValueError("Reset a valid fixture before forking")
        branch = type(self)(self.db, self.ruleset, timer_ms=self.timer_ms,
                            provenance=self.provenance, fixture=self.fixture)
        branch.reset(seed=self._seed)
        for kind, item in self._timeline:
            if kind == "action":
                branch.step(item)
            else:
                request = item["request"]
                branch.combat_request(pairings=request["pairings"], seed=request["seed"])
                branch.advance_combat(item["receipt"])
        if self._pending_request is not None:
            branch.combat_request(pairings=self._pending_request["pairings"],
                                  seed=self._pending_request["seed"])
        return branch


class TimedTwoTurnOpeningEngine(TimedRecruitEngine):
    @classmethod
    def for_fixture(cls, engine, profile, ruleset):
        report = engine.coverage()
        if (not isinstance(engine, TwoTurnOpeningRecruitEngine)
                or report.get("scope") != TRANSITION_SCOPE
                or report.get("full_game_ready") is not False
                or report.get("ruleset_sha256") != ruleset_digest(ruleset)):
            raise RecruitCoverageError("Explicit matching two-turn opening fixture required")
        result = cls.__new__(cls)
        result._initialize(engine, profile)
        result._fixture_ruleset = dict(ruleset)
        return result

    def advance_combat(self, receipt):
        if self._visible is not None:
            raise ValueError("Finish all recruit decisions before applying combat")
        return self._accept(self.engine.advance_combat(receipt))
