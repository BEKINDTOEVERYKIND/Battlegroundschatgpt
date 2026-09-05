"""Explicit, fail-closed recruitment curriculum on the pinned external engine.

This is a deliberately restricted two-turn fixture, not full-game Battlegrounds.
No source files from the external simulator are modified or vendored.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable, Mapping, Sequence

from bg_ai.recruiting import (RecruitAction, RecruitObservation,
                             RecruitCoverageError, ruleset_digest)

FIXTURE_SCOPE = "current-early-recruit-fixture-v1"
SUPPORTED_MINIONS = frozenset({"BGS_119", "BG25_001", "BG20_100", "BG26_135",
                              "BG23_002", "BG32_237", "BG36_345"})
SUPPORTED_SPELLS = frozenset({"BG28_810", "BG28_897", "BG28_503"})
SUPPORTED_GENERATED = frozenset({"BG20_GEM"})
SUPPORTED_HEROES = frozenset({"TB_BaconShop_HERO_34", "TB_BaconShop_HERO_39"})
_EMPTY_EIGHT = ((),) * 8


def _fixture_game_class(Game, Minion, Spell, GameTag, Zone):
    """Narrow compatibility fixes, only for this declared effect closure.

    Blizzard's 28.2 announcement specifies a dedicated spell slot and immediate
    pool return when a Tavern spell is obtained. The pinned upstream differs.
    """
    class FixtureGame(Game):
        def refresh_tavern(self, hero, *, auto=False, free=False, spells_only=False):
            from hsrl2.constants import TAVERN_OFFERS
            if spells_only:
                raise UnsupportedRecruitTransition("Spell-only refresh is outside the fixture")
            frozen = auto and hero.frozen_tavern
            if not auto:
                hero.clear(GameTag.FROZEN)
            super().refresh_tavern(hero, auto=auto, free=free, spells_only=False)
            if frozen:
                need = TAVERN_OFFERS[hero.tavern_tier] - sum(isinstance(x, Minion) for x in hero.tavern)
                for cid in self.minion_pool.draw_tavern(hero.tavern_tier, max(0, need)):
                    m = self.create_minion(cid, controller=hero)
                    m.zone = Zone.TAVERN
                    hero.tavern.append(m)

        def buy_from_tavern(self, hero, entity):
            result = super().buy_from_tavern(hero, entity)
            if result and isinstance(entity, Spell):
                self.spell_pool.release(entity.card_id)
            return result

        def _finish_play_spell(self, hero, spell, target, card_def=None, choose=None):
            # The supported Coin/Banana/Fortify/Gem effects never draw or return
            # pool spells. Restore this card's count after the old redundant
            # play-time return; generated copies also have no pool occupancy.
            if spell.card_id not in SUPPORTED_SPELLS | SUPPORTED_GENERATED:
                raise UnsupportedRecruitTransition("Unsupported spell pool transition")
            count = self.spell_pool.available(spell.card_id)
            result = super()._finish_play_spell(hero, spell, target, card_def, choose)
            if spell.card_id in SUPPORTED_SPELLS:
                self.spell_pool._available[spell.card_id] = count
            return result
    return FixtureGame


class UnsupportedRecruitTransition(RecruitCoverageError):
    """A fixture crossed its audited transition boundary; discard the rollout."""


@dataclass(frozen=True)
class EarlyRecruitFixtureSpec:
    """Synthetic setup and horizon, always recorded as curriculum assumptions."""
    max_turns: int = 2
    hero_ids: tuple[str, ...] = ("TB_BaconShop_HERO_34",) * 8
    initial_boards: tuple[tuple[str, ...], ...] = _EMPTY_EIGHT
    initial_hands: tuple[tuple[str, ...], ...] = _EMPTY_EIGHT

    def __post_init__(self):
        if type(self.max_turns) is not int or not 1 <= self.max_turns <= 2:
            raise ValueError("Fixture must stop before turn-three Dark Gifts")
        if len(self.hero_ids) != 8 or set(self.hero_ids) - SUPPORTED_HEROES:
            raise ValueError("Exactly eight supported fixture heroes are required")
        for groups, limit in ((self.initial_boards, 7), (self.initial_hands, 10)):
            if len(groups) != 8 or any(len(cards) > limit for cards in groups):
                raise ValueError("Fixture zones must contain eight legal-sized lists")
        if any(set(cards) - SUPPORTED_MINIONS for cards in self.initial_boards):
            raise ValueError("Unsupported initial board minion")
        allowed = SUPPORTED_MINIONS | SUPPORTED_SPELLS | SUPPORTED_GENERATED
        if any(set(cards) - allowed for cards in self.initial_hands):
            raise ValueError("Unsupported initial hand card")
        for board, hand in zip(self.initial_boards, self.initial_hands):
            all_cards = board + hand
            if any(all_cards.count(cid) >= 3 for cid in SUPPORTED_MINIONS):
                raise ValueError("Golden/triple transitions are outside this fixture")


class EarlyRecruitFixtureEngine:
    """Actual hsrl2 commands with one policy action per buy/play/choice/etc.

    Pass a current CardDB from ``build_current_database`` restricted to the
    declared fixture pools. Always wrap this object with
    ``TimedRecruitEngine.for_fixture`` before training or policy evaluation.
    ``timer_ms(player_id, turn)`` supplies remaining time; there is no implicit
    timer table or wall-clock speed assumption.
    """
    def __init__(self, db: Any, ruleset: Mapping[str, Any], *,
                 timer_ms: Callable[[int, int], int],
                 provenance: Mapping[str, Any],
                 fixture: EarlyRecruitFixtureSpec | None = None):
        from hsrl2.game import Game
        from hsrl2.hero import Hero
        from hsrl2.minion import Minion
        from hsrl2.spell import Spell
        from hsrl2.tags import GameTag, Zone
        self._Game = _fixture_game_class(Game, Minion, Spell, GameTag, Zone)
        self._Hero = Hero
        self._Minion, self._Spell = Minion, Spell
        self._T, self._Zone = GameTag, Zone
        self.db, self.ruleset = db, dict(ruleset)
        self.provenance = dict(provenance)
        self.timer_ms = timer_ms
        self.fixture = fixture or EarlyRecruitFixtureSpec()
        self._validate_database()
        self.game = None
        self.player_id = 0
        self.finished_players: set[int] = set()
        self.terminal = False
        self._seed: int | None = None
        self._history: list[RecruitAction] = []
        self._entity_ids: dict[str, str] = {}
        self._entities: dict[str, Any] = {}
        self._last_observation = None
        self._invalid = False

    def _validate_database(self):
        from bg_ai.recruiting import HSBRSIM_REVISION
        revision = self.provenance.get("engine_revision", self.provenance.get("revision"))
        if revision != HSBRSIM_REVISION:
            raise RecruitCoverageError("Fixture requires a provenance-verified pinned engine")
        minions = {d.id for d in self.db.pool_minions()}
        spells = {d.id for d in self.db.pool_spells()}
        if minions != SUPPORTED_MINIONS or spells != SUPPORTED_SPELLS:
            raise RecruitCoverageError("CardDB must use the exact declared restricted fixture pools")
        active = self.ruleset["active"]
        if minions - set(active.get("shop_minion_ids", active["minion_ids"])):
            raise RecruitCoverageError("Fixture contains a minion outside the current shop pool")
        if spells - set(active["tavern_spell_ids"]):
            raise RecruitCoverageError("Fixture contains a retired tavern spell")
        if set(self.fixture.hero_ids) - set(active["hero_ids"]):
            raise RecruitCoverageError("Fixture contains a retired hero")
        for cid in minions | spells | SUPPORTED_GENERATED:
            d = self.db.get(cid)
            if d is None:
                raise RecruitCoverageError(f"Missing current fixture definition: {cid}")
            if len(d.raw.get("races", [])) > 1:
                raise RecruitCoverageError("Dual-tribe engine conformance is unresolved")

    def coverage(self):
        return {"scope": FIXTURE_SCOPE, "fixture_only": True,
                "fixture_transition_contract": "explicit-actions-v1",
                "full_game_ready": False, "conformance_validated": False,
                "ruleset_sha256": ruleset_digest(self.ruleset),
                "blockers": ["restricted_curriculum_pool", "no_full_game_or_placements",
                             "golden_and_triple_transitions_unvalidated",
                             "current_seasonal_transitions_unvalidated",
                             "client_pending_choice_timeout_unvalidated"],
                "turn_timing": {"external_player_timer_required": True,
                                "pending_choice_timeout": "raise_and_abort"},
                "fixture": {"max_turns": self.fixture.max_turns,
                            "minion_ids": sorted(SUPPORTED_MINIONS),
                            "tavern_spell_ids": sorted(SUPPORTED_SPELLS),
                            "hero_ids": list(self.fixture.hero_ids)},
                "provenance": self.provenance}

    def reset(self, *, seed: int):
        self._seed = seed
        self._history.clear()
        self._entity_ids.clear()
        self._entities.clear()
        self.finished_players.clear()
        self.player_id, self.terminal, self._invalid = 0, False, False
        heroes = [self._Hero(cid, f"Player {i}", armor=self.db.get(cid).armor)
                  for i, cid in enumerate(self.fixture.hero_ids)]
        self.game = self._Game(heroes, self.db, seed=seed)
        self.game.choice_policy = None
        for i, hero in enumerate(heroes):
            for cid in self.fixture.initial_boards[i]:
                self._initial_card(hero, cid, board=True)
            for cid in self.fixture.initial_hands[i]:
                self._initial_card(hero, cid, board=False)
        self.game.start_game()
        self._check_scope()
        return self.observe()

    def _initial_card(self, hero, cid, *, board):
        if cid in SUPPORTED_MINIONS:
            if not self.game.minion_pool.acquire(cid):
                raise ValueError("Fixture initial minions exceed the shared pool")
            entity = self.game.create_minion(cid, controller=hero)
        else:
            # Already obtained Tavern spells are back in the shared pool.
            entity = self.game.create_spell(cid, controller=hero)
        # Initial fixture placement is a setup, not a free played battlecry.
        if board:
            self.game.summon(hero, entity)
        else:
            hero.add_to_hand(entity)

    def _id(self, entity):
        key = entity.uuid
        if key not in self._entity_ids:
            eid = f"entity-{len(self._entity_ids)}"
            self._entity_ids[key] = eid
            self._entities[eid] = entity
        return self._entity_ids[key]

    def _card(self, entity):
        d = self.db.get(entity.card_id)
        numbers = [d.num(i) for i in range(4)]
        resolved = re.sub(r"\{([0-3])\}", lambda match: str(numbers[int(match[1])])
                          if numbers[int(match[1])] is not None else match[0], d.text)
        row = {"entity_id": self._id(entity), "card_id": entity.card_id,
               "name": d.name, "text": resolved, "tier": d.tech_level,
               "script_data_numbers": numbers,
               "card_type": "minion" if isinstance(entity, self._Minion) else "spell",
               "cost": self._buy_cost(entity.controller, entity),
               "keywords": sorted(d.keywords)}
        if isinstance(entity, self._Minion):
            row.update(attack=entity.atk, health=entity.health,
                       max_health=entity.max_health, race=d.race.name,
                       divine_shield=entity.divine_shield, taunt=entity.taunt,
                       windfury=entity.windfury, reborn=entity.reborn,
                       golden=entity.is_golden,
                       activate_used=entity.has(self._T.ACTIVATE_USED_THIS_TURN))
        return row

    def _choice(self):
        if not self.game.pending_choices:
            return None
        choice = self.game.pending_choices[0]
        if choice.owner is not self.game.heroes[self.player_id]:
            raise UnsupportedRecruitTransition("Cross-player pending choice is not supported")
        return choice

    def _choice_option(self, option):
        if hasattr(option, "card_id"):
            return self._card(option)
        if isinstance(option, (str, int, float, bool)) or option is None:
            return option
        raise UnsupportedRecruitTransition("Unserializable or hidden choice payload")

    def _buy_cost(self, hero, entity):
        if isinstance(entity, self._Minion):
            return 3
        return max(0, entity.get(self._T.COST, 0) -
                   hero.get(self._T.NEXT_SPELL_COST_REDUCTION, 0))

    def legal_actions(self) -> tuple[RecruitAction, ...]:
        if self.terminal:
            return ()
        self._check_scope()
        hero = self.game.heroes[self.player_id]
        choice = self._choice()
        if choice is not None:
            return tuple(RecruitAction("choose", choice_index=i,
                                       target_id=self._id(option) if hasattr(option, "card_id") else None)
                         for i, option in enumerate(choice.options))
        actions = [RecruitAction("end_turn")]
        if not hero.hand_full():
            actions.extend(RecruitAction("buy", entity_id=self._id(card))
                           for card in hero.tavern if hero.gold >= self._buy_cost(hero, card))
        # The client sells battlefield minions; upstream also accepts hand sales,
        # but this adapter never exposes that engine shortcut.
        actions.extend(RecruitAction("sell", entity_id=self._id(m)) for m in hero.board)
        for card in hero.hand:
            if self.db.get(card.card_id).unplayable or card.has(self._T.LOCKED_IN_HAND):
                continue
            if isinstance(card, self._Minion):
                if not hero.board_full():
                    actions.extend(RecruitAction("play", entity_id=self._id(card), position=i)
                                   for i in range(len(hero.board) + 1))
            else:
                script = card.scripts
                # Targeted spells cannot be thrown away without a valid target.
                if getattr(script, "needs_target", False) and not self._spell_targets(hero):
                    continue
                actions.append(RecruitAction("cast_spell", entity_id=self._id(card)))
        if hero.gold >= 1 or hero.get(self._T.FREE_REFRESH_REMAINING, 0) > 0:
            actions.append(RecruitAction("refresh"))
        actions.append(RecruitAction("freeze"))  # toggles freeze/unfreeze
        if hero.tavern_tier < 2 and hero.gold >= hero.upgrade_cost:
            actions.append(RecruitAction("upgrade"))
        for m in hero.board:
            if (m.has(self._T.ACTIVATE) and not m.has(self._T.ACTIVATE_USED_THIS_TURN)
                    and hero.gold >= m.get(self._T.ACTIVATE_COST, 1)
                    and any(other is not m for other in hero.board)):
                actions.append(RecruitAction("activate", entity_id=self._id(m)))
            actions.extend(RecruitAction("move", entity_id=self._id(m), position=i)
                           for i in range(len(hero.board)) if hero.board.index(m) != i)
        from hsrl2.scripts import REGISTRY
        power = self.game.hero_power_def(hero)
        script = REGISTRY.get(power.id) if power else None
        if script is not None and not getattr(script, "passive", False):
            available = getattr(script, "available", lambda h, g: True)(hero, self.game)
            if (available and hero.get(self._T.HERO_POWER_USED_THIS_TURN, 0) <
                    getattr(script, "uses_per_turn", 1) and
                    hero.gold >= self.game.hero_power_cost(hero, power, script) and
                    not hero.hand_full() and any(isinstance(c, self._Minion) for c in hero.tavern)):
                actions.append(RecruitAction("hero_power"))
        return tuple(actions)

    def _spell_targets(self, hero):
        # These three verified targeted buffs say "a minion"; tavern minions
        # are eligible. Explicitly fix the upstream friendly-board-only default.
        return [m for m in hero.board + hero.tavern if isinstance(m, self._Minion) and not m.dead]

    def observe(self):
        if self.terminal:
            self._last_observation = None
            return None
        hero = self.game.heroes[self.player_id]
        remaining = self.timer_ms(self.player_id, self.game.turn)
        if type(remaining) is not int or remaining < 0:
            raise ValueError("Timer callback must supply nonnegative integer milliseconds")
        public = {"scope": FIXTURE_SCOPE, "players": [
            {"player_id": i, "hero_id": h.card_id, "health": h.health,
             "armor": h.armor, "tavern_tier": h.tavern_tier, "alive": h.is_alive}
            for i, h in enumerate(self.game.heroes)]}
        private = {"gold": hero.gold, "health": hero.health, "armor": hero.armor,
                   "tavern_tier": hero.tavern_tier,
                   "upgrade_cost": hero.upgrade_cost, "frozen": hero.frozen_tavern,
                   "board": [self._card(m) for m in hero.board],
                   "hand": [self._card(m) for m in hero.hand],
                   "shop": [self._card(m) for m in hero.tavern],
                   "counters": {name: hero.get(getattr(self._T, name), 0) for name in (
                       "GOLD_SPENT_THIS_TURN", "CARDS_PLAYED_THIS_TURN",
                       "TAVERN_SPELLS_CAST_THIS_GAME", "TAVERN_SPELL_EXTRA_ATK",
                       "TAVERN_SPELL_EXTRA_HEALTH", "HERO_POWER_USED_THIS_TURN")}}
        choice = self._choice()
        if choice:
            private["pending_choice"] = {"kind": choice.kind,
                                         "options": [self._choice_option(x) for x in choice.options]}
        self._last_observation = RecruitObservation(
            self.player_id, self.game.turn, public, private, self.legal_actions(),
            ruleset_digest(self.ruleset), time_remaining_ms=remaining)
        return self._last_observation

    def step(self, action: RecruitAction):
        if self._invalid:
            raise UnsupportedRecruitTransition("Rollout already failed; reset is required")
        if action not in self.legal_actions():
            raise ValueError("Action is not legal in this observation")
        try:
            result = self._apply(action)
            self._history.append(action)
            self._check_scope()
            return result
        except Exception:
            self._invalid = True
            raise

    def _apply(self, action):
        hero = self.game.heroes[self.player_id]
        entity = self._entities.get(action.entity_id) if action.entity_id else None
        success = True
        if action.kind == "end_turn":
            return self._end_player()
        if action.kind == "buy":
            if isinstance(entity, self._Minion):
                copies = sum(m.card_id == entity.card_id for m in hero.board + hero.hand)
                if copies >= 2:
                    raise UnsupportedRecruitTransition("Triple transition exceeds this curriculum")
            success = self.game.buy_from_tavern(hero, entity)
        elif action.kind == "sell":
            success = self.game.sell_minion(hero, entity)
        elif action.kind == "play":
            success = self.game.play_minion(hero, entity, position=action.position)
        elif action.kind == "cast_spell":
            if getattr(entity.scripts, "needs_target", False):
                from hsrl2.game import PendingChoice
                # Adapter owns the explicit target choice, then executes the real
                # upstream spell effect. Both commands are charged separately.
                self.game.pending_choices.append(PendingChoice(
                    hero, self._spell_targets(hero), "spell_target",
                    resolve_callback=lambda target: self.game.play_spell(hero, entity, target=target)))
            else:
                success = self.game.play_spell(hero, entity)
        elif action.kind == "choose":
            self.game.pending_choices.pop(0).choose(action.choice_index)
        elif action.kind == "refresh":
            hero.clear(self._T.FROZEN)
            self.game.refresh_tavern(hero)
        elif action.kind == "freeze":
            if hero.frozen_tavern:
                hero.clear(self._T.FROZEN)
            else:
                self.game.freeze_tavern(hero)
        elif action.kind == "upgrade":
            success = self.game.upgrade_tavern(hero)
        elif action.kind == "activate":
            success = self.game.use_activate(hero, entity)
        elif action.kind == "hero_power":
            success = self.game.use_hero_power(hero)
        elif action.kind == "move":
            hero.board.remove(entity)
            hero.board.insert(action.position, entity)
            self.game._update_positions(hero)
        else:
            raise UnsupportedRecruitTransition(f"Unimplemented command: {action.kind}")
        if success is False:
            raise RuntimeError("Enumerated legal action was rejected by the engine")
        self._check_scope()
        return self.observe()

    def _end_player(self):
        if self._choice() is not None:
            raise UnsupportedRecruitTransition("Pending-choice timeout is unverified; abort rollout")
        self.finished_players.add(self.player_id)
        remaining = [i for i, h in enumerate(self.game.heroes)
                     if h.is_alive and i not in self.finished_players]
        if remaining:
            self.player_id = remaining[0]
            return self.observe()
        if self.game.turn >= self.fixture.max_turns:
            self.terminal = True
            self._last_observation = None
            return None
        # Only after ALL eight finish, execute the upstream global phase once.
        self.game.end_recruit_phase()
        self.finished_players.clear()
        self.player_id = next(i for i, h in enumerate(self.game.heroes) if h.is_alive)
        self._check_scope()
        return self.observe()

    def expire_turn(self):
        # Timeout is not an automatic choice policy; no fake random/first pick.
        if self._choice() is not None:
            self._invalid = True
            raise UnsupportedRecruitTransition("Pending-choice timeout is unverified; abort rollout")
        return self.step(RecruitAction("end_turn"))

    def _check_scope(self):
        if self.game is None:
            return
        if self._invalid:
            raise UnsupportedRecruitTransition("Rollout already failed")
        if self.game.turn > 2 or self.game.dark_gift_state or self.game.dark_gift_audit_log:
            raise UnsupportedRecruitTransition("Seasonal transition exceeds the early fixture")
        allowed = SUPPORTED_MINIONS | SUPPORTED_SPELLS | SUPPORTED_GENERATED
        for hero in self.game.heroes:
            if hero.trinkets or getattr(hero, "quest_rewards", []):
                raise UnsupportedRecruitTransition("Seasonal effects are outside the fixture")
            for card in hero.board + hero.hand + hero.tavern:
                if card.card_id not in allowed:
                    raise UnsupportedRecruitTransition(f"Unsupported generated card: {card.card_id}")
                if isinstance(card, self._Minion) and card.is_golden:
                    raise UnsupportedRecruitTransition("Golden transition exceeds the fixture")
        if self.game.pending_hand_queue:
            raise UnsupportedRecruitTransition("Full-hand generated-card handling is outside the fixture")

    def terminal_boards(self):
        """Trusted evaluation output, unavailable until ALL fixture players finish."""
        if not self.terminal:
            raise ValueError("Fixture has not terminated")
        return [self.player_board(i) for i in range(8)]

    def player_board(self, player_id: int):
        """Trusted evaluator state; do not feed opposing private zones to policy."""
        h = self.game.heroes[player_id]
        return {"player_id": player_id, "hero_id": h.card_id,
                "tavern_tier": h.tavern_tier, "gold": h.gold,
                "board": [self._card(m) for m in h.board],
                "hand": [self._card(m) for m in h.hand]}

    def final_placements(self):
        raise RecruitCoverageError("Early recruitment fixtures do not produce final placements")

    def fork(self):
        """Replay into fresh callback closures; deepcopy of hsrl2 is unsafe."""
        if self._seed is None or self._invalid:
            raise ValueError("Reset a valid fixture before forking")
        branch = type(self)(self.db, self.ruleset, timer_ms=self.timer_ms,
                            provenance=self.provenance, fixture=self.fixture)
        branch.reset(seed=self._seed)
        for action in self._history:
            branch.step(action)
        return branch
