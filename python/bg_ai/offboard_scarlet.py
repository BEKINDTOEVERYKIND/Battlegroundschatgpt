"""Opt-in hand-only Scarlet Survivor conformance; Tavern ambiguity stays closed.

The official 35.6 handbuff fix establishes the hand path. Client tag1724 alone
does not settle Tavern ownership, so it never authorizes that path here.
Existing trainer adapters and receipt workers are unchanged.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from types import MethodType

from .hsbrsim_adapter import UnsupportedRecruitTransition
from .hsbrsim_opening import OpeningFixtureSpec, TIER1_SPELLS, GENERATED_RECRUIT, ALWAYS_GOLDEN
from .opening_transition import TwoTurnOpeningRecruitEngine, _transition_game_type
from .recruiting import RecruitCoverageError, ruleset_digest
from .timed_recruiting import TimedRecruitEngine

SCARLET = "BG35_814"
VERSION = "current-hand-scarlet-opening-v1"


@dataclass(frozen=True)
class ScarletHandFixtureSpec(OpeningFixtureSpec):
    # Synthetic visible initial states for conformance. Each tuple identifies
    # (hand index, permanent attack buff, permanent health buff). These do not
    # assert that the old Tier-1 opening can itself generate a handbuff.
    initial_scarlet_hand_buffs: tuple = ((),) * 8

    def __post_init__(self):
        super().__post_init__()
        if len(self.initial_scarlet_hand_buffs) != 8:
            raise ValueError("Eight initial hand-buff lists are required")
        for hand, buffs in zip(self.initial_hands, self.initial_scarlet_hand_buffs):
            for record in buffs:
                if (len(record) != 3 or any(type(x) is not int for x in record)
                        or record[0] < 0 or record[0] >= len(hand)
                        or hand[record[0]] != SCARLET or min(record[1:]) < 0):
                    raise ValueError("Initial hand buffs must identify a current Scarlet and nonnegative stats")


def _hand_scarlet_game_type():
    from hsrl2.entity import Entity
    from hsrl2.actions.stats import GainKeyword
    from hsrl2.tags import GameTag, Zone

    def grant(source, game):
        if source.zone not in (Zone.HAND, Zone.PLAY) or getattr(source, "_scarlet_done", False):
            return
        if source.has(GameTag.SILENCED) or source.atk < game.db.get(SCARLET).num(0):
            return
        if game.in_combat:
            raise UnsupportedRecruitTransition("Native Scarlet combat persistence is outside this fixture")
        source._scarlet_done = True
        game.run_actions(GainKeyword(source, GameTag.DIVINE_SHIELD))

    class CurrentHandScarlet:
        @staticmethod
        def on_summon(source, game, ctx):
            grant(source, game)

    class HandScarletGame(_transition_game_type()):
        def create_minion(self, card_id, **kwargs):
            card = super().create_minion(card_id, **kwargs)
            if card.card_id == SCARLET:
                card._scarlet_done = False
                card.scripts = CurrentHandScarlet

                def add_buff(this, buff):
                    if (this.zone == Zone.TAVERN and not this._scarlet_done
                            and this.atk + buff.atk >= self.db.get(SCARLET).num(0)):
                        raise UnsupportedRecruitTransition(
                            "Tavern Scarlet threshold requires direct client conformance")
                    # Replace only the old off-board guard for this instance;
                    # preserve the real engine buff application and events.
                    Entity.add_buff(this, buff)
                    grant(this, self)
                card.add_buff = MethodType(add_buff, card)
            return card

    return HandScarletGame


class HandScarletOpeningEngine(TwoTurnOpeningRecruitEngine):
    def __init__(self, *args, fixture=None, **kwargs):
        super().__init__(*args, fixture=fixture or ScarletHandFixtureSpec(), **kwargs)
        d = self.db.get(SCARLET)
        expected = ("Once this reaches {0} Attack, gain <b>Divine Shield</b>.@"
                    "Once this reaches {0} Attack, gain <b>Divine Shield</b>. <i>(Done!)</i>")
        if d.num(0) != 6 or d.text != expected or d.raw.get("raw_tags", {}).get("1724") != 1:
            raise RecruitCoverageError("Current Scarlet text, threshold or wherever tag changed")
        self._Game = _hand_scarlet_game_type()

    def _initial_card(self, hero, cid, *, board):
        index = len(hero.hand)
        super()._initial_card(hero, cid, board=board)
        if not board and isinstance(self.fixture, ScarletHandFixtureSpec):
            from hsrl2.entity import Buff
            for hand_index, attack, health in self.fixture.initial_scarlet_hand_buffs[self.game.heroes.index(hero)]:
                if hand_index == index:
                    hero.hand[index].add_buff(Buff(attack, health, source_id="synthetic-initial-handbuff"))

    def coverage(self):
        report = super().coverage()
        report.update(scope=VERSION, hand_scarlet_threshold_supported=True,
            tavern_scarlet_threshold_supported=False,
            tavern_fugitive_spell_trigger_supported=False,
            production_training_enabled=False,
            combat_effect_transport=VERSION)
        return report

    def observe(self):
        observation = super().observe()
        if observation is None:
            return None
        self._last_observation = replace(observation,
            public_state=dict(observation.public_state, scope=VERSION))
        return self._last_observation

    def _card(self, card):
        row = super()._card(card)
        if card.card_id == SCARLET:
            row["scarlet_trigger_consumed"] = bool(getattr(card, "_scarlet_done", False))
        return row

    def _check_scope(self):
        # Same existing two-turn boundaries, with only the directly supported
        # Scarlet-in-hand branch admitted. Do not suppress arbitrary parent
        # exceptions or temporarily remove entities to fool its checks.
        if self.game is None:
            return
        if self._invalid:
            raise UnsupportedRecruitTransition("Hand Scarlet rollout already failed")
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
                    raise UnsupportedRecruitTransition("Unsupported opening card: " + card.card_id)
                if isinstance(card, self._Minion) and card.is_golden and card.card_id != ALWAYS_GOLDEN:
                    raise UnsupportedRecruitTransition("Ordinary goldens exceed the opening frontier")
                if card.card_id == SCARLET and card.atk >= self.db.get(SCARLET).num(0):
                    if card in hero.tavern:
                        raise UnsupportedRecruitTransition("Tavern Scarlet threshold requires direct client conformance")
                    if not getattr(card, "_scarlet_done", False):
                        raise UnsupportedRecruitTransition("Unrecorded Scarlet threshold crossing")
        if self.game.pending_hand_queue:
            raise UnsupportedRecruitTransition("Opening full-hand generated queue requires conformance")

    def combat_snapshot(self, player_id):
        snapshot = super().combat_snapshot(player_id)
        snapshot["effect_transport_version"] = VERSION
        return snapshot

    def combat_request(self, **kwargs):
        request = super().combat_request(**kwargs)
        request["effect_transport_version"] = VERSION
        self._pending_request = deepcopy(request)
        return request

    def advance_combat(self, receipt):
        if receipt.get("effect_transport_version") != VERSION:
            raise ValueError("A Scarlet-aware sampled receipt is required")
        return super().advance_combat(receipt)


class TimedHandScarletOpeningEngine(TimedRecruitEngine):
    @classmethod
    def for_fixture(cls, engine, profile, ruleset):
        report = engine.coverage()
        if (not isinstance(engine, HandScarletOpeningEngine) or report.get("scope") != VERSION
                or report.get("full_game_ready") is not False
                or report.get("ruleset_sha256") != ruleset_digest(ruleset)):
            raise RecruitCoverageError("Explicit current hand-Scarlet fixture required")
        result = cls.__new__(cls)
        result._initialize(engine, profile)
        result._fixture_ruleset = dict(ruleset)
        return result

    def advance_combat(self, receipt):
        if self._visible is not None:
            raise ValueError("Finish all recruit decisions before applying combat")
        return self._accept(self.engine.advance_combat(receipt))
