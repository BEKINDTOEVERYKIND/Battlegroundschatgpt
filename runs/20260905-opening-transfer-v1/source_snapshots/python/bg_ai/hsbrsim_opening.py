"""Current complete Tier-1 pool with explicit first-turn recruitment commands.

A separate curriculum version; the previously trained seven-card fixture is
unchanged. Coverage of the pool is not a claim of full-game rules conformance.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import MethodType
from typing import Any, Mapping

from .hsbrsim_adapter import (EarlyRecruitFixtureEngine, UnsupportedRecruitTransition)
from .recruiting import RecruitAction, RecruitCoverageError, ruleset_digest, HSBRSIM_REVISION
from .timed_recruiting import TimedRecruitEngine

OPENING_SCOPE = "current-complete-tier1-opening-v1"
TIER1_MINIONS = frozenset({
    "BG20_100", "BG23_000", "BG25_001", "BG25_013", "BG26_135", "BG26_146",
    "BG28_300", "BG29_611", "BG29_888", "BG31_330", "BG31_803", "BG32_236",
    "BG32_330", "BG33_140", "BG33_886", "BG35_814", "BG36_200", "BG36_345",
    "BG36_921", "BGS_004", "BGS_119", "BGS_127",
})
TIER1_SPELLS = frozenset({"BG28_503", "BG28_504", "BG28_512", "BG28_810",
                          "BG28_897", "BG28_966", "BG31_880", "BG33_101"})
GENERATED_RECRUIT = frozenset({"BG20_GEM", "BG23_000t"})
COMBAT_TOKENS = frozenset({"BG_ICC_026t", "BG_BOT_312t", "BG28_603t", "BG36_200t"})
TRIBES = frozenset({"BEAST", "DEMON", "DRAGON", "ELEMENTAL", "MECH",
                   "MURLOC", "NAGA", "PIRATE", "QUILBOAR", "UNDEAD"})
TARGETED_SPELLS = frozenset({"BG28_503", "BG28_897", "BG31_880", "BG20_GEM", "BG23_000t"})
ALWAYS_GOLDEN = "BG32_236"
UNVERIFIED_SELF_PLAY = frozenset()
_EMPTY_EIGHT = ((),) * 8


@dataclass(frozen=True)
class OpeningFixtureSpec:
    valid_tribes: tuple[str, ...] = ("BEAST", "DEMON", "MECH", "NAGA", "PIRATE")
    max_turns: int = 1
    hero_ids: tuple[str, ...] = ("TB_BaconShop_HERO_34",) * 8
    initial_boards: tuple[tuple[str, ...], ...] = _EMPTY_EIGHT
    initial_hands: tuple[tuple[str, ...], ...] = _EMPTY_EIGHT

    def __post_init__(self):
        object.__setattr__(self, "valid_tribes", tuple("MECH" if t == "MECHANICAL" else t
                                                     for t in self.valid_tribes))
        if len(self.valid_tribes) != 5 or len(set(self.valid_tribes)) != 5 or set(self.valid_tribes) - TRIBES:
            raise ValueError("Exactly five distinct current lobby tribes are required")
        if self.max_turns != 1 or self.hero_ids != ("TB_BaconShop_HERO_34",) * 8:
            raise ValueError("Opening fixture supports one recruit turn and eight Patchwerk players")
        for groups, limit in ((self.initial_boards, 7), (self.initial_hands, 10)):
            if len(groups) != 8 or any(len(cards) > limit for cards in groups):
                raise ValueError("Opening fixture zones must contain eight legal-sized lists")
        if any(set(cards) - TIER1_MINIONS for cards in self.initial_boards):
            raise ValueError("Initial board contains a non-Tier1 current minion")
        if any(set(cards) - (TIER1_MINIONS | TIER1_SPELLS | GENERATED_RECRUIT) for cards in self.initial_hands):
            raise ValueError("Initial hand contains an unsupported opening card")


def lobby_minion_ids(definitions, valid_tribes):
    tribes = set(valid_tribes)
    return frozenset(d["id"] for d in definitions if d["id"] in TIER1_MINIONS
                     and (not d.get("races") or "ALL" in d["races"] or tribes.intersection(
                         "MECH" if r == "MECHANICAL" else r for r in d["races"])))


def build_opening_database(engine_root: Path, *, valid_tribes, **kwargs):
    """Exact current Tier1 pool, including a dual-tribe card if either tribe is in."""
    from .hsbrsim_data import export_current_definitions, build_current_database
    spec = OpeningFixtureSpec(valid_tribes=tuple(valid_tribes))
    exported = export_current_definitions(**kwargs)
    tier1 = {d["id"] for d in exported.definitions if d["is_pool_minion"] and d["tech_level"] == 1}
    spells = {d["id"] for d in exported.definitions if d["is_pool_spell"] and d["tech_level"] == 1}
    if tier1 != TIER1_MINIONS or spells != TIER1_SPELLS:
        raise RecruitCoverageError("The current Tier1 pool changed; opening conformance must be updated")
    return build_current_database(engine_root,
        fixture_minion_ids=lobby_minion_ids(exported.definitions, spec.valid_tribes),
        fixture_spell_ids=TIER1_SPELLS, **kwargs)


def _opening_game_type():
    from hsrl2.game import Game, PendingChoice
    from hsrl2.minion import Minion
    from hsrl2.spell import Spell
    from hsrl2.tags import GameTag, Zone

    class CurrentMoltenRock:
        @staticmethod
        def on_summon(source, game, ctx):
            from hsrl2.events import Listener
            from hsrl2.entity import Buff
            from hsrl2.tags import Race
            gain = game.db.get(source.card_id).num(1)
            if type(gain) is not int or gain <= 0:
                raise UnsupportedRecruitTransition("Missing current Molten Rock health parameter")
            def played(g, card=None, **kw):
                source.add_buff(Buff(health=gain, source_id=source.card_id))
            game.events.register(Listener(
                event="card_played", owner=source,
                condition=lambda card=None, **kw: isinstance(card, Minion)
                    and card is not source and card.controller is source.controller
                    and card.race in (Race.ELEMENTAL, Race.ALL), callback=played))

    class CurrentWrathWeaver:
        @staticmethod
        def on_summon(source, game, ctx):
            from hsrl2.events import Listener
            from hsrl2.entity import Buff
            from hsrl2.tags import Race
            d = game.db.get(source.card_id)
            attack, health = d.num(0), d.num(1)
            if any(type(n) is not int or n <= 0 for n in (attack, health)):
                raise UnsupportedRecruitTransition("Missing current Wrath Weaver buff parameters")
            def played(g, card=None, **kw):
                source.controller.take_damage(1)
                source.add_buff(Buff(atk=attack, health=health, source_id=source.card_id))
            game.events.register(Listener(
                event="card_played", owner=source,
                condition=lambda card=None, **kw: isinstance(card, Minion)
                    and card is not source and card.controller is source.controller
                    and card.race in (Race.DEMON, Race.ALL), callback=played))

    class OpeningGame(Game):
        def play_minion(self, hero, minion, *args, **kwargs):
            if minion.card_id in UNVERIFIED_SELF_PLAY:
                raise UnsupportedRecruitTransition(
                    f"Self-play trigger conformance is unverified: {minion.card_id}")
            return super().play_minion(hero, minion, *args, **kwargs)

        def create_minion(self, card_id, *, controller=None, golden=False):
            if card_id not in self._opening_allowed_minions:
                raise UnsupportedRecruitTransition(f"Generated minion exceeds opening pool: {card_id}")
            if golden and card_id != ALWAYS_GOLDEN:
                raise UnsupportedRecruitTransition("Ordinary goldens exceed the opening frontier")
            m = super().create_minion(card_id, controller=controller, golden=golden)
            # Always Golden is true in the Tavern as well as after acquisition.
            if card_id == ALWAYS_GOLDEN:
                m.set(GameTag.GOLDEN, True)
                m.set(GameTag.GILDED_NOT_TRIPLED, True)
                self._gilded_no_reward_uuids.add(m.uuid)
            if card_id == "BGS_127":
                # Independently tested historic simulator excludes the source
                # from this unchanged After-you-play trigger. Stats/health gain
                # are read from CURRENT definitions, never historic base values.
                m.scripts = CurrentMoltenRock
            if card_id == "BGS_004":
                # Public post-Demon-change raw client replay: a newly played
                # Weaver does not trigger itself; the previous Weaver does.
                m.scripts = CurrentWrathWeaver
            if card_id == "BG35_814":
                original_add = m.add_buff
                threshold = self.db.get(card_id).num(0)
                def add_guarded(this, buff):
                    if this.zone in (Zone.TAVERN, Zone.HAND) and this.atk + buff.atk >= threshold:
                        raise UnsupportedRecruitTransition("Off-board Scarlet threshold conformance is unverified")
                    return original_add(buff)
                m.add_buff = MethodType(add_guarded, m)
            return m

        def check_for_triple(self, hero, acquired):
            if acquired.card_id == ALWAYS_GOLDEN or acquired.is_golden:
                return
            copies = [m for m in hero.hand + hero.board
                      if isinstance(m, Minion) and m.card_id == acquired.card_id and not m.is_golden]
            if len(copies) >= 3:
                raise UnsupportedRecruitTransition("Triple/reward-tier frontier reached; discard rollout")

        def pending_hand_add(self, hero, entity):
            if isinstance(entity, Minion) and entity.card_id != ALWAYS_GOLDEN:
                if sum(m.card_id == entity.card_id for m in hero.hand + hero.board) >= 2:
                    raise UnsupportedRecruitTransition("Generated triple acquisition exceeds opening frontier")
            return super().pending_hand_add(hero, entity)

        def refresh_tavern(self, hero, *, auto=False, free=False, spells_only=False):
            if hero.tavern_tier != 1 or spells_only:
                raise UnsupportedRecruitTransition("Refreshing beyond Tier1 requires the full next-tier pool")
            if not auto:
                hero.clear(GameTag.FROZEN)
            # Only one turn, so no automatic frozen refill can occur here.
            return super().refresh_tavern(hero, auto=auto, free=free, spells_only=False)

        def buy_from_tavern(self, hero, entity):
            if isinstance(entity, Minion) and entity.card_id != ALWAYS_GOLDEN:
                if sum(m.card_id == entity.card_id for m in hero.hand + hero.board) >= 2:
                    raise UnsupportedRecruitTransition("Triple acquisition exceeds the opening frontier")
            result = super().buy_from_tavern(hero, entity)
            if result and isinstance(entity, Spell):
                self.spell_pool.release(entity.card_id)
            return result

        def _finish_play_spell(self, hero, spell, target, card_def=None, choose=None):
            if spell.card_id not in TIER1_SPELLS | GENERATED_RECRUIT:
                raise UnsupportedRecruitTransition("Unsupported opening spell")
            before = self.spell_pool.available(spell.card_id)
            result = super()._finish_play_spell(hero, spell, target, card_def, choose)
            # Current T1 effects never draw or release pool spells themselves.
            # Obtained spells already returned to their pool before being played.
            if spell.card_id in TIER1_SPELLS:
                self.spell_pool._available[spell.card_id] = before
            return result

        def attach_magnetic(self, hero, magnetic_minion, host):
            if magnetic_minion.card_id != "BG26_146":
                raise UnsupportedRecruitTransition("Only current opening Lullabot magnetic is supported")
            bonus = [tag for tag in (GameTag.TAUNT, GameTag.DIVINE_SHIELD,
                     GameTag.WINDFURY, GameTag.REBORN, GameTag.VENOMOUS,
                     GameTag.POISONOUS, GameTag.STEALTH) if magnetic_minion.has(tag)]
            super().attach_magnetic(hero, magnetic_minion, host)
            for tag in bonus:
                host.set(tag, True)

        def sell_minion(self, hero, minion):
            attached = list(self._magnetic_stack.get(minion.uuid, ()))
            result = super().sell_minion(hero, minion)
            if result:
                for cid in attached:
                    self.minion_pool.release(cid)
                self._magnetic_stack.pop(minion.uuid, None)
            return result

    return OpeningGame


class OpeningRecruitEngine(EarlyRecruitFixtureEngine):
    """Complete current Tier1 offering pool, explicit first-turn frontier guards."""
    def __init__(self, db, ruleset: Mapping[str, Any], *, timer_ms, provenance,
                 fixture: OpeningFixtureSpec | None = None):
        self.opening_spec = fixture or OpeningFixtureSpec()
        self.allowed_minions = lobby_minion_ids([d.raw for d in db._by_id.values()], self.opening_spec.valid_tribes)
        super().__init__(db, ruleset, timer_ms=timer_ms, provenance=provenance, fixture=self.opening_spec)
        self._Game = _opening_game_type()

    def _validate_database(self):
        if self.provenance.get("engine_revision") != HSBRSIM_REVISION or not self.provenance.get("engine_checkout_clean"):
            raise RecruitCoverageError("Pinned clean current-data provenance is required")
        if ({d.id for d in self.db.pool_minions()} != self.allowed_minions or
                {d.id for d in self.db.pool_spells()} != TIER1_SPELLS):
            raise RecruitCoverageError("Database must contain the complete five-tribe Tier1 opening pool")
        active = self.ruleset["active"]
        current_t1 = {cid for cid in active["minion_ids"] if self.db.get(cid).tech_level == 1}
        current_s1 = {cid for cid in active["tavern_spell_ids"] if self.db.get(cid).tech_level == 1}
        if current_t1 != TIER1_MINIONS or current_s1 != TIER1_SPELLS:
            raise RecruitCoverageError("Current opening pool changed")
        for groups in (self.fixture.initial_boards, self.fixture.initial_hands):
            if any(set(cards).intersection(TIER1_MINIONS) - self.allowed_minions for cards in groups):
                raise ValueError("Fixture initial cards contain a tribe absent from this lobby")
        for cid in GENERATED_RECRUIT | COMBAT_TOKENS | {"BG31_880t", "BG31_880t2"}:
            if self.db.get(cid) is None:
                raise RecruitCoverageError(f"Missing current generated opening definition: {cid}")

    def coverage(self):
        return {"scope": OPENING_SCOPE, "fixture_only": True, "full_game_ready": False,
                "full_opening_rules_ready": False, "conformance_validated": False,
                "ruleset_sha256": ruleset_digest(self.ruleset),
                "current_tier1_minions": len(TIER1_MINIONS), "current_tier1_spells": len(TIER1_SPELLS),
                "tested_playable_tier1_minions": len(TIER1_MINIONS - UNVERIFIED_SELF_PLAY),
                "blocked_play_minion_ids": sorted(UNVERIFIED_SELF_PLAY),
                "lobby_minion_count": len(self.allowed_minions),
                "valid_tribes": list(self.fixture.valid_tribes),
                "lobby_minion_ids": sorted(self.allowed_minions),
                "tavern_spell_ids": sorted(TIER1_SPELLS),
                "generated_recruit_ids": sorted(GENERATED_RECRUIT),
                "combat_token_ids": sorted(COMBAT_TOKENS),
                "blockers": ["ordinary_triples_and_next_tier_reward_pool",
                             "refresh_after_tier1", "pending_choice_timeout_unverified",
                             "full_hand_generated_card_queue", "later_turns_and_seasonal_effects",
                             "shop_threshold_conformance_unverified",
                             "tavern_fugitive_spell_trigger_unverified"],
                "provenance": self.provenance}

    def reset(self, *, seed):
        # The base reset constructs its Game before generating entities. Supply
        # the lobby pool on the Game class so factories fail closed immediately.
        self._Game._opening_allowed_minions = self.allowed_minions
        return super().reset(seed=seed)

    def _initial_card(self, hero, cid, *, board):
        if cid in self.allowed_minions:
            if not self.game.minion_pool.acquire(cid):
                raise ValueError("Initial fixture exceeds minion pool")
            entity = self.game.create_minion(cid, controller=hero)
        else:
            entity = self.game.create_spell(cid, controller=hero)
        if board:
            self.game.summon(hero, entity)
        else:
            hero.add_to_hand(entity)
        self.game.check_for_triple(hero, entity) if isinstance(entity, self._Minion) else None

    def _card(self, entity):
        row = super()._card(entity)
        row["races"] = ["MECH" if r == "MECHANICAL" else r
                        for r in self.db.get(entity.card_id).raw.get("races", [])]
        row["temporary_spellcraft"] = entity.has(self._T.SPELLCRAFT) and isinstance(entity, self._Spell)
        row["magnetic_attachments"] = list(self.game._magnetic_stack.get(entity.uuid, ()))
        row["enchantments"] = [{"attack": b.atk, "health": b.health,
                                 "temporary": b.temporary, "source_id": b.source_id,
                                 "blood_gem": b.gem} for b in entity._buffs]
        return row

    def _choice_option(self, option):
        if isinstance(option, str) and self.db.get(option) is not None:
            d = self.db.get(option)
            import re
            text = re.sub(r"\{([0-3])\}", lambda m: str(d.num(int(m[1])))
                          if d.num(int(m[1])) is not None else m[0], d.text)
            return {"card_id": option, "name": d.name, "text": text,
                    "attack": d.atk, "health": d.health, "cost": d.cost,
                    "tier": d.tech_level, "races": list(d.raw.get("races", []))}
        return super()._choice_option(option)

    def observe(self):
        observation = super().observe()
        if observation is None:
            return None
        public = dict(observation.public_state, scope=OPENING_SCOPE,
                      valid_tribes=list(self.fixture.valid_tribes))
        private = dict(observation.private_state,
            deferred_next_turn_gold=sum(1 for h, _ in self.game.deferred_actions
                                        if h is self.game.heroes[self.player_id]))
        self._last_observation = replace(observation, public_state=public, private_state=private)
        return self._last_observation

    def legal_actions(self):
        actions = list(super().legal_actions())
        if self.terminal or self.game.pending_choices:
            return tuple(actions)
        hero = self.game.heroes[self.player_id]
        # Upstream Mini-Myrmidon omits needs_target: provide the real target step.
        if not self._spell_targets(hero):
            actions = [a for a in actions if not (a.kind == "cast_spell" and
                       self._entities[a.entity_id].card_id in TARGETED_SPELLS)]
        for m in hero.hand:
            if m.card_id == "BG26_146":
                for host in hero.board:
                    if self.game._magnetic_valid(m, host):
                        actions.append(RecruitAction("play", entity_id=self._id(m), target_id=self._id(host)))
        # Upgrading is real, but refreshing afterwards aborts at a declared
        # next-tier pool boundary rather than drawing an incomplete Tier2 shop.
        if 2 <= hero.tavern_tier < 6 and hero.gold >= hero.upgrade_cost:
            actions.append(RecruitAction("upgrade"))
        return tuple(actions)

    def _apply(self, action):
        hero = self.game.heroes[self.player_id]
        entity = self._entities.get(action.entity_id)
        if action.kind == "choose" and self._choice() is not None:
            choice = self._choice()
            if choice.kind == "spell_target":
                target = choice.options[action.choice_index]
                if getattr(target, "card_id", None) == "BG36_921" and target not in hero.board:
                    raise UnsupportedRecruitTransition("Tavern Fugitive spell-trigger conformance is unverified")
        if action.kind == "buy":
            if not self.game.buy_from_tavern(hero, entity):
                raise RuntimeError("Opening legal purchase was rejected")
            self._check_scope()
            return self.observe()
        if action.kind == "play" and action.target_id is not None:
            host = self._entities[action.target_id]
            if not self.game.play_minion(hero, entity, magnetic_target=host):
                raise RuntimeError("Opening legal magnetic play was rejected")
            self._check_scope()
            return self.observe()
        if action.kind == "cast_spell" and entity.card_id in TARGETED_SPELLS:
            from hsrl2.game import PendingChoice
            if entity.card_id == "BG31_880":
                # Choose One first, target second, cast/buff/events only after
                # both choices resolve. A rope expiry between either aborts.
                def choose_branch(branch):
                    self.game.pending_choices.append(PendingChoice(
                        hero, self._spell_targets(hero), "spell_target",
                        resolve_callback=lambda target: self._cast_alliance(entity, target, branch)))
                self.game.pending_choices.append(PendingChoice(
                    hero, ["BG31_880t", "BG31_880t2"], "choose_one", choose_branch))
            else:
                self.game.pending_choices.append(PendingChoice(
                    hero, self._spell_targets(hero), "spell_target",
                    resolve_callback=lambda target: self.game.play_spell(hero, entity, target=target)))
            return self.observe()
        return super()._apply(action)

    def _cast_alliance(self, spell, target, branch):
        from hsrl2.actions.racefx import tavern_spell_buff
        class ResolvedAlliance:
            @staticmethod
            def on_play(source, game, ctx):
                d = game.db.get(source.card_id)
                i = 0 if branch == "BG31_880t" else 2
                return tavern_spell_buff(ctx["target"], d.num(i), d.num(i+1))
        spell.scripts = ResolvedAlliance
        self.game._finish_play_spell(spell.controller, spell, target,
                                     card_def=self.db.get(spell.card_id))

    def _end_player(self):
        if self._choice() is not None:
            raise UnsupportedRecruitTransition("Pending-choice timeout is unverified; abort rollout")
        self.finished_players.add(self.player_id)
        remaining = [i for i in range(8) if i not in self.finished_players]
        if remaining:
            self.player_id = remaining[0]
            return self.observe()
        # Complete the genuine opening end-of-turn effects before exporting a
        # combat snapshot. Only Lullabot has an EOT hook in the current T1 pool.
        self.game.events.fire(self.game, "turn_end", turn=1)
        for hero in self.game.heroes:
            for minion in list(hero.board):
                self.game.run_script_hook(minion, "end_of_turn")
                self.game.check_deaths()
            for card in list(hero.hand):
                if isinstance(card, self._Spell) and card.has(self._T.SPELLCRAFT):
                    hero.hand.remove(card)
                    card.zone = self._Zone.REMOVED
        self._check_scope()
        self.terminal = True
        self._last_observation = None
        return None

    def _check_scope(self):
        if self.game is None:
            return
        if self._invalid:
            raise UnsupportedRecruitTransition("Opening rollout already failed")
        if self.game.turn != 1 or self.game.dark_gift_state or self.game.dark_gift_audit_log:
            raise UnsupportedRecruitTransition("Opening seasonal/turn frontier reached")
        allowed = self.allowed_minions | TIER1_SPELLS | GENERATED_RECRUIT
        for hero in self.game.heroes:
            if hero.trinkets or getattr(hero, "quest_rewards", []):
                raise UnsupportedRecruitTransition("Unexpected opening seasonal state")
            for card in hero.board + hero.hand + hero.tavern:
                if card.card_id not in allowed:
                    raise UnsupportedRecruitTransition(f"Unsupported opening card: {card.card_id}")
                if isinstance(card, self._Minion) and card.is_golden and card.card_id != ALWAYS_GOLDEN:
                    raise UnsupportedRecruitTransition("Ordinary golden reached opening frontier")
                if card.card_id == "BG35_814" and card not in hero.board:
                    threshold = self.db.get(card.card_id).num(0)
                    if card.atk >= threshold:
                        raise UnsupportedRecruitTransition("Off-board Scarlet threshold conformance is unverified")
        if self.game.pending_hand_queue:
            raise UnsupportedRecruitTransition("Opening generated full-hand queue requires conformance")

    def combat_snapshot(self, player_id):
        """Complete trusted precombat payload, including actual private hand.

        Later engines must apply combat-to-recruitment persistence separately;
        this first-combat evaluator is not a source of next-turn state.
        """
        if not self.terminal:
            raise ValueError("End every player's recruit turn before taking a combat snapshot")
        hero = self.game.heroes[player_id]
        return {"scope": OPENING_SCOPE, "player_id": player_id,
                "heroId": hero.card_id, "turn": 1, "tavernTier": hero.tavern_tier,
                "validTribes": list(self.fixture.valid_tribes),
                "board": [self._card(m) for m in hero.board],
                "hand": [self._card(m) for m in hero.hand],
                "health": hero.health, "armor": hero.armor,
                "ruleset_sha256": ruleset_digest(self.ruleset),
                "reference_cards_sha256": self.provenance.get("reference_cards_sha256"),
                "recruit_return_state": {"next_turn_state_available": False,
                    "persistent_enchantments_preserved_in_export": True,
                    "combat_changes_applied_to_recruitment": False}}


class TimedOpeningEngine(TimedRecruitEngine):
    @classmethod
    def for_fixture(cls, engine, profile, ruleset):
        report = engine.coverage()
        if (not isinstance(engine, OpeningRecruitEngine) or report.get("scope") != OPENING_SCOPE
                or report.get("full_game_ready") is not False
                or report.get("ruleset_sha256") != ruleset_digest(ruleset)):
            raise RecruitCoverageError("Explicit matching current opening fixture required")
        result = cls.__new__(cls)
        result._initialize(engine, profile)
        result._fixture_ruleset = dict(ruleset)
        return result
