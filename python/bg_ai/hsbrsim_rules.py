"""Original, bounded repairs to ordinary triples and playable Triple Rewards.

Applies to one external hsrl2 Game instance, without editing vendor files or
claiming that all card effects / golden Battlecries are ready for self-play.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from types import MethodType
from typing import Any, Iterator

TRIPLE_REWARD = "TB_BaconShop_Triples_01"
CONSOLATION_POUCH = "BG27_Anomaly_574t2"


class UnsupportedTripleTransition(RuntimeError):
    """A merge needs state semantics outside the audited ordinary-triple scope."""


def grant_triple_reward(game: Any, hero: Any, *, reward_tier: int | None = None,
                        source: Any = None) -> Any:
    """Put a reward CARD into hand; creating it does not Discover or play it.

    Default tier is the player's current Tavern Tier + 1, capped at ordinary
    Solo Tier 6. Explicit tiers are for separately audited card/hero effects.
    The tier is frozen now, so later upgrades do not improve an existing card.
    """
    from hsrl2.tags import GameTag
    tier = min(hero.tavern_tier + 1, 6) if reward_tier is None else reward_tier
    if type(tier) is not int or not 1 <= tier <= 7:
        raise ValueError("Triple Reward tier must be an integer from 1 through 7")
    if game.db.get(TRIPLE_REWARD) is None:
        raise UnsupportedTripleTransition("Current Triple Reward definition is missing")
    spell = game.create_spell(TRIPLE_REWARD, controller=hero)
    spell.scripts = TripleRewardScript
    spell._bg_ai_triple_reward_tier = tier
    spell._bg_ai_reward_source = getattr(source, "card_id", source)
    spell.set(GameTag.TECH_LEVEL, tier)  # Explicitly available for observation.
    game.pending_hand_add(hero, spell)
    return spell


class ConsolationPouchScript:
    """The actual 3-Gold fallback spell requires its own later play action."""

    @staticmethod
    def on_play(source: Any, game: Any, ctx: Any) -> Any:
        from hsrl2.actions.economy import GainGold
        return GainGold(source.controller, 3) if source.controller is not None else None


class TripleRewardScript:
    """Manual Discover at the reward card's frozen tier, with real pool removal."""

    @staticmethod
    def on_play(source: Any, game: Any, ctx: Any) -> None:
        from hsrl2.game import PendingChoice
        hero = source.controller
        tier = getattr(source, "_bg_ai_triple_reward_tier", None)
        if hero is None or type(tier) is not int or not 1 <= tier <= 7:
            raise UnsupportedTripleTransition("Triple Reward is missing its frozen reward tier")
        options = sorted(game.minion_pool.candidates(max_tier=tier, min_tier=tier))
        if not options:
            if game.db.get(CONSOLATION_POUCH) is None:
                raise UnsupportedTripleTransition("Empty reward pool requires the current consolation pouch")
            pouch = game.create_spell(CONSOLATION_POUCH, controller=hero)
            pouch.scripts = ConsolationPouchScript
            game.pending_hand_add(hero, pouch)
            return
        picks = game.rng.sample(options, min(3, len(options)), label="current_triple_reward")
        resolved = False

        def resolve(card_id: str) -> None:
            nonlocal resolved
            if resolved:
                raise UnsupportedTripleTransition("Triple Reward choice already resolved")
            if card_id not in picks:
                raise UnsupportedTripleTransition("Triple Reward choice was not offered")
            if not game.minion_pool.acquire(card_id):
                raise UnsupportedTripleTransition("Offered Triple Reward minion is no longer available")
            resolved = True
            minion = game.create_minion(card_id, controller=hero)
            game.pending_hand_add(hero, minion)

        game.pending_choices.append(PendingChoice(hero, picks, "triple_reward", resolve_callback=resolve))


def _validate_participants(game: Any, hero: Any, base_def: Any,
                           participants: list[Any]) -> None:
    """Validate everything before removing one of the three existing entities."""
    from hsrl2.tags import GameTag
    ordinary_keys = {"uuid", "card_id", "game", "controller", "tags", "_buffs",
                     "scripts", "_script_overrides"}
    bonus_keywords = {GameTag.TAUNT, GameTag.DIVINE_SHIELD, GameTag.WINDFURY,
                      GameTag.REBORN, GameTag.POISONOUS, GameTag.VENOMOUS,
                      GameTag.STEALTH}
    allowed_tags = {
        GameTag.CARD_ID, GameTag.NAME, GameTag.CARDTYPE, GameTag.ZONE,
        GameTag.ZONE_POSITION, GameTag.BASE_ATK, GameTag.BASE_HEALTH,
        GameTag.HEALTH, GameTag.RACE, GameTag.TECH_LEVEL,
        GameTag.GEMS_PLAYED_ON,
    } | bonus_keywords
    from hsrl2.game import _KEYWORD_TAG_MAP
    allowed_tags.update(_KEYWORD_TAG_MAP[k] for k in base_def.keywords
                        if k in _KEYWORD_TAG_MAP)
    if base_def.spellcraft_id is not None:
        allowed_tags.add(GameTag.SPELLCRAFT)
    if base_def.avenge_target:
        allowed_tags.update({GameTag.AVENGE, GameTag.AVENGE_TARGET})
    if base_def.activate_cost is not None:
        allowed_tags.update({GameTag.ACTIVATE, GameTag.ACTIVATE_COST})
    for m in participants:
        if m.controller is not hero or m.dead:
            raise UnsupportedTripleTransition("Triple participant is not a living owned minion")
        if game._magnetic_stack.get(m.uuid) or m.has(GameTag.DARK_GIFT) or m._script_overrides:
            raise UnsupportedTripleTransition("Magnetic, Dark Gift or custom-hook triple merge is not audited")
        if set(vars(m)) - ordinary_keys:
            raise UnsupportedTripleTransition("Stateful per-minion counters need an explicit triple migration")
        if set(m.tags) - allowed_tags:
            raise UnsupportedTripleTransition("Triple includes unaudited persistent or temporary tags")
        if m.get(GameTag.BASE_ATK) != base_def.atk or m.get(GameTag.BASE_HEALTH) != base_def.health:
            raise UnsupportedTripleTransition("Set-stat/base-stat triple merge is not audited")
        if any(b.temporary or b.dark_gift for b in m._buffs):
            raise UnsupportedTripleTransition("Temporary or Dark Gift buff triple merge is not audited")
        if any(type(b.atk) is not int or type(b.health) is not int for b in m._buffs):
            raise UnsupportedTripleTransition("Triple requires numeric ordinary buff enchantments")


def combine_ordinary_triple(game: Any, hero: Any, acquired: Any) -> Any | None:
    """Combine exactly three same-card ordinary minions into one golden in hand."""
    from hsrl2.minion import Minion
    from hsrl2.tags import GameTag, Race, Zone
    if not isinstance(acquired, Minion):
        return None
    definition = game.db.get(acquired.card_id)
    if definition is None or acquired.is_golden or definition.is_golden_def:
        return None
    owned = [m for m in hero.hand + hero.board if isinstance(m, Minion)]
    copies = [m for m in owned if m.card_id == acquired.card_id and not m.is_golden]
    if hero.get(GameTag.TRIPLE_THRESHOLD, 3) != 3 or hero.get(GameTag.TRIPLE_REWARD_COINS, 0):
        if len(copies) >= hero.get(GameTag.TRIPLE_THRESHOLD, 3):
            raise UnsupportedTripleTransition("Alternative triple threshold/reward is not audited")
        return None
    if len(copies) < 3:
        if any(m.card_id == "BG26_175" for m in owned):
            elementals = [m for m in owned if m.race in (Race.ELEMENTAL, Race.ALL) and not m.is_golden]
            if len(elementals) >= 3:
                raise UnsupportedTripleTransition("Elemental of Surprise substitute triples are not audited")
        return None
    if not definition.is_pool_minion:
        raise UnsupportedTripleTransition("Generated-token triple rules need an explicit audit")
    golden_def = game.db.golden_version(definition)
    if golden_def is None:
        raise UnsupportedTripleTransition("Triple is missing its golden definition")
    participants = copies[:3]
    _validate_participants(game, hero, definition, participants)
    merged_buffs = deepcopy([b for m in participants for b in m._buffs])
    inherited = {tag for tag in (GameTag.TAUNT, GameTag.DIVINE_SHIELD, GameTag.WINDFURY,
                                 GameTag.REBORN, GameTag.POISONOUS, GameTag.VENOMOUS,
                                 GameTag.STEALTH) if any(m.has(tag) for m in participants)}
    gems = sum(m.get(GameTag.GEMS_PLAYED_ON, 0) for m in participants)
    golden = game.create_minion(golden_def.id, controller=hero, golden=True)
    # Transfer enchantments, not fresh buff applications: no artificial buff events.
    golden._buffs = merged_buffs
    for tag in inherited:
        golden.set(tag, True)
    if gems:
        golden.set(GameTag.GEMS_PLAYED_ON, gems)
    golden.set(GameTag.HEALTH, golden.max_health)
    golden.set(GameTag.TRIPLE_BASE_CARD_ID, definition.id)
    golden.set(GameTag.TRIPLE_REWARD_PENDING, True)
    for m in participants:
        if m in hero.hand:
            hero.hand.remove(m)
        if m in hero.board:
            hero.board.remove(m)
        game.events.unregister_owner(m)
        m.zone = Zone.REMOVED
    game._update_positions(hero)
    game.pending_hand_add(hero, golden)
    game.events.fire(game, "triple_combined", golden=golden)
    return golden


@dataclass(frozen=True)
class RecruitRuleInstallation:
    ordinary_triples: bool = True
    playable_triple_rewards: bool = True
    complex_triple_merges: bool = False
    golden_battlecry_play: bool = False
    full_game_ready: bool = False


@contextmanager
def installed_recruit_rules(game: Any) -> Iterator[RecruitRuleInstallation]:
    """Reversibly patch this Game only; keep active for the whole game lifetime."""
    from hsrl2.tags import GameTag
    for cid in (TRIPLE_REWARD, CONSOLATION_POUCH):
        if game.db.get(cid) is None:
            raise UnsupportedTripleTransition(f"Current generated definition missing: {cid}")
    names = ("check_for_triple", "_grant_triple_reward", "play_minion")
    missing = object()
    saved = {name: game.__dict__.get(name, missing) for name in names}
    original_play = game.play_minion

    def check(this: Any, hero: Any, acquired: Any) -> None:
        combine_ordinary_triple(this, hero, acquired)

    def reward(this: Any, hero: Any, golden: Any) -> None:
        if not golden.has(GameTag.TRIPLE_REWARD_PENDING):
            return
        grant_triple_reward(this, hero, source=golden)
        golden.clear(GameTag.TRIPLE_REWARD_PENDING)

    def play(this: Any, hero: Any, minion: Any, *args: Any, **kwargs: Any) -> bool:
        d = this.db.get(minion.card_id)
        migrated = (getattr(this, "_bg_ai_battlecry_dispatcher_version", None) == 1
                    and minion.card_id in getattr(this, "_bg_ai_migrated_battlecry_ids", ()))
        if d is not None and d.is_golden_def and not migrated and (
                "battlecry" in d.keywords
                or callable(getattr(minion.scripts, "battlecry", None))
                or getattr(minion.scripts, "choose_options", None)):
            raise UnsupportedTripleTransition("Golden Battlecry/Choose One play dispatcher is not yet migrated")
        return original_play(hero, minion, *args, **kwargs)

    game.check_for_triple = MethodType(check, game)
    game._grant_triple_reward = MethodType(reward, game)
    game.play_minion = MethodType(play, game)
    try:
        yield RecruitRuleInstallation()
    finally:
        for name, value in saved.items():
            if value is missing:
                delattr(game, name)
            else:
                setattr(game, name, value)
