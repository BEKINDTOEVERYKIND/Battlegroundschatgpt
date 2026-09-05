"""Audited current-card effects for an external, pinned ``hsrl2`` checkout.

No upstream source is copied or edited. Installation is explicit, process-local,
and reversible. These handlers close concrete effect gaps; they do not certify
all recruitment, generated-card, golden-battlecry, or combat behavior.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterator

PATCH = "36.4.2.251332"
DEFAULT_CARDS = Path(__file__).resolve().parents[2] / "data/reference_cards.json"
CORRUPTED_CUPCAKES = "BG28_607"
JAILBIRD_GOLEM = "BG30_MagicItem_442t"
DEFERRED_EFFECTS = {
    "BG33_319": "Rimescale Priestess: exact stat-giving spell generation pool unverified.",
    "BG33_319_G": "Rimescale Priestess: exact stat-giving spell generation pool unverified.",
    "BG33_319t": "Rime or Reason: exact stat-giving spell generation pool unverified.",
    "BG33_319_Gt": "Rime or Reason: exact stat-giving spell generation pool unverified.",
    "EBG_Spell_037": "Unmasked Identity: power eligibility and replacement lifecycle unverified.",
    "BG36_MidGameEffect_000t65": (
        "Polarization: random-Mech generation and arbitrary inherited effects require "
        "a correct magnetic source identity/lifecycle implementation."
    ),
}


class EffectValidationError(RuntimeError):
    """The loaded definitions do not match this audited implementation."""


def _number(source: Any, game: Any, index: int) -> int:
    definition = game.db.get(source.card_id)
    value = definition.num(index) if definition is not None else None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise EffectValidationError(f"{source.card_id}: missing positive num({index})")
    return value


class CurrentSoulRewinder:
    """Restore the actual pre-damage snapshot once; every Rewinder gets +2/+4 HP."""

    @staticmethod
    def on_summon(source: Any, game: Any, ctx: Any) -> None:
        from hsrl2.actions.stats import Buff
        from hsrl2.events import HERO_DAMAGE_TAKEN, Listener
        if source.controller is None:
            return
        gain = _number(source, game, 2)

        def damaged(g: Any, hero: Any = None, amount: int = 0,
                    health_before: int | None = None,
                    armor_before: int | None = None, **kw: Any) -> None:
            if hero is not source.controller or amount <= 0:
                return
            if health_before is None or armor_before is None:
                raise EffectValidationError("Soul Rewinder requires per-instance pre-damage HP/armor")
            # Set, rather than heal: multiple rewinders cannot restore twice.
            hero.health = health_before
            hero.armor = armor_before
            g.run_actions(Buff(source, health=gain))

        game.events.register(Listener(HERO_DAMAGE_TAKEN, source, damaged))


class CurrentEredarEscapist:
    """Each four cumulative damage grants Cupcakes into hand; never auto-cast."""

    @staticmethod
    def on_summon(source: Any, game: Any, ctx: Any) -> None:
        from hsrl2.events import HERO_DAMAGE_TAKEN, Listener
        if source.controller is None:
            return
        threshold = _number(source, game, 1)
        copies = 2 if game.db.get(source.card_id).is_golden_def else 1
        definition = game.db.get(source.card_id)
        result = game.db.by_dbf(definition.evolution_card_id)
        if result is None or result.id != CORRUPTED_CUPCAKES:
            raise EffectValidationError("Eredar Escapist evolution must resolve to Corrupted Cupcakes")

        def damaged(g: Any, hero: Any = None, amount: int = 0, **kw: Any) -> None:
            if hero is not source.controller or amount <= 0:
                return
            total = getattr(source, "_escapist_dmg_taken", 0) + amount
            completed, remainder = divmod(total, threshold)
            source._escapist_dmg_taken = remainder
            for _ in range(completed * copies):
                # A generated copy is distinct from buying/removing a shop offer.
                g.pending_hand_add(hero, g.create_spell(CORRUPTED_CUPCAKES, controller=hero))

        game.events.register(Listener(HERO_DAMAGE_TAKEN, source, damaged))


class CurrentJailbirdJuggernaut:
    """Copy current attack and maximum Health, including all buffs, into a Golem."""

    @staticmethod
    def rally(source: Any, game: Any, ctx: Any) -> Any:
        from hsrl2.combat import execute_immediate_attack
        from hsrl2.queue import Action
        from hsrl2.tags import GameTag, Zone
        target = (ctx or {}).get("target")
        hero = source.controller
        if hero is None or target is None:
            return None
        mult = 2 if game.db.get(source.card_id).is_golden_def else 1
        attack = max(0, source.atk) * mult
        health = max(0, source.max_health) * mult

        class SummonAndAttack(Action):
            def do(self, g: Any) -> None:
                token = g.create_minion(JAILBIRD_GOLEM, controller=hero)
                token.set(GameTag.BASE_ATK, attack)
                token.set(GameTag.BASE_HEALTH, health)
                token.set(GameTag.HEALTH, health)
                position = hero.board.index(source) + 1 if source in hero.board else None
                g.summon(hero, token, position)
                if not token.dead and token.zone == Zone.PLAY and token.atk > 0:
                    execute_immediate_attack(g, token, target)

        return SummonAndAttack()


class CurrentSanguineChampion:
    """One effect trigger improves gems +2/+1, or +4/+2 for a golden source.

    Upstream's generic golden-play path incorrectly calls golden battlecries
    twice. The adapter must reject that path unless the separate scoped
    installed_battlecry_rules dispatcher is active. This handler itself never halves
    a golden effect to hide a dispatcher error.
    """

    @staticmethod
    def battlecry(source: Any, game: Any, ctx: Any) -> Any:
        from hsrl2.actions.bloodgem import ImproveBloodGems
        if source.controller is None:
            return None
        return ImproveBloodGems(source.controller, _number(source, game, 0),
                                _number(source, game, 1))

    @staticmethod
    def deathrattle(source: Any, game: Any, ctx: Any) -> None:
        action = CurrentSanguineChampion.battlecry(source, game, ctx)
        if action is not None:
            action.do(game)


EFFECT_OVERRIDES = {
    "BG26_174": CurrentSoulRewinder,
    "BG26_174_G": CurrentSoulRewinder,
    "BG36_733": CurrentEredarEscapist,
    "BG36_733_G": CurrentEredarEscapist,
    "BG36_333": CurrentJailbirdJuggernaut,
    "BG36_333_G": CurrentJailbirdJuggernaut,
    "BG23_017": CurrentSanguineChampion,
    "BG23_017_G": CurrentSanguineChampion,
}


@dataclass(frozen=True)
class EffectInstallation:
    patch: str
    registered_ids: tuple[str, ...]
    full_game_ready: bool = False
    golden_battlecry_play_ready: bool = False


def validate_effect_definitions(db: Any, cards_path: Path | str = DEFAULT_CARDS) -> None:
    """Fail before mutating the registry if old or incompatible data was loaded."""
    cards = {c["id"]: c for c in json.loads(Path(cards_path).read_text())}
    for card_id in (*EFFECT_OVERRIDES, CORRUPTED_CUPCAKES, JAILBIRD_GOLEM):
        reference = cards.get(card_id)
        d = db.get(card_id)
        if reference is None or d is None:
            raise EffectValidationError(f"Missing current definition: {card_id}")
        if reference.get("snapshotBuild") != 251332:
            raise EffectValidationError(f"{card_id}: source build changed; re-audit effects")
        if d.text != reference.get("text", ""):
            raise EffectValidationError(f"{card_id}: effect text differs from the current snapshot")
        tags = reference.get("tags", {})
        for i in range(4):
            if d.num(i) != tags.get(f"TAG_SCRIPT_DATA_NUM_{i + 1}"):
                raise EffectValidationError(f"{card_id}: stale or missing script parameter {i + 1}")
        if d.atk != reference.get("attack", 0) or d.health != reference.get("health", 0):
            raise EffectValidationError(f"{card_id}: stale base stats")
        if d.evolution_card_id != tags.get("BACON_EVOLUTION_CARD_ID"):
            raise EffectValidationError(f"{card_id}: stale generated-card chain")


@contextmanager
def installed_current_effects(db: Any, *, cards_path: Path | str = DEFAULT_CARDS
                              ) -> Iterator[EffectInstallation]:
    """Install for the lifetime of newly created games, then restore exactly.

    Existing entities retain their original bound script class. Never enter or
    leave this context midway through a live game. Registry mutation is global
    inside one process: create separate worker processes for differing rulesets.
    """
    from hsrl2.scripts import REGISTRY
    validate_effect_definitions(db, cards_path)
    absent = object()
    old = {cid: REGISTRY.get(cid, absent) for cid in EFFECT_OVERRIDES}
    REGISTRY.update(EFFECT_OVERRIDES)
    try:
        yield EffectInstallation(PATCH, tuple(sorted(EFFECT_OVERRIDES)))
    finally:
        for cid, value in old.items():
            if value is absent:
                REGISTRY.pop(cid, None)
            else:
                REGISTRY[cid] = value
