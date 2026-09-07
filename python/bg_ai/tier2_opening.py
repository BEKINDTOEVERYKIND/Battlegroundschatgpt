"""Versioned Tier-2 inventory and a scoped Forest Rover recruitment repair.

This is not a Tier-2 training adapter. Every current Tier-2 card remains in the
inventory; no difficult card is removed to manufacture a complete shop pool.
The mixin is additive and never changes the pinned upstream registry or files.
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from types import MethodType
from typing import Any

VERSION = "current-tier2-frontier-v1"
ROVERS = frozenset({"BG31_801", "BG31_801_G"})
BEETLES = frozenset({"BG28_603t", "BG28_603t_G"})

# These are missing integration/conformance requirements, not proof that the
# registered upstream handler is wrong in every context.
BLOCKERS = {
    "BG20_101": "Sampled combat-generated Blood Gems must return to the actual hand.",
    "BG21_015": "Exact permanent combat stat/keyword delta writeback and golden doubling.",
    "BG22_202": "Random Murloc acquisition needs its complete current generation pool and all generated effects.",
    "BG23_002": "Existing migrated Battlecry must be integrated with the complete Tier-2 adapter.",
    "BG23_009": "Spellcraft permanence, per-turn counters, hand-cast identity and magnetic lifetime integration.",
    "BG23_357": "Targeted consume, shared-pool return, dual-type Demon matching and consume listeners.",
    "BG24_715": "Upstream derives Discover tier from global game.turn, not this Scout's upgrades; entity-local age and golden merge required.",
    "BG25_008": "Sampled combat deaths must update the permanent family counter, including reborn copies.",
    "BG25_011": "Permanent Undead aura export, dual-type matching and intrinsic golden Battlecry migration.",
    "BG25_022": "Complete Tier-2 combat closure, reborn and random friendly Undead buffs.",
    "BG26_174": "Existing current Soul Rewinder repair must compose with recruit self-damage and combat damage boundaries.",
    "BG26_805": "Combat-only Beast aura export and complete Tier-2 combat conformance.",
    "BG26_963": "Intrinsic golden Battlecry migration, current Dragon matching and combat-only Start of Combat gains.",
    "BG27_002": "Generated Slimy Shields, explicit hand play/target commands and spell-buff modifiers.",
    "BG28_518": "Complete current same-type generation pool, source exclusion, dual types and pool conservation.",
    "BG28_571": "Health-based purchase must charge health rather than gold and compose with all damage listeners.",
    "BG28_805": "Upstream changes only the eventual income cap; current maximum-gold progression requires explicit early-turn conformance.",
    "BG28_827": "Free refresh counter must be visible and consumed without waiving refresh action time.",
    "BG29_300": "Every sampled damage trigger must persist the exact selected hand-card buff; aggregate combat output is insufficient.",
    "BG29_810": "Combat keyword/stat export, left-most Dragon targeting and Tarecgosa writeback interaction.",
    "BG31_177": "Play versus magnetic single-trigger conformance, all current Mech types, source identity and permanent attachment.",
    "BG31_320": "Choose One routing once per play, explicit Blood Gem/Gem Day hand commands and current generation effects.",
    "BG31_801": "Beetle-only recruitment repair supplied here; Firestone Beetle global counters still need export and persistent receipt integration.",
    "BG31_816": "Exact shared Baller improvement state, golden contributions, sell listeners and public feature transport.",
    "BG31_818": "Exact shared Baller improvement state, golden contributions, sell listeners and public feature transport.",
    "BG31_924": "All-spell cumulative counter, generated Spellcraft strength, expiry and Lava Lurker permanence.",
    "BG32_170": "Combat-generated Pointy Arrows must persist into the actual hand with correct queue/overflow semantics.",
    "BG32_235": "Current end-turn adjacency, Golden count, repeat modifiers and contextual text branch.",
    "BG32_237": "Existing Choose One repair plus spell modifiers must compose with every current Tier-2 Tavern spell.",
    "BG33_430": "Combat Blood Gem counters/bonuses and exact attack-trigger ordering in complete Tier-2 combat closure.",
    "BG34_140": "Exact hand snapshot, highest-Attack tie sampling, summon counters and removal of combat-only copies.",
    "BG34_330": "Complete same-tier Discover pool, explicit choice action and locked-card expiration lifecycle.",
    "BG35_150": "Fodder additions to exactly the next refreshes, shared-pool identities, shop capacity and current golden Battlecry.",
    "BG35_951": "Four distinct/random friendly targets, current +1/+2 parameters and Tavern-spell modifiers.",
    "BG36_201": "Activate target selection, Fishbait replacement and recruit-phase attack need full damage/death/persistence conformance.",
    "BG36_342": "Activate/Discover with complete current legal Tavern-spell pool and separately timed choice.",
    "BG36_354": "Highest-Attack shop tie, steal acquisition/triples and pool conservation with a separately charged Activate.",
    "BG36_520": "Current Lockbox countdown, existing-box acceleration, full reward pool and pending hand lifecycle.",
    "BG36_883": "Exact next-combat winner event and next-turn target-bound buff; outcome receipt alone does not fire this lifecycle.",
    "BGS_115": "Current 3/3 Water Droplet generation, explicit hand play and golden acquisition/triple behavior.",
    "BG_TTN_401": "Permanent summon-family counter, combat-only summons, copy identity, dynamic stats and snapshot normalization.",
}


class Tier2ConformanceError(RuntimeError):
    pass


def validate_rover_definitions(db: Any) -> None:
    """Refuse stale numeric/text meanings before constructing the additive game."""
    from .hsbrsim_battlecries import _plain
    for cid, params, count in (("BG31_801", (2, 2, 2, 1), "a"),
                               ("BG31_801_G", (2, 2, 4, 2), "two")):
        d = db.get(cid)
        expected = ("Battlecry: Your Beetles have +{2}/+{3} this game. "
                    f"Deathrattle: Summon {count} {{0}}/{{1}} Beetle" +
                    ("s." if count == "two" else "."))
        if d is None or tuple(d.num(i) for i in range(4)) != params or _plain(d.text) != expected:
            raise Tier2ConformanceError("Forest Rover definition changed: " + cid)
    for cid, stats in (("BG28_603t", (2, 2)), ("BG28_603t_G", (4, 4))):
        d = db.get(cid)
        if d is None or d.name != "Beetle" or (d.atk, d.health) != stats or d.text:
            raise Tier2ConformanceError("Beetle definition changed: " + cid)


def beetle_bonus(hero: Any) -> tuple[int, int]:
    return getattr(hero, "_bg_ai_beetle_bonus_v1", (0, 0))


class CurrentForestRover:
    @staticmethod
    def battlecry(source: Any, game: Any, ctx: Any) -> None:
        from hsrl2.tags import GameTag
        if getattr(game, "effect_version", None) != VERSION:
            raise Tier2ConformanceError("Forest Rover requires the matching Beetle-aware entity factory")
        d = game.db.get(source.card_id)
        hero = source.controller
        attack, health = d.num(2), d.num(3)
        old_attack, old_health = beetle_bonus(hero)
        hero._bg_ai_beetle_bonus_v1 = (old_attack + attack, old_health + health)
        # Dynamic player aura changes maximum Health. Preserve existing damage
        # while raising current Health by the exact new bonus once.
        for card in hero.board + hero.hand:
            if card.card_id in BEETLES:
                card.set(GameTag.HEALTH, card.health + health)

    @staticmethod
    def deathrattle(source: Any, game: Any, ctx: Any) -> None:
        # The focused tests use the actual upstream death processing. Combat
        # snapshots and persistent global-info export are a separate frontier.
        d = game.db.get(source.card_id)
        count = 2 if d.is_golden_def else 1
        for offset in range(count):
            token = game.create_minion("BG28_603t", controller=source.controller)
            game.summon(source.controller, token, source.zone_position + offset)


def tier2_beetle_game_type():
    """An isolated actual-engine repair fixture, never a full-game certificate.

    Construct only with a clean current database. Other upstream card behavior
    is not certified by this class. Frozen opening adapters never use it.
    """
    from hsrl2.game import Game
    from hsrl2.minion import Minion
    from hsrl2.queue import Action
    from hsrl2.actions.trigger import TriggerBattlecry
    from hsrl2.tags import GameTag
    from .hsbrsim_battlecries import battlecry_multiplier

    def aura_attack(card):
        return Minion._aura_atk(card) + (beetle_bonus(card.controller)[0] if card.controller else 0)

    def aura_health(card):
        return Minion._aura_health(card) + (beetle_bonus(card.controller)[1] if card.controller else 0)

    class RoverRetrigger(Action):
        def __init__(self, source):
            self.source = source
        def do(self, game):
            game._dispatch_current_rover(self.source)

    class Tier2BeetleGame(Game):
        full_game_ready = False
        tier2_pool_ready = False
        effect_version = VERSION

        def __init__(self, heroes, db, **kwargs):
            validate_rover_definitions(db)
            super().__init__(heroes, db, **kwargs)

        def create_minion(self, card_id, **kwargs):
            card = super().create_minion(card_id, **kwargs)
            if card.card_id in ROVERS:
                card.scripts = CurrentForestRover
            if card.card_id in BEETLES:
                card._aura_atk = MethodType(aura_attack, card)
                card._aura_health = MethodType(aura_health, card)
                card.set(GameTag.HEALTH, card.max_health)
            return card

        def _dispatch_current_rover(self, source):
            for _ in range(battlecry_multiplier(self, source.controller)):
                self.run_script_hook(source, "battlecry")
                hero = source.controller
                hero.set(GameTag.COUNTER_BATTLECRIES,
                         hero.get(GameTag.COUNTER_BATTLECRIES, 0) + 1)
                self.events.fire(self, "battlecry_trigger", minion=source)

        def _run_play_effects(self, hero, minion, target=None, choose=None):
            if minion.card_id not in ROVERS:
                return super()._run_play_effects(hero, minion, target, choose)
            self._dispatch_current_rover(minion)
            self.check_deaths()
            if self.db.get(minion.card_id).is_golden_def:
                hero.set(GameTag.GOLDEN_MINIONS_PLAYED,
                         hero.get(GameTag.GOLDEN_MINIONS_PLAYED, 0) + 1)
            self.events.fire(self, "card_played", card=minion)
            if minion.has(GameTag.TRIPLE_REWARD_PENDING):
                raise Tier2ConformanceError("Golden merge/reward integration is outside this effect fixture")
            if not self.db.get(minion.card_id).is_golden_def:
                self.check_for_triple(hero, minion)
            return True

        def run_actions(self, value):
            def convert(item):
                if isinstance(item, TriggerBattlecry) and item.target.card_id in ROVERS:
                    return RoverRetrigger(item.target)
                if isinstance(item, (list, tuple)):
                    return [convert(x) for x in item]
                return item
            return super().run_actions(convert(value))

        def run_combat(self, *args, **kwargs):
            raise Tier2ConformanceError("Tier-2 combat receipts and persistent Beetle globals are not integrated")

    return Tier2BeetleGame


def build_frontier_manifest(engine_root: Path) -> dict[str, Any]:
    from .hsbrsim_data import build_current_database
    current = build_current_database(engine_root)
    from hsrl2.scripts import REGISTRY
    definitions = [d for d in current.definitions if d.get("tech_level") == 2
                   and (d["is_pool_minion"] or d["is_pool_spell"])]
    if {d["id"] for d in definitions} != set(BLOCKERS):
        raise Tier2ConformanceError("Current Tier-2 identities changed; re-audit the complete frontier")
    cards = []
    for d in definitions:
        handler = REGISTRY.get(d["id"])
        path = Path(inspect.getsourcefile(handler)) if handler else None
        cards.append({"card_id": d["id"], "name": d["name"], "tier": 2,
            "kind": "minion" if d["is_pool_minion"] else "tavern_spell",
            "text": d["text"], "races": d["races"],
            "official_card_url": f"https://hearthstone.blizzard.com/en-us/battlegrounds/{d['dbf_id']}/",
            "client_numeric_tags": d["raw_tags"],
            "upstream_handler_registered": handler is not None,
            "upstream_handler": handler.__qualname__ if handler else None,
            "upstream_source": str(path.relative_to(engine_root.resolve())) if path else None,
            "upstream_source_sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path else None,
            "opening_integrated": False, "remaining_requirement": BLOCKERS[d["id"]]})
    provenance_keys = ("engine_revision", "engine_checkout_clean", "ruleset_sha256",
        "ruleset_file_sha256", "reference_cards_sha256", "client_xml_sha256",
        "client_build", "patch", "source_client_commit", "definitions_sha256")
    return {"schema_version": 1, "scope": VERSION,
        "provenance": {k: current.provenance[k] for k in provenance_keys},
        "full_game_ready": False, "tier2_pool_ready": False,
        "all_current_tier2_cards_in_inventory": True,
        "pool_counts": {"minions": sum(c["kind"] == "minion" for c in cards),
                        "tavern_spells": sum(c["kind"] == "tavern_spell" for c in cards)},
        "card_count": len(cards), "cards": cards,
        "repair": {"card_ids": sorted(ROVERS), "beetle_token_ids": sorted(BEETLES),
                   "behavior": "Beetle-only permanent player bonus; one intrinsic effect per true Battlecry trigger",
                   "integrated_into_training": False},
        "next_integration_requirements": [
            "Complete current Tier-1 plus Tier-2 lobby-filtered shop, with separate spell slot and pool conservation",
            "Complete generated-card closures; never shrink legal pools to supported cards",
            "Exact sampled combat writeback for permanent board/hand changes, generated cards and global counters",
            "Per-entity public counters and timed choices in policy observations",
            "Full action-to-action conformance before removing the first-turn upgrade guard"]}
