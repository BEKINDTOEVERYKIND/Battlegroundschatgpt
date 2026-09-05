"""Audited intrinsic Battlecries and a per-game dispatcher with correct repeats.

The current opening-card migration is explicit. Unknown Battlecry/Choose One
handlers fail before play; no global registry or upstream source is modified.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import re
from types import MethodType
from typing import Any, Iterator

from bg_ai.hsbrsim_effects import CurrentSanguineChampion

DEFAULT_CARDS = Path(__file__).resolve().parents[2] / "data/reference_cards.json"
DISPATCHER_VERSION = 1


class UnmigratedBattlecry(RuntimeError):
    pass


def _multiplier(source: Any, game: Any) -> int:
    return 2 if game.db.get(source.card_id).is_golden_def else 1


class RazorfenGeomancer:
    @staticmethod
    def battlecry(source: Any, game: Any, ctx: Any) -> Any:
        from hsrl2.actions.bloodgem import GetBloodGems
        return GetBloodGems(source.controller, 2 * _multiplier(source, game))


class SouthseaBusker:
    @staticmethod
    def battlecry(source: Any, game: Any, ctx: Any) -> Any:
        from hsrl2.actions.economy import GainGold, ScheduleNextTurn
        return ScheduleNextTurn(source.controller, GainGold(source.controller, _multiplier(source, game)))


class OminousSeer:
    @staticmethod
    def battlecry(source: Any, game: Any, ctx: Any) -> None:
        from hsrl2.tags import GameTag
        hero = source.controller
        tag = GameTag.NEXT_SPELL_COST_REDUCTION
        hero.set(tag, hero.get(tag, 0) + _multiplier(source, game))


class ShellCollector:
    @staticmethod
    def battlecry(source: Any, game: Any, ctx: Any) -> None:
        for _ in range(_multiplier(source, game)):
            coin = game.create_spell("BG28_810", controller=source.controller)
            game.pending_hand_add(source.controller, coin)


class IntrepidBotanist:
    # This is a Choose One play effect, not a Battlecry.
    choose_options = [("atk", "Improve Tavern spell Attack"),
                      ("health", "Improve Tavern spell Health")]

    @staticmethod
    def on_choose(source: Any, game: Any, ctx: Any) -> None:
        from hsrl2.tags import GameTag
        choice = (ctx or {}).get("choose")
        if choice not in {"atk", "health", "both"}:
            raise UnmigratedBattlecry("Intrepid Botanist requires an explicit valid choice")
        hero = source.controller
        amount = _multiplier(source, game)
        if choice in {"atk", "both"}:
            tag = GameTag.TAVERN_SPELL_EXTRA_ATK
            hero.set(tag, hero.get(tag, 0) + amount)
        if choice in {"health", "both"}:
            tag = GameTag.TAVERN_SPELL_EXTRA_HEALTH
            hero.set(tag, hero.get(tag, 0) + amount)


BATTLECRY_SCRIPTS = {
    "BG20_100": RazorfenGeomancer, "BG20_100_G": RazorfenGeomancer,
    "BG26_135": SouthseaBusker, "BG26_135_G": SouthseaBusker,
    "BG31_330": OminousSeer, "BG31_330_G": OminousSeer,
    "BG23_002": ShellCollector, "BG23_002_G": ShellCollector,
    "BG23_017": CurrentSanguineChampion, "BG23_017_G": CurrentSanguineChampion,
}
CHOOSE_ONE_SCRIPTS = {"BG32_237": IntrepidBotanist, "BG32_237_G": IntrepidBotanist}
MIGRATED_SCRIPTS = {**BATTLECRY_SCRIPTS, **CHOOSE_ONE_SCRIPTS}
BRANN_MULTIPLIERS = {"BG_LOE_077": 2, "TB_BaconUps_045": 3,
                     "BG27_518": 2, "BG27_518_G": 3}
_LITERAL_TEXT = {
    "BG20_100": "Battlecry: Get 2 Blood Gems.",
    "BG20_100_G": "Battlecry: Get 4 Blood Gems.",
    "BG26_135": "Battlecry: Gain 1 Gold next turn.",
    "BG26_135_G": "Battlecry: Gain 2 Gold next turn.",
    "BG31_330": "Battlecry: The next Tavern spell you buy costs (1) less.",
    "BG31_330_G": "Battlecry: The next Tavern spell you buy costs (2) less.",
    "BG23_002": "Battlecry: Get a Tavern Coin.",
    "BG23_002_G": "Battlecry: Get 2 Tavern Coins.",
    "BG32_237": "Choose One - Your Tavern spells give an extra +1 Attack this game; or +1 Health.",
    "BG32_237_G": "Choose One - Your Tavern spells give an extra +2 Attack this game; or +2 Health.",
    "BG_LOE_077": "Your Battlecries trigger twice.",
    "TB_BaconUps_045": "Your Battlecries trigger three times.",
    "BG27_518": "Your Battlecries and Deathrattles trigger twice.",
    "BG27_518_G": "Your Battlecries and Deathrattles trigger three times.",
}


def _plain(text: str) -> str:
    return " ".join(re.sub(r"<[^>]*>", "", text.replace("[x]", "")).split())


def validate_migration(db: Any, cards_path: Path | str = DEFAULT_CARDS) -> None:
    reference = {c["id"]: c for c in json.loads(Path(cards_path).read_text())}
    for cid in {*MIGRATED_SCRIPTS, *BRANN_MULTIPLIERS}:
        d, ref = db.get(cid), reference.get(cid)
        if d is None or ref is None or ref.get("snapshotBuild") != 251332 or d.text != ref["text"]:
            raise UnmigratedBattlecry(f"{cid}: current definition validation failed")
        if cid in _LITERAL_TEXT and _plain(d.text) != _LITERAL_TEXT[cid]:
            raise UnmigratedBattlecry(f"{cid}: audited intrinsic amount or meaning changed")
        if cid.startswith("BG23_017"):
            expected = (4, 2) if d.is_golden_def else (2, 1)
            if (d.num(0), d.num(1)) != expected:
                raise UnmigratedBattlecry(f"{cid}: current Sanguine Champion values changed")


def battlecry_multiplier(game: Any, hero: Any) -> int:
    """Highest applicable Brann/Moira value wins; extra copies do not add."""
    from hsrl2.tags import GameTag
    result = 1
    for minion in hero.board:
        if minion.dead or minion.has(GameTag.SILENCED):
            continue
        if minion.has(GameTag.BATTLECRY_DOUBLER):
            value = BRANN_MULTIPLIERS.get(minion.card_id)
            if value is None:
                raise UnmigratedBattlecry("Unmigrated Battlecry repeat source: " + minion.card_id)
            result = max(result, value)
    return result


def dispatch_battlecry(game: Any, source: Any, ctx: Any = None) -> int:
    """Resolve real Battlecry triggers, counting each once, on play or retrigger."""
    from hsrl2.tags import GameTag
    d = game.db.get(source.card_id)
    if d is None or "battlecry" not in d.keywords:
        return 0  # Choose One and ordinary minions cannot be retriggered as Battlecries.
    script = BATTLECRY_SCRIPTS.get(source.card_id)
    if script is None:
        raise UnmigratedBattlecry("Intrinsic Battlecry has not been migrated: " + source.card_id)
    if source._script_overrides.get("battlecry") is not None:
        raise UnmigratedBattlecry("Custom Battlecry override needs separate migration")
    hero = source.controller
    if hero is None:
        raise UnmigratedBattlecry("Battlecry source is missing its controller")
    repeats = battlecry_multiplier(game, hero)
    for _ in range(repeats):
        game.run_actions(script.battlecry(source, game, ctx))
        hero.set(GameTag.COUNTER_BATTLECRIES, hero.get(GameTag.COUNTER_BATTLECRIES, 0) + 1)
        game.events.fire(game, "battlecry_trigger", minion=source)
    return repeats


def resolve_play_effects(game: Any, hero: Any, minion: Any, target: Any = None,
                         choose: str | None = None) -> bool:
    """Post-summon play bookkeeping with one intrinsic effect per true trigger."""
    from hsrl2.tags import GameTag
    d = game.db.get(minion.card_id)
    context = {"target": target, "choose": choose}
    if minion.card_id in CHOOSE_ONE_SCRIPTS:
        CHOOSE_ONE_SCRIPTS[minion.card_id].on_choose(minion, game, context)
    elif d is not None and "battlecry" in d.keywords:
        dispatch_battlecry(game, minion, context)
    elif callable(getattr(minion.scripts, "battlecry", None)) or (
            d is not None and "choose one" in _plain(d.text).lower()):
        raise UnmigratedBattlecry("Unclassified play-effect handler: " + minion.card_id)
    game.check_deaths()
    golden = d is not None and d.is_golden_def
    if golden:
        hero.set(GameTag.GOLDEN_MINIONS_PLAYED, hero.get(GameTag.GOLDEN_MINIONS_PLAYED, 0) + 1)
    game.events.fire(game, "card_played", card=minion)
    if golden and minion.has(GameTag.TRIPLE_REWARD_PENDING):
        game._grant_triple_reward(hero, minion)
    if not golden:
        game.check_for_triple(hero, minion)
    return True


@dataclass(frozen=True)
class BattlecryInstallation:
    migrated_ids: tuple[str, ...]
    dispatcher_version: int = DISPATCHER_VERSION
    full_game_ready: bool = False


@contextmanager
def installed_battlecry_rules(game: Any, *, cards_path: Path | str = DEFAULT_CARDS
                             ) -> Iterator[BattlecryInstallation]:
    """Install only on this Game; compose inside installed_recruit_rules(game)."""
    from hsrl2.actions.trigger import TriggerBattlecry
    from hsrl2.queue import Action
    validate_migration(game.db, cards_path)
    names = ("create_minion", "play_minion", "_run_play_effects", "run_actions",
             "_bg_ai_migrated_battlecry_ids", "_bg_ai_battlecry_dispatcher_version")
    absent = object()
    saved = {name: game.__dict__.get(name, absent) for name in names}
    original_create, original_play, original_actions = game.create_minion, game.play_minion, game.run_actions
    previous_scripts = []
    for hero in game.heroes:
        for entity in [*hero.board, *hero.hand, *hero.tavern]:
            if entity.card_id in MIGRATED_SCRIPTS:
                previous_scripts.append((entity, entity.scripts))
                entity.scripts = MIGRATED_SCRIPTS[entity.card_id]

    def create(this: Any, card_id: str, **kwargs: Any) -> Any:
        minion = original_create(card_id, **kwargs)
        if minion.card_id in MIGRATED_SCRIPTS:
            minion.scripts = MIGRATED_SCRIPTS[minion.card_id]
        return minion

    def play(this: Any, hero: Any, minion: Any, *args: Any, **kwargs: Any) -> bool:
        d = this.db.get(minion.card_id)
        has_bc = d is not None and "battlecry" in d.keywords
        has_choice = bool(getattr(minion.scripts, "choose_options", None)) or (
            d is not None and "choose one" in _plain(d.text).lower())
        has_old_handler = callable(getattr(minion.scripts, "battlecry", None))
        if (has_bc or has_choice or has_old_handler) and minion.card_id not in MIGRATED_SCRIPTS:
            raise UnmigratedBattlecry("Play effect has not been migrated: " + minion.card_id)
        if minion._script_overrides.get("battlecry") is not None:
            raise UnmigratedBattlecry("Custom Battlecry play override needs separate migration")
        if has_bc:
            battlecry_multiplier(this, hero)  # Validate repeat sources before entering the board.
        return original_play(hero, minion, *args, **kwargs)

    class MigratedTrigger(Action):
        def __init__(self, target: Any):
            self.target = target
        def do(self, g: Any) -> None:
            dispatch_battlecry(g, self.target)

    def actions(this: Any, value: Any) -> None:
        def convert(item: Any) -> Any:
            if isinstance(item, TriggerBattlecry):
                return MigratedTrigger(item.target)
            if isinstance(item, (tuple, list)):
                return [convert(child) for child in item]
            return item
        original_actions(convert(value))

    game.create_minion = MethodType(create, game)
    game.play_minion = MethodType(play, game)
    game._run_play_effects = MethodType(resolve_play_effects, game)
    game.run_actions = MethodType(actions, game)
    game._bg_ai_migrated_battlecry_ids = frozenset(MIGRATED_SCRIPTS)
    game._bg_ai_battlecry_dispatcher_version = DISPATCHER_VERSION
    try:
        yield BattlecryInstallation(tuple(sorted(MIGRATED_SCRIPTS)))
    finally:
        for entity, script in previous_scripts:
            entity.scripts = script
        for name, value in saved.items():
            if value is absent:
                delattr(game, name)
            else:
                setattr(game, name, value)
