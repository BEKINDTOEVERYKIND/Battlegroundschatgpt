"""Audited hero extensions for the pinned external hsrl2 engine.

This module does not authorize full-game training. Install its registry entries
only for the lifetime of a game built from the current-data bridge. There are no
imports of, or mutations to, an external checkout until the context is entered.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[2]
GALEWING = "BG20_HERO_283p"
WESTFALL = "BG20_HERO_283p_t1"
IRONFORGE = "BG20_HERO_283p_t2"
PLAGUELANDS = "BG20_HERO_283p_t3"
FLIGHTPATHS = (WESTFALL, IRONFORGE, PLAGUELANDS)
SUPPORTED_HERO_POWER_IDS = (GALEWING,)
EXTENSION_IDS = (GALEWING, *FLIGHTPATHS)
PREFIX = "bg_ai.galewing."


class HeroExtensionError(ValueError):
    """Current definitions or transition prerequisites failed verification."""


def hero_extension_state(hero: Any) -> dict[str, Any]:
    """Public state for action masks, observations and readable traces."""
    return {
        "galewing_path": hero.get(PREFIX + "path", None),
        "galewing_due_turn": hero.get(PREFIX + "due_turn", None),
        "galewing_last_path": hero.get(PREFIX + "last_path", None),
        "galewing_choice_pending": bool(hero.get(PREFIX + "choice_pending", False)),
    }


def _current_power(hero: Any, game: Any) -> str | None:
    definition = game.hero_power_def(hero)
    return definition.id if definition else None


def _validate(db: Any, cards_path: Path) -> None:
    from hsrl2.tags import CardType

    refs = {card["id"]: card for card in json.loads(cards_path.read_text())}
    for card_id in EXTENSION_IDS:
        card = refs.get(card_id)
        definition = db.get(card_id)
        if card is None or definition is None:
            raise HeroExtensionError(f"Missing current Galewing definition: {card_id}")
        expected_cost = card.get("cost", 0)
        expected_numbers = tuple(card.get("rawTags", {}).get(str(tag))
                                 for tag in (2, 3, 2889, 2919))
        if (definition.card_type != CardType.HERO_POWER
                or definition.text != card["text"]
                or definition.cost != expected_cost
                or tuple(definition.num(i) for i in range(4)) != expected_numbers):
            raise HeroExtensionError(f"Stale or incompatible Galewing definition: {card_id}")
    # These transitions were reviewed against this exact ruleset. A changed
    # source file is not itself evidence that the implementation remains valid.
    if (refs[WESTFALL]["rawTags"].get("2") != 1
            or refs[IRONFORGE]["rawTags"].get("2") != 2
            or refs[PLAGUELANDS]["rawTags"].get("2") != 3
            or "1-Cost Tavern spell" not in refs[WESTFALL]["text"]
            or "gain 2 Gold" not in refs[IRONFORGE]["text"]
            or "minion of your Tier" not in refs[PLAGUELANDS]["text"]):
        raise HeroExtensionError("Galewing flightpaths changed; a new rules audit is required")


def _make_scripts() -> dict[str, type]:
    from hsrl2.actions.discover import Discover
    from hsrl2.events import Listener, TURN_START
    from hsrl2.game import PendingChoice

    class FlightInProgress:
        passive = True

    class GalewingScript:
        @staticmethod
        def cost_override(hero, game):
            return game.db.get(GALEWING).cost

        @staticmethod
        def available(hero, game):
            return (hero.is_alive and _current_power(hero, game) == GALEWING
                    and hero.get(PREFIX + "path", None) is None
                    and not hero.get(PREFIX + "choice_pending", False))

        @staticmethod
        def on_bind(hero, game):
            # Binding twice must not duplicate start-of-turn rewards/listeners.
            if hero.get(PREFIX + "bound", False):
                return
            hero.set(PREFIX + "bound", True)

            def on_turn(g, **event):
                path = hero.get(PREFIX + "path", None)
                if path is None or not hero.is_alive:
                    return
                # Replacing a hero power deactivates its listeners' effects.
                # Do not resurrect Galewing after Unmasked Identity/etc.
                if _current_power(hero, g) != path:
                    hero.clear(PREFIX + "path")
                    hero.clear(PREFIX + "due_turn")
                    return
                if event.get("turn", g.turn) < hero.get(PREFIX + "due_turn"):
                    return
                hero.set(PREFIX + "last_path", path)
                hero.clear(PREFIX + "path")
                hero.clear(PREFIX + "due_turn")
                g.replace_hero_power(hero, GALEWING)
                if path == IRONFORGE:
                    # TURN_START fires after base income is reset.
                    hero.gold += 2
                elif path == PLAGUELANDS:
                    tier = hero.tavern_tier
                    g.run_actions(Discover(hero, min_tier=tier, max_tier=tier,
                                           kind="discover_galewing_minion"))
                elif path == WESTFALL:
                    # Text says cost, not Tavern tier: eligible higher-Tier
                    # 1-cost spells remain eligible at a Tier-1 Tavern.
                    candidates = sorted(d.id for d in g.db.pool_spells()
                                        if d.cost == 1
                                        and g.spell_pool.available(d.id) > 0)
                    if not candidates:
                        raise HeroExtensionError("Westfall has no available current 1-cost spell")
                    pick = g.rng.choice(candidates, label="galewing_westfall_spell")
                    if not g.spell_pool.acquire(pick):
                        raise HeroExtensionError(f"Westfall spell acquisition failed: {pick}")
                    # Blizzard 28.2: Tavern spells return to the shared pool
                    # immediately when obtained, including before hand play.
                    g.spell_pool.release(pick)
                    g.pending_hand_add(hero, g.create_spell(pick, controller=hero))

            game.events.register(Listener(
                TURN_START, hero, on_turn,
                condition=lambda **event: event.get("hero") is hero))

        @staticmethod
        def hero_power(hero, game, ctx):
            # Also bind if a controlled fixture invokes a power without
            # start_game(). It does not bypass the normal action/timer wrapper.
            GalewingScript.on_bind(hero, game)
            options = [path for path in FLIGHTPATHS
                       if path != hero.get(PREFIX + "last_path", None)]
            hero.set(PREFIX + "choice_pending", True)

            def choose(path):
                if not hero.get(PREFIX + "choice_pending", False):
                    raise HeroExtensionError("Galewing choice was already resolved")
                if not hero.is_alive or _current_power(hero, game) != GALEWING:
                    raise HeroExtensionError("Galewing power changed while its choice was pending")
                if path not in options:
                    raise HeroExtensionError("Invalid or consecutive Galewing flightpath")
                hero.clear(PREFIX + "choice_pending")
                hero.set(PREFIX + "path", path)
                hero.set(PREFIX + "due_turn", game.turn + game.db.get(path).num(0))
                game.replace_hero_power(hero, path)

            game.pending_choices.append(PendingChoice(
                hero, options, "galewing_flightpath", resolve_callback=choose))

    return {GALEWING: GalewingScript,
            **{path: FlightInProgress for path in FLIGHTPATHS}}


@contextmanager
def installed_current_hero_extensions(
    db: Any, *, cards_path: Path = ROOT / "data/reference_cards.json",
) -> Iterator[dict[str, Any]]:
    """Temporarily install reviewed scripts; restore every prior registry entry.

    The current-data builder verifies the external revision before this call.
    Nesting is supported. Runtime game objects must stay inside the context.
    """
    _validate(db, Path(cards_path))
    from hsrl2.scripts import REGISTRY

    scripts = _make_scripts()
    absent = object()
    previous = {card_id: REGISTRY.get(card_id, absent) for card_id in scripts}
    REGISTRY.update(scripts)
    try:
        yield {
            "hero_power_ids": list(SUPPORTED_HERO_POWER_IDS),
            "generated_power_ids": list(FLIGHTPATHS),
            "full_game_training_ready": False,
            "scope": "Galewing recruitment transitions; external engine gates still apply",
            "unverified": [
                "random Tavern spell generation distribution beyond card text",
                "unpatched native spell casting redundantly returns held spells; a validated cast adapter is required",
                "Buddy/replaced/second-power acquisition lifecycle outside explicit base-hero scope",
            ],
        }
    finally:
        for card_id, prior in previous.items():
            if prior is absent:
                REGISTRY.pop(card_id, None)
            else:
                REGISTRY[card_id] = prior
