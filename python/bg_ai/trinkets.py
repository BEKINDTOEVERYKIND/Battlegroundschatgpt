"""Current-trinket primitives and an explicit partial recruitment integration.

This module does not promote a reduced pool to a full-game environment. Offering
probabilities and card-specific eligibility metadata are partly unpublished.
Recorded offers can be replayed; synthetic offers require explicit assumptions.
Every effect here uses a card-ID dispatch and real host mutations. Unknown IDs
raise before payment. Automatic discoveries and automatic hand plays are absent.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
import math
import random
from typing import Any, Mapping, Protocol, Sequence

RULES_URL = "https://us.forums.blizzard.com/en/hearthstone/t/battlegrounds-developer-insights-updated-trinket-rules/158955"
SEASON_URL = "https://us.forums.blizzard.com/en/hearthstone/t/battlegrounds-season-14-trinket-updates/163710"
PATCH_URL = "https://hearthstone.blizzard.com/en-us/news/24296231/3642-patch-notes"
SPHERE_PATCH_URL = "https://hearthstone.blizzard.com/en-gb/news/24293284/36-2-2-patch-notes"
CURRENT_BUILD = 251332
SPHERE_ID = "BG36_MagicItem_372"
BLESSING_ID = "BG32_MagicItem_894"
TRIBES = frozenset({"BEAST", "DEMON", "DRAGON", "ELEMENTAL", "MECH", "MURLOC", "NAGA", "PIRATE", "QUILBOAR", "UNDEAD"})


class TrinketError(RuntimeError):
    pass


@dataclass(frozen=True)
class MinionView:
    entity_id: str
    card_id: str
    types: frozenset[str] = frozenset()
    attack: int = 0
    health: int = 1
    tier: int = 1
    keywords: frozenset[str] = frozenset()


@dataclass(frozen=True)
class OfferContext:
    turn: int
    tavern_tier: int
    hero_id: str
    board: tuple[MinionView, ...]
    hand: tuple[MinionView, ...]
    lobby_types: frozenset[str]
    owned_ids: frozenset[str] = frozenset()
    # Hero-power substitutions must supply affinity explicitly, not infer from
    # the original hero portrait (Finley, Nguyen, Unmasked Identity, etc.).
    hero_affinity: frozenset[str] = frozenset()
    health_and_armor: int = 30
    history: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if type(self.turn) is not int or self.turn < 1 or type(self.tavern_tier) is not int or not 1 <= self.tavern_tier <= 7:
            raise ValueError("Invalid current turn or Tavern tier")
        if len(self.board) > 7 or len(self.hand) > 10 or len(self.lobby_types) != 5 or not self.lobby_types <= TRIBES:
            raise ValueError("Invalid Solo board, hand, or five-type lobby")

    def counts(self) -> Counter[str]:
        out: Counter[str] = Counter()
        for m in self.board:
            types = m.types & TRIBES
            if "ALL" in m.types:
                types = TRIBES
            for tribe in types:
                out[tribe] += 1
            if not types:
                out["NO_TYPE"] += 1
        if len(set(out) & TRIBES) >= 3:
            out["MENAGERIE"] = len(set(out) & TRIBES)
        return out

    def in_types(self, stage: str) -> frozenset[str]:
        threshold = 2 if stage == "lesser" else 3
        counts = self.counts()
        result = {t for t, n in counts.items() if n >= (3 if t in {"NO_TYPE", "MENAGERIE"} else threshold)}
        return frozenset(result | self.hero_affinity)

    def most_common_types(self, stage: str) -> frozenset[str]:
        counts = self.counts()
        active = self.in_types(stage) - {"MENAGERIE"}
        if not active:
            return frozenset()
        highest = max(counts[t] for t in active)
        return frozenset(t for t in active if counts[t] == highest)


@dataclass(frozen=True)
class TrinketSpec:
    card_id: str
    name: str
    stage: str
    cost: int
    affiliated_types: frozenset[str]
    # None means the client tag has not been behaviorally validated. Synthetic
    # offers require an explicit fallback; this is never silently interpreted.
    requires_in_type: bool | None = None
    parameters: tuple[int, ...] = ()
    source_status: str = "gallery_client_agree"


# These card IDs have implemented effect transitions below. This is an effect
# inventory, not an offerable pool or a claim that all interactions are covered.
CORE_EFFECTS: Mapping[str, str] = {
    "BG30_MagicItem_420": "book_lesser",
    "BG30_MagicItem_420t": "book_greater",
    "BG30_MagicItem_435": "goldenizer_supply",
    "BG30_MagicItem_547": "coffin_lesser",
    "BG30_MagicItem_547t": "coffin_greater",
    "BG30_MagicItem_705": "oilcan",
    "BG30_MagicItem_847": "wallet",
    "BG30_MagicItem_914": "phrasebook_lesser",
    "BG30_MagicItem_914t": "phrasebook_greater",
    "BG30_MagicItem_996": "tip_jar",
    "BG32_MagicItem_276": "headstone",
    "BG35_MagicItem_814": "coin_purse",
    "BG35_MagicItem_818": "mysterious_orb",
    "BG36_MagicItem_300": "frigid_blossom",
    "BG36_MagicItem_307": "wand",
    "BG36_MagicItem_811": "defensive_shell",
    "BG30_MagicItem_422": "lorewalker_lesser",
    "BG30_MagicItem_422t": "lorewalker_greater",
    SPHERE_ID: "sphere",
    "BG30_MagicItem_406": "butchers_sickle",
    "BG31_MagicItem_903": "wisdomball_supply",
    "BG30_MagicItem_544": "nomi_lesser",
    "BG30_MagicItem_544t": "nomi_greater",
    "BG30_MagicItem_988": "great_boar_lesser",
    "BG30_MagicItem_988t": "great_boar_greater",
    "BG35_MagicItem_851": "water_wheel",
    "BG36_MagicItem_840": "secret_schematic",
    "BG36_MagicItem_850": "cookie_rod",
    "BG36_MagicItem_800": "demonic_bloodletter",
    "BG30_MagicItem_900": "dragonwing_glider",
    "BG30_MagicItem_924": "booty_bay_lesser",
    "BG30_MagicItem_924t": "booty_bay_greater",
    "BG36_MagicItem_202": "warcry_totem",
    "BG36_MagicItem_220": "glowing_crystal",
    "BG32_MagicItem_888": "recycling_sticker",
    "BG32_MagicItem_428": "worn_treasure_map",
    "BG32_MagicItem_271": "ornate_clock",
    "BG30_MagicItem_430": "rockin_music_box",
    "BG30_MagicItem_951": "lava_lamp",
}


def catalog_from_snapshot(cards: Sequence[Mapping[str, Any]], ruleset: Mapping[str, Any],
                          *, resolve_sphere_from_official_history: bool = False) -> dict[str, TrinketSpec]:
    """Read current candidates; callers still must choose an explicit scope.

    Sphere inclusion is an explicit evidence-based resolution, not an implicit
    override of the frozen manifest. The original training manifest is unchanged.
    """
    if int(ruleset.get("build", 0)) != CURRENT_BUILD:
        raise TrinketError("Trinket implementation has not been reviewed for this build")
    by_id = {c["id"]: c for c in cards}
    sources = ruleset.get("active", {})
    candidate = ruleset.get("candidate", {})
    result = {}
    for stage in ("lesser", "greater"):
        ids = set(sources.get(f"{stage}_trinket_ids", ())) | set(candidate.get(f"{stage}_trinket_ids", ()))
        if stage == "greater" and resolve_sphere_from_official_history:
            ids.add(SPHERE_ID)
        for cid in sorted(ids):
            if cid == BLESSING_ID:
                raise TrinketError("Latest official patch removed Blessing Portrait")
            c = by_id[cid]
            tags = c.get("tags", {})
            if int(c.get("snapshotBuild", 0)) != CURRENT_BUILD or not tags.get("BACON_TRINKET"):
                raise TrinketError(f"Invalid current trinket definition: {cid}")
            types = set()
            for key, value in tags.items():
                if value and key.startswith("BACON_SUBSET_"):
                    t = key.removeprefix("BACON_SUBSET_")
                    types.add({"ELEMENTALS": "ELEMENTAL", "QUILLBOAR": "QUILBOAR"}.get(t, t))
            params = tuple(int(tags.get(f"TAG_SCRIPT_DATA_NUM_{i}", 0)) for i in range(1, 7))
            result[cid] = TrinketSpec(cid, c["name"], stage, int(c.get("cost", 0)),
                frozenset(types), None if types else False, params,
                "official_history_over_gallery_omission" if cid == SPHERE_ID else "gallery_client_agree")
    return result


@dataclass(frozen=True)
class OfferingAssumptions:
    """Required, serialized assumptions for a synthetic offering experiment.

    Weight values, tie selection, and inference from the discount client tag are
    not certified Blizzard probabilities. Fixtures must retain this provenance.
    """
    card_weights: Mapping[str, float]
    unknown_requires_in_type: bool
    most_common_tie_rule: str = "seeded_uniform"
    distribution: str = "weighted_without_replacement_conditioned_on_constraints"
    label: str = "experimental_unpublished_offering_distribution"

    def __post_init__(self):
        if not self.label or self.most_common_tie_rule != "seeded_uniform":
            raise ValueError("Explicit supported offering assumptions are required")
        if self.distribution != "weighted_without_replacement_conditioned_on_constraints":
            raise ValueError("Unsupported offering distribution")
        if any(not math.isfinite(float(w)) or float(w) <= 0 for w in self.card_weights.values()):
            raise ValueError("Offering weights must be finite and positive")


@dataclass(frozen=True)
class TrinketOffer:
    card_id: str
    price: int


def _special_allowed(spec: TrinketSpec, ctx: OfferContext) -> bool:
    # These are published individual restrictions, not a complete hidden table.
    name, h = spec.name, ctx.history
    if name in {"Shrine of Evolution", "Sacrificial Altar"} and len(ctx.board) < 6:
        return False
    if name in {"Demonic Tapestry", "The Eye of Sargeras", "Nether Pendant", "Felburned Ledger"} and not h.get("health_rewinder"):
        return False
    if name in {"Scraper Sticker", "Beatboxer Portrait"} and ctx.tavern_tier < 3:
        return False
    if name == "Magician's Top Hat" and len(ctx.board) + len(ctx.hand) >= 5:
        return False
    if name == "Lens Case" and not any(m.tier == 3 for m in ctx.board):
        return False
    if name == "Bob-blehead" and "DEMON" in ctx.lobby_types:
        return False
    if name == "Baller Portrait" and not h.get("baller_sold"):
        return False
    if name in {"Ur'zul Sticker", "Flaming Portrait"} and not h.get("tavern_buffed"):
        return False
    if name == "Murky Sticker" and h.get("battlecries_triggered", 0) < 2:
        return False
    if name == "Dramaloc Sticker" and not any(m.attack >= 10 for m in ctx.hand):
        return False
    if name in {"Herald Sticker", "Blood Amulet", "Unholy Sanctum", "Thornspike Pauldron"} and not any("DEATHRATTLE" in m.keywords for m in ctx.board):
        return False
    if name == "Safety Patch" and ctx.health_and_armor >= 16:
        return False
    if name in {"Fancy Spellbook", "Heart of the Forest"} and not h.get("tavern_spells_buffed"):
        return False
    # Affinity and exclusions follow the CURRENT hero power identity supplied by
    # adapter history. A portrait alone is unsafe after power replacement.
    hero_key = h.get("hero_power_owner", "")
    excluded = {
        "Souvenir Stand": {"Marin", "Buttons"}, "Ornate Clock": {"Marin", "Buttons"},
        "Trip Vouchers": {"Marin", "Buttons"}, "Cho'gall Sticker": {"Cho", "Gall"},
        "Corrupted Tome": {"Mr. Clocksworth"}, "Kaleidoscope": {"Voone", "E.T.C."},
        "Goblin Wallet": {"Gallywix"}, "Bob's Tip Jar": {"Gallywix"},
        "Fish Portrait": {"Greybough"},
    }
    return hero_key not in excluded.get(name, set())


def _price(spec: TrinketSpec, ctx: OfferContext, stage: str) -> int:
    discount = 2 if spec.affiliated_types and not (spec.affiliated_types & ctx.in_types(stage)) else 0
    return max(0, spec.cost - discount)


def validate_offering(offers: Sequence[TrinketOffer], catalog: Mapping[str, TrinketSpec],
                      ctx: OfferContext, stage: str, *, most_common: str | None = None,
                      require_in_type: Mapping[str, bool] | None = None) -> None:
    if len(offers) != 4 or len({o.card_id for o in offers}) != 4:
        raise TrinketError("Exactly four distinct trinkets must be offered")
    if stage not in {"lesser", "greater"}:
        raise TrinketError("Invalid trinket stage")
    if most_common is not None and most_common not in ctx.most_common_types(stage):
        raise TrinketError("Invalid most-common type selection")
    specs = [catalog[o.card_id] for o in offers]
    in_types = ctx.in_types(stage)
    for offer, spec in zip(offers, specs):
        if spec.stage != stage or spec.card_id in ctx.owned_ids or not _special_allowed(spec, ctx):
            raise TrinketError("Ineligible or already owned trinket")
        if spec.affiliated_types & TRIBES and not (spec.affiliated_types & ctx.lobby_types):
            raise TrinketError("Trinket refers to an absent lobby type")
        if offer.price != _price(spec, ctx, stage):
            raise TrinketError("Trinket cost or pivot discount mismatch")
        restricted = (require_in_type or {}).get(spec.card_id, spec.requires_in_type)
        if restricted and spec.affiliated_types and not (spec.affiliated_types & in_types):
            raise TrinketError("Trinket requires an established minion type")
    if not any(not spec.affiliated_types for spec in specs):
        raise TrinketError("Offering lacks the guaranteed neutral trinket")
    if min(o.price for o in offers) > 2:
        raise TrinketError("Offering lacks the guaranteed affordable trinket")
    tied_types = ctx.most_common_types(stage)
    if most_common is None and tied_types:
        # A recorded offer need only match some legal tie resolution. Synthetic
        # generation samples one tie explicitly and records it.
        if not any(_set_constraints(specs, t) for t in sorted(tied_types)):
            raise TrinketError("Offering violates type guarantees or exclusivity")
    elif not _set_constraints(specs, most_common):
        raise TrinketError("Offering violates type guarantees or exclusivity")


def _set_constraints(specs: Sequence[TrinketSpec], most_common: str | None) -> bool:
    if most_common and not any(most_common in s.affiliated_types for s in specs):
        return False
    counts = Counter(t for s in specs for t in s.affiliated_types)
    if any(n > 1 for t, n in counts.items() if t != most_common) or counts["MENAGERIE"] > 1:
        return False
    names = {s.name for s in specs}
    return (len(names & {"Shrine of Evolution", "Sacrificial Altar"}) <= 1 and
            len(names & {"Souvenir Stand", "Trip Vouchers", "Ornate Clock"}) <= 1)


def offer_trinkets(catalog: Mapping[str, TrinketSpec], ctx: OfferContext, *, stage: str,
                   candidate_ids: Sequence[str], rng: random.Random,
                   assumptions: OfferingAssumptions, max_attempts: int = 20000) -> tuple[TrinketOffer, ...]:
    """Generate a bounded synthetic offering, with explicit incomplete metadata.

    This function deliberately requires candidate_ids; effect support is never
    used to silently trim an alleged full pool. Callers must label reduced tests.
    """
    if stage not in {"lesser", "greater"} or max_attempts < 1:
        raise ValueError("Invalid trinket stage or attempt bound")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Duplicate candidate IDs")
    if set(candidate_ids) != set(assumptions.card_weights):
        raise TrinketError("Every explicit candidate requires an offering weight")
    restricted = {}
    eligible = []
    for cid in sorted(candidate_ids):
        spec = catalog[cid]
        restricted[cid] = assumptions.unknown_requires_in_type if spec.requires_in_type is None else spec.requires_in_type
        if spec.stage != stage or cid in ctx.owned_ids or not _special_allowed(spec, ctx):
            continue
        if spec.affiliated_types & TRIBES and not (spec.affiliated_types & ctx.lobby_types):
            continue
        if restricted[cid] and spec.affiliated_types and not (spec.affiliated_types & ctx.in_types(stage)):
            continue
        eligible.append(cid)
    if len(eligible) < 4:
        raise TrinketError("Too few eligible cards for a legal offering; refusing fallback")
    tied = sorted(ctx.most_common_types(stage))
    common = rng.choice(tied) if tied else None
    for _ in range(max_attempts):
        remaining = list(eligible)
        chosen = []
        for _ in range(4):
            cid = rng.choices(remaining, weights=[assumptions.card_weights[c] for c in remaining], k=1)[0]
            remaining.remove(cid)
            chosen.append(TrinketOffer(cid, _price(catalog[cid], ctx, stage)))
        try:
            validate_offering(chosen, catalog, ctx, stage, most_common=common, require_in_type=restricted)
        except TrinketError:
            continue
        return tuple(chosen)
    raise TrinketError("Offering constraints cannot be satisfied within bounded sampling")


class TrinketHost(Protocol):
    @property
    def gold(self) -> int: ...
    @property
    def tavern_tier(self) -> int: ...
    def spend_gold(self, amount: int, *, source_id: str | None = None) -> None: ...
    def gain_gold(self, amount: int) -> None: ...
    def increase_income_cap(self, amount: int) -> None: ...
    def reduce_upgrade_cost(self, amount: int) -> None: ...
    def add_undead_attack(self, amount: int) -> None: ...
    def give_card(self, card_id: str, source_id: str) -> None:
        """Generate into hand or protected queue, then check triples."""
        ...
    def discover_spell_options(self) -> Sequence[str]:
        """Return eligible current IDs; do not peek at future RNG or shops."""
        ...
    def card_discovered(self, card_id: str) -> None: ...
    def minion_view(self, entity_id: str) -> MinionView: ...
    def board_minions(self) -> Sequence[MinionView]: ...
    def increase_gem_bonus(self, attack: int, health: int) -> None: ...
    def gain_free_refresh(self, amount: int) -> None: ...
    def buff_tavern(self, attack: int, health: int, source_id: str, tribe: str | None) -> None: ...
    def minion_generation_options(self, *, tribe: str | None = None, keyword: str | None = None) -> Sequence[str]: ...
    def give_pool_minion(self, card_id: str, source_id: str) -> None: ...
    def buff_minion(self, entity_id: str, attack: int, health: int, source_id: str) -> None: ...
    def give_divine_shield(self, entity_id: str) -> None: ...
    def leftmost_hand_minion(self) -> str | None: ...


@dataclass
class OwnedTrinket:
    card_id: str
    acquired_turn: int
    counter: int = 0
    done: bool = False
    turn_counter: int = 0


@dataclass(frozen=True)
class TrinketChoice:
    source_id: str
    options: tuple[str, ...]


class TrinketController:
    """Per-player seasonal controller. Adapter must deliver each event once.

    Purchase and discovery each correspond to ONE timed action. All game calls
    must stop for pending_choice before further recruit actions. Automatic effect
    grants consume no extra clicks, but never auto-play a generated card.
    """
    full_game_ready = False

    def __init__(self, host: TrinketHost, catalog: Mapping[str, TrinketSpec], *,
                 allowed_ids: Sequence[str], seed: int):
        if not set(allowed_ids) <= set(CORE_EFFECTS) or not set(allowed_ids) <= set(catalog):
            raise TrinketError("Scope contains unsupported effects or noncurrent IDs")
        self.host, self.catalog = host, catalog
        self.allowed_ids = frozenset(allowed_ids)
        self.rng = random.Random(seed)
        self.goldenizer_id = "BG26_813t"
        self.owned: list[OwnedTrinket] = []
        self.offering: tuple[TrinketOffer, ...] = ()
        self.turn = 0
        self.last_tavern_spell: str | None = None
        self._choices: deque[TrinketChoice] = deque()
        self._events: set[str] = set()
        self._offered_slots: set[str] = set()
        self._active_slot: str | None = None
        self._active_stage: str | None = None
        self._purchased_slots: set[str] = set()
        self._minions_played = 0
        self._turn_ended = True
        self.next_stage_override: str | None = None
        self.greater_offer_turn = 9
        self.trace: list[dict[str, Any]] = []

    @property
    def pending_choice(self) -> TrinketChoice | None:
        return self._choices[0] if self._choices else None

    def _record_event(self, event_id: str) -> None:
        if not event_id or event_id in self._events:
            raise TrinketError("Missing or duplicate event ID; abort the trajectory")
        self._events.add(event_id)

    def scheduled_offering(self) -> tuple[str, str] | None:
        """Return a due slot/stage; the adapter still must construct four offers."""
        for slot, due, stage in (("normal_lesser", 6, "lesser"),
                                 ("normal_greater", self.greater_offer_turn, "greater")):
            if self.turn == due and slot not in self._offered_slots:
                return slot, self.next_stage_override or stage
        return None

    def install_offering(self, offers: Sequence[TrinketOffer], ctx: OfferContext, *,
                         slot: str, stage: str, special_schedule: bool = False) -> None:
        if not special_schedule and slot not in {"normal_lesser", "normal_greater"}:
            raise TrinketError("Unrecognized normal offering slot")
        if self._turn_ended or self.offering or self.pending_choice or slot in self._offered_slots:
            raise TrinketError("Pending choice or repeated offering slot")
        if ctx.turn != self.turn or ctx.owned_ids != frozenset(t.card_id for t in self.owned):
            raise TrinketError("Stale offering context")
        required_stage = self.next_stage_override or stage
        if required_stage != stage:
            raise TrinketError("Trinket stage override must be honored")
        if not special_schedule and ctx.turn != (6 if slot == "normal_lesser" else self.greater_offer_turn):
            raise TrinketError("Normal trinket offers occur only on turns 6 and 9")
        if slot == "normal_lesser" and stage != "lesser":
            raise TrinketError("Normal turn 6 slot must offer Lesser trinkets")
        if slot == "normal_greater" and stage != (self.next_stage_override or "greater"):
            raise TrinketError("Normal turn 9 slot has incorrect trinket stage")
        if any(o.card_id not in self.allowed_ids for o in offers):
            raise TrinketError("Unimplemented trinket offered; refusing reduced/no-op trajectory")
        validate_offering(offers, self.catalog, ctx, stage)
        self.offering = tuple(offers)
        self._active_slot, self._active_stage = slot, stage
        self._offered_slots.add(slot)

    def buy_trinket(self, card_id: str) -> None:
        if self._turn_ended or self.pending_choice:
            raise TrinketError("No open turn or a pending discovery must be resolved")
        offer = next((o for o in self.offering if o.card_id == card_id), None)
        if offer is None or card_id not in self.allowed_ids:
            raise TrinketError("Trinket is not in the current offering")
        if offer.price > self.host.gold:
            raise TrinketError("Insufficient gold")
        self.host.spend_gold(offer.price, source_id=card_id)
        self._gold_spent_effects(offer.price)
        item = OwnedTrinket(card_id, self.turn)
        self.owned.append(item)
        self._purchased_slots.add(self._active_slot)
        self.offering = ()
        self.next_stage_override = None
        self.trace.append({"event": "buy_trinket", "turn": self.turn, "card_id": card_id, "price": offer.price})
        self._apply(item, "acquire")

    def choose(self, index: int) -> str:
        choice = self.pending_choice
        if choice is None or isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(choice.options):
            raise TrinketError("Invalid pending discovery choice")
        cid = choice.options[index]
        self.host.give_card(cid, choice.source_id)
        self.host.card_discovered(cid)
        self._choices.popleft()
        self.trace.append({"event": "choose", "turn": self.turn, "source_id": choice.source_id, "card_id": cid})
        return cid

    def start_turn(self, turn: int, *, event_id: str) -> None:
        if (type(turn) is not int or turn < 1 or (self.turn and turn != self.turn + 1)
                or not self._turn_ended or self.offering or self.pending_choice):
            raise TrinketError("Cannot advance through an unresolved or nonincreasing turn")
        self._record_event(event_id)
        self.turn = turn
        self._turn_ended = False
        self._minions_played = 0
        for item in self.owned:
            item.turn_counter = 0
            self._apply(item, "start")

    def end_turn(self, *, event_id: str) -> None:
        if self._turn_ended or self.offering or self.pending_choice:
            raise TrinketError("Timeout resolution requires an explicit validated adapter policy")
        self._record_event(event_id)
        self._turn_ended = True
        for item in self.owned:
            self._apply(item, "end")

    def _require_action_window(self) -> None:
        if self._turn_ended or self.offering or self.pending_choice:
            raise TrinketError("Recruit event outside an open action window")

    def on_refresh(self, *, event_id: str) -> None:
        self._require_action_window()
        self._record_event(event_id)
        for item in self.owned:
            if CORE_EFFECTS[item.card_id] == "frigid_blossom":
                self.host.reduce_upgrade_cost(1)

    def on_tier_changed(self, *, event_id: str) -> None:
        self._require_action_window()
        self._record_event(event_id)
        for item in self.owned:
            self._apply(item, "tier_changed")

    def quote_minion_buy(self, minion: MinionView, base_price: int, *, payment_resource: str = "gold") -> int:
        """Pure price query. Consume Warcry's allowance only after a real buy."""
        self._require_action_window()
        if type(base_price) is not int or base_price < 0:
            raise TrinketError("Invalid base buy price")
        for item in self.owned:
            if CORE_EFFECTS[item.card_id] == "warcry_totem" and "BATTLECRY" in minion.keywords:
                if payment_resource != "gold":
                    raise TrinketError("Warcry Totem with Health-payment replacement requires conformance")
                if item.turn_counter < 2:
                    return 0
        return base_price

    def on_minion_bought(self, minion: MinionView, *, base_price: int, paid_gold: int,
                         event_id: str, payment_resource: str = "gold") -> None:
        self._require_action_window()
        expected = self.quote_minion_buy(minion, base_price, payment_resource=payment_resource)
        if type(paid_gold) is not int or paid_gold != expected:
            raise TrinketError("Adapter paid a price inconsistent with the Trinket quote")
        self._record_event(event_id)
        for item in self.owned:
            kind = CORE_EFFECTS[item.card_id]
            if kind == "warcry_totem" and "BATTLECRY" in minion.keywords:
                item.turn_counter += 1
            elif kind == "secret_schematic" and _has_type(minion, "MECH"):
                self._random_spell(item.card_id)

    def on_minion_sold(self, minion: MinionView, *, event_id: str) -> None:
        self._require_action_window()
        self._record_event(event_id)
        for item in self.owned:
            if CORE_EFFECTS[item.card_id] == "lava_lamp":
                item.counter += 1
                if item.counter == self.catalog[item.card_id].parameters[1]:
                    item.counter = 0
                    self._random_pool_minion(item.card_id, tribe="ELEMENTAL")

    def on_card_played(self, *, event_id: str) -> None:
        """One hand-play event. Automatic spell casts do not call this hook."""
        self._require_action_window()
        self._record_event(event_id)
        for item in self.owned:
            if CORE_EFFECTS[item.card_id] == "dragonwing_glider":
                self._buff_random_board(item, "DRAGON", 1)

    def on_gold_spent(self, amount: int, *, event_id: str, source_kind: str = "recruit_action") -> None:
        """One payment event; trinket-purchase echoes are handled internally."""
        if source_kind == "trinket_purchase":
            self._record_event(event_id)
            return
        self._require_action_window()
        if type(amount) is not int or not 0 <= amount <= 99:
            raise TrinketError("Gold-spend event must be an integer within the held-gold cap")
        self._record_event(event_id)
        self._gold_spent_effects(amount)

    def _gold_spent_effects(self, amount: int) -> None:
        for _ in range(amount):
            for item in self.owned:
                if CORE_EFFECTS[item.card_id].startswith("booty_bay"):
                    self._buff_random_board(item, "PIRATE", 2)

    def _buff_random_board(self, item: OwnedTrinket, tribe: str, count: int) -> None:
        candidates = [m.entity_id for m in self.host.board_minions() if _has_type(m, tribe)]
        a, h = self.catalog[item.card_id].parameters[:2]
        for target in self.rng.sample(candidates, min(count, len(candidates))):
            self.host.buff_minion(target, a, h, item.card_id)

    def _random_spell(self, source_id: str) -> None:
        candidates = sorted(set(self.host.discover_spell_options()))
        if not candidates:
            raise TrinketError("No validated current random-spell pool")
        self.host.give_card(self.rng.choice(candidates), source_id)

    def _random_pool_minion(self, source_id: str, *, tribe: str | None = None, keyword: str | None = None) -> None:
        candidates = sorted(set(self.host.minion_generation_options(tribe=tribe, keyword=keyword)))
        if not candidates:
            self.trace.append({"event": "generation_pool_exhausted", "source_id": source_id, "turn": self.turn})
            return
        self.host.give_pool_minion(self.rng.choice(candidates), source_id)

    def on_minion_played(self, entity_id: str, *, event_id: str, minion: MinionView | None = None) -> None:
        self._require_action_window()
        self._record_event(event_id)
        self._minions_played += 1
        played = minion or self.host.minion_view(entity_id)
        if played.entity_id != entity_id:
            raise TrinketError("Played minion view identity mismatch")
        for item in self.owned:
            kind = CORE_EFFECTS[item.card_id]
            if kind.startswith("nomi") and _has_type(played, "ELEMENTAL"):
                a, h = self.catalog[item.card_id].parameters[:2]
                self.host.buff_tavern(a, h, item.card_id, "ELEMENTAL")
            if kind == "water_wheel" and _has_type(played, "ELEMENTAL") and item.turn_counter < 2:
                item.turn_counter += 1
                self._random_spell(item.card_id)
            if kind == "recycling_sticker" and _has_type(played, "ELEMENTAL"):
                self.host.gain_free_refresh(1)
            if kind == "cookie_rod" and _has_type(played, "MURLOC"):
                item.counter += 1
                if item.counter == self.catalog[item.card_id].parameters[0]:
                    item.counter = 0
                    self._random_spell(item.card_id)
                    self._random_spell(item.card_id)
            if kind == "defensive_shell" and self._minions_played == 1:
                self.host.give_divine_shield(entity_id)
            if kind.startswith("phrasebook"):
                target = self.host.leftmost_hand_minion()
                if target is not None:
                    amount = 3 if kind.endswith("lesser") else 6
                    self.host.buff_minion(target, amount, amount, item.card_id)

    def on_spell_cast(self, card_id: str, *, tavern_spell: bool, target_minion_id: str | None,
                      event_id: str, in_combat: bool = False) -> None:
        self._require_action_window()
        self._record_event(event_id)
        if in_combat:
            raise TrinketError("Combat-cast persistence and event ordering require a separate validated bridge")
        if tavern_spell:
            self.last_tavern_spell = card_id
        for item in self.owned:
            kind = CORE_EFFECTS[item.card_id]
            if tavern_spell and kind == "demonic_bloodletter":
                a, h = self.catalog[item.card_id].parameters[:2]
                self.host.buff_tavern(a, h, item.card_id, None)
            if tavern_spell and kind.startswith("coffin"):
                self.host.add_undead_attack(1 if kind.endswith("lesser") else 2)
            if target_minion_id is not None and kind.startswith("lorewalker"):
                a, h = self.catalog[item.card_id].parameters[:2]
                self.host.buff_minion(target_minion_id, a, h, item.card_id)
            if target_minion_id is not None and kind == "wand":
                item.counter += 1
                if item.counter == 3:
                    item.counter = 0
                    self.host.gain_gold(1)

    def _discover_spell(self, source_id: str) -> None:
        candidates = sorted(set(self.host.discover_spell_options()))
        if not candidates:
            raise TrinketError("No validated spell-discovery candidates")
        options = tuple(self.rng.sample(candidates, min(3, len(candidates))))
        self._choices.append(TrinketChoice(source_id, options))

    def _apply(self, item: OwnedTrinket, event: str) -> None:
        kind = CORE_EFFECTS[item.card_id]
        spec = self.catalog[item.card_id]
        if kind in {"butchers_sickle", "wisdomball_supply"} and event in {"acquire", "start"}:
            token = "BG28_604" if kind == "butchers_sickle" else "BG30_802"
            self.host.give_card(token, item.card_id)
        elif kind.startswith("great_boar") and event == "acquire":
            self.host.increase_gem_bonus(*spec.parameters[:2])
            for _ in range(3 if kind.endswith("lesser") else 5):
                self.host.give_card("BG20_GEM", item.card_id)
        elif kind == "glowing_crystal" and event == "start":
            self.host.gain_gold(distinct_minion_type_count(self.host.board_minions()))
        elif kind == "worn_treasure_map" and event == "start" and not item.done:
            item.counter += 1
            if item.counter == 2:
                item.done = True
                self.host.gain_gold(10)
        elif kind == "ornate_clock" and event == "acquire":
            if "normal_greater" in self._offered_slots:
                raise TrinketError("Ornate Clock cannot reschedule an already offered Greater slot")
            self.host.gain_gold(2)
            self.greater_offer_turn = self.turn + 1
        elif kind == "rockin_music_box" and event in {"acquire", "start"}:
            self._random_pool_minion(item.card_id, keyword="BATTLECRY")
        elif kind == "oilcan" and event in {"acquire", "start"}:
            self.host.reduce_upgrade_cost(3)
        elif kind == "wallet" and event == "end":
            self.host.increase_income_cap(1)
        elif kind == "tip_jar" and event == "acquire":
            self.host.increase_income_cap(4)
            self.host.gain_gold(4)
        elif kind == "headstone" and event == "end":
            self.host.add_undead_attack(2)
        elif kind == "coin_purse" and event in {"acquire", "tier_changed"} and not item.done:
            if self.host.tavern_tier >= spec.parameters[0]:
                item.done = True
                self.host.gain_gold(spec.parameters[1])
        elif kind == "mysterious_orb" and event == "acquire":
            self.host.gain_gold(spec.parameters[0])
            self.next_stage_override = "lesser"
        elif kind.startswith("book") and event in {"acquire", "start"}:
            for _ in range(1 if kind.endswith("lesser") else 2):
                self._discover_spell(item.card_id)
        elif kind == "sphere" and event == "end" and self.last_tavern_spell is not None:
            for _ in range(spec.parameters[0]):
                self.host.give_card(self.last_tavern_spell, item.card_id)
        elif kind == "goldenizer_supply" and event == "end":
            item.counter += 1
            if item.counter == spec.parameters[1]:
                item.counter = 0
                self.host.give_card(self.goldenizer_id, item.card_id)


def _has_type(minion: MinionView, tribe: str) -> bool:
    return tribe in minion.types or "ALL" in minion.types


def distinct_minion_type_count(minions: Sequence[MinionView]) -> int:
    """Maximum one-minion-per-type matching; an Amalgam contributes at most one.

    This avoids treating one dual-type minion as two separate representatives.
    Matching changes no game RNG state, since only the cardinality is needed.
    """
    assignments: dict[str, int] = {}
    types = [TRIBES if "ALL" in m.types else m.types & TRIBES for m in minions]
    def augment(index: int, seen: set[str]) -> bool:
        for tribe in sorted(types[index]):
            if tribe in seen:
                continue
            seen.add(tribe)
            if tribe not in assignments or augment(assignments[tribe], seen):
                assignments[tribe] = index
                return True
        return False
    return sum(augment(i, set()) for i in range(len(minions)))


@dataclass(frozen=True)
class CurrentTavernBuff:
    """hsrl2-compatible descriptor retaining authoritative dual-type matching."""
    atk: int
    health: int
    source_id: str
    tribe: str | None = None

    def matches(self, minion_def: Any) -> bool:
        if "races" not in minion_def.raw:
            raise TrinketError("Persistent Tavern buffs require authoritative current race metadata")
        races = set(minion_def.raw["races"])
        return self.tribe is None or self.tribe in races or "ALL" in races


class HSRLTrinketHost:
    """Mutations against an external hsrl2 Game; no source is copied or patched.

    The caller supplies verified spell discovery IDs and must call
    flush_generated_cards after each action that creates hand space. The upstream
    queue is otherwise flushed only on turn start, which is insufficient here.
    This bridge supplies no broad conformance or full-game authorization.
    """
    def __init__(self, game: Any, hero: Any, *, spell_ids: Sequence[str],
                 minion_ids: Sequence[str] | None = None):
        from hsrl2.tags import GameTag
        self.game, self.hero, self.tags = game, hero, GameTag
        self.spell_ids = tuple(spell_ids)
        self.minion_ids = tuple(minion_ids) if minion_ids is not None else None
        for cid in self.minion_ids or ():
            card = game.db.get(cid)
            if card is None or not card.is_pool_minion or "shop_minion_ids" not in card.raw.get("current_roles", ()):
                raise TrinketError(f"Invalid explicit current minion-generation candidate {cid}")
        for cid in self.spell_ids:
            card = game.db.get(cid)
            if card is None or not card.is_pool_spell:
                raise TrinketError(f"Invalid explicit spell candidate {cid}")

    @property
    def gold(self) -> int:
        return self.hero.gold

    @property
    def tavern_tier(self) -> int:
        return self.hero.tavern_tier

    def spend_gold(self, amount: int, *, source_id: str | None = None) -> None:
        if amount < 0 or self.hero.gold < amount:
            raise TrinketError("Invalid trinket payment")
        self.hero.gold -= amount
        self.hero.set(self.tags.GOLD_SPENT_THIS_TURN, self.hero.get(self.tags.GOLD_SPENT_THIS_TURN) + amount)
        if amount:
            self.game.events.fire(self.game, "gold_spent", hero=self.hero, amount=amount,
                                  source_kind="trinket_purchase" if source_id else "recruit_action", source_id=source_id)

    def gain_gold(self, amount: int) -> None:
        self.hero.gold += amount

    def increase_income_cap(self, amount: int) -> None:
        tag = self.tags.INCOME_CAP_BONUS
        self.hero.set(tag, self.hero.get(tag) + amount)

    def reduce_upgrade_cost(self, amount: int) -> None:
        self.hero.set(self.tags.UPGRADE_COST, max(0, self.hero.upgrade_cost - amount))

    def add_undead_attack(self, amount: int) -> None:
        from hsrl2.tags import Race
        if not hasattr(self.hero, "race_auras"):
            self.hero.race_auras = {}
        a, h = self.hero.race_auras.get(Race.UNDEAD, (0, 0))
        self.hero.race_auras[Race.UNDEAD] = (a + amount, h)

    def give_card(self, card_id: str, source_id: str) -> None:
        from hsrl2.tags import CardType
        card = self.game.db.get(card_id)
        if card is None:
            raise TrinketError(f"Generated card definition missing: {card_id}")
        if card.card_type == CardType.SPELL:
            entity = self.game.create_spell(card_id, controller=self.hero)
        else:
            raise TrinketError("This bridge currently grants only audited spell tokens")
        entity.trinket_source_id = source_id
        self.game.pending_hand_add(self.hero, entity)

    def flush_generated_cards(self) -> None:
        if not self.hero.hand_full() and any(
                h is self.hero and not getattr(e, "trinket_source_id", None)
                for h, e in self.game.pending_hand_queue):
            raise TrinketError("Mixed-source generated-card queue requires an integrated FIFO adapter")
        from hsrl2.minion import Minion
        pending = list(self.game.pending_hand_queue)
        self.game.pending_hand_queue = []
        waiting = []
        # Preserve FIFO, and retain new queue entries created by a triple hook.
        for hero, entity in pending:
            if hero is self.hero and getattr(entity, "trinket_source_id", None) and not hero.hand_full():
                hero.add_to_hand(entity)
                if isinstance(entity, Minion):
                    self.game.check_for_triple(hero, entity)
            else:
                waiting.append((hero, entity))
        self.game.pending_hand_queue = waiting + self.game.pending_hand_queue

    def discover_spell_options(self) -> Sequence[str]:
        return [cid for cid in self.spell_ids if self.game.db.get(cid).tech_level <= self.hero.tavern_tier]

    def minion_view(self, entity_id: str) -> MinionView:
        from hsrl2.tags import Race
        entity = self._entity(entity_id)
        definition = self.game.db.get(entity.card_id)
        if definition is None or "races" not in definition.raw:
            raise TrinketError("Minion view requires authoritative current type metadata")
        races = frozenset(definition.raw["races"])
        if entity.race != definition.race:
            races = frozenset() if entity.race == Race.NONE else frozenset({entity.race.name})
        keywords = frozenset(k.upper() for k in definition.keywords)
        if entity.has(self.tags.SILENCED):
            keywords = frozenset()
        return MinionView(entity.uuid, entity.card_id, races, entity.atk, entity.health,
                          entity.tech_level, keywords)

    def board_minions(self) -> Sequence[MinionView]:
        return [self.minion_view(m.uuid) for m in self.hero.board if not m.dead]

    def increase_gem_bonus(self, attack: int, health: int) -> None:
        for tag, value in ((self.tags.BLOOD_GEM_BONUS_ATK, attack),
                           (self.tags.BLOOD_GEM_BONUS_HEALTH, health)):
            self.hero.set(tag, self.hero.get(tag) + value)

    def gain_free_refresh(self, amount: int) -> None:
        self.hero.set(self.tags.FREE_REFRESH_REMAINING, self.hero.get(self.tags.FREE_REFRESH_REMAINING) + amount)

    def buff_tavern(self, attack: int, health: int, source_id: str, tribe: str | None) -> None:
        from hsrl2.entity import Buff
        from hsrl2.minion import Minion
        buff = CurrentTavernBuff(attack, health, source_id, tribe)
        matching = [m for m in self.hero.tavern if isinstance(m, Minion) and buff.matches(self.game.db.get(m.card_id))]
        if not hasattr(self.hero, "tavern_buffs"):
            self.hero.tavern_buffs = []
        self.hero.tavern_buffs.append(buff)
        for minion in matching:
            minion.add_buff(Buff(attack, health, source_id=source_id))

    def minion_generation_options(self, *, tribe: str | None = None, keyword: str | None = None) -> Sequence[str]:
        if self.minion_ids is None:
            raise TrinketError("Generated-minion eligibility must be supplied explicitly")
        pool = self.game.minion_pool
        result = []
        for cid in self.minion_ids:
            card = self.game.db.get(cid)
            if pool.available(cid) <= 0:
                continue
            races = set(card.raw["races"])
            if pool.active_races is not None:
                active = {race.name for race in pool.active_races}
                if races and "ALL" not in races and not (races & active):
                    continue
            if tribe and tribe not in card.raw["races"] and "ALL" not in card.raw["races"]:
                continue
            if keyword and keyword.lower() not in card.keywords:
                continue
            result.append(cid)
        return result

    def give_pool_minion(self, card_id: str, source_id: str) -> None:
        if self.minion_ids is None or card_id not in self.minion_ids:
            raise TrinketError("Minion grant is outside the explicit generation pool")
        if not self.game.minion_pool.acquire(card_id):
            raise TrinketError("Generated minion has no remaining shared-pool copy")
        entity = self.game.create_minion(card_id, controller=self.hero)
        entity.trinket_source_id = source_id
        self.game.pending_hand_add(self.hero, entity)

    def card_discovered(self, card_id: str) -> None:
        self.game.events.fire(self.game, "card_discovered", kind="discover_trinket_spell", hero=self.hero, option=card_id)

    def _entity(self, entity_id: str) -> Any:
        for entity in self.hero.board + self.hero.hand + self.hero.tavern:
            if entity.uuid == entity_id:
                return entity
        raise TrinketError("Target is not a visible friendly or Tavern entity")

    def buff_minion(self, entity_id: str, attack: int, health: int, source_id: str) -> None:
        from hsrl2.entity import Buff
        self._entity(entity_id).add_buff(Buff(attack, health, source_id=source_id))

    def give_divine_shield(self, entity_id: str) -> None:
        self._entity(entity_id).set(self.tags.DIVINE_SHIELD, True)

    def leftmost_hand_minion(self) -> str | None:
        from hsrl2.minion import Minion
        return next((x.uuid for x in self.hero.hand if isinstance(x, Minion)), None)
