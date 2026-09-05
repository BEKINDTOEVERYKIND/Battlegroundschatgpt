#!/usr/bin/env python3
"""Inspect one experimental finite-budget recruit decision from a saved snapshot.

This advises one first action in the explicitly restricted early fixture. It
neither executes the action nor claims a complete recruit-turn or lobby policy.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from bg_ai.hsbrsim_adapter import (FIXTURE_SCOPE, SUPPORTED_MINIONS, SUPPORTED_SPELLS,
                                  SUPPORTED_GENERATED)
from bg_ai.learning import Ranker
from bg_ai.recruit_features import (RECRUIT_FEATURE_NAMES, RECRUIT_FEATURE_VERSION,
    RECRUIT_V2_FEATURE_NAMES, encode_legal_actions, encode_legal_actions_v2, action_card)
from bg_ai.recruiting import RecruitAction, RecruitObservation, ruleset_digest
from bg_ai.turn_budget import TimingProfile
from train_recruit_curriculum import candidates_for, heuristic_action


def recommend(record, checkpoint, ruleset, profile):
    record = dict(record)
    record["legal_actions"] = tuple(RecruitAction(**a) for a in record["legal_actions"])
    observation = RecruitObservation(**record)
    if observation.ruleset_sha256 != ruleset_digest(ruleset):
        raise ValueError("Observation belongs to another ruleset")
    if observation.public_state.get("scope") != FIXTURE_SCOPE or observation.turn not in (1, 2):
        raise ValueError("This checkpoint is restricted to the declared early recruit fixture")
    player = next((p for p in observation.public_state.get("players", ()) if p["player_id"] == observation.player_id), None)
    if player is None or player.get("hero_id") != "TB_BaconShop_HERO_34":
        raise ValueError("This recruit policy was trained only with Patchwerk")
    if not observation.action_budget or observation.action_budget.get("profile_sha256") != profile.fingerprint:
        raise ValueError("Unknown or missing timing profile; no unbounded advice is allowed")
    budget = observation.action_budget
    for key in ("remaining_ms", "remaining_actions", "actions_used", "action_limit", "initial_available_ms"):
        if type(budget.get(key)) is not int or budget[key] < 0:
            raise ValueError("Invalid finite action budget")
    expected_limit = math.floor(max(0, budget["initial_available_ms"] - profile.reserve_ms) * profile.actions_per_second / 1000)
    if budget["action_limit"] != expected_limit or budget["remaining_actions"] > budget["action_limit"] - budget["actions_used"]:
        raise ValueError("Snapshot budget exceeds its supplied turn window")
    if any(a.kind != "end_turn" and (budget["remaining_actions"] < 1 or profile.action_ms(a.kind) > budget["remaining_ms"]) for a in observation.legal_actions):
        raise ValueError("Snapshot legal mask contains an unaffordable action")
    for zone in ("board", "shop", "hand"):
        permitted = SUPPORTED_MINIONS if zone == "board" else SUPPORTED_MINIONS | SUPPORTED_SPELLS | (SUPPORTED_GENERATED if zone == "hand" else frozenset())
        if any(c.get("card_id") not in permitted for c in observation.private_state.get(zone, ())):
            raise ValueError(f"The {zone} contains a card outside the trained fixture scope")
    raw = json.loads(Path(checkpoint).read_text())
    version = raw.get("metadata", {}).get("recruit_feature_version")
    if version == "recruit-visible-action-v2-choice-zone":
        names, encoder = RECRUIT_V2_FEATURE_NAMES, encode_legal_actions_v2
    elif version == RECRUIT_FEATURE_VERSION:
        names, encoder = RECRUIT_FEATURE_NAMES, encode_legal_actions
    else:
        raise ValueError("Unsupported recruit checkpoint feature version")
    model = Ranker.load(checkpoint, names)
    if model.metadata.get("timing_profile_sha256") != profile.fingerprint:
        raise ValueError("Checkpoint belongs to a different timing profile")
    actions = candidates_for(observation)
    features = encoder(observation, actions)
    values = model.predict(features)
    chosen = actions[int(np.argmax(values))]
    def describe(action):
        item = asdict(action)
        source = action_card(observation, action)
        target = action_card(observation, action, target=True)
        if source: item["card_name"] = source.get("name")
        if target: item["target_name"] = target.get("name")
        return item
    return {"scope": "experimental first action in the restricted current early-turn recruit fixture",
        "full_game_policy": False, "action_executed": False,
        "feature_version": version, "ruleset_sha256": observation.ruleset_sha256,
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "chosen_action": describe(chosen), "practical_heuristic_action": describe(heuristic_action(observation)),
        "remaining_budget_before_action": dict(observation.action_budget),
        "chosen_action_cost_ms": 0 if chosen.kind == "end_turn" else profile.action_ms(chosen.kind),
        "candidate_rank_values": [{"action": describe(actions[i]), "value": float(values[i])}
            for i in np.argsort(-values)],
        "value_definition": "Uncalibrated relative ranking values; not win probabilities or full-game EV",
        "limitations": ["A fixed heuristic follows the first learned action in the benchmark",
            "Seven-card fixture scope; broader pool and full-game strength require separate evaluation",
            "Consult the run results before treating this experimental checkpoint as stronger than the practical heuristic"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--allow-historical", action="store_true")
    args = parser.parse_args()
    if not args.allow_historical:
        subprocess.run([sys.executable, str(ROOT / "scripts/check_live_ruleset.py")], cwd=ROOT,
                       stdout=subprocess.DEVNULL, check=True)
    result = recommend(json.loads(args.observation.read_text()), args.checkpoint,
        json.loads((ROOT / "data/ruleset.json").read_text()), TimingProfile.load(ROOT / "config/turn-budget.json"))
    result["current_sources_checked"] = not args.allow_historical
    body = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(body)
    print(body)

if __name__ == "__main__":
    main()
