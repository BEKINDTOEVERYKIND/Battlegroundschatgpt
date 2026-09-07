#!/usr/bin/env python3
"""Recommend one offline opening action, honoring the run's deployment decision."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
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

from bg_ai.choice_timing import CHOICE_TIMING_VERSION, opening_choice_completion
from bg_ai.hsbrsim_opening import GENERATED_RECRUIT, TIER1_MINIONS, TIER1_SPELLS
from bg_ai.learning import Ranker
from bg_ai.recruit_features import action_card
from bg_ai.recruiting import RecruitAction, RecruitObservation, ruleset_digest
from bg_ai.turn_budget import TimingProfile
from bg_ai.two_turn_features_v2 import (
    TWO_TURN_FEATURE_NAMES, TWO_TURN_FEATURE_VERSION, encode_two_turn_actions,
    enrich_two_turn_observation, two_turn_schema_id,
)
from train_recruit_curriculum import candidates_for, heuristic_action


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _count(value, field):
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be an explicit nonnegative integer")
    return value


def _followups(observation, action):
    if action.kind == "end_turn":
        return 0
    if action.kind == "choose":
        return observation.action_budget["reserved_choice_commands"] - 1
    completion = opening_choice_completion(observation, action)
    return len(completion.kinds) if completion else 0


def _fits(observation, action, profile):
    if action.kind == "end_turn":
        return True
    following = _followups(observation, action)
    return (observation.action_budget["remaining_actions"] >= 1 + following
        and observation.action_budget["remaining_ms"] >= profile.action_ms(action.kind) + following * profile.action_ms("choose"))


def validate_budget(observation, profile, remaining_ms=None):
    """Validate the archived clock, then optionally tighten it with a new timer."""
    budget = observation.action_budget
    if (not isinstance(budget, dict) or budget.get("profile_sha256") != profile.fingerprint
            or budget.get("choice_timing_version") != CHOICE_TIMING_VERSION
            or budget.get("actions_per_second") != profile.actions_per_second
            or budget.get("reserve_ms") != profile.reserve_ms
            or budget.get("action_cost_ms") != {kind: profile.action_ms(kind) for kind in profile.extra_ms}):
        raise ValueError("Missing or incompatible timing/choice profile")
    for field in ("initial_available_ms", "action_limit", "actions_used", "remaining_actions",
                  "remaining_ms", "charged_ms", "reserved_choice_commands", "reserved_choice_ms"):
        _count(budget.get(field), field)
    observed = _count(observation.time_remaining_ms, "time_remaining_ms")
    usable = max(0, budget["initial_available_ms"] - profile.reserve_ms)
    expected_limit = math.floor(usable * profile.actions_per_second / 1000)
    if (budget["action_limit"] != expected_limit or budget["actions_used"] > expected_limit
            or not budget["actions_used"] * profile.base_ms <= budget["charged_ms"] <= usable
            or budget["remaining_ms"] > min(usable - budget["charged_ms"], max(0, observed-profile.reserve_ms))
            or budget["remaining_actions"] != min(expected_limit-budget["actions_used"], budget["remaining_ms"] // profile.base_ms)):
        raise ValueError("Snapshot budget exceeds or disagrees with its finite turn window")
    pending = observation.private_state.get("pending_choice")
    count = budget["reserved_choice_commands"]
    if pending:
        required = {"choose_one": 2, "spell_target": 1, "activate_target": 1, "discover_minion": 1}.get(pending.get("kind"))
        if count != required or any(a.kind != "choose" for a in observation.legal_actions):
            raise ValueError("Pending choice and reserved explicit commands disagree")
    elif count or any(a.kind == "choose" for a in observation.legal_actions):
        raise ValueError("Choice reservation lacks its visible pending choice")
    if budget["reserved_choice_ms"] != count * profile.action_ms("choose"):
        raise ValueError("Reserved choice cost disagrees with the timing profile")
    if not observation.legal_actions or len(set(observation.legal_actions)) != len(observation.legal_actions):
        raise ValueError("A unique nonempty recorded legal mask is required")
    if any(not _fits(observation, action, profile) for action in observation.legal_actions):
        raise ValueError("Recorded legal mask contains an unaffordable complete command sequence")
    if remaining_ms is not None:
        remaining_ms = _count(remaining_ms, "remaining_ms override")
        budget = dict(budget)
        budget["remaining_ms"] = min(budget["remaining_ms"], max(0, remaining_ms-profile.reserve_ms))
        budget["remaining_actions"] = min(budget["action_limit"]-budget["actions_used"], budget["remaining_ms"] // profile.base_ms)
        observation = replace(observation, time_remaining_ms=min(observed, remaining_ms), action_budget=budget)
        observation = replace(observation, legal_actions=tuple(a for a in observation.legal_actions if _fits(observation, a, profile)))
        if not observation.legal_actions:
            raise ValueError("Observed timer no longer permits the reserved choice; request a fresh engine observation")
    return observation


def load_run_policy(run_dir, ruleset_path, profile, *, experimental=False):
    selection = json.loads((run_dir / "deployment_policy.json").read_text())
    checkpoint = run_dir / "experimental_model.json"
    expected_hash = selection.get("experimental_checkpoint_sha256")
    if not isinstance(expected_hash, str) or digest(checkpoint) != expected_hash:
        raise ValueError("Frozen experimental checkpoint checksum mismatch")
    frozen = json.loads((run_dir / "frozen_evaluation_plan.json").read_text())
    frozen_hash = frozen.get("experimental_checkpoint_sha256", frozen.get("selection", {}).get("experimental_checkpoint_sha256"))
    if frozen.get("selection_complete_before_test") is not True or frozen_hash != expected_hash:
        raise ValueError("Checkpoint does not match the frozen evaluation selection")
    expected = {"feature_version": TWO_TURN_FEATURE_VERSION,
        "feature_schema_sha256": two_turn_schema_id(), "ruleset_file_sha256": digest(ruleset_path),
        "timing_profile_sha256": profile.fingerprint, "choice_timing_version": CHOICE_TIMING_VERSION,
        "full_game_ready": False}
    def validated_model(path):
        model = Ranker.load(path, TWO_TURN_FEATURE_NAMES)
        for key, value in expected.items():
            if model.metadata.get(key) != value:
                raise ValueError(f"Checkpoint {key} is incompatible with this offline fixture")
        if model.metadata.get("training_objective") not in {
                "positive_action_set_softmax_v1", "complete_episode_clipped_policy_gradient_frozen_bc_kl_v1"}:
            raise ValueError("This adviser requires an action-policy checkpoint, not a combat-value ranker")
        return model
    model = validated_model(checkpoint)
    declared = selection.get("deployment_policy")
    if declared == "bc_reference" and selection.get("existing_baseline") == "bc_reference":
        prior = json.loads((run_dir / "frozen_bc_prior_selection.json").read_text())
        prereg = json.loads((run_dir / "preregistration.json").read_text())
        reference_path = run_dir / "frozen_bc_reference.json"
        reference_hash = prereg.get("reference_sha256")
        if (not isinstance(prior, dict) or prior.get("experimental_checkpoint_sha256") != reference_hash
                or prior.get("independent_test_gate_passed") is not True
                or prior.get("learner_promoted_within_fixture") is not True
                or digest(reference_path) != reference_hash):
            raise ValueError("Retained BC reference lacks its exact previously passing selection")
        reference = validated_model(reference_path)
        if reference.metadata.get("training_objective") != "positive_action_set_softmax_v1":
            raise ValueError("Retained reference must be the frozen behavior-cloning actor")
        if not experimental:
            return reference, selection, reference_hash, "bc_reference"
    elif declared != "practical" and (declared != selection.get("experimental_learner")
            or selection.get("learner_promoted_within_fixture") is not True
            or selection.get("independent_test_gate_passed") is not True):
        raise ValueError("Declared actor deployment lacks a passing frozen-test decision")
    use_model = experimental or declared != "practical"
    return (model if use_model else None), selection, expected_hash, (selection["experimental_learner"] if use_model else "practical")


def recommend(record, run_dir, ruleset_path, profile, *, experimental=False, remaining_ms=None):
    model, selection, checkpoint_hash, policy_name = load_run_policy(Path(run_dir), Path(ruleset_path), profile, experimental=experimental)
    raw = dict(record)
    raw["legal_actions"] = tuple(RecruitAction(**action) for action in raw["legal_actions"])
    observation = RecruitObservation(**raw)
    ruleset = json.loads(Path(ruleset_path).read_text())
    if observation.ruleset_sha256 != ruleset_digest(ruleset):
        raise ValueError("Observation belongs to a different ruleset")
    if observation.turn not in (1, 2) or observation.public_state.get("scope") != "current-two-turn-tier1-opening-v1":
        raise ValueError("Observation is outside the trained two-turn opening scope")
    players = observation.public_state.get("players", [])
    if (len(players) != 8 or {p.get("player_id") for p in players} != set(range(8))
            or observation.player_id not in range(8)
            or any(p.get("hero_id") != "TB_BaconShop_HERO_34" or p.get("alive") is not True for p in players)):
        raise ValueError("The fixture requires eight living Patchwerk players")
    cards = {c["id"]: c for c in json.loads((ROOT / "data/reference_cards.json").read_text())}
    tribes = set(observation.public_state.get("valid_tribes", []))
    for zone in ("board", "hand", "shop"):
        permitted = TIER1_MINIONS if zone == "board" else TIER1_MINIONS | TIER1_SPELLS | (GENERATED_RECRUIT if zone == "hand" else set())
        for card in observation.private_state.get(zone, []):
            cid = card.get("card_id")
            if cid not in permitted:
                raise ValueError(f"Visible {zone} card exceeds this trained opening pool")
            races = set(cards[cid].get("races", []))
            if cid in TIER1_MINIONS and races and "ALL" not in races and not races & tribes:
                raise ValueError("Visible minion is outside the declared lobby tribes")
    observation = enrich_two_turn_observation(validate_budget(observation, profile, remaining_ms))
    actions = candidates_for(observation)
    features = encode_two_turn_actions(observation, actions)
    values = model.predict(features) if model is not None else None
    chosen = actions[int(np.argmax(values))] if model is not None else heuristic_action(observation)
    if chosen not in actions:
        raise ValueError("Chosen action escaped the canonical legal shortlist")
    cost = 0 if chosen.kind == "end_turn" else profile.action_ms(chosen.kind)
    following = _followups(observation, chosen)
    before = dict(observation.action_budget)
    after = dict(before)
    after.update(actions_used=before["actions_used"] + (chosen.kind != "end_turn"),
                 charged_ms=before["charged_ms"] + cost, remaining_ms=before["remaining_ms"] - cost,
                 reserved_choice_commands=following, reserved_choice_ms=following * profile.action_ms("choose"))
    after["remaining_actions"] = min(after["action_limit"]-after["actions_used"], after["remaining_ms"] // profile.base_ms)
    def describe(action):
        result = asdict(action)
        source, target = action_card(observation, action), action_card(observation, action, target=True)
        if source:
            result["card_name"] = source.get("name")
        if target:
            result["target_name"] = target.get("name")
        return result
    return {"scope": "offline current Tier1 two-turn action advice", "full_game_ready": False,
        "action_executed": False, "policy_used": policy_name,
        "declared_deployment_policy": selection["deployment_policy"], "experimental_override": experimental,
        "checkpoint_sha256": checkpoint_hash, "feature_version": TWO_TURN_FEATURE_VERSION,
        "chosen_action": describe(chosen), "chosen_action_cost_ms": cost,
        "forced_choice_commands_after_action": following,
        "reserved_choice_cost_ms_after_action": after["reserved_choice_ms"],
        "minimum_complete_sequence_cost_ms": cost + after["reserved_choice_ms"],
        "remaining_budget_before_action": before, "remaining_budget_after_command": after,
        "turn_closed_by_action": chosen.kind == "end_turn",
        "candidate_count": len(actions),
        "candidate_policy_logits": [{"action": describe(actions[int(i)]), "logit": float(values[int(i)])}
                                     for i in np.argsort(-values)] if values is not None else None,
        "score_definition": "Uncalibrated action-policy logits, not combat probabilities or full-game EV",
        "limitations": ["Saved fixture observations only; no live client integration or full-game MMR claim",
            "Recheck the actual timer and legal state immediately before any manual action",
            "Only the chosen command is advised; mandatory choices need fresh explicit decisions and separate charges",
            "A practical fallback remains active unless the run records a passing independent-test decision or --experimental is explicit"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--experimental", action="store_true")
    parser.add_argument("--remaining-ms", type=int, help="New actual timer reading; can only reduce the recorded budget")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--allow-historical", action="store_true")
    args = parser.parse_args(argv)
    if not args.allow_historical:
        subprocess.run([sys.executable, str(ROOT / "scripts/check_live_ruleset.py")], cwd=ROOT,
                       stdout=subprocess.DEVNULL, check=True)
    result = recommend(json.loads(args.observation.read_text()), args.run_dir, ROOT / "data/ruleset.json",
        TimingProfile.load(ROOT / "config/turn-budget.json"), experimental=args.experimental, remaining_ms=args.remaining_ms)
    result["current_sources_checked"] = not args.allow_historical
    body = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(body)
    print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
