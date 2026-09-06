#!/usr/bin/env python3
"""Bounded two-turn policy-improvement pilot with actual persistent recruitment.

Candidate values use a complete two-combat rollout, and the trained policy then
acts at every decision in independently seeded held-out episodes. This is one
policy-improvement iteration, not full Battlegrounds training or a strength claim.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import numpy as np
from bg_ai.learning import Dataset, Ranker, Scenario
from bg_ai.hsbrsim_adapter import UnsupportedRecruitTransition
from bg_ai.hsbrsim_opening import OpeningFixtureSpec, TIER1_MINIONS, TIER1_SPELLS, TRIBES, build_opening_database
from bg_ai.opening_transition import TwoTurnOpeningRecruitEngine, TimedTwoTurnOpeningEngine
from bg_ai.recruit_features import observation_record
from bg_ai.turn_budget import TimingProfile
from bg_ai.two_turn_features import TWO_TURN_FEATURE_NAMES, TWO_TURN_FEATURE_VERSION, encode_two_turn_actions, two_turn_schema_id
from train_recruit_curriculum import candidates_for, heuristic_action, paired_summary, sha, write_json

SCOPE = "current complete Tier-1 pool, two recruit turns and sampled combats, eight Patchwerk players"
PAIRINGS = {1: ((0, 1), (2, 3), (4, 5), (6, 7)),
            2: ((0, 2), (1, 3), (4, 6), (5, 7))}


class DuplicateTrajectoryState(ValueError):
    """Expected whole-trajectory rejection for train/validation overlap."""


def load_policy_checkpoint(path, ruleset_path, profile):
    model = Ranker.load(path, TWO_TURN_FEATURE_NAMES)
    expected = {"feature_version": TWO_TURN_FEATURE_VERSION,
        "two_turn_feature_schema_sha256": two_turn_schema_id(),
        "ruleset_file_sha256": sha(ruleset_path), "timing_profile_sha256": profile.fingerprint}
    for key, value in expected.items():
        if model.metadata.get(key) != value:
            raise ValueError(f"Behavior checkpoint {key} does not match the verified pilot")
    return model


def training_model(behavior_model, hidden):
    if behavior_model is None:
        return Ranker(TWO_TURN_FEATURE_NAMES, hidden=hidden, seed=17)
    if behavior_model.params["w1"].shape[1] != hidden:
        raise ValueError("Requested hidden width must match the behavior checkpoint")
    return deepcopy(behavior_model)


def previous_evaluation_seeds(checkpoint, model):
    seeds = set(model.metadata.get("reserved_evaluation_seeds", []))
    sidecar = Path(checkpoint).parent / "frozen_evaluation_plan.json"
    if sidecar.exists():
        plan = json.loads(sidecar.read_text())
        if plan.get("checkpoint_sha256") == sha(checkpoint):
            seeds.update(plan["seeds"])
    return seeds


def reserve_evaluation_seeds(generation_seeds, evaluation_seeds, inherited):
    """An earlier holdout cannot become either new training or a reused test."""
    if (set(generation_seeds) | set(evaluation_seeds)) & set(inherited):
        raise ValueError("A fresh --seed is required: generation/evaluation overlaps inherited reserved test seeds")
    return sorted(set(inherited) | set(evaluation_seeds))


def episode_split(index):
    """Every decision and counterfactual sibling inherits its episode split."""
    return "validation" if index % 5 == 4 else "train"


def combat_seed(episode_seed, turn, sample=0):
    return (episode_seed * 2654435761 + turn * 32452843 + sample * 49979687) & 0xFFFFFFFF


def outcome_score(receipt, player=0):
    rows = [row for row in receipt["outcomes"] if player in (row["player_a"], row["player_b"])]
    if len(rows) != 1:
        raise ValueError("Player must occur exactly once per sampled phase")
    winner = rows[0]["winner_id"]
    return .5 if winner is None else float(winner == player)


def episode_score(receipts):
    if len(receipts) != 2 or [r["turn"] for r in receipts] != [1, 2]:
        raise ValueError("Objective requires both actual sampled combats")
    return sum(outcome_score(receipt) for receipt in receipts) / 2


class CombatBridge:
    """Persistent normal JSONL interface; receipts never enter model features."""
    def __init__(self, stderr_path):
        self.stderr = Path(stderr_path).open("w")
        self.process = subprocess.Popen(["node", "simulator/opening-transition-firestone.mjs"],
            cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr,
            text=True, bufsize=1)
        self.requests = 0

    def sample(self, request):
        self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Firestone receipt worker exited; inspect worker log")
        result = json.loads(line)
        if "error" in result:
            raise RuntimeError("Firestone receipt failure: " + result["error"])
        self.requests += 1
        return result

    def close(self):
        self.process.stdin.close()
        try:
            status = self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=5)
            raise RuntimeError("Receipt worker did not exit")
        finally:
            self.stderr.close()
        if status:
            raise RuntimeError(f"Receipt worker exited with status {status}")


def policy_action(observation, model=None):
    if model is None:
        return heuristic_action(observation)
    actions = candidates_for(observation)
    return actions[int(np.argmax(model.predict(encode_two_turn_actions(observation, actions))))]


def apply_combat(timed, bridge, episode_seed, sample=0):
    turn = timed.engine.game.turn
    request = timed.engine.combat_request(pairings=PAIRINGS[turn],
        seed=combat_seed(episode_seed, turn, sample))
    receipt = bridge.sample(request)
    timed.advance_combat(receipt)
    return {"request": request, "receipt": receipt}


def rollout(timed, bridge, episode_seed, *, model=None, first=None, sample=0):
    commands, combats = [], []
    before_trace = len(timed.trace)
    if first is not None:
        commands.append({"player_id": timed._visible.player_id,
            "turn": timed._visible.turn, "action": asdict(first)})
        timed.step(first)
    for _ in range(1500):
        obs = timed._visible
        if obs is None:
            if timed.engine.phase == "complete":
                return {"score": episode_score(timed.engine.combat_receipts),
                    "receipts": timed.engine.combat_receipts, "combats": combats,
                    "commands": commands, "timing_trace": timed.trace[before_trace:],
                    "final_players": [timed.engine.player_board(i) for i in range(8)]}
            if timed.engine.phase != "awaiting_combat":
                raise RuntimeError("Unexpected missing recruit observation")
            combats.append(apply_combat(timed, bridge, episode_seed, sample))
            continue
        action = policy_action(obs, model if obs.player_id == 0 else None)
        commands.append({"player_id": obs.player_id, "turn": obs.turn, "action": asdict(action)})
        timed.step(action)
    raise RuntimeError("Finite two-turn rollout exceeded 1500 transitions")


class Factory:
    def __init__(self, engine_root, ruleset, profile):
        self.engine_root, self.ruleset, self.profile = engine_root, ruleset, profile
        self.databases = {}

    def setup(self, seed):
        rng = random.Random(seed)
        tribes = tuple(sorted(rng.sample(sorted(TRIBES), 5)))
        timers = (rng.choice((15000, 20000, 30000)), rng.choice((15000, 20000, 30000)))
        if tribes not in self.databases:
            self.databases[tribes] = build_opening_database(self.engine_root, valid_tribes=tribes)
        database = self.databases[tribes]
        raw = TwoTurnOpeningRecruitEngine(database.db, self.ruleset,
            timer_ms=lambda player, turn: timers[turn - 1] if player == 0 else 60000,
            provenance=database.provenance, fixture=OpeningFixtureSpec(valid_tribes=tribes))
        timed = TimedTwoTurnOpeningEngine.for_fixture(raw, self.profile, self.ruleset)
        timed.reset(seed=seed)
        return timed, {"seed": seed, "valid_tribes": tribes, "player_zero_timer_ms": timers,
            "other_player_timer_ms": 60000, "provenance": database.provenance}


def state_fingerprint(vectors):
    return hashlib.sha256(json.dumps(sorted(vectors.tolist()), separators=(",", ":")).encode()).hexdigest()


def build_trajectory(factory, bridge, *, index, seed, samples,
                     behavior_model=None, continuation_model=None):
    """Any unsupported shortlisted action rejects all decisions from this seed."""
    timed, setup = factory.setup(seed)
    rows, actual_commands, actual_combats = [], [], []
    split = episode_split(index)
    for _ in range(1500):
        obs = timed._visible
        if obs is None:
            if timed.engine.phase == "complete":
                return {"episode_id": f"two-turn-{seed}", "index": index, "split": split,
                    "setup": setup, "decisions": rows,
                    "behavior_policy": "frozen_checkpoint" if behavior_model is not None else "practical_heuristic_v1",
                    "continuation_policy": "frozen_checkpoint" if continuation_model is not None else "practical_heuristic_v1",
                    "commands": actual_commands, "combats": actual_combats,
                    "timing_trace": timed.trace, "score": episode_score(timed.engine.combat_receipts)}
            actual_combats.append(apply_combat(timed, bridge, seed))
            continue
        action = policy_action(obs, behavior_model if obs.player_id == 0 else None)
        if obs.player_id == 0:
            choices = candidates_for(obs)
            vectors = encode_two_turn_actions(obs, choices)
            decision = {"decision_id": f"two-turn-{seed}-decision-{len(rows)}", "split": split,
                "episode_id": f"two-turn-{seed}", "turn": obs.turn,
                "observation": observation_record(obs), "candidates": [],
                "state_fingerprint": state_fingerprint(vectors)}
            for candidate, vector in zip(choices, vectors):
                branches = [rollout(timed.fork(), bridge, seed, first=candidate, sample=sample,
                                    model=continuation_model)
                            for sample in range(samples)]
                decision["candidates"].append({"action": asdict(candidate),
                    "features": vector.tolist(), "score": float(np.mean([b["score"] for b in branches])),
                    "rollouts": branches})
            rows.append(decision)
        actual_commands.append({"player_id": obs.player_id, "turn": obs.turn, "action": asdict(action)})
        timed.step(action)
    raise RuntimeError("Finite behavior trajectory exceeded 1500 transitions")


def check_split_overlap(trajectory, seen):
    """Reject whole trajectory on a duplicate visible decision across splits."""
    for decision in trajectory["decisions"]:
        previous = seen.get(decision["state_fingerprint"])
        if previous is not None and previous != trajectory["split"]:
            raise DuplicateTrajectoryState("Equivalent visible decision across trajectory splits")


def write_line(handle, record):
    handle.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
    handle.flush()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/20260906-two-turn-recruit-v1")
    parser.add_argument("--trajectories", type=int, default=40)
    parser.add_argument("--label-samples", type=int, default=2)
    parser.add_argument("--evaluation-episodes", type=int, default=16)
    parser.add_argument("--evaluation-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=202609061)
    parser.add_argument("--epochs", type=int, nargs="+", default=[5, 15])
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--policy-checkpoint", type=Path,
                        help="Exact-schema current checkpoint for every collection decision and independent warm fitting")
    parser.add_argument("--continuation-policy", choices=("practical", "checkpoint"), default="practical",
                        help="Declared candidate-label continuation; checkpoint requires --policy-checkpoint")
    parser.add_argument("--allow-historical", action="store_true")
    args = parser.parse_args(argv)
    if min(args.trajectories, args.label_samples, args.evaluation_episodes, args.evaluation_samples, args.hidden) < 1:
        parser.error("Counts must be positive")
    if args.trajectories < 10 or args.epochs != sorted(set(args.epochs)) or min(args.epochs) < 1:
        parser.error("Need at least ten trajectories and positive increasing epochs")
    if args.continuation_policy == "checkpoint" and args.policy_checkpoint is None:
        parser.error("Checkpoint continuation requires --policy-checkpoint")
    if (args.out / "preregistration.json").exists():
        raise FileExistsError("Use a new run directory; completed or partial evidence is immutable")
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if not args.allow_historical:
        subprocess.run([sys.executable, "scripts/check_live_ruleset.py", "--report", str(args.out / "live_preflight.json")], cwd=ROOT, check=True)
    ruleset = json.loads((ROOT / "data/ruleset.json").read_text())
    profile = TimingProfile.load(ROOT / "config/turn-budget.json")
    behavior_model = load_policy_checkpoint(args.policy_checkpoint, ROOT / "data/ruleset.json", profile) if args.policy_checkpoint else None
    evaluation_seeds = [args.seed + 10000000 + i*1009 for i in range(args.evaluation_episodes)]
    inherited_test_seeds = previous_evaluation_seeds(args.policy_checkpoint, behavior_model) if behavior_model is not None else set()
    reserved_test_seeds = reserve_evaluation_seeds(
        [args.seed + i*1009 for i in range(args.trajectories)], evaluation_seeds, inherited_test_seeds)
    continuation_model = behavior_model if args.continuation_policy == "checkpoint" else None
    model = training_model(behavior_model, args.hidden)
    preregistration = {"scope": SCOPE, "full_game_ready": False, "seed": args.seed,
        "trajectory_attempts": args.trajectories, "label_samples_per_candidate": args.label_samples,
        "evaluation_episodes": args.evaluation_episodes, "evaluation_samples_per_policy": args.evaluation_samples,
        "evaluation_seed_offset": 10000000, "split": "episode-index modulo5: 0-3 train, 4 validation; new seeds for test",
        "objective": "mean of first and second sampled combat scores: win1, tie0.5, loss0",
        "label_policy": "one candidate action then " + args.continuation_policy + " policy until both combats complete",
        "behavior_policy": "frozen_checkpoint" if behavior_model is not None else "practical_heuristic_v1",
        "continuation_policy": args.continuation_policy,
        "policy_checkpoint": str(args.policy_checkpoint) if args.policy_checkpoint else None,
        "policy_checkpoint_sha256": sha(args.policy_checkpoint) if args.policy_checkpoint else None,
        "initial_optimizer_step": model.step,
        "inherited_reserved_test_seeds": sorted(inherited_test_seeds),
        "decision_sampling": "every player-zero decision on complete declared-behavior trajectories",
        "evaluation_policy": "selected neural model at every player-zero decision across both turns",
        "shortlist": "inherited candidates_for max16; canonical playposition0, no move; legal upgrades retained",
        "unsupported_rule": "any shortlisted branch unsupported rejects entire training trajectory; any eval policy unsupported rejects whole paired episode and is reported",
        "pairings": PAIRINGS, "feature_version": TWO_TURN_FEATURE_VERSION,
        "feature_names": TWO_TURN_FEATURE_NAMES, "timing_profile_sha256": profile.fingerprint,
        "timers": "explicit synthetic15/20/30 seconds player0,60seconds others; not claimed client durations",
        "candidate_models": {"hidden": args.hidden, "epochs": args.epochs, "seed": 17, "tie_tolerance": 0},
        "selection": "validation mean candidate score only; full-policy test starts after checkpoint frozen",
        "policy_promotion": "none: engineering pilot, no full-game strength claim",
        "ruleset_file_sha256": sha(ROOT / "data/ruleset.json"),
        "reference_cards_sha256": sha(ROOT / "data/reference_cards.json")}
    preregistration["source_sha256"] = {str(path.relative_to(ROOT)): sha(path) for path in
        [Path(__file__).resolve(), ROOT / "python/bg_ai/opening_transition.py",
         ROOT / "python/bg_ai/two_turn_features.py", ROOT / "simulator/opening-transition-firestone.mjs"]}
    write_json(args.out / "preregistration.json", preregistration)
    factory = Factory(args.engine_root, ruleset, profile)
    bridge = CombatBridge(args.out / "firestone-worker.log")
    trajectories, failures, seen = [], [], {}
    try:
        with gzip.open(args.out / "training_trajectories.jsonl.gz", "wt", encoding="utf8") as archive:
            for index in range(args.trajectories):
                seed = args.seed + index * 1009
                before = bridge.requests
                try:
                    trajectory = build_trajectory(factory, bridge, index=index, seed=seed, samples=args.label_samples,
                        behavior_model=behavior_model, continuation_model=continuation_model)
                    check_split_overlap(trajectory, seen)
                    seen.update({d["state_fingerprint"]: trajectory["split"] for d in trajectory["decisions"]})
                    write_line(archive, trajectory)
                    # Full requests, sampled receipts and command/time traces
                    # stay in the archive; optimization retains only inputs and
                    # labels so memory does not scale with all rollout traces.
                    trajectories.append({"episode_id": trajectory["episode_id"], "split": trajectory["split"],
                        "decisions": [{**{k: v for k, v in d.items() if k != "candidates"},
                            "candidates": [{k: v for k, v in c.items() if k != "rollouts"}
                                           for c in d["candidates"]]} for d in trajectory["decisions"]]})
                except (UnsupportedRecruitTransition, DuplicateTrajectoryState) as error:
                    failures.append({"index": index, "seed": seed, "split": episode_split(index),
                        "error": f"{type(error).__name__}: {error}", "sampled_combats_before_rejection": (bridge.requests-before)*4})
                progress = {"stage": "generation", "attempted": index+1, "accepted": len(trajectories),
                    "rejected": len(failures), "sampled_combats": bridge.requests*4, "elapsed_seconds": time.time()-started}
                write_json(args.out / "progress.json", progress)
                write_json(args.out / "generation_failures.json", failures)
                print(json.dumps(progress), flush=True)
        scenarios = [Scenario(d["decision_id"], t["split"],
            np.asarray([c["features"] for c in d["candidates"]]),
            np.asarray([c["score"] for c in d["candidates"]], dtype=np.float64),
            {"episode_id": t["episode_id"], "turn": d["turn"]}) for t in trajectories for d in t["decisions"]
            if len(d["candidates"]) >= 2]
        dataset = Dataset(scenarios, TWO_TURN_FEATURE_NAMES)
        if not dataset.split("train") or not dataset.split("validation"):
            raise RuntimeError("Supported complete episodes did not leave both train and validation splits")
        best, best_value, prior, sweep = None, -float("inf"), 0, []
        for epoch in args.epochs:
            losses = model.fit(dataset, epochs=epoch-prior, tie_tolerance=0)
            prior = epoch
            # Average decisions within each episode, then average episodes.
            episode_values = {}
            for scenario in dataset.split("validation"):
                value = float(scenario.scores[int(np.argmax(model.predict(scenario.features)))])
                episode_values.setdefault(scenario.metadata["episode_id"], []).append(value)
            value = float(np.mean([np.mean(values) for values in episode_values.values()]))
            model.metadata.update(scope=SCOPE, full_game_ready=False, feature_version=TWO_TURN_FEATURE_VERSION,
                two_turn_feature_schema_sha256=two_turn_schema_id(),
                decision_policy="every decision at evaluation", label_continuation=args.continuation_policy,
                behavior_policy=preregistration["behavior_policy"],
                behavior_checkpoint_sha256=preregistration["policy_checkpoint_sha256"],
                collection_seed=args.seed, reserved_evaluation_seeds=reserved_test_seeds,
                objective=preregistration["objective"], timing_profile_sha256=profile.fingerprint,
                ruleset_file_sha256=preregistration["ruleset_file_sha256"])
            path = args.out / f"model-h{args.hidden}-e{epoch}.json"
            model.save(path)
            sweep.append({"epochs": epoch, "checkpoint": path.name, "validation_mean": value,
                "final_epoch_loss": losses[-1]})
            if value > best_value:
                best_value, best = value, Ranker.load(path, TWO_TURN_FEATURE_NAMES)
        write_json(args.out / "validation_sweep.json", sweep)
        best.save(args.out / "selected_model.json")
        evaluation_plan = {"checkpoint_sha256": sha(args.out / "selected_model.json"),
            "seeds": evaluation_seeds,
            "samples_per_policy": args.evaluation_samples, "policies": ["model", "practical_heuristic"],
            "selection_complete_before_test": True}
        write_json(args.out / "frozen_evaluation_plan.json", evaluation_plan)
        evaluations, evaluation_failures = [], []
        generation_combats = bridge.requests*4
        with gzip.open(args.out / "evaluation_trajectories.jsonl.gz", "wt", encoding="utf8") as archive:
            for index, seed in enumerate(evaluation_plan["seeds"]):
                result = {"episode_id": f"two-turn-test-{seed}", "seed": seed, "policies": {}}
                try:
                    for name, policy in (("model", best), ("practical_heuristic", None)):
                        outputs = []
                        for sample in range(args.evaluation_samples):
                            timed, setup = factory.setup(seed)
                            output = rollout(timed, bridge, seed, model=policy, sample=sample)
                            outputs.append(output)
                        result["policies"][name] = {"score": float(np.mean([o["score"] for o in outputs])), "rollouts": outputs}
                        result["setup"] = setup
                    evaluations.append(result)
                    write_line(archive, result)
                except UnsupportedRecruitTransition as error:
                    failure = {"seed": seed, "error": f"{type(error).__name__}: {error}",
                        "failed_policy": name, "failed_sample": sample, "completed_policy_names": list(result["policies"])}
                    evaluation_failures.append(failure)
                print(json.dumps({"stage": "evaluation", "attempted": index+1, "accepted": len(evaluations),
                    "failed": len(evaluation_failures), "elapsed_seconds": time.time()-started}), flush=True)
                write_json(args.out / "evaluation_failures.json", evaluation_failures)
        if not evaluations:
            raise RuntimeError("No complete paired full-policy evaluation episodes survived")
        differences = [r["policies"]["model"]["score"] - r["policies"]["practical_heuristic"]["score"] for r in evaluations]
        own_traces = [trace for r in evaluations for p in r["policies"].values() for rollout_ in p["rollouts"]
                      for trace in rollout_["timing_trace"] if trace["player_id"] == 0]
        violations = sum(1 for trace in own_traces if trace["event"] == "action" and
            (trace["after"]["remaining_ms"] < 0 or trace["after"]["remaining_actions"] < 0))
        cards_seen = set()
        for trajectory in trajectories:
            for decision in trajectory["decisions"]:
                for zone in ("board", "hand", "shop"):
                    cards_seen.update(c["card_id"] for c in decision["observation"]["private_state"].get(zone, []))
        results = {"scope": SCOPE, "full_game_ready": False, "policy_promoted": False,
            "training_trajectory_attempts": args.trajectories, "accepted_training_trajectories": len(trajectories),
            "trajectory_splits": dict(Counter(t["split"] for t in trajectories)),
            "decision_splits": dict(Counter(s.split for s in scenarios)),
            "generation_failures": len(failures), "generation_failure_reasons": dict(Counter(f["error"] for f in failures)),
            "available_pool": {"minions": len(TIER1_MINIONS), "spells": len(TIER1_SPELLS)},
            "accepted_own_observation_coverage": {"minions": sorted(cards_seen & TIER1_MINIONS), "spells": sorted(cards_seen & TIER1_SPELLS)},
            "evaluation_episodes": len(evaluations), "evaluation_attempts": args.evaluation_episodes,
            "evaluation_failures": evaluation_failures, "objective": preregistration["objective"],
            "scores": {name: float(np.mean([r["policies"][name]["score"] for r in evaluations]))
                for name in ("model", "practical_heuristic")},
            "paired_model_minus_practical": paired_summary(differences, np.random.default_rng(86028121)),
            "bootstrap": {"unit": "complete paired episode, averaging samples first", "samples": 10000, "seed": 86028121},
            "generation_sampled_combats_including_rejections": generation_combats,
            "evaluation_sampled_combats_including_rejections": bridge.requests*4-generation_combats,
            "timing_budget_violations": violations,
            "model_action_counts": dict(Counter(command["action"]["kind"] for r in evaluations
                for roll in r["policies"]["model"]["rollouts"] for command in roll["commands"] if command["player_id"] == 0)),
            "checkpoint_sha256": evaluation_plan["checkpoint_sha256"],
            "elapsed_seconds": time.time()-started,
            "limitations": ["Small noisy engineering pilot, no measured full-game strength or MMR",
                "Only two turns; no valuation beyond second combat, including second-turn tier upgrades",
                "All players Patchwerk; fixed declared pairings; opponents use a practical heuristic",
                "Synthetic turn times; one command/sec plus configured costs and five-second reserve",
                "One fitted iteration with declared " + args.continuation_policy + " continuation; full-policy evaluation may shift state distribution",
                "Visible temporary enchantment duration flags are not encoded; equal current stats can alias different next-turn persistence",
                "Conditional supported-trajectory sample; complete rejection counts reported, no policy promoted",
                "Tier-2 next-turn shops, ordinary triples and unverified contextual effects remain explicit engine frontiers"]}
        if not args.allow_historical:
            subprocess.run([sys.executable, "scripts/check_live_ruleset.py", "--report", str(args.out / "live_postflight.json")], cwd=ROOT, check=True)
        results["current_snapshot_verified"] = not args.allow_historical
        write_json(args.out / "results.json", results)
        write_json(args.out / "progress.json", {"stage": "complete", "elapsed_seconds": time.time()-started})
        print(json.dumps(results, indent=2), flush=True)
    finally:
        bridge.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
