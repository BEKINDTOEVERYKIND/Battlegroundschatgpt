#!/usr/bin/env python3
"""Imitate complete practical opening trajectories, select by closed-loop play.

Binary action-preference labels are imitation targets, never combat estimates.
Every checkpoint is validated as a complete two-turn actor before a fresh test.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import numpy as np

from bg_ai.choice_timing import CHOICE_TIMING_VERSION, ChoiceTimedTwoTurnOpeningEngine
from bg_ai.hsbrsim_adapter import UnsupportedRecruitTransition
from bg_ai.hsbrsim_opening import OpeningFixtureSpec, TRIBES, build_opening_database
from bg_ai.learning import Dataset, Ranker, Scenario
from bg_ai.opening_transition import TwoTurnOpeningRecruitEngine
from bg_ai.policy_learning import POLICY_OBJECTIVE, fit_policy
from bg_ai.recruit_features import observation_record
from bg_ai.recruiting import RecruitAction, RecruitObservation
from bg_ai.turn_budget import TimingProfile
from bg_ai.two_turn_features_v2 import (
    TWO_TURN_FEATURE_NAMES, TWO_TURN_FEATURE_VERSION, encode_two_turn_actions,
    enrich_two_turn_observation, two_turn_schema_id,
)
from train_recruit_curriculum import (
    candidates_for, heuristic_action, heuristic_value, paired_summary, sha, write_json,
)
from train_two_turn_recruit import CombatBridge, PAIRINGS, SCOPE, apply_combat, episode_score, write_line

EXPERT_DATA_VERSION = "complete-opening-expert-preferred-action-sets-v1"
NONINFERIORITY_MARGIN = .025


class PolicyFactory:
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
        timed = ChoiceTimedTwoTurnOpeningEngine.for_fixture(raw, self.profile, self.ruleset)
        timed.reset(seed=seed)
        return timed, {"seed": seed, "valid_tribes": tribes, "player_zero_timer_ms": timers,
            "other_player_timer_ms": 60000, "provenance": database.provenance}


def seed_plan(seed, trajectories, validation_episodes, test_episodes):
    plan = {"train": [seed + i * 1009 for i in range(trajectories)],
            "validation": [seed + 20000000 + i * 1009 for i in range(validation_episodes)],
            "test": [seed + 30000000 + i * 1009 for i in range(test_episodes)]}
    flat = [value for group in plan.values() for value in group]
    if len(flat) != len(set(flat)) or any(type(x) is not int or x < 0 for x in flat):
        raise ValueError("Train/validation/test seed families must be nonnegative and disjoint")
    return plan


def teacher_targets(observation, actions):
    if not actions or actions[0] != heuristic_action(observation):
        raise ValueError("Candidate zero must preserve the frozen practical teacher action")
    values = np.asarray([heuristic_value(observation, action) for action in actions])
    if not np.isfinite(values).all():
        raise ValueError("Teacher values must be finite")
    return (values == values.max()).astype(np.float64)


def choose_action(observation, model=None):
    if model is None:
        return heuristic_action(observation)
    actions = candidates_for(observation)
    encoded = encode_two_turn_actions(enrich_two_turn_observation(observation), actions)
    return actions[int(np.argmax(model.predict(encoded)))]


def play_episode(factory, bridge, seed, *, model=None, sample=0, collect=False, partial=None,
                 action_selector=None):
    """One actual trajectory: all commands are charged; no counterfactual search."""
    timed, setup = factory.setup(seed)
    result = partial if partial is not None else {}
    result.update(seed=seed, sample=sample, setup=setup, commands=[], combats=[], decisions=[])
    for _ in range(1500):
        observation = timed._visible
        if observation is None:
            if timed.engine.phase == "complete":
                result.update(score=episode_score(timed.engine.combat_receipts),
                    timing_trace=timed.trace, complete=True)
                return result
            if timed.engine.phase != "awaiting_combat":
                raise RuntimeError("Unexpected missing recruit observation")
            result["combats"].append(apply_combat(timed, bridge, seed, sample))
            continue
        visible = enrich_two_turn_observation(observation)
        action = (action_selector(visible, model) if action_selector is not None and visible.player_id == 0
                  else choose_action(visible, model if visible.player_id == 0 else None))
        if collect and visible.player_id == 0:
            actions = candidates_for(visible)
            vectors = encode_two_turn_actions(visible, actions)
            labels = teacher_targets(visible, actions)
            result["decisions"].append({
                "decision_id": f"expert-{seed}-{len(result['decisions'])}", "split": "train",
                "observation": observation_record(visible),
                "candidates": [{"action": asdict(a), "features": v.tolist(), "preferred": float(y)}
                               for a, v, y in zip(actions, vectors, labels)]})
        result["commands"].append({"player_id": visible.player_id, "turn": visible.turn,
            "action": asdict(action), "observation": observation_record(visible)})
        result["timing_trace"] = timed.trace
        timed.step(action)
    raise RuntimeError("Finite two-turn trajectory exceeded 1500 transitions")


def trajectory_scenarios(trajectory, *, verify=False):
    if trajectory.get("complete") is not True or len(trajectory.get("combats", [])) != 2:
        raise ValueError("Only complete two-combat expert trajectories can become training data")
    scenarios = []
    for decision in trajectory["decisions"]:
        candidates = decision["candidates"]
        features = np.asarray([c["features"] for c in candidates], dtype=np.float64)
        targets = np.asarray([c["preferred"] for c in candidates], dtype=np.float64)
        if decision.get("split") != "train":
            raise ValueError("Expert data reuse cannot import validation/test decisions")
        if verify:
            record = dict(decision["observation"])
            record["legal_actions"] = tuple(RecruitAction(**a) for a in record["legal_actions"])
            observation = RecruitObservation(**record)
            actions = candidates_for(observation)
            if [asdict(a) for a in actions] != [c["action"] for c in candidates]:
                raise ValueError("Reused expert candidate actions changed")
            if (not np.array_equal(features, encode_two_turn_actions(observation, actions))
                    or not np.array_equal(targets, teacher_targets(observation, actions))):
                raise ValueError("Reused expert features or preferred-action labels changed")
        scenarios.append(Scenario(decision["decision_id"], "train", features, targets,
            {"episode_seed": trajectory["seed"], "turn": decision["observation"]["turn"]}))
    return scenarios


def evaluation_summary(rows, policy_names, bootstrap_seed=86028121):
    """Every attempted seed is retained; missing complete episodes score zero."""
    if not rows:
        raise ValueError("At least one attempted evaluation episode is required")
    baseline = "practical"
    reference = np.asarray([r["policies"][baseline]["penalized_score"] for r in rows])
    result = {}
    for name in policy_names:
        records = [row["policies"][name] for row in rows]
        values = np.asarray([record["penalized_score"] for record in records])
        supported = [i for i, row in enumerate(rows)
                     if row["policies"][name]["complete"] and row["policies"][baseline]["complete"]]
        result[name] = {"attempted_episodes": len(rows),
            "complete_episodes": sum(r["complete"] for r in records),
            "failed_episodes": sum(not r["complete"] for r in records),
            "failure_penalized_mean": float(values.mean()),
            "timing_budget_violations": sum(record.get("timing_budget_violations", 0) for record in records),
            "paired_all_attempts": paired_summary(values-reference, np.random.default_rng(bootstrap_seed)),
            "support_intersection_episodes": len(supported),
            "paired_supported_intersection": paired_summary(values[supported]-reference[supported],
                np.random.default_rng(bootstrap_seed)) if supported else None}
    return result


def select_validation(summary, learner_names):
    """Freeze an experimental learner; validation never promotes deployment."""
    if not learner_names:
        raise ValueError("At least one experimental learner is required")
    best = max(learner_names, key=lambda name: summary[name]["failure_penalized_mean"])
    eligible = [name for name in learner_names if summary[name]["failed_episodes"] == 0
        and summary["practical"]["failed_episodes"] == 0
        and summary[name]["paired_all_attempts"]["ci95"][0] >= -NONINFERIORITY_MARGIN]
    if eligible:
        best = max(eligible, key=lambda name: summary[name]["failure_penalized_mean"])
    return {"deployment_policy": "practical", "experimental_learner": best,
            "learner_promoted_within_fixture": False,
            "full_game_ready": False, "eligible_learners": eligible,
            "reason": "Experimental learner frozen from full-policy validation; practical retained pending the single independent test"}


def test_promotion(selection, summary, *, current_snapshot_verified):
    """One preregistered superiority gate; never choose another model on test."""
    result = dict(selection)
    name = result["experimental_learner"]
    learner, control = summary[name], summary["practical"]
    promote = (current_snapshot_verified and learner["failed_episodes"] == 0
        and control["failed_episodes"] == 0 and learner["timing_budget_violations"] == 0
        and control["timing_budget_violations"] == 0
        and learner["paired_all_attempts"]["ci95"][0] > 0)
    result.update(deployment_policy=name if promote else "practical",
        learner_promoted_within_fixture=promote, independent_test_gate_passed=promote,
        reason="Frozen learner passed the preregistered independent-test superiority gate within this fixture"
            if promote else "Practical retained: the frozen learner did not pass every independent-test superiority/currentness gate")
    return result


def evaluate_policies(factory, bridge, seeds, policies, samples, archive_path, progress=None):
    rows = []
    with gzip.open(archive_path, "wt", encoding="utf8") as archive:
        for index, seed in enumerate(seeds):
            row = {"seed": seed, "policies": {}}
            for name, model in policies.items():
                outputs, failures = [], []
                for sample in range(samples):
                    partial = {}
                    try:
                        outputs.append(play_episode(factory, bridge, seed, model=model, sample=sample, partial=partial))
                    except UnsupportedRecruitTransition as error:
                        failures.append({"sample": sample, "error": str(error), "partial": partial})
                complete = len(outputs) == samples
                score = float(np.mean([output["score"] for output in outputs])) if complete else 0.0
                full = {"complete": complete, "penalized_score": score, "rollouts": outputs, "failures": failures}
                write_line(archive, {"seed": seed, "policy": name, **full})
                row["policies"][name] = {"complete": complete, "penalized_score": score,
                    "failed_samples": len(failures),
                    "failure_reasons": dict(Counter(f["error"] for f in failures)),
                    "action_counts": dict(Counter(c["action"]["kind"] for output in outputs
                        for c in output["commands"] if c["player_id"] == 0)),
                    "timing_budget_violations": sum(
                        t["after"]["remaining_ms"] < 0 or t["after"]["remaining_actions"] < 0
                        for output in outputs for t in output["timing_trace"] if t["event"] == "action")}
            rows.append(row)
            if progress:
                progress(index + 1, len(seeds))
    return rows, evaluation_summary(rows, tuple(policies))


def source_snapshot(out):
    files = set((ROOT / "python/bg_ai").glob("*.py")) | set((ROOT / "simulator").glob("*.mjs"))
    files.update(ROOT / "scripts" / name for name in (
        "train_recruit_policy.py", "train_two_turn_recruit.py", "train_recruit_curriculum.py",
        "check_live_ruleset.py"))
    payload = {str(path.relative_to(ROOT)): {"sha256": sha(path), "content": path.read_bytes().decode("utf8")}
               for path in sorted(files)}
    with gzip.open(out / "execution_sources.json.gz", "wt", encoding="utf8") as stream:
        json.dump(payload, stream, separators=(",", ":"))
    return {path: record["sha256"] for path, record in payload.items()}


def reuse_expert_data(source, expected, destination):
    directory = source if source.is_dir() else source.parent
    archive = directory / "expert_trajectories.jsonl.gz" if source.is_dir() else source
    manifest = json.loads((directory / "expert_data_manifest.json").read_text())
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Reused expert data {key} does not match this preregistration")
    if manifest.get("archive_sha256") != sha(archive):
        raise ValueError("Reused expert archive checksum mismatch")
    scenarios, seeds, identifiers = [], [], set()
    with gzip.open(archive, "rt", encoding="utf8") as stream:
        for line in stream:
            trajectory = json.loads(line)
            if trajectory["seed"] not in expected["attempted_seeds"] or trajectory["seed"] in seeds:
                raise ValueError("Reused expert episode seed is outside the unique training plan")
            seeds.append(trajectory["seed"])
            current = trajectory_scenarios(trajectory, verify=True)
            for scenario in current:
                if scenario.scenario_id in identifiers:
                    raise ValueError("Reused expert decision IDs must be unique")
                identifiers.add(scenario.scenario_id)
            scenarios.extend(current)
    if len(seeds) != manifest["accepted_trajectories"]:
        raise ValueError("Reused expert accepted-trajectory count mismatch")
    shutil.copyfile(archive, destination / "expert_trajectories.jsonl.gz")
    write_json(destination / "expert_data_manifest.json", manifest)
    return scenarios, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--trajectories", type=int, default=512)
    parser.add_argument("--validation-episodes", type=int, default=32)
    parser.add_argument("--validation-samples", type=int, default=4)
    parser.add_argument("--test-episodes", type=int, default=128)
    parser.add_argument("--test-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=202609071)
    parser.add_argument("--hidden", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--epochs", type=int, nargs="+", default=[10, 30])
    parser.add_argument("--data", type=Path, help="Reuse matching expert archive/directory; rules/schema/seed plan verified")
    parser.add_argument("--allow-historical", action="store_true")
    args = parser.parse_args(argv)
    if min(args.trajectories, args.validation_episodes, args.validation_samples,
           args.test_episodes, args.test_samples, *args.hidden, *args.epochs) < 1:
        parser.error("All counts, widths and epochs must be positive")
    if args.epochs != sorted(set(args.epochs)) or len(set(args.hidden)) != len(args.hidden):
        parser.error("Epochs must increase and hidden widths must be distinct")
    plan = seed_plan(args.seed, args.trajectories, args.validation_episodes, args.test_episodes)
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("Use a new empty run directory; evidence is immutable")
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    profile = TimingProfile.load(ROOT / "config/turn-budget.json")
    expected = {"version": EXPERT_DATA_VERSION, "feature_schema_sha256": two_turn_schema_id(),
        "ruleset_file_sha256": sha(ROOT / "data/ruleset.json"),
        "reference_cards_sha256": sha(ROOT / "data/reference_cards.json"),
        "timing_profile_sha256": profile.fingerprint, "choice_timing_version": CHOICE_TIMING_VERSION,
        "seed": args.seed, "attempted_seeds": plan["train"]}
    prereg = {"scope": SCOPE, "full_game_ready": False, "expert_data": expected,
        "feature_version": TWO_TURN_FEATURE_VERSION, "feature_names": TWO_TURN_FEATURE_NAMES,
        "seed_plan": plan, "training_target": "binary set of all maximal frozen heuristic values among candidates_for",
        "training_objective": POLICY_OBJECTIVE, "training_interpretation": "behavior cloning only; labels are not combat values",
        "candidate_models": {"hidden": args.hidden, "epochs": args.epochs, "model_seed": 17},
        "validation_samples": args.validation_samples, "test_samples": args.test_samples,
        "selection": {"evaluation": "complete actor-controlled two-turn episodes against simultaneous frozen practical control",
            "failure_penalty": "all samples must complete; otherwise episode score zero; every attempted seed in denominator",
            "experimental_eligibility": "zero learner/control failures, paired lower95 bound >= -margin",
            "noninferiority_margin": NONINFERIORITY_MARGIN, "ranking": "highest failure-penalized validation mean",
            "fallback": "practical remains recommended; best learned candidate is evaluated as experimental even if all declined",
            "independent_test_promotion": "single frozen learner only: paired lower95 > 0, zero learner/control failures and timing violations, live snapshot verified; fixture only"},
        "objective": "mean score over the first two sampled combats: win1, tie0.5, loss0",
        "pairings": PAIRINGS, "shortlist": "unchanged candidates_for max16, preserves practical at index0",
        "unsupported": "reject whole expert trajectory; retain all validation/test attempts with per-policy failures",
        "bootstrap": {"unit": "episode after averaging its samples", "samples": 10000, "seed": 86028121},
        "historical_mode": args.allow_historical, "data_reuse": str(args.data) if args.data else None,
        "source_sha256": source_snapshot(args.out)}
    write_json(args.out / "preregistration.json", prereg)
    if not args.allow_historical:
        subprocess.run([sys.executable, "scripts/check_live_ruleset.py", "--report", str(args.out / "live_preflight.json")], cwd=ROOT, check=True)
    ruleset = json.loads((ROOT / "data/ruleset.json").read_text())
    factory = PolicyFactory(args.engine_root, ruleset, profile)
    bridge = CombatBridge(args.out / "firestone-worker.log")
    def progress(stage, completed, total):
        record = {"stage": stage, "completed": completed, "total": total,
                  "sampled_combats": bridge.requests * 4, "elapsed_seconds": time.time() - started}
        write_json(args.out / "progress.json", record)
        print(json.dumps(record), flush=True)
    try:
        if args.data:
            scenarios, manifest = reuse_expert_data(args.data, expected, args.out)
        else:
            scenarios, failures, accepted = [], [], 0
            with gzip.open(args.out / "expert_trajectories.jsonl.gz", "wt", encoding="utf8") as archive, \
                 gzip.open(args.out / "expert_rejected_trajectories.jsonl.gz", "wt", encoding="utf8") as rejected:
                for index, seed in enumerate(plan["train"]):
                    partial = {}
                    try:
                        trajectory = play_episode(factory, bridge, seed, collect=True, partial=partial)
                        current = trajectory_scenarios(trajectory)
                        write_line(archive, trajectory)
                        scenarios.extend(current)
                        accepted += 1
                    except UnsupportedRecruitTransition as error:
                        failures.append({"seed": seed, "error": str(error)})
                        write_line(rejected, {"seed": seed, "error": str(error), "partial": partial})
                    progress("expert_generation", index + 1, len(plan["train"]))
            manifest = {**expected, "accepted_trajectories": accepted, "decisions": len(scenarios),
                "failures": failures, "failure_reasons": dict(Counter(f["error"] for f in failures)),
                "archive_sha256": sha(args.out / "expert_trajectories.jsonl.gz")}
            write_json(args.out / "expert_data_manifest.json", manifest)
        if not scenarios:
            raise RuntimeError("No complete expert trajectories produced training decisions")
        dataset = Dataset(scenarios, TWO_TURN_FEATURE_NAMES)
        policies, model_paths, training = {"practical": None}, {}, []
        for width in args.hidden:
            model = Ranker(TWO_TURN_FEATURE_NAMES, hidden=width, seed=17)
            prior = 0
            for epoch in args.epochs:
                losses = fit_policy(model, dataset, epochs=epoch - prior)
                prior = epoch
                name = f"policy-h{width}-e{epoch}"
                model.metadata.update(full_game_ready=False, scope=SCOPE, feature_version=TWO_TURN_FEATURE_VERSION,
                    feature_schema_sha256=two_turn_schema_id(), ruleset_file_sha256=expected["ruleset_file_sha256"],
                    timing_profile_sha256=profile.fingerprint, choice_timing_version=CHOICE_TIMING_VERSION,
                    collection_seed=args.seed, reserved_evaluation_seeds=plan["validation"] + plan["test"],
                    training_target="binary preferred teacher action set; imitation only")
                path = args.out / f"{name}.json"
                model.save(path)
                policies[name], model_paths[name] = Ranker.load(path, TWO_TURN_FEATURE_NAMES), path
                training.append({"candidate": name, "checkpoint_sha256": sha(path), "epochs": epoch,
                    "last_loss": losses[-1], "optimizer_step": model.step})
                progress("fit", len(training), len(args.hidden) * len(args.epochs))
        write_json(args.out / "training.json", training)
        validation_rows, validation = evaluate_policies(factory, bridge, plan["validation"], policies,
            args.validation_samples, args.out / "validation_trajectories.jsonl.gz",
            lambda done, total: progress("validation", done, total))
        write_json(args.out / "validation_results.json", {"policies": validation, "episodes": validation_rows})
        selection = select_validation(validation, list(model_paths))
        experimental = selection["experimental_learner"]
        shutil.copyfile(model_paths[experimental], args.out / "experimental_model.json")
        selection["experimental_checkpoint_sha256"] = sha(args.out / "experimental_model.json")
        selection["deployment_checkpoint"] = None
        write_json(args.out / "deployment_policy.json", selection)
        frozen = {"selection": selection, "seeds": plan["test"], "samples": args.test_samples,
            "experimental_checkpoint_sha256": selection["experimental_checkpoint_sha256"],
            "selection_complete_before_test": True}
        write_json(args.out / "frozen_evaluation_plan.json", frozen)
        test_policies = {"practical": None, experimental: policies[experimental]}
        test_rows, test = evaluate_policies(factory, bridge, plan["test"], test_policies,
            args.test_samples, args.out / "test_trajectories.jsonl.gz",
            lambda done, total: progress("test", done, total))
        write_json(args.out / "test_results.json", {"policies": test, "episodes": test_rows})
        if not args.allow_historical:
            subprocess.run([sys.executable, "scripts/check_live_ruleset.py", "--report", str(args.out / "live_postflight.json")], cwd=ROOT, check=True)
        selection = test_promotion(selection, test, current_snapshot_verified=not args.allow_historical)
        selection["deployment_checkpoint"] = (model_paths[experimental].name
                                              if selection["learner_promoted_within_fixture"] else None)
        write_json(args.out / "deployment_policy.json", selection)
        result = {"full_game_ready": False, "scope": SCOPE, "selection": selection,
            "training": {"attempted_trajectories": args.trajectories, "accepted_trajectories": manifest["accepted_trajectories"],
                         "decisions": len(scenarios), "failure_reasons": manifest["failure_reasons"]},
            "validation": validation, "test": test, "current_snapshot_verified": not args.allow_historical,
            "sampled_combats_including_rejections": bridge.requests * 4,
            "elapsed_seconds": time.time() - started,
            "limitations": ["Imitates a fixed practical teacher; combat improvement is measured separately, not assumed",
                "Only two-turn current Tier1 fixture; no full-game MMR or seasonal coverage",
                "Synthetic 15/20/30 second own timers, 60 second opponents; one command/sec plus costs and 5 second reserve",
                "Shortlist excludes most board positioning choices; legal upgrades remain possible but unsupported next-tier draws abort",
                "Small validation samples and repeated model comparisons do not establish a population guarantee",
                "Failed episodes are penalized, while supported-intersection estimates remain conditional"]}
        write_json(args.out / "results.json", result)
        progress("complete", 1, 1)
        print(json.dumps(result, indent=2), flush=True)
        return 0
    except Exception as error:
        write_json(args.out / "failure.json", {"type": type(error).__name__, "error": str(error),
            "elapsed_seconds": time.time() - started})
        raise
    finally:
        bridge.close()


if __name__ == "__main__":
    raise SystemExit(main())
