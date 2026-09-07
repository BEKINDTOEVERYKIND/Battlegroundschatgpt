#!/usr/bin/env python3
"""Bounded complete-episode policy gradients initialized from a frozen BC actor."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import numpy as np

from bg_ai.choice_timing import CHOICE_TIMING_VERSION
from bg_ai.hsbrsim_adapter import UnsupportedRecruitTransition
from bg_ai.learning import Ranker
from bg_ai.policy_gradient import (
    GradientConfig, PolicyEpisode, fit_policy_gradient,
    initialize_from_behavior_cloning, sample_decision,
)
from bg_ai.turn_budget import TimingProfile
from bg_ai.policy_factory_cached import CachedPolicyFactory
from bg_ai.two_turn_features_v2 import (
    TWO_TURN_FEATURE_NAMES, TWO_TURN_FEATURE_VERSION, encode_two_turn_actions, two_turn_schema_id,
)
from train_recruit_curriculum import candidates_for, paired_summary, sha, write_json
from train_recruit_policy import (
    CombatBridge, evaluate_policies, play_episode, seed_plan,
    select_validation, source_snapshot, test_promotion, SCOPE, write_line,
)


def load_reference(path, profile):
    model = Ranker.load(path, TWO_TURN_FEATURE_NAMES)
    expected = {"feature_version": TWO_TURN_FEATURE_VERSION,
        "feature_schema_sha256": two_turn_schema_id(),
        "ruleset_file_sha256": sha(ROOT / "data/ruleset.json"),
        "timing_profile_sha256": profile.fingerprint, "choice_timing_version": CHOICE_TIMING_VERSION}
    for key, value in expected.items():
        if model.metadata.get(key) != value:
            raise ValueError(f"Frozen BC reference {key} differs from this verified fixture")
    # Also rejects legacy combat-Q and first-action checkpoints.
    initialize_from_behavior_cloning(model, sha(path))
    return model


def reserve_seeds(plan, source):
    inherited = set(source.metadata.get("reserved_evaluation_seeds", []))
    trained = {row["episode_seed"] for row in source.metadata.get("training_sources", [])
               if "episode_seed" in row}
    attempted = {seed for family in plan.values() for seed in family}
    if attempted & (inherited | trained):
        raise ValueError("New policy-gradient seeds overlap BC training or reserved evaluation seeds")
    return sorted(inherited | set(plan["validation"]) | set(plan["test"]))


def decision_record(decision):
    return {"features": decision.features.tolist(), "action_index": decision.action_index,
        "old_log_probabilities": decision.old_log_probabilities.tolist(),
        "reference_log_probabilities": decision.reference_log_probabilities.tolist()}


def collect_pair(factory, bridge, seed, actor, reference, config, sampling_seed):
    """Record every attempt; only two fully completed rollouts produce an episode."""
    decisions = []
    rng = np.random.default_rng(np.random.SeedSequence([sampling_seed, seed, 823]))

    def selector(observation, model):
        actions = candidates_for(observation)
        decision = sample_decision(model, reference, encode_two_turn_actions(observation, actions),
                                   rng, config.temperature)
        decisions.append(decision)
        return actions[decision.action_index]

    record = {"seed": seed, "sampling_seed": sampling_seed, "policies": {}}
    for name, model in (("actor", actor), ("practical", None)):
        partial = {}
        try:
            result = play_episode(factory, bridge, seed, model=model, sample=0, partial=partial,
                                  action_selector=selector if name == "actor" else None)
            if result.get("complete") is not True or len(result.get("combats", [])) != 2:
                raise RuntimeError("Runner returned without a complete two-combat trajectory")
            record["policies"][name] = {"complete": True, "trajectory": result}
        except UnsupportedRecruitTransition as error:
            record["policies"][name] = {"complete": False, "error": str(error), "partial": partial}
    record["frozen_decisions"] = [decision_record(d) for d in decisions]
    record["complete"] = all(row["complete"] for row in record["policies"].values())
    if not record["complete"]:
        return None, record
    episode = PolicyEpisode(seed, record["policies"]["actor"]["trajectory"]["score"],
        record["policies"]["practical"]["trajectory"]["score"], tuple(decisions))
    record["advantage"] = episode.score - episode.practical_score
    return episode, record


def update_complete_batch(actor, episodes, failures, attempted_episodes, config):
    """An unsupported paired trajectory rejects the whole planned update batch."""
    if attempted_episodes < 1 or len(episodes) + len(failures) != attempted_episodes:
        raise ValueError("Every attempted training pair must be accounted for exactly once")
    if failures:
        return []
    return fit_policy_gradient(actor, episodes, config)


def gradient_test_promotion(selection, summary, comparison, *, current_snapshot_verified,
                            existing_baseline="practical"):
    """Approve only a frozen PG actor superior to both predeclared controls."""
    if existing_baseline not in ("practical", "bc_reference"):
        raise ValueError("Existing deployment baseline must be practical or frozen BC")
    result = test_promotion(selection, summary, current_snapshot_verified=current_snapshot_verified)
    reference = summary["bc_reference"]
    promote = (result["independent_test_gate_passed"] and result["experimental_learner"] != "bc_reference"
        and reference["failed_episodes"] == 0 and reference["timing_budget_violations"] == 0
        and comparison["ci95"][0] > 0)
    result.update(deployment_policy=result["experimental_learner"] if promote else existing_baseline,
        learner_promoted_within_fixture=promote, independent_test_gate_passed=promote,
        existing_baseline=existing_baseline, policy_gradient_promoted=promote,
        reason="Frozen policy-gradient actor passed the preregistered independent-test superiority gates versus both practical and frozen BC"
            if promote else "Existing baseline retained: no frozen gradient actor passed every independent-test superiority/currentness gate versus both controls")
    return result


def existing_baseline(path, digest):
    """Preserve an already approved BC baseline, bound to its exact checkpoint."""
    sidecar = path.parent / "deployment_policy.json"
    if not sidecar.exists():
        return "practical", None
    record = json.loads(sidecar.read_text())
    approved = (record.get("experimental_checkpoint_sha256") == digest
        and record.get("independent_test_gate_passed") is True
        and record.get("learner_promoted_within_fixture") is True)
    return ("bc_reference" if approved else "practical"), record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-root", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True, help="Frozen lifetime-v2 BC checkpoint")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026090717)
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--episodes-per-iteration", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=4)
    parser.add_argument("--validation-episodes", type=int, default=32)
    parser.add_argument("--validation-samples", type=int, default=4)
    parser.add_argument("--test-episodes", type=int, default=128)
    parser.add_argument("--test-samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=.5)
    parser.add_argument("--clip-epsilon", type=float, default=.15)
    parser.add_argument("--reference-kl", type=float, default=.1)
    parser.add_argument("--learning-rate", type=float, default=.0001)
    parser.add_argument("--update-epochs", type=int, default=2)
    parser.add_argument("--allow-historical", action="store_true")
    args = parser.parse_args(argv)
    positive = (args.iterations, args.episodes_per_iteration, args.checkpoint_every,
                args.validation_episodes, args.validation_samples, args.test_episodes, args.test_samples)
    if any(value < 1 for value in positive) or args.iterations > 128 or args.episodes_per_iteration > 128:
        parser.error("Positive bounds required, at most 128 iterations and 128 episodes per iteration")
    if args.validation_episodes < 2 or args.test_episodes < 2:
        parser.error("Validation and test each require at least two independently seeded episodes")
    config = GradientConfig(temperature=args.temperature, clip_epsilon=args.clip_epsilon,
        reference_kl=args.reference_kl, learning_rate=args.learning_rate, epochs=args.update_epochs,
        batch_episodes=args.episodes_per_iteration)
    config.validate()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Output directory must be empty; never overwrite a recorded experiment")
    args.out.mkdir(parents=True, exist_ok=True)
    profile = TimingProfile.load(ROOT / "config/turn-budget.json")
    reference = load_reference(args.initial, profile)
    reference_hash = sha(args.initial)
    fallback, source_selection = existing_baseline(args.initial, reference_hash)
    actor = initialize_from_behavior_cloning(reference, reference_hash)
    plan = seed_plan(args.seed, args.iterations * args.episodes_per_iteration,
                     args.validation_episodes, args.test_episodes)
    actor.metadata["reserved_evaluation_seeds"] = reserve_seeds(plan, reference)
    actor.metadata["policy_gradient_config"] = asdict(config)
    shutil.copyfile(args.initial, args.out / "frozen_bc_reference.json")
    write_json(args.out / "frozen_bc_prior_selection.json", source_selection)
    shutil.copyfile(__file__, args.out / "executed_train_recruit_policy_gradient.py")
    prereg = {"scope": SCOPE, "full_game_ready": False, "config": asdict(config),
        "iterations": args.iterations, "episodes_per_iteration": args.episodes_per_iteration,
        "checkpoint_every": args.checkpoint_every, "seed_plan": plan,
        "reference_sha256": reference_hash, "feature_version": TWO_TURN_FEATURE_VERSION,
        "feature_schema_sha256": two_turn_schema_id(), "feature_count": len(TWO_TURN_FEATURE_NAMES),
        "ruleset_file_sha256": sha(ROOT / "data/ruleset.json"),
        "reference_cards_sha256": sha(ROOT / "data/reference_cards.json"),
        "timing_profile_sha256": profile.fingerprint, "choice_timing_version": CHOICE_TIMING_VERSION,
        "objective": "Sum clipped action-ratio times complete two-combat score minus paired practical score; average episodes; add KL(frozen BC || actor)",
        "sampling": "softmax over unchanged canonical max16 legal candidates; RNG separate from fixture/combat RNG",
        "baseline": "independent practical-action rollout of same fixture and combat seeds; no actor observations of its outcomes",
        "unsupported": "archive every attempt; reject entire update batch if any actor or practical rollout fails; no replacement seeds or partial return targets",
        "validation_samples": args.validation_samples, "test_samples": args.test_samples,
        "selection": "complete deterministic actors on fixed fresh validation fixtures; practical fallback and BC candidate retained; select_validation gates unchanged",
        "existing_deployment_baseline": fallback,
        "factory": "CachedPolicyFactory: verified immutable export cache, independent tribe database views, bounded LRU of 16 views; actual complete-trajectory conformance checked",
        "independent_test_promotion": "single frozen PG learner only: paired lower95 > 0 versus BOTH practical and frozen BC, zero learner/both-control failures and timing violations, live snapshot verified; fixture only; selected BC cannot count as PG promotion",
        "historical_mode": args.allow_historical, "source_sha256": source_snapshot(args.out),
        "executed_script_sha256": sha(__file__)}
    write_json(args.out / "preregistration.json", prereg)
    started = time.time()
    if not args.allow_historical:
        subprocess.run([sys.executable, "scripts/check_live_ruleset.py", "--report", str(args.out / "live_preflight.json")], cwd=ROOT, check=True)
    factory = CachedPolicyFactory(args.engine_root, json.loads((ROOT / "data/ruleset.json").read_text()), profile)
    bridge = CombatBridge(args.out / "firestone-worker.log")
    training, candidates, paths = [], {"practical": None, "bc_reference": reference}, {
        "bc_reference": args.out / "frozen_bc_reference.json"}

    def progress(stage, completed, total):
        row = {"stage": stage, "completed": completed, "total": total,
               "sampled_combats": bridge.requests * 4, "elapsed_seconds": time.time() - started}
        write_json(args.out / "progress.json", row)
        print(json.dumps(row), flush=True)

    try:
        with gzip.open(args.out / "training_trajectories.jsonl.gz", "wt", encoding="utf8") as archive:
            for iteration in range(1, args.iterations + 1):
                batch, failures = [], []
                seeds = plan["train"][(iteration-1)*args.episodes_per_iteration:iteration*args.episodes_per_iteration]
                for seed in seeds:
                    episode, record = collect_pair(factory, bridge, seed, actor, reference, config, args.seed)
                    write_line(archive, {"iteration": iteration, **record})
                    if episode is not None:
                        batch.append(episode)
                    else:
                        failures.append({"seed": seed, "errors": {name: value["error"]
                            for name, value in record["policies"].items() if not value["complete"]}})
                # Reject the complete preregistered batch rather than learn from
                # a policy-selected successful subset or fabricated frontier loss.
                updates = update_complete_batch(actor, batch, failures, len(seeds), config)
                training.append({"iteration": iteration, "attempted_episodes": len(seeds),
                    "complete_pairs": len(batch), "rejected_batch": bool(failures), "failures": failures,
                    "updates": updates, "mean_complete_pair_advantage": float(np.mean([
                        episode.score-episode.practical_score for episode in batch])) if batch else None})
                if iteration % args.checkpoint_every == 0 or iteration == args.iterations:
                    name = f"pg-iteration-{iteration:03d}"
                    path = args.out / f"{name}.json"
                    actor.save(path)
                    candidates[name], paths[name] = Ranker.load(path, TWO_TURN_FEATURE_NAMES), path
                write_json(args.out / "training.json", training)
                progress("on_policy_training", iteration, args.iterations)
        validation_rows, validation = evaluate_policies(factory, bridge, plan["validation"], candidates,
            args.validation_samples, args.out / "validation_trajectories.jsonl.gz",
            lambda done, total: progress("validation", done, total))
        write_json(args.out / "validation_results.json", {"policies": validation, "episodes": validation_rows})
        selection = select_validation(validation, list(paths))
        experimental = selection["experimental_learner"]
        shutil.copyfile(paths[experimental], args.out / "experimental_model.json")
        selection["experimental_checkpoint_sha256"] = sha(args.out / "experimental_model.json")
        write_json(args.out / "deployment_policy.json", selection)
        write_json(args.out / "frozen_evaluation_plan.json", {"selection": selection, "seeds": plan["test"],
            "samples": args.test_samples, "selection_complete_before_test": True,
            "candidate_sha256": {name: sha(path) for name, path in paths.items()}})
        test_policies = {"practical": None, "bc_reference": reference, experimental: candidates[experimental]}
        test_rows, test = evaluate_policies(factory, bridge, plan["test"], test_policies,
            args.test_samples, args.out / "test_trajectories.jsonl.gz",
            lambda done, total: progress("test", done, total))
        write_json(args.out / "test_results.json", {"policies": test, "episodes": test_rows})
        if sha(args.initial) != reference_hash or sha(args.out / "frozen_bc_reference.json") != reference_hash:
            raise RuntimeError("Frozen reference checkpoint changed during training")
        if not args.allow_historical:
            subprocess.run([sys.executable, "scripts/check_live_ruleset.py", "--report", str(args.out / "live_postflight.json")], cwd=ROOT, check=True)
        reference_differences = np.array([row["policies"][experimental]["penalized_score"]
            - row["policies"]["bc_reference"]["penalized_score"] for row in test_rows])
        reference_comparison = paired_summary(reference_differences, np.random.default_rng(86028121))
        selection = gradient_test_promotion(selection, test, reference_comparison,
            current_snapshot_verified=not args.allow_historical, existing_baseline=fallback)
        write_json(args.out / "deployment_policy.json", selection)
        result = {"scope": SCOPE, "full_game_ready": False, "selection": selection,
            "completed_update_batches": sum(any(update["optimizer_updated"] for update in row["updates"]) for row in training),
            "accepted_update_batches": sum(not row["rejected_batch"] for row in training),
            "optimizer_updates": sum(update["optimizer_updated"] for row in training for update in row["updates"]),
            "skipped_zero_signal_updates": sum(not update["optimizer_updated"] for row in training for update in row["updates"]),
            "rejected_update_batches": sum(row["rejected_batch"] for row in training),
            "attempted_training_pairs": len(plan["train"]),
            "failed_training_pairs": sum(len(row["failures"]) for row in training),
            "failure_reasons": dict(Counter(error for row in training for failure in row["failures"]
                for error in failure["errors"].values())), "validation": validation, "test": test,
            "frozen_experimental_vs_bc_reference": reference_comparison,
            "current_snapshot_verified": not args.allow_historical,
            "sampled_combats_including_rejections": bridge.requests * 4,
            "elapsed_seconds": time.time() - started,
            "limitations": ["Only audited current Tier1 two-turn fixture with synthetic 15/20/30 second own timers",
                "BC reference and practical opponents remain fixed; no claim of full-game strength",
                "Unsupported batches provide no gradient; supported-batch learning remains conditional on engine scope",
                "Stochastic training optimizes complete returns; deterministic argmax actors are separately evaluated",
                "Monte Carlo returns are noisy; single preregistered test gate only, no test tuning or alternate model selection"]}
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
