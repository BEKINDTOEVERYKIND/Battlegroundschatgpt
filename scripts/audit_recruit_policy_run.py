#!/usr/bin/env python3
"""Independently audit completed BC/PG reports from their frozen raw evidence.

Does not import the trainer, execute archived source, rerun combat, select another
checkpoint, or inspect an unfinished run's holdout. Failure-penalized episode
scores are checked separately from actual completed sampled combat outcomes.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AUDIT_VERSION = "recruit-policy-evidence-audit-v1"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def equivalent(actual, expected, context):
    if type(expected) in (int, float):
        check(type(actual) in (int, float) and math.isfinite(actual)
              and abs(actual - expected) <= 1e-12, context)
    elif isinstance(expected, dict):
        check(isinstance(actual, dict) and set(actual) == set(expected), context)
        for key in expected:
            equivalent(actual[key], expected[key], f"{context}.{key}")
    elif isinstance(expected, list):
        check(isinstance(actual, list) and len(actual) == len(expected), context)
        for index, value in enumerate(expected):
            equivalent(actual[index], value, f"{context}[{index}]")
    else:
        check(type(actual) is type(expected) and actual == expected, context)


class Evidence:
    def __init__(self, directory):
        self.directory = directory
        self.consumed = {}

    def path(self, name):
        path = self.directory / name
        check(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(self.directory.resolve()),
              f"Missing or unsafe evidence file: {name}")
        self.consumed[name] = {"sha256": sha(path), "bytes": path.stat().st_size}
        return path

    def json(self, name):
        path = self.path(name)
        opener = gzip.open if name.endswith(".gz") else open
        with opener(path, "rt", encoding="utf8") as stream:
            return json.load(stream)

    def rows(self, name):
        with gzip.open(self.path(name), "rb") as stream:
            while line := stream.readline(256 * 1024 * 1024 + 1):
                check(len(line) <= 256 * 1024 * 1024, "Episode evidence exceeds bounded row size")
                yield json.loads(line)

    def unchanged(self):
        for name, record in self.consumed.items():
            check(sha(self.directory / name) == record["sha256"], f"Evidence changed during audit: {name}")


def bootstrap(differences):
    values = np.asarray(differences, dtype=np.float64)
    check(len(values) > 0 and np.isfinite(values).all(), "Invalid bootstrap inputs")
    rng = np.random.default_rng(86028121)
    draws = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return {"mean_delta": float(values.mean()), "ci95": np.quantile(draws, [.025, .975]).tolist()}


def timing_counts(trajectory):
    traces = [trace for trace in trajectory.get("timing_trace", []) if trace["event"] == "action"]
    violated = lambda trace: trace["after"]["remaining_ms"] < 0 or trace["after"]["remaining_actions"] < 0
    return {"all_players": sum(violated(trace) for trace in traces),
            "player_zero": sum(violated(trace) for trace in traces if trace["player_id"] == 0)}


def own_actions(trajectory):
    return Counter(command["action"]["kind"] for command in trajectory.get("commands", []) if command["player_id"] == 0)


def actual_score(trajectory, seed, sample):
    check(trajectory.get("complete") is True and trajectory.get("seed") == seed
          and trajectory.get("sample") == sample, "Completed trajectory identity differs from plan")
    combats = trajectory.get("combats", [])
    check(len(combats) == 2, "Complete score requires both actual sampled combats")
    scores = []
    for turn, combat in enumerate(combats, 1):
        receipt, request = combat["receipt"], combat["request"]
        expected_seed = (seed * 2654435761 + turn * 32452843 + sample * 49979687) & 0xFFFFFFFF
        check(receipt["turn"] == turn and receipt["seed"] == expected_seed
              and request["seed"] == expected_seed, "Combat seed or turn differs from planned sample")
        outcomes = receipt["outcomes"]
        players = [player for row in outcomes for player in (row["player_a"], row["player_b"])]
        check(sorted(players) == list(range(8)), "Sampled combat does not cover all eight players exactly once")
        for row in outcomes:
            check(row["samples"] == 1 and row["winner_id"] in (None, row["player_a"], row["player_b"]),
                  "Invalid sampled combat outcome")
        own = next(row for row in outcomes if 0 in (row["player_a"], row["player_b"]))
        scores.append(.5 if own["winner_id"] is None else float(own["winner_id"] == 0))
    value = math.fsum(scores) / 2
    equivalent(trajectory["score"], value, "Recorded rollout score differs from both combat receipts")
    return value


def audit_evaluation(evidence, phase, seeds, names, samples):
    report = evidence.json(f"{phase}_results.json")
    rows, raw, diagnostics = {}, set(), {}
    for record in evidence.rows(f"{phase}_trajectories.jsonl.gz"):
        seed, name = record["seed"], record["policy"]
        key = (seed, name)
        check(seed in seeds and name in names and key not in raw, "Duplicate or undeclared policy/seed evaluation")
        raw.add(key)
        successful, failed, actions = [], [], Counter()
        complete_scores, partial_actions = [], Counter()
        timing, partial_timing = Counter(), Counter()
        for trajectory in record["rollouts"]:
            sample = trajectory["sample"]
            check(type(sample) is int and 0 <= sample < samples and sample not in successful, "Repeated or undeclared completed sample")
            successful.append(sample)
            complete_scores.append(actual_score(trajectory, seed, sample))
            actions.update(own_actions(trajectory))
            timing.update(timing_counts(trajectory))
        for failure in record["failures"]:
            sample = failure["sample"]
            check(type(sample) is int and 0 <= sample < samples and sample not in successful + failed,
                  "Repeated, completed or undeclared failed sample")
            partial = failure["partial"]
            check(partial.get("seed") == seed and partial.get("sample") == sample
                  and partial.get("complete") is not True, "Failed sample contains a complete or unrelated trajectory")
            failed.append(sample)
            partial_actions.update(own_actions(partial))
            partial_timing.update(timing_counts(partial))
        check(set(successful + failed) == set(range(samples)), "An attempted sample disappeared from evidence")
        complete = not failed
        penalized = math.fsum(complete_scores) / samples if complete else 0.0
        equivalent(record["complete"], complete, "Episode completion flag")
        equivalent(record["penalized_score"], penalized, "Failure penalty differs from preregistered all-samples rule")
        compact = {"complete": complete, "penalized_score": penalized, "failed_samples": len(failed),
                   "failure_reasons": dict(Counter(f["error"] for f in record["failures"])),
                   "action_counts": dict(actions), "timing_budget_violations": timing["all_players"]}
        rows.setdefault(seed, {})[name] = compact
        d = diagnostics.setdefault(name, {"completed_sample_scores": [], "completed_sample_action_counts": Counter(),
            "failed_partial_action_counts": Counter(), "completed_sample_timing_violations": Counter(),
            "failed_partial_timing_violations": Counter(), "failed_samples": 0})
        d["completed_sample_scores"].extend(complete_scores)
        d["completed_sample_action_counts"].update(actions)
        d["failed_partial_action_counts"].update(partial_actions)
        d["completed_sample_timing_violations"].update(timing)
        d["failed_partial_timing_violations"].update(partial_timing)
        d["failed_samples"] += len(failed)
    check(set(raw) == {(seed, name) for seed in seeds for name in names}, "Evaluation policy/seed grid is incomplete")
    expected_rows = [{"seed": seed, "policies": rows[seed]} for seed in seeds]
    equivalent(report["episodes"], expected_rows, f"{phase} compact episode report")
    summaries = {}
    for name in names:
        values = [rows[seed][name]["penalized_score"] for seed in seeds]
        differences = [rows[seed][name]["penalized_score"] - rows[seed]["practical"]["penalized_score"] for seed in seeds]
        supported = [i for i, seed in enumerate(seeds) if rows[seed][name]["complete"] and rows[seed]["practical"]["complete"]]
        count = sum(rows[seed][name]["complete"] for seed in seeds)
        summaries[name] = {"attempted_episodes": len(seeds), "complete_episodes": count,
            "failed_episodes": len(seeds) - count, "failure_penalized_mean": math.fsum(values) / len(values),
            "timing_budget_violations": sum(rows[seed][name]["timing_budget_violations"] for seed in seeds),
            "paired_all_attempts": bootstrap(differences), "support_intersection_episodes": len(supported),
            "paired_supported_intersection": bootstrap([differences[i] for i in supported]) if supported else None}
        d = diagnostics[name]
        scores = d.pop("completed_sample_scores")
        d.update(completed_samples=len(scores), actual_complete_sample_mean=math.fsum(scores) / len(scores) if scores else None,
                 failure_penalized_episode_mean=summaries[name]["failure_penalized_mean"])
    equivalent(report["policies"], summaries, f"{phase} policy report and paired bootstrap")
    return summaries, diagnostics, expected_rows


def frozen_source_pool(sources, name):
    module = ast.parse(sources["python/bg_ai/hsbrsim_opening.py"]["content"])
    for node in module.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            check(isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                  and node.value.func.id == "frozenset", "Unexpected frozen pool declaration")
            return set(ast.literal_eval(node.value.args[0]))
    raise ValueError("Frozen source does not declare the opening pool")


def audit_expert(evidence, prereg, results, sources, *, verified_recovery=False):
    prefix = "recovered_expert_data/" if verified_recovery else ""
    recovery = None
    if verified_recovery:
        recovery = evidence.json(prefix + "recovery_manifest.json")
        check(recovery["status"] == "verified_recovery" and recovery["holdout_trajectories_read"] is False,
              "Expert recovery was not completely verified without holdout trajectories")
        check(sha(evidence.path("expert_trajectories.jsonl.gz")) == recovery["original_archive_sha256"], "Original truncated bytes changed")
        check(sha(evidence.path("expert_data_manifest.json")) == recovery["original_manifest_sha256"], "Original expert manifest changed")
        candidates = [f"policy-h{width}-e{epoch}.json" for width in prereg["candidate_models"]["hidden"]
                      for epoch in prereg["candidate_models"]["epochs"]]
        check([row["checkpoint"] for row in recovery["checkpoint_reproduction"]] == candidates,
              "Recovery omitted a preregistered checkpoint reproduction")
        for row in recovery["checkpoint_reproduction"]:
            original = sha(evidence.path(row["checkpoint"]))
            regenerated = sha(evidence.path(prefix + row["checkpoint"]))
            check(row["byte_identical"] is True and original == regenerated == row["original_sha256"] == row["recovered_sha256"],
                  "Recovered training data did not reproduce every original checkpoint byte")
        check(recovery["original_rows_json_identical"] == recovery["prefix"]["complete_original_rows"],
              "Not every preserved original trajectory matched recovery")
    manifest = evidence.json(prefix + "expert_data_manifest.json")
    archive = evidence.path(prefix + "expert_trajectories.jsonl.gz")
    check(manifest["archive_sha256"] == sha(archive), "Expert archive differs from manifest SHA256")
    plan = prereg["seed_plan"]["train"]
    equivalent(manifest["attempted_seeds"], plan, "Expert attempted-seed manifest")
    accepted, ids, visible = set(), set(), set()
    for trajectory in evidence.rows(prefix + "expert_trajectories.jsonl.gz"):
        seed = trajectory["seed"]
        check(seed in plan and seed not in accepted, "Repeated or nontraining expert episode")
        accepted.add(seed)
        actual_score(trajectory, seed, 0)
        for decision in trajectory["decisions"]:
            identifier = decision["decision_id"]
            check(decision["split"] == "train" and identifier not in ids, "Repeated or nontraining expert decision")
            ids.add(identifier)
            observation = decision["observation"]
            check(observation["player_id"] == 0, "Expert training contains another player's private observation")
            for zone in ("board", "hand", "shop"):
                visible.update(card["card_id"] for card in observation["private_state"].get(zone, []))
    failures = manifest["failures"]
    failed = [row["seed"] for row in failures]
    check(len(set(failed)) == len(failed) and not accepted.intersection(failed)
          and accepted.union(failed) == set(plan), "Expert attempts are not completely accounted for")
    equivalent(manifest["accepted_trajectories"], len(accepted), "Expert accepted count")
    equivalent(manifest["decisions"], len(ids), "Expert decision count")
    summary = {"attempted_trajectories": len(plan), "accepted_trajectories": len(accepted),
               "decisions": len(ids), "failure_reasons": dict(Counter(row["error"] for row in failures))}
    equivalent(results["training"], summary, "Training result counts")
    if (evidence.directory / (prefix + "expert_rejected_trajectories.jsonl.gz")).exists():
        rejected = list(evidence.rows(prefix + "expert_rejected_trajectories.jsonl.gz"))
        equivalent([{k: row[k] for k in ("seed", "error")} for row in rejected], failures, "Rejected expert evidence")
    coverage = {}
    for label, declaration in (("minions", "TIER1_MINIONS"), ("spells", "TIER1_SPELLS")):
        pool = frozen_source_pool(sources, declaration)
        coverage[label] = {"observed_count": len(visible & pool), "active_pool_count": len(pool),
                           "observed_ids": sorted(visible & pool), "missing_ids": sorted(pool - visible)}
    output = {**summary, "accepted_visible_training_coverage": coverage,
              "coverage_scope": "Player-zero board/hand/shop in accepted expert training decisions; frozen active Tier1 declarations"}
    if recovery is not None:
        check(recovery["recovered_archive_sha256"] == sha(archive)
              and recovery["regenerated_trajectories"] == len(accepted) and recovery["regenerated_decisions"] == len(ids),
              "Recovered archive or counts differ from verified recovery manifest")
        output["verified_recovery"] = {"original_archive_integrity": "incomplete original preserved unchanged",
            "original_rows_matched": recovery["original_rows_json_identical"],
            "all_original_checkpoints_byte_identical": True, "recovered_archive_sha256": sha(archive)}
    return output


def audit_gradient_training(evidence, prereg, results):
    records, batches = {}, {}
    plan = prereg["seed_plan"]["train"]
    for row in evidence.rows("training_trajectories.jsonl.gz"):
        seed = row["seed"]
        check(seed in plan and seed not in records, "Repeated or undeclared gradient training seed")
        records[seed] = row
        expected_iteration = plan.index(seed) // prereg["episodes_per_iteration"] + 1
        check(row["iteration"] == expected_iteration and set(row["policies"]) == {"actor", "practical"}, "Gradient batch identity")
        scores = {}
        for policy, value in row["policies"].items():
            if value["complete"]:
                scores[policy] = actual_score(value["trajectory"], seed, 0)
            else:
                check(value["partial"].get("complete") is not True, "Failed gradient trajectory marked complete")
        check(row["complete"] == (len(scores) == 2), "Gradient paired completion differs from evidence")
        if row["complete"]:
            equivalent(row["advantage"], scores["actor"] - scores["practical"], "Gradient return advantage")
        batches.setdefault(expected_iteration, []).append(row)
    check(set(records) == set(plan), "Missing gradient training attempts")
    training = evidence.json("training.json")
    check(len(training) == prereg["iterations"], "Missing gradient training batches")
    for batch in training:
        rows = batches[batch["iteration"]]
        complete = [row for row in rows if row["complete"]]
        failures = [{"seed": row["seed"], "errors": {name: value["error"] for name, value in row["policies"].items() if not value["complete"]}}
                    for row in rows if not row["complete"]]
        equivalent(batch["attempted_episodes"], len(rows), "Gradient attempted batch count")
        equivalent(batch["complete_pairs"], len(complete), "Gradient complete-pair count")
        equivalent(batch["failures"], failures, "Gradient failure evidence")
        equivalent(batch["rejected_batch"], bool(failures), "Gradient whole-batch rejection")
        if failures:
            check(batch["updates"] == [], "Rejected gradient batch nevertheless updated optimizer")
        value = math.fsum(row["advantage"] for row in complete) / len(complete) if complete else None
        equivalent(batch["mean_complete_pair_advantage"], value, "Gradient batch mean return")
    computed = {"completed_update_batches": sum(any(update["optimizer_updated"] for update in row["updates"]) for row in training),
        "accepted_update_batches": sum(not row["rejected_batch"] for row in training),
        "optimizer_updates": sum(update["optimizer_updated"] for row in training for update in row["updates"]),
        "skipped_zero_signal_updates": sum(not update["optimizer_updated"] for row in training for update in row["updates"]),
        "rejected_update_batches": sum(row["rejected_batch"] for row in training), "attempted_training_pairs": len(plan),
        "failed_training_pairs": sum(len(row["failures"]) for row in training),
        "failure_reasons": dict(Counter(error for row in training for failure in row["failures"] for error in failure["errors"].values()))}
    for key, value in computed.items():
        equivalent(results[key], value, "Gradient result." + key)
    return computed


def audit_run(directory, *, evaluation_only=False, verified_recovery=False):
    check(not (evaluation_only and verified_recovery), "Evaluation-only and recovered-training modes are mutually exclusive")
    # This check must precede every access to validation or holdout evidence.
    check((directory / "results.json").is_file(), "Run is unfinished; refusing to inspect validation or holdout evidence")
    evidence = Evidence(directory)
    result, prereg = evidence.json("results.json"), evidence.json("preregistration.json")
    frozen, deployment = evidence.json("frozen_evaluation_plan.json"), evidence.json("deployment_policy.json")
    sources = evidence.json("execution_sources.json.gz")
    equivalent({name: record["sha256"] for name, record in sources.items()}, prereg["source_sha256"], "Archived execution-source inventory")
    for name, record in sources.items():
        check(hashlib.sha256(record["content"].encode("utf8")).hexdigest() == record["sha256"], "Archived source content SHA256: " + name)
    plan = prereg["seed_plan"]
    check(set(plan) == {"train", "validation", "test"}, "Unexpected seed families")
    seeds = [seed for family in plan.values() for seed in family]
    check(all(type(seed) is int and seed >= 0 for seed in seeds) and len(seeds) == len(set(seeds)), "Seed families overlap or repeat")
    equivalent(frozen["seeds"], plan["test"], "Frozen test seeds")
    equivalent(frozen["samples"], prereg["test_samples"], "Frozen test sample count")
    check(frozen["selection_complete_before_test"] is True, "Evaluation plan was not declared frozen")
    selected = frozen["selection"]["experimental_learner"]
    checkpoint_hash = sha(evidence.path("experimental_model.json"))
    check(checkpoint_hash == frozen["selection"]["experimental_checkpoint_sha256"], "Frozen experimental checkpoint SHA256")
    gradient = "reference_sha256" in prereg
    check(not (gradient and verified_recovery), "Expert recovery is only supported for behavior cloning")
    if gradient:
        reference_path = evidence.path("frozen_bc_reference.json")
        check(sha(reference_path) == prereg["reference_sha256"], "Frozen BC reference SHA256")
        reference = evidence.json("frozen_bc_reference.json")
        inherited = set(reference["metadata"].get("reserved_evaluation_seeds", []))
        inherited.update(row["episode_seed"] for row in reference["metadata"].get("training_sources", []) if "episode_seed" in row)
        check(not inherited.intersection(seeds), "Gradient seeds reuse the BC training or reserved evaluation families")
        check(sha(evidence.path("executed_train_recruit_policy_gradient.py")) == prereg["executed_script_sha256"], "Archived gradient runner SHA256")
        candidates = list(frozen["candidate_sha256"])
        for name, digest in frozen["candidate_sha256"].items():
            path = "frozen_bc_reference.json" if name == "bc_reference" else name + ".json"
            check(sha(evidence.path(path)) == digest, "Frozen gradient candidate SHA256")
        test_names = list(dict.fromkeys(["practical", "bc_reference", selected]))
        training = (audit_gradient_training(evidence, prereg, result) if not evaluation_only
                    else {"status": "not_audited", "reason": "Explicit evaluation-only mode"})
    else:
        training_rows = evidence.json("training.json")
        candidates = [row["candidate"] for row in training_rows]
        for row in training_rows:
            check(sha(evidence.path(row["candidate"] + ".json")) == row["checkpoint_sha256"], "Frozen BC candidate SHA256")
        check(checkpoint_hash == frozen["experimental_checkpoint_sha256"], "Frozen BC evaluation SHA256")
        test_names = ["practical", selected]
        training = (audit_expert(evidence, prereg, result, sources, verified_recovery=verified_recovery) if not evaluation_only
                    else {"status": "not_audited", "reason": "Explicit evaluation-only mode"})
    selected_path = "frozen_bc_reference.json" if selected == "bc_reference" else selected + ".json"
    check(sha(evidence.path(selected_path)) == checkpoint_hash, "Experimental checkpoint is not the frozen selected candidate")
    validation, validation_diagnostics, _ = audit_evaluation(evidence, "validation", plan["validation"], ["practical", *candidates], prereg["validation_samples"])
    test, test_diagnostics, test_rows = audit_evaluation(evidence, "test", plan["test"], test_names, prereg["test_samples"])
    equivalent(result["validation"], validation, "Final validation summary")
    equivalent(result["test"], test, "Final test summary")
    equivalent(result["selection"], deployment, "Final deployment record")
    check(deployment["experimental_learner"] == selected and deployment["experimental_checkpoint_sha256"] == checkpoint_hash,
          "Final deployment changed the learner chosen before test")
    if gradient:
        differences = [row["policies"][selected]["penalized_score"] - row["policies"]["bc_reference"]["penalized_score"] for row in test_rows]
        equivalent(result["frozen_experimental_vs_bc_reference"], bootstrap(differences), "Frozen experimental versus BC bootstrap")
    current = not prereg["historical_mode"]
    equivalent(result["current_snapshot_verified"], current, "Historical/current run labeling")
    if current:
        for phase in ("preflight", "postflight"):
            check_ = evidence.json(f"live_{phase}.json")
            check(check_["current"] is True and check_["network_checked"] is True and check_["failures"] == [], "Original live check failed")
    evidence.unchanged()
    status = "passed_evaluation_only" if evaluation_only else "passed_with_verified_training_recovery" if verified_recovery else "passed"
    return {"audit_version": AUDIT_VERSION, "status": status, "run": directory.name,
            "kind": "policy_gradient" if gradient else "behavior_cloning", "full_game_ready": False,
            "independent_seed_families": {name: len(values) for name, values in plan.items()},
            "experimental_checkpoint_sha256": checkpoint_hash, "training": training,
            "validation": validation_diagnostics, "test": test_diagnostics,
            "reported_bootstrap_recomputed": True, "consumed_files": evidence.consumed,
            "checks": ["Archived source contents match declared SHA256 without importing them",
                "Exact planned policy/seed/sample grid including every failed sample",
                "Actual complete scores recomputed from both sampled combat receipts",
                "Failure-penalized episode scores separated from complete sampled combat scores",
                "Own action counts and own/all-player timing counts, with failed partial traces separate",
                "Both evaluation summaries and paired bootstrap intervals independently recomputed",
                "Frozen experimental checkpoint identity and disjoint seed families"],
            "limitations": ["Evidence integrity audit; no simulator reexecution or proof of simulator correctness",
                "Archives declare chronology; hashes alone cannot prove historical write timestamps",
                "Original run currentness is preserved; no claim of currentness on the audit date"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--evaluation-only", action="store_true",
                        help="Explicitly exclude training-archive checks; writes a separately labeled evaluation-only report")
    parser.add_argument("--verified-recovery", action="store_true",
                        help="Explicitly audit separately recovered BC expert data after all original checkpoints reproduce byte-for-byte")
    args = parser.parse_args()
    output = audit_run(args.run, evaluation_only=args.evaluation_only, verified_recovery=args.verified_recovery)
    name = "independent_evaluation_audit.json" if args.evaluation_only else "independent_recovery_audit.json" if args.verified_recovery else "independent_audit.json"
    path = args.run / name
    contents = json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        check(path.read_text() == contents, "Existing independent audit differs; refusing to overwrite evidence")
    else:
        path.write_text(contents)
    print(json.dumps({"status": output["status"], "run": output["run"], "kind": output["kind"],
                      "training": output["training"], "test": output["test"]}, allow_nan=False))


if __name__ == "__main__":
    main()
