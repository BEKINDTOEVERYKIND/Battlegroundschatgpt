#!/usr/bin/env python3
"""Read-only train/validation audit and fixed behavioral-cloning diagnostic.

No held-out evaluation archive is opened, no existing checkpoint is modified,
and no diagnostic policy is promoted. The optional executable check uses only
the source run's already designated validation episodes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import numpy as np
from bg_ai.learning import Dataset, Ranker, Scenario
from bg_ai.recruiting import RecruitAction, RecruitObservation
from bg_ai.turn_budget import TimingProfile
from bg_ai.two_turn_features import TWO_TURN_FEATURE_NAMES
from train_two_turn_recruit import CombatBridge, Factory, rollout
from train_recruit_curriculum import heuristic_action


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def observation(record):
    raw = dict(record)
    raw["legal_actions"] = tuple(RecruitAction(**a) for a in raw["legal_actions"])
    return RecruitObservation(**raw)


def commands_used(roll):
    rows = [r for r in roll["timing_trace"] if r["player_id"] == 0 and r["event"] == "action"]
    return {"commands": len(rows), "paid_ms": sum(r["before"]["remaining_ms"] - r["after"]["remaining_ms"] for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/20260906-two-turn-recruit-v1")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/policy-collapse-audit-sept7.json")
    parser.add_argument("--engine-root", type=Path)
    args = parser.parse_args()
    archive = args.run / "training_trajectories.jsonl.gz"
    checkpoint = args.run / "selected_model.json"
    files = [archive, checkpoint, Path(__file__).resolve(), ROOT / "scripts/train_two_turn_recruit.py",
        ROOT / "python/bg_ai/learning.py", ROOT / "python/bg_ai/two_turn_features.py",
        ROOT / "python/bg_ai/opening_transition.py", ROOT / "simulator/opening-transition-firestone.mjs",
        ROOT / "data/ruleset.json", ROOT / "data/reference_cards.json"]
    before = {str(p.relative_to(ROOT)): sha(p) for p in files}
    frozen = Ranker.load(checkpoint, TWO_TURN_FEATURE_NAMES)
    counts = {s: Counter() for s in ("train", "validation")}
    predictions = {s: Counter() for s in counts}
    examples, scenarios, rows, validation_seeds = [], [], [], []
    with gzip.open(archive, "rt", encoding="utf8") as stream:
        for line in stream:
            trajectory = json.loads(line)
            split = trajectory["split"]
            if split not in counts:
                raise ValueError("Audit refuses archives containing test rows")
            counts[split]["trajectories"] += 1
            if split == "validation":
                validation_seeds.append(trajectory["setup"]["seed"])
            for decision in trajectory["decisions"]:
                candidates = decision["candidates"]
                obs = observation(decision["observation"])
                if candidates[0]["action"] != asdict(heuristic_action(obs)):
                    raise ValueError("Candidate zero was not the declared practical expert")
                if len(candidates) < 2:
                    counts[split]["single_candidate_decisions"] += 1
                    continue
                stats = counts[split]
                stats["decisions"] += 1
                scores = np.asarray([c["score"] for c in candidates])
                vectors = np.asarray([c["features"] for c in candidates])
                predicted = frozen.predict(vectors)
                selected = int(np.argmax(predicted))
                predictions[split][candidates[selected]["action"]["kind"]] += 1
                stats["all_candidate_labels_tied"] += int(np.all(scores == scores[0]))
                stats["practical_action_tied_for_best"] += int(scores[0] == scores.max())
                stats["nonexpert_candidates_tied_with_expert"] += int(np.sum(scores[1:] == scores[0]))
                stats["selected_nonexpert_tied_with_expert"] += int(selected != 0 and scores[selected] == scores[0])
                left, right = np.triu_indices(len(candidates), 1)
                stats["candidate_pairs"] += len(left)
                stats["zero_loss_exact_tied_pairs"] += int(np.sum(scores[left] == scores[right]))
                for i, candidate in enumerate(candidates):
                    if candidate["action"]["kind"] != "freeze":
                        continue
                    stats["freeze_available"] += 1
                    stats["freeze_tied_with_expert"] += int(scores[i] == scores[0])
                    stats["freeze_strictly_beats_expert"] += int(scores[i] > scores[0])
                    stats["freeze_strictly_loses_to_expert"] += int(scores[i] < scores[0])
                    same_inputs = all(canonical([r["request"] for r in a["combats"]]) == canonical([r["request"] for r in b["combats"]])
                                      for a, b in zip(candidate["rollouts"], candidates[0]["rollouts"], strict=True))
                    stats["freeze_same_complete_remaining_combat_requests_as_expert"] += int(same_inputs)
                    if same_inputs:
                        extra = [commands_used(a)["paid_ms"] - commands_used(b)["paid_ms"]
                                 for a, b in zip(candidate["rollouts"], candidates[0]["rollouts"], strict=True)]
                        stats["same_combat_freeze_extra_paid_ms_total"] += sum(extra)
                        stats["same_combat_freeze_rollout_pairs"] += len(extra)
                    others = scores[np.arange(len(scores)) != i]
                    stats["freeze_positive_label_comparisons"] += int(np.sum(scores[i] > others))
                    stats["freeze_negative_label_comparisons"] += int(np.sum(scores[i] < others))
                    if selected == i and len(examples) < 4 and same_inputs:
                        examples.append({"decision_id": decision["decision_id"], "split": split,
                            "turn": decision["turn"], "expert_action": candidates[0]["action"],
                            "freeze_action": candidate["action"], "expert_label": float(scores[0]),
                            "freeze_label": float(scores[i]), "prediction_margin_over_expert": float(predicted[i]-predicted[0]),
                            "identical_remaining_combat_requests": True, "extra_paid_ms_per_sample": extra,
                            "complete_recruit_states_claimed_equal": False})
                expert_targets = np.zeros(len(candidates))
                expert_targets[0] = 1.0
                scenarios.append(Scenario(decision["decision_id"], split, vectors, expert_targets,
                    {"episode_id": trajectory["episode_id"], "turn": decision["turn"]}))
                rows.append((split, vectors, scores, [c["action"]["kind"] for c in candidates], trajectory["episode_id"]))
    # Fixed in advance: a fresh semantic ranker, same 16-unit architecture,
    # 15 epochs and usual optimizer defaults. The expert is only a training
    # target. It neither overrides nor masks actions during learned inference.
    bc = Ranker(TWO_TURN_FEATURE_NAMES, hidden=16, seed=170907)
    losses = bc.fit(Dataset(scenarios, TWO_TURN_FEATURE_NAMES), epochs=15, tie_tolerance=0)
    static = {}
    for name, model in (("frozen_one_deviation_ranker", frozen), ("expert_imitation_15_epochs", bc)):
        static[name] = {}
        for split in counts:
            subset = [r for r in rows if r[0] == split]
            selected = [int(np.argmax(model.predict(r[1]))) for r in subset]
            episode_values = {}
            for r, i in zip(subset, selected):
                episode_values.setdefault(r[4], []).append(float(r[2][i]))
            static[name][split] = {"decisions": len(subset),
                "expert_agreement": float(np.mean([i == 0 for i in selected])),
                "one_deviation_label_mean": float(np.mean([r[2][i] for r, i in zip(subset, selected)])),
                "one_deviation_episode_weighted_mean": float(np.mean([np.mean(v) for v in episode_values.values()])),
                "action_counts": dict(Counter(r[3][i] for r, i in zip(subset, selected)))}
    report = {"purpose": "train/validation diagnostic, no held-out test consumption or policy promotion",
        "source_sha256": before, "splits": {s: dict(c) for s, c in counts.items()},
        "frozen_action_counts": {s: dict(c) for s, c in predictions.items()}, "examples": examples,
        "fixed_imitation_fit": {"hidden": 16, "seed": 170907, "epochs": 15,
            "target": "1 for verified practical candidate zero, 0 for all other candidates",
            "loss": "existing gap-weighted pairwise logistic", "final_epoch_loss": losses[-1],
            "legal_action_masks_changed": False, "warm_start": False},
        "static_diagnostics": static, "full_policy_validation": None,
        "inferences": ["One candidate then practical continuation is evaluated differently from repeating learned argmax.",
            "Exact label ties have zero pairwise-loss gradient even with tie tolerance zero.",
            "One-deviation candidate validation cannot establish full-policy control quality.",
            "Imitation is a behavior initialization/control diagnostic, not demonstrated improvement over its expert."]}
    report["limitations"] = ["Six development validation episodes are too small for a strength claim.",
        "Old fitted Q ranker and fresh imitation differ in initialization/training history; this is a diagnostic, not a matched causal ablation.",
        "The fixed practical expert does not strategically use Freeze; imitating it is initialization rather than a general solution to future shop value.",
        "Same remaining combat requests do not imply identical future recruitment states beyond the truncated objective."]
    if args.engine_root:
        profile = TimingProfile.load(ROOT / "config/turn-budget.json")
        factory = Factory(args.engine_root, json.loads((ROOT / "data/ruleset.json").read_text()), profile)
        validations, failures = [], []
        with tempfile.TemporaryDirectory(prefix="bg-collapse-audit-") as temporary:
            bridge = CombatBridge(Path(temporary) / "firestone.log")
            try:
                for seed in validation_seeds:
                    entry = {"seed": seed, "policies": {}}
                    try:
                        for name, model in (("frozen_one_deviation_ranker", frozen), ("expert_imitation_15_epochs", bc), ("practical", None)):
                            outputs = []
                            for sample in (10, 11, 12, 13):
                                timed, _ = factory.setup(seed)
                                outputs.append(rollout(timed, bridge, seed, model=model, sample=sample))
                            entry["policies"][name] = {"scores": [o["score"] for o in outputs],
                                "mean_score": float(np.mean([o["score"] for o in outputs])),
                                "commands": dict(Counter(c["action"]["kind"] for o in outputs for c in o["commands"] if c["player_id"] == 0)),
                                "mean_paid_ms": float(np.mean([commands_used(o)["paid_ms"] for o in outputs])),
                                "budget_violations": sum(t["after"]["remaining_ms"] < 0 or t["after"]["remaining_actions"] < 0 for o in outputs for t in o["timing_trace"] if t["player_id"] == 0 and t["event"] == "action")}
                        validations.append(entry)
                    except Exception as error:
                        from bg_ai.hsbrsim_adapter import UnsupportedRecruitTransition
                        if not isinstance(error, UnsupportedRecruitTransition):
                            raise
                        failures.append({"seed": seed, "failed_policy": name, "error": str(error)})
                    print(json.dumps({"validation_seed": seed, "accepted": len(validations), "failed": len(failures)}), flush=True)
            finally:
                bridge.close()
        report["full_policy_validation"] = {"seeds": validation_seeds, "samples": [10, 11, 12, 13],
            "split": "source pilot validation only, already development data; not a fresh test",
            "episodes": validations, "failures": failures,
            "means": {name: float(np.mean([e["policies"][name]["mean_score"] for e in validations]))
                      for name in ("frozen_one_deviation_ranker", "expert_imitation_15_epochs", "practical")} if validations else {},
            "policy_promoted": False}
    after = {str(p.relative_to(ROOT)): sha(p) for p in files}
    if after != before:
        raise RuntimeError("Audited source changed during the audit")
    report["sources_unchanged"] = True
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"report": str(args.report), "splits": report["splits"], "static": static,
        "validation_means": (report["full_policy_validation"] or {}).get("means")}, indent=2))


if __name__ == "__main__":
    main()
