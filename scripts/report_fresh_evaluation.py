#!/usr/bin/env python3
"""Audit and summarize fresh simulation of previously frozen selections.

All declared baselines are reported, and a positioning benchmark advantage is
claimed only if every paired 95% CI is positive on >=30 final test scenarios.
This gate is an intersection of tests, not a choice of the easiest baseline.
It cannot establish full-game strength or current ranked-game representativeness.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from bg_ai.learning import _paired_summary


BASELINES = ("random", "attack", "health", "taunt_last")


def _read(path):
    return [json.loads(line) for line in _read_bytes(path).decode().splitlines() if line.strip()]


def _read_bytes(path):
    data = Path(path).read_bytes()
    return gzip.decompress(data) if str(path).endswith(".gz") else data


def summarize(fresh_path, selections_path, seed=20260905, bootstrap_samples=10000):
    fresh, frozen = _read(fresh_path), _read(selections_path)
    if not fresh or bootstrap_samples < 100:
        raise ValueError("Need nonempty evaluation and >=100 bootstrap samples")
    selections = {r["scenario_id"]: r for r in frozen}
    if len(selections) != len(frozen):
        raise ValueError("Duplicate frozen scenario IDs")
    if len({r["scenario_id"] for r in fresh}) != len(fresh):
        raise ValueError("Duplicate fresh scenario IDs")
    if {r["scenario_id"] for r in fresh} != set(selections):
        raise ValueError("Fresh evaluation must contain every frozen scenario exactly once")
    selection_hash = hashlib.sha256(_read_bytes(selections_path)).hexdigest()
    checkpoints, dataset_hashes, engine_versions, trials = set(), set(), set(), set()
    policies, search_trials, proposal_hashes = set(), set(), set()
    for row in fresh:
        frozen_row = selections[row["scenario_id"]]
        provenance = frozen_row["provenance"]
        meta = row["metadata"]
        if row["split"] != frozen_row["split"] or row["split"] not in ("validation", "test"):
            raise ValueError("Fresh split must match frozen held-out split")
        if meta["selectionsSha256"] != selection_hash:
            raise ValueError("Fresh evaluation is not bound to this frozen selection file")
        if meta["combatSeed"] == provenance["label_combat_seed"]:
            raise ValueError("Fresh evaluation reused the training-label RNG stream")
        policy = provenance.get("evaluated_policy", "neural_model_only")
        if policy not in ("neural_model_only", "neural_proposals_plus_simulation_search"):
            raise ValueError("Unknown evaluated policy")
        if meta.get("evaluatedPolicy", policy) != policy:
            raise ValueError("Fresh evaluation changed the frozen policy identity")
        policies.add(policy)
        if policy == "neural_proposals_plus_simulation_search":
            if meta["combatSeed"] == provenance["search_seed"]:
                raise ValueError("Fresh evaluation reused the search RNG stream")
            if provenance["search_seed"] == provenance["label_combat_seed"]:
                raise ValueError("Search reused the training-label RNG stream")
            if provenance["search_trials"] != 128 or provenance["search_top_k"] != 3:
                raise ValueError("Search differs from the fixed top3/128-trial policy")
            searched = frozen_row["search_outcomes"]
            if not searched or len(searched) > 5:
                raise ValueError("Search must include one to five distinct candidates")
            if len({r["candidateIndex"] for r in searched}) != len(searched):
                raise ValueError("Duplicate search candidates")
            if any(r["simulations"]["seed"] != provenance["search_seed"] or r["simulations"]["n"] != 128 for r in searched):
                raise ValueError("Search outcomes differ from frozen RNG/trial settings")
            if max(searched, key=lambda r: r["score"])["candidateIndex"] != frozen_row["selections"]["model"]:
                raise ValueError("Frozen search selection is not the stable best search outcome")
            search_trials.add(provenance["search_trials"])
            proposal_hashes.add(provenance["proposal_file_sha256"])
        if (meta["cardsSha256"] != provenance["cards_sha256"] or
                meta["rulesetSha256"] != provenance["ruleset_sha256"]):
            raise ValueError("Fresh evaluation changed card/ruleset snapshot")
        if provenance["primary_baseline"] != "attack" or set(provenance["all_baselines"]) != set(BASELINES):
            raise ValueError("Unexpected predeclared baseline comparison")
        if set(row["scores"]) != {"model", *BASELINES}:
            raise ValueError("Must report every frozen model/baseline result")
        outcomes_by_candidate = {}
        for name, result in row["scores"].items():
            if result["candidateIndex"] != frozen_row["selections"][name]:
                raise ValueError("Evaluated action differs from frozen selection")
            counts = result["simulations"]
            n = counts["n"]
            if n != meta["trials"] or n < 1 or sum(counts[k] for k in ("won", "tied", "lost")) != n:
                raise ValueError("Invalid simulation counts")
            if any(not isinstance(counts[k], int) or counts[k] < 0 for k in ("won", "tied", "lost", "n")):
                raise ValueError("Simulation counts must be nonnegative integers")
            expected = (counts["won"] + counts["tied"] * 0.5) / n
            if not np.isfinite(result["score"]) or abs(result["score"] - expected) > 1e-12:
                raise ValueError("Reported score does not match simulation outcomes")
            if "seed" in counts and counts["seed"] != meta["combatSeed"]:
                raise ValueError("Simulation seed mismatch")
            candidate = result["candidateIndex"]
            if candidate in outcomes_by_candidate and outcomes_by_candidate[candidate] != result:
                raise ValueError("Identical selected candidates have inconsistent outcomes")
            outcomes_by_candidate[candidate] = result
        checkpoints.add(provenance["checkpoint_sha256"])
        dataset_hashes.add(provenance["dataset_sha256"])
        engine_versions.add(meta["engine"])
        trials.add(meta["trials"])
    if len(checkpoints) != 1 or len(dataset_hashes) != 1 or len(engine_versions) != 1 or len(policies) != 1:
        raise ValueError("Cannot pool different checkpoints, datasets, engine versions, or policies")
    evaluated_policy = next(iter(policies))
    rng = np.random.default_rng(seed)
    results = {}
    for split in ("validation", "test"):
        rows = [r for r in fresh if r["split"] == split]
        if not rows:
            continue
        values = {name: np.array([r["scores"][name]["score"] for r in rows])
                  for name in ("model", *BASELINES)}
        comparisons = {name: _paired_summary(values["model"] - values[name], rng, bootstrap_samples)
                       for name in BASELINES}
        enough = len(rows) >= 30
        beats_all = enough and all(comparisons[b]["ci95"][0] > 0 for b in BASELINES)
        results[split] = {"scenario_count": len(rows),
                          "mean_scores": {k: float(v.mean()) for k, v in values.items()},
                          "paired_model_minus_baseline": comparisons,
                          "positioning_benchmark_gate": {
                              "eligible_final_test": split == "test" and enough,
                              "passes": split == "test" and beats_all,
                              "rule": "At least 30 final test scenarios and positive lower 95% paired CI against every predeclared baseline",
                              "status": "benchmark_advantage" if split == "test" and beats_all else
                                        "validation_only" if split == "validation" else
                                        "insufficient_scenarios" if not enough else "no_clear_advantage"},
                          "scenario_details": rows}
    return {"scope": "combat positioning on generated legal-pool boards; not full-game playing strength",
            "evaluated_policy": evaluated_policy,
            "policy_interpretation": "Combined neural proposals and simulation search; any improvement cannot be attributed to the neural model alone"
                if evaluated_policy == "neural_proposals_plus_simulation_search" else "Neural model only; no simulation search at selection time",
            "search_trials_per_candidate": sorted(search_trials), "proposal_file_sha256": sorted(proposal_hashes),
            "score_definition": "P(combat win) + 0.5 * P(combat tie)",
            "primary_baseline": "attack", "all_baselines": list(BASELINES),
            "selections_sha256": selection_hash, "checkpoint_sha256": next(iter(checkpoints)),
            "dataset_sha256": next(iter(dataset_hashes)), "engine": next(iter(engine_versions)),
            "trials_per_selected_candidate": sorted(trials),
            "bootstrap": {"seed": seed, "samples": bootstrap_samples, "unit": "scenario",
                          "method": "paired percentile bootstrap", "confidence": 0.95},
            "limitations": ["Generated boards are not a ranked-player game-state distribution",
                            "Only candidate board permutations were evaluated",
                            "Independent combat RNG does not make combat EV equal full-game EV",
                            "Repeated model selection on final test results invalidates its held-out status"],
            "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--selections", required=True)
    parser.add_argument("--out", "--report", dest="out", required=True)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    report = summarize(args.input, args.selections, args.seed, args.bootstrap_samples)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    compact = {"output": str(target), "scope": report["scope"], "evaluated_policy": report["evaluated_policy"], "results": {
        split: {k: v for k, v in result.items() if k != "scenario_details"}
        for split, result in report["results"].items()}}
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
