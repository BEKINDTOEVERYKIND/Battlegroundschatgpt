#!/usr/bin/env python3
"""Independently verify and summarize the frozen positioning adapter audit.

This script performs no model fitting, candidate search, or policy selection.
The six policy scores use the same per-scenario combat seeds. Bootstrap rows
are also shared across every comparison, with the settings fixed in audit_plan.
"""

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/20260905-positioning-corrected-audit"
BASELINES = ("random", "attack", "health", "taunt_last")
POLICIES = ("hybrid", "model", *BASELINES)
ORIGINAL_REPORT_HASHES = {
    "hybrid": "3ddf0ad005d90fbcc1e575d8150f8bf4d359c935f1e15824440258e7f729765a",
    "model": "d8c2cf56de236ac679f62bf5a03c22c79187cc9b1c66d9210c4e26f8517b4355",
}


def digest(value):
    return hashlib.sha256(value).hexdigest()


def read_bytes(path):
    value = Path(path).read_bytes()
    return gzip.decompress(value) if str(path).endswith(".gz") else value


def read_rows(path):
    return [json.loads(line) for line in read_bytes(path).splitlines() if line.strip()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(plan, rows, scenarios, lines, selections):
    require(len(rows) == len(scenarios) == plan["scenarios"] == 1000, "Expected all 1,000 frozen scenarios")
    require(len({r["scenario_id"] for r in rows}) == 1000, "Duplicate audit IDs")
    total = 0
    for index, (row, scenario, raw_line) in enumerate(zip(rows, scenarios, lines)):
        meta = row["metadata"]
        require(row["input_index"] == index and row["scenario_id"] == scenario["scenario_id"], "Changed input order or scenario")
        require(row["split"] == scenario["split"] == "test", "Non-test scenario")
        require(row["source_scenario_sha256"] == digest(raw_line), "Changed raw source scenario")
        source_boards = json.dumps({"candidates": scenario["candidates"], "opponent": scenario["opponent"]}, separators=(",", ":"), ensure_ascii=False).encode()
        require(row["source_boards_sha256"] == digest(source_boards), "Changed source candidates or opponent")
        require(meta["adapterVersion"] == plan["adapter"]["adapterVersion"] and meta["datasetSchemaVersion"] == 2, "Incorrect adapter version")
        require(meta["source_dataset_adapter"] == "legacy-v1" and meta["source_dataset_schema"] == 1 and meta["audit_only_reinterpretation"] is True, "Missing explicit legacy reinterpretation")
        for name in ("cards_sha256", "ruleset_sha256", "adapter_source_sha256", "engine", "referencePackage"):
            require(meta[name] == plan[name], f"Mixed audit provenance: {name}")
        for name in ("hybrid", "model"):
            require(meta["selections_sha256"][name] == plan["expected_raw_input_sha256"][name], "Changed frozen selection file")
        old_tribes = scenario["metadata"]["validTribes"]
        corrected = ["MECH" if tribe == "MECHANICAL" else tribe for tribe in old_tribes]
        require(row["legacy_lobby_tribes"] == old_tribes and row["corrected_lobby_tribes"] == corrected, "Unexpected lobby change")
        require(len(set(corrected)) == 5, "Invalid corrected lobby")
        expected_seed = (plan["base_seed"] + (index + 1) * 2654435761) & 0xFFFFFFFF
        original_seed = (982451653 + (index + 1) * 2654435761) & 0xFFFFFFFF
        hybrid, model = selections["hybrid"][row["scenario_id"]], selections["model"][row["scenario_id"]]
        require(hybrid["provenance"]["model_only_candidate"] == model["selections"]["model"], "Hybrid proposal mismatch")
        require(hybrid["provenance"]["proposal_file_sha256"] == plan["expected_raw_input_sha256"]["model"], "Changed hybrid proposal provenance")
        require(max(hybrid["search_outcomes"], key=lambda r: r["score"])["candidateIndex"] == hybrid["selections"]["model"], "Changed original hybrid choice")
        require(meta["combat_seed"] == expected_seed and meta["original_evaluation_seed"] == original_seed, "Changed evaluation seed")
        require(expected_seed not in (original_seed, scenario["metadata"]["combatSeed"], hybrid["provenance"]["search_seed"]), "Reused label/search/original RNG")
        require(set(row["scores"]) == set(POLICIES), "Missing policy or baseline")
        outcomes = {}
        for name in POLICIES:
            value = row["scores"][name]
            candidate = hybrid["selections"]["model"] if name == "hybrid" else model["selections"][name]
            if name in BASELINES:
                require(hybrid["selections"][name] == candidate, "Original baseline orders differ")
            require(value["candidate_index"] == candidate, "Changed frozen action")
            counts = value["simulations"]
            require(counts["seed"] == expected_seed and counts["n"] == meta["trials"] == plan["trials_per_distinct_order"] == 1024, "Unequal seed or trial count")
            require(all(type(counts[k]) is int and counts[k] >= 0 for k in ("won", "tied", "lost", "n")), "Invalid counts")
            require(sum(counts[k] for k in ("won", "tied", "lost")) == counts["n"], "Counts do not sum to trials")
            require(value["score"] == (counts["won"] + counts["tied"] / 2) / counts["n"], "Score does not match outcomes")
            require(np.isfinite(counts["meanNetDamage"]), "Nonfinite damage")
            if candidate in outcomes:
                require(value == outcomes[candidate], "Identical frozen orders have unequal outcomes")
            outcomes[candidate] = value
        require(row["actual_combat_simulations"] == len(outcomes) * 1024, "Incorrect actual compute")
        total += row["actual_combat_simulations"]
    return total


def summary(rows, bootstrap):
    values = np.array([[r["scores"][name]["score"] for name in POLICIES] for r in rows])
    pairs = [(policy, baseline) for policy in ("hybrid", "model") for baseline in BASELINES] + [("hybrid", "model")]
    deltas = np.column_stack([values[:, POLICIES.index(a)] - values[:, POLICIES.index(b)] for a, b in pairs])
    rng = np.random.default_rng(bootstrap["seed"])
    draws = []
    for offset in range(0, bootstrap["samples"], 200):
        indices = rng.integers(0, len(rows), size=(min(200, bootstrap["samples"] - offset), len(rows)))
        draws.append(deltas[indices].mean(axis=1))
    limits = np.quantile(np.concatenate(draws), [0.025, 0.975], axis=0)
    comparisons = {f"{a}_minus_{b}": {"mean": float(deltas[:, i].mean()), "ci95": limits[:, i].tolist(), "scenarios": len(rows)} for i, (a, b) in enumerate(pairs)}
    return {
        "scenario_count": len(rows),
        "mean_scores": {name: float(values[:, i].mean()) for i, name in enumerate(POLICIES)},
        "paired_comparisons": comparisons,
        "beats_every_predeclared_baseline": {policy: len(rows) >= 30 and all(comparisons[f"{policy}_minus_{baseline}"]["ci95"][0] > 0 for baseline in BASELINES) for policy in ("hybrid", "model")},
    }


def legacy_audit(plan, selections):
    """Verify the exact originally published actions and score means from counts."""
    result = {}
    legacy_baselines = {}
    for policy, folder in (("hybrid", "20260905-positioning-search-v1"), ("model", "20260905-positioning-v2")):
        path = ROOT / "runs" / folder / "fresh_evaluation.jsonl"
        require(digest((path.parent / "evaluation.json").read_bytes()) == ORIGINAL_REPORT_HASHES[policy], "Original published evaluation report changed")
        original = read_rows(path)
        published = json.loads((path.parent / "evaluation.json").read_text())
        require(original == published["results"]["test"]["scenario_details"], "Original raw evaluation differs from published report")
        require(len(original) == 1000 and len({r["scenario_id"] for r in original}) == 1000, "Original evaluation incomplete")
        require({r["scenario_id"] for r in original} == set(selections[policy]), "Original scenario set changed")
        for index, row in enumerate(original):
            frozen = selections[policy][row["scenario_id"]]
            require(row["metadata"]["selectionsSha256"] == plan["expected_raw_input_sha256"][policy], "Original selection hash mismatch")
            seed = (982451653 + (index + 1) * 2654435761) & 0xFFFFFFFF
            require(row["metadata"]["inputIndex"] == index and row["metadata"]["combatSeed"] == seed, "Original evaluation seed changed")
            for name, value in row["scores"].items():
                require(value["candidateIndex"] == frozen["selections"][name], "Published evaluation did not use frozen action")
                c = value["simulations"]
                require(c["n"] == 1024 and c["seed"] == seed and c["won"] + c["tied"] + c["lost"] == c["n"], "Invalid original counts or RNG")
                require(value["score"] == (c["won"] + c["tied"] / 2) / c["n"], "Original scores differ from counts")
                if name in BASELINES:
                    key = (row["scenario_id"], name)
                    require(key not in legacy_baselines or legacy_baselines[key] == value, "Original baseline outcomes differ between published runs")
                    legacy_baselines[key] = value
        means = {name: float(np.mean([r["scores"][name]["score"] for r in original])) for name in ("model", *BASELINES)}
        require(means == published["results"]["test"]["mean_scores"], "Published aggregate means differ from raw outcomes")
        legacy_rng = np.random.default_rng(published["bootstrap"]["seed"])
        for baseline in BASELINES:
            differences = np.array([r["scores"]["model"]["score"] - r["scores"][baseline]["score"] for r in original])
            delta = float(differences.mean())
            require(delta == published["results"]["test"]["paired_model_minus_baseline"][baseline]["mean"], "Published paired means differ")
            boot_means = np.concatenate([differences[legacy_rng.integers(0, len(original), size=(min(256, 10000 - offset), len(original)))].mean(axis=1) for offset in range(0, 10000, 256)])
            limits = np.quantile(boot_means, [0.025, 0.975]).tolist()
            require(limits == published["results"]["test"]["paired_model_minus_baseline"][baseline]["ci95"], "Published original confidence intervals do not reproduce")
        result[policy] = {"raw_evaluation_sha256": digest(path.read_bytes()), "frozen_actions_verified": 1000, "original_paired_confidence_intervals_reproduced": True, "published_mean_scores": means, "published_paired_comparisons": published["results"]["test"]["paired_model_minus_baseline"]}
    return result


def coverage():
    coverage_path = ROOT / "runs/20260905-positioning-v1/positions.jsonl.coverage.json"
    declared = json.loads(coverage_path.read_text())
    eligible = {r["id"] for r in declared["rows"] if r["positioningExclusion"] is None}
    reference = {r["id"]: r for r in json.loads((ROOT / "data/reference_cards.json").read_text())}
    output = {"declared_active_minions": declared["activeMinions"], "nominal_positioning_eligible_minions": len(eligible), "coverage_manifest_sha256": digest(coverage_path.read_bytes()), "datasets": {}}
    for name, relative in (("legacy_training_dataset", "runs/20260905-positioning-v1/positions.jsonl.gz"), ("frozen_test", "runs/20260905-holdout-v2/positions.jsonl.gz")):
        observed = set()
        split_counts = Counter()
        count = alias_lobbies = 0
        sha = hashlib.sha256()
        with gzip.open(ROOT / relative, "rb") as stream:
            for line in stream:
                sha.update(line)
                row = json.loads(line)
                count += 1
                split_counts[row["split"]] += 1
                alias_lobbies += "MECHANICAL" in row["metadata"]["validTribes"]
                observed.update(e["cardId"] for e in row["candidates"][0]["board"] + row["opponent"])
        require(observed <= eligible, "Dataset contains a card outside its nominal positioning subset")
        absent = sorted(eligible - observed)
        output["datasets"][name] = {"file": relative, "raw_dataset_sha256": sha.hexdigest(), "scenarios": count, "splits": dict(split_counts), "legacy_mechanical_lobbies": alias_lobbies, "observed_minions": len(observed), "observed_minion_ids": sorted(observed), "missing_nominal_minions": [{"id": card_id, "name": reference[card_id]["name"], "races": reference[card_id].get("races", [])} for card_id in absent]}
    return output


def main():
    plan = json.loads((RUN / "audit_plan.json").read_text())
    for name, expected in plan["expected_raw_input_sha256"].items():
        require(digest(read_bytes(ROOT / plan["input_files"][name])) == expected, f"Frozen {name} file changed")
    require(digest((ROOT / "simulator/firestone.mjs").read_bytes()) == plan["adapter_source_sha256"], "Audit adapter source changed")
    require(digest((ROOT / "scripts/reevaluate_positioning_v2.mjs").read_bytes()) == plan["evaluator_source_sha256"], "Audit evaluator source changed")
    lines = read_bytes(ROOT / plan["input_files"]["dataset"]).splitlines()
    scenarios = [json.loads(line) for line in lines]
    selections = {name: {r["scenario_id"]: r for r in read_rows(ROOT / plan["input_files"][name])} for name in ("hybrid", "model")}
    rows = read_rows(RUN / "fresh_evaluation.jsonl")
    total = verify(plan, rows, scenarios, lines, selections)
    primary = summary(rows, plan["bootstrap"])
    legacy = legacy_audit(plan, selections)
    actual_coverage = coverage()
    best = max(BASELINES, key=primary["mean_scores"].get)
    report = {
        "audit": plan["audit"], "scope": "Frozen legacy synthetic positioning boards and choices rescored with corrected canonical-MECH semantics; not new current-complete training or full-game strength",
        "protocol": plan, "raw_evaluation_sha256": digest((RUN / "fresh_evaluation.jsonl").read_bytes()),
        "actual_combat_simulations": total, "policy_order_evaluations": len(rows) * len(POLICIES),
        "all_1000_frozen_scenarios_and_choices_verified": True,
        "legacy_published_evaluation_audit": legacy, "coverage": actual_coverage,
        "results": primary,
        "diagnostics_not_policy_selection": {
            "legacy_mechanical_lobbies": summary([r for r in rows if "MECHANICAL" in r["legacy_lobby_tribes"]], plan["bootstrap"]),
            "other_lobbies": summary([r for r in rows if "MECHANICAL" not in r["legacy_lobby_tribes"]], plan["bootstrap"]),
        },
        "claim_assessment": {
            "previous_hybrid_advantage_over_best_heuristic": legacy["hybrid"]["published_paired_comparisons"]["taunt_last"],
            "corrected_best_heuristic_by_mean": best,
            "corrected_hybrid_advantage_over_best_heuristic": primary["paired_comparisons"][f"hybrid_minus_{best}"],
            "restricted_hybrid_advantage_survives": primary["beats_every_predeclared_baseline"]["hybrid"],
            "neural_model_alone_beats_every_baseline": primary["beats_every_predeclared_baseline"]["model"],
            "old_138_observed_card_coverage_claim_supported": False,
            "promotion_or_retraining_performed": False,
        },
        "limitations": [
            "Frozen legacy boards contain 123 of the nominal 138 positioning minions; 15 pure Mechs were never sampled. Rescoring cannot restore absent boards.",
            "Only the lobby alias and corrected adapter interpretation change. All own and opponent entities and every policy permutation remain frozen.",
            "The hybrid includes original legacy simulation search. This audit tests its frozen decisions; it does not test a newly optimized corrected-search policy.",
            "Paired scenario bootstrap quantifies this synthetic sample; it does not establish ranked-player representativeness or full-game value.",
            "Independent fresh RNG and adapter semantics both differ from the original evaluation, so score changes cannot be attributed quantitatively to the adapter alone.",
            "The test has now been inspected for an audit and must not be reused for model selection or a fresh promotion claim.",
        ],
    }
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    (RUN / "evaluation.json").write_text(payload)
    (ROOT / "reports/positioning_corrected_audit.json").write_text(payload)
    print(json.dumps({"actual_combat_simulations": total, "results": primary, "claim_assessment": report["claim_assessment"], "coverage": {k: {f: v[f] for f in ("scenarios", "observed_minions", "legacy_mechanical_lobbies")} for k, v in actual_coverage["datasets"].items()}}, indent=2))


if __name__ == "__main__":
    main()
