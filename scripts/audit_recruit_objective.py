#!/usr/bin/env python3
"""Read-only objective diagnostics on completed recruit-v2 train/validation.

No held-out individual rows are analyzed, and no labels, parameters, candidate
lists, source files, or running experiments are changed. Test context is copied
only from the already published aggregate result.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from bg_ai.learning import Ranker
from archive_recruit_runs import read_rows

RUN = ROOT / "runs/20260905-recruit-v2"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(value):
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def label_board(candidate):
    # Exact LABEL_JS cache projection, independently reconstructed.
    return key([[c["card_id"], c["attack"], c["health"], c["divine_shield"], c["taunt"], c["reborn"], c["windfury"]] for c in candidate["terminal"]["board"]])


def without_entity_ids(value):
    if isinstance(value, list):
        return [without_entity_ids(v) for v in value]
    if isinstance(value, dict):
        return {k: without_entity_ids(v) for k, v in value.items() if k != "entity_id"}
    return value


def charges(row, candidate):
    elapsed = commands = 0
    for event in candidate["timing_trace"]:
        if event.get("player_id") != row["observation"]["player_id"] or event.get("turn") != row["turn"]:
            continue
        if event["event"] == "action":
            elapsed += event["after"]["charged_ms"] - event["before"]["charged_ms"]
            commands += event["after"]["actions_used"] - event["before"]["actions_used"]
    return elapsed, commands


def distribution(values):
    if not values:
        return {"count": 0}
    a = np.asarray(values, dtype=float)
    return {"count": len(a), "min": float(a.min()), "mean": float(a.mean()), "median": float(np.median(a)), "p95": float(np.quantile(a, .95)), "max": float(a.max())}


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported report value {type(value).__name__}")


def analyze(rows, labels, model):
    counts = Counter()
    actions = Counter()
    selection = Counter()
    legal_actions = Counter()
    missing_actions = Counter()
    freezes = Counter()
    margins = []
    selected_freeze_margins = []
    extra_ms, extra_commands = [], []
    freeze_extra_ms, freeze_extra_commands = [], []
    groupsizes = Counter()
    action_pair_signal = defaultdict(Counter)
    examples = []
    for row in rows:
        candidates = row["candidates"]
        n = len(candidates)
        counts["scenarios"] += 1
        counts["candidates"] += n
        legal = row["observation"]["legal_actions"]
        counts["legal_actions"] += len(legal)
        counts["scenarios_with_shortlist_omissions"] += len(legal) > n
        chosen_actions = {key(c["action"]) for c in candidates}
        legal_actions.update(a["kind"] for a in legal)
        missing_actions.update(a["kind"] for a in legal if key(a) not in chosen_actions)
        scores = np.array([labels[row["scenario_id"]]["results"][str(i)]["score"] for i in range(n)])
        values = model.predict(np.asarray([c["features"] for c in candidates]))
        selected = int(np.argmax(values))
        selection[candidates[selected]["action"]["kind"]] += 1
        sorted_values = np.sort(values)
        margin = float(sorted_values[-1] - sorted_values[-2])
        margins.append(margin)
        counts["prediction_top_exact_ties"] += margin == 0
        counts["prediction_top_ties_within_1e_12"] += margin <= 1e-12
        groups = defaultdict(list)
        costs = [charges(row, c) for c in candidates]
        for i, candidate in enumerate(candidates):
            actions[candidate["action"]["kind"]] += 1
            groups[label_board(candidate)].append(i)
        counts["unique_label_board_groups"] += len(groups)
        counts["duplicate_candidate_board_rows"] += n - len(groups)
        counts["scenarios_with_duplicate_board_groups"] += len(groups) < n
        counts["scenarios_all_identical_label_board"] += len(groups) == 1
        counts["scenarios_all_equal_scores"] += bool(np.all(scores == scores[0]))
        counts["scenarios_all_pairs_discarded_at_0_02"] += float(scores.max() - scores.min()) <= .02
        for group in groups.values():
            groupsizes[len(group)] += 1
            if len({scores[i] for i in group}) != 1:
                raise ValueError("Same cache key received different training scores")
        all_pairs = n * (n - 1) // 2
        counts["all_pairs"] += all_pairs
        for i in range(n):
            for j in range(i + 1, n):
                delta = float(scores[i] - scores[j])
                counts["exact_score_tie_pairs"] += delta == 0
                counts["pairs_discarded_at_0_02"] += abs(delta) <= .02
                counts["identical_label_board_pairs"] += label_board(candidates[i]) == label_board(candidates[j])
                if abs(delta) > .02:
                    for idx, sign in ((i, delta), (j, -delta)):
                        kind = candidates[idx]["action"]["kind"]
                        action_pair_signal[kind]["positive_pairs" if sign > 0 else "negative_pairs"] += 1
                        action_pair_signal[kind]["positive_gap_weight" if sign > 0 else "negative_gap_weight"] += abs(delta) / all_pairs
        group = groups[label_board(candidates[selected])]
        cheaper = min(group, key=lambda i: costs[i])
        ms_difference = costs[selected][0] - costs[cheaper][0]
        command_difference = costs[selected][1] - costs[cheaper][1]
        counts["selected_has_cheaper_identical_label_board"] += ms_difference > 0
        if ms_difference > 0:
            extra_ms.append(ms_difference)
            extra_commands.append(command_difference)
        freeze_indices = [i for i, c in enumerate(candidates) if c["action"]["kind"] == "freeze"]
        if not freeze_indices:
            continue
        if len(freeze_indices) != 1:
            raise ValueError("Unexpected duplicate freeze candidates")
        i = freeze_indices[0]
        baseline = row["baseline_indices"]["practical_heuristic"]
        same_board = label_board(candidates[i]) == label_board(candidates[baseline])
        freezes["available"] += 1
        freezes["same_label_board_as_practical"] += same_board
        freezes["same_score_as_practical"] += scores[i] == scores[baseline]
        freezes["same_recorded_terminal_as_practical"] += key(without_entity_ids(candidates[i]["terminal"])) == key(without_entity_ids(candidates[baseline]["terminal"]))
        freezes["score_lower_than_practical"] += scores[i] < scores[baseline]
        freezes["score_higher_than_practical"] += scores[i] > scores[baseline]
        freezes["label_at_scenario_max"] += scores[i] == scores.max()
        freezes["within_0_02_of_scenario_max"] += scores.max() - scores[i] <= .02
        freezes["root_already_frozen"] += bool(row["observation"]["private_state"]["frozen"])
        if same_board:
            freezes["same_board_costlier_than_practical"] += costs[i][0] > costs[baseline][0]
            freeze_extra_ms.append(costs[i][0] - costs[baseline][0])
            freeze_extra_commands.append(costs[i][1] - costs[baseline][1])
        if selected == i:
            freezes["selected"] += 1
            freezes["selected_same_board_as_practical"] += same_board
            freezes["selected_has_cheaper_identical_label_board"] += ms_difference > 0
            freezes["selected_label_below_scenario_max"] += scores[i] < scores.max()
            selected_freeze_margins.append(margin)
        # Fixed first five qualifying training/validation examples, never extremes.
        if selected == i and same_board and len(examples) < 5:
            examples.append({"scenario_id": row["scenario_id"], "split": row["split"], "turn": row["turn"], "remaining_ms": row["observation"]["action_budget"]["remaining_ms"], "freeze_candidate": i, "practical_candidate": baseline, "freeze_ranking_value": float(values[i]), "practical_ranking_value": float(values[baseline]), "top_rank_margin": margin, "combat_label": float(scores[i]), "freeze_cost_ms": costs[i][0], "practical_cost_ms": costs[baseline][0], "freeze_commands": candidates[i]["continuation"], "practical_commands": candidates[baseline]["continuation"], "same_combat_projection": True, "same_recorded_terminal": key(without_entity_ids(candidates[i]["terminal"])) == key(without_entity_ids(candidates[baseline]["terminal"]))})
    return {"counts": dict(counts), "legal_action_kinds": dict(legal_actions), "shortlisted_action_kinds": dict(actions), "omitted_legal_action_kinds": dict(missing_actions), "selected_action_kinds": dict(selection), "terminal_label_group_size_counts": dict(groupsizes), "freeze": dict(freezes), "top_prediction_margin": distribution(margins), "selected_freeze_top_prediction_margin": distribution(selected_freeze_margins), "selected_extra_ms_over_cheapest_same_combat_projection": distribution(extra_ms), "selected_extra_commands_over_cheapest_same_combat_projection": distribution(extra_commands), "freeze_extra_ms_over_practical_when_same_combat_projection": distribution(freeze_extra_ms), "freeze_extra_commands_over_practical_when_same_combat_projection": distribution(freeze_extra_commands), "action_pair_label_signal": {k: dict(v) for k, v in action_pair_signal.items()}, "first_qualifying_examples": examples}


def main():
    source_paths = [RUN / "selected_model.json", RUN / "scenarios.compact.jsonl.gz", RUN / "scenario_provenance.json", RUN / "scenario_archive.json", RUN / "labels.jsonl.gz", RUN / "results.json", ROOT / "reports/opening_transfer_results.json", ROOT / "scripts/train_recruit_curriculum.py", ROOT / "python/bg_ai/learning.py", ROOT / "python/bg_ai/timed_recruiting.py", ROOT / "python/bg_ai/hsbrsim_opening.py", ROOT / "python/bg_ai/recruit_features.py", ROOT / "config/turn-budget.json", Path(__file__).resolve()]
    before = {str(p.relative_to(ROOT)): sha(p) for p in source_paths}
    # The restoration iterator hashes the whole archive, but test candidate
    # records are discarded immediately and do not enter any diagnostics.
    rows = [r for r in read_rows(RUN) if r["split"] in ("train", "validation")]
    ids = {r["scenario_id"] for r in rows}
    with gzip.open(RUN / "labels.jsonl.gz", "rt") as stream:
        labels = {r["scenario_id"]: r for line in stream if (r := json.loads(line))["scenario_id"] in ids}
    if set(labels) != ids:
        raise ValueError("Missing train/validation labels")
    model = Ranker.load(RUN / "selected_model.json")
    result = {"scope": "Read-only completed recruit-v2 train/validation objective and command-cost audit; no training or test-driven policy selection", "source_sha256": before, "analyzed_splits": ["train", "validation"], "heldout_individual_rows_analyzed": False, "frozen_model": {"training_objective": model.metadata["training_objective"], "tie_tolerance": model.metadata["last_fit"]["tie_tolerance"], "feature_version": model.metadata["recruit_feature_version"], "training_scenarios": model.metadata["last_fit"]["training_scenarios"]}, "diagnostics": {split: analyze([r for r in rows if r["split"] == split], labels, model) for split in ("train", "validation")}}
    transfer = json.loads((ROOT / "reports/opening_transfer_results.json").read_text())
    result["previously_published_transfer_aggregate_only"] = {k: transfer[k] for k in ("test_states", "selected_model_action_counts", "scores", "paired_model_minus_baseline", "passes_opening_transfer_benchmark", "model_sha256")}
    result["limitations"] = ["Equal label-board projections are not equivalent complete game states: frozen shops, economy, hand, counters, future effects and RNG can differ.", "Same score can occur for different boards, especially deterministic early combats; score ties alone do not establish state equivalence.", "Predictions on training and previously used validation data describe an existing model, not an unbiased improvement estimate.", "The existing v2 shortlist omits some legal movement/insertion actions; this audit does not justify further pruning.", "The fixed continuation policy provides all decisions after the selected first command; useful first-command labels do not establish recurrent neural control."]
    result["recommended_priority"] = {"first": "Implement and verify actual combat-to-next-recruit transition, then learn a declared two-turn opening objective that observes frozen-shop, hand and next-turn income consequences. More same-turn labels cannot supply missing future value.", "second": "Generate per-decision labels with complete affordable legal actions and whole-game splits, using fixed continuation initially and separately evaluated policy iteration for autonomous whole-turn control.", "optional_bounded_ablation": "The cost-aware same-combat auxiliary loss below can test execution efficiency while the transition is built. It is not a substitute for economic value and must not classify Freeze as globally dominated."}
    result["minimal_next_experiment"] = {"status": "proposal_only_requires_separate_preregistration_and_root_coordination", "factor_changed": "Add auxiliary command-efficiency pair loss only within verified identical complete combat inputs; keep the existing combat loss unchanged", "keep_fixed": ["visible feature schema and all existing candidate actions including freeze/upgrade/end", "training/validation split", "continuation and opponent policies", "simulation budget and engine snapshot", "optimizer/architecture budget and evaluation seeds across compared arms"], "primary": "Keep the present combat-gap loss and its 0.02 resolution unchanged in this first ablation. Duplicate-group reweighting is a distinct possible later experiment, not a simultaneous change.", "secondary": "For verified same-combat-input pairs with different paid continuation milliseconds, add a separately normalized logistic preference for the cheaper continuation. Proposed preregistration: lambda=0.01, each auxiliary pair weighted by min(1, absolute time difference / root usable remaining time). Do not apply the combat tie_tolerance to this cost loss. These are proposed settings, not fitted or evaluated settings.", "deployment": "Retain all candidate commands. Use the same ordinary argmax of root-visible observation/action scores; no terminal snapshot, future shop, continuation trace, RNG, realized cost, or oracle equivalence-class lookup at inference.", "validation": "Matched control and auxiliary-loss ablation at equal compute on train only; compare validation combat score and charged milliseconds separately. Before running, freeze a combat non-inferiority margin (proposed 0.25 percentage points), efficiency acceptance rule, and checkpoint choice; use an entirely new heldout set after choices are frozen. A claim of stronger combat still requires a positive paired lower confidence bound against the practical baseline. Reduced freeze frequency alone never passes.", "scope": "Same-turn first-action command efficiency only. This explicit preference under a truncated target does not establish that freezing is inferior over later turns. Existing shortlist omissions remain disclosed; a new broader benchmark should enumerate every affordable legal command and legal placement, not remove economic actions.", "next_horizon": "First implement verified combat-to-recruitment persistence: current opening snapshots explicitly report next-turn state unavailable and combat changes unapplied. Then generate complete-current-opening two-turn trajectories carrying frozen shop, hand, economy, exact deferred income, combat damage and engine RNG. Expose active lobby tribes and exact deferred income through a versioned public feature schema. Average multiple hidden futures consistent with visible state, shared in distribution across candidate arms and independently replicated. Label a declared cumulative two-combat target with separate cost; later upgrades remain horizon-truncated. Train recurrent neural control only in a separately trained policy-iteration experiment using per-decision labels and whole-game splits."}
    after = {str(p.relative_to(ROOT)): sha(p) for p in source_paths}
    if before != after:
        raise RuntimeError("Audited input changed during read-only diagnostics")
    result["audited_sources_unchanged"] = True
    target = ROOT / "reports/recruit_objective_audit.json"
    target.write_text(json.dumps(result, indent=2, allow_nan=False, default=json_default) + "\n")
    print(json.dumps({split: {k: value[k] for k in ("counts", "selected_action_kinds", "freeze", "selected_freeze_top_prediction_margin", "selected_extra_ms_over_cheapest_same_combat_projection")} for split, value in result["diagnostics"].items()}, indent=2, default=json_default))


if __name__ == "__main__":
    main()
