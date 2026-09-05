#!/usr/bin/env python3
"""Use a trained shared-feature ranker to order an observed combat board."""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import numpy as np
from bg_ai.features import BOARD_FEATURE_NAMES, encode_board
from bg_ai.learning import Ranker
from bg_ai.provenance import file_sha256, load_snapshot
from bg_ai.position_actions import affordable_orders, drag_plan
from bg_ai.turn_budget import TimingProfile, TurnBudget


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="JSON object with board, opponent, and metadata from a supported scenario")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--candidates", type=int, default=10)
    parser.add_argument("--pure-model", action="store_true")
    parser.add_argument("--timing-profile", type=Path, default=ROOT / "config/turn-budget.json")
    parser.add_argument("--remaining-seconds", type=float,
                        help="Observed recruit time remaining; overrides input recruitTimeRemainingMs")
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.candidates < 5:
        parser.error("--candidates must be at least five")
    snapshot = load_snapshot(ROOT / "data/ruleset.json")
    row = json.loads(args.input.read_text())
    available_ms = row.get("recruitTimeRemainingMs")
    timer_source = "input.recruitTimeRemainingMs"
    if args.remaining_seconds is not None:
        if not math.isfinite(args.remaining_seconds * 1000) or args.remaining_seconds < 0:
            parser.error("--remaining-seconds must be finite and nonnegative")
        available_ms = math.floor(args.remaining_seconds * 1000)
        timer_source = "--remaining-seconds"
    if type(available_ms) is not int or available_ms < 0:
        parser.error("Supply actual recruitTimeRemainingMs or --remaining-seconds; turn number alone is insufficient")
    profile = TimingProfile.load(args.timing_profile)
    budget = TurnBudget(profile, available_ms)
    initial_budget = budget.snapshot()
    max_drags = min(budget.remaining_actions, budget.remaining_ms // profile.action_ms("move"))
    board, opponent = row["board"], row["opponent"]
    if not 1 <= len(board) <= 7:
        parser.error("Require one to seven friendly minions")
    metadata = row.get("metadata", {})
    if metadata.get("source") != "synthetic_current_pool_combat":
        parser.error("This initial CLI requires a saved supported combat scenario with explicit scope metadata")
    if metadata.get("rulesetSha256") != file_sha256(ROOT / "data/ruleset.json"):
        parser.error("Scenario belongs to a different ruleset; regenerate it before recommending an order")
    active = set(snapshot["active"]["minion_ids"])
    if any(c.get("cardId") not in active for c in board + opponent):
        parser.error("Board contains a card outside the verified active snapshot")
    permutations = affordable_orders(itertools.permutations(range(len(board))), max_drags)
    rng = random.Random(args.seed)
    identity = tuple(range(len(board)))
    rng.shuffle(permutations)
    heuristics = {
        "attack": tuple(sorted(identity, key=lambda i: -board[i]["attack"])),
        "health": tuple(sorted(identity, key=lambda i: -board[i]["health"])),
        "taunt_last": tuple(sorted(identity, key=lambda i: (bool(board[i].get("taunt", False)), -board[i]["attack"]))),
    }
    orders = []
    for order in affordable_orders([identity, *heuristics.values(), *permutations], max_drags):
        if order not in orders:
            orders.append(order)
        if len(orders) >= args.candidates:
            break
    features = np.stack([encode_board([board[i] for i in order], opponent) for order in orders])
    model = Ranker.load(args.checkpoint, BOARD_FEATURE_NAMES)
    scores = model.predict(features)
    ranking = np.argsort(-scores, kind="stable").tolist()
    chosen = ranking[0]
    search_result = None
    # An unreachable heuristic maps to staying put inside the search interface;
    # it never introduces an order that exceeds the actual drag budget.
    heuristic_indices = {name: orders.index(order) if order in orders else 0
                         for name, order in heuristics.items()}
    if not args.pure_model and len(orders) > 1:
        scenario = {"scenario_id": "recommendation", "split": "test", "opponent": opponent,
                    "metadata": metadata,
                    "candidates": [{"board": [board[i] for i in order]} for order in orders]}
        proposal = {"scenario_id": "recommendation", "split": "test", "ranked_candidates": ranking,
                    "selections": {"model": chosen, "random": 0,
                                   **heuristic_indices},
                    "provenance": {"cards_sha256": metadata["cardsSha256"],
                                   "ruleset_sha256": metadata["rulesetSha256"],
                                   "label_combat_seed": metadata["combatSeed"],
                                   "checkpoint_sha256": file_sha256(args.checkpoint)}}
        js = """import fs from 'node:fs';
import {FirestoneCombat,sha256} from './simulator/firestone.mjs';
import {refineScenario} from './scripts/refine_positions.mjs';
const bytes=fs.readFileSync(process.argv[1]); const x=JSON.parse(bytes);
console.log(JSON.stringify(refineScenario(FirestoneCombat.fromFiles(),x.scenario,x.proposal,0,sha256(bytes))));"""
        with tempfile.TemporaryDirectory(prefix="bg-recommend-") as temporary:
            path = Path(temporary) / "input.json"
            path.write_text(json.dumps({"scenario": scenario, "proposal": proposal}))
            search_result = json.loads(subprocess.check_output(
                ["node", "--input-type=module", "-e", js, str(path)], cwd=ROOT, text=True))
        chosen = search_result["selections"]["model"]
    order = orders[chosen]
    actions = drag_plan(order)
    for action in actions:
        action["estimated_ms"] = budget.charge("move")
        if "entityId" in board[action["original_index"]]:
            action["entity_id"] = board[action["original_index"]]["entityId"]
    result = {
        "scope": "Experimental positioning only, conditioned on the supplied opponent and supported combat context",
        "ruleset_id": snapshot["ruleset_id"], "candidates_ranked": len(orders),
        "indices": list(order), "order": [board[i] for i in order],
        "policy": ("keep_current_order" if len(orders) == 1 else
                   "neural_model_only" if args.pure_model else "neural_proposals_plus_simulation_search"),
        "actions": actions,
        "action_indexing": "Zero-based current slots; to_index is the insertion slot after removing the minion.",
        "timing": {"mode": "offline_advisory", "timer_source": timer_source,
                   "before_actions": initial_budget, "after_actions": budget.snapshot(),
                   "max_feasible_drags": max_drags,
                   "inference_elapsed_included": False,
                   "unavailable_heuristics": [name for name, value in heuristics.items() if value not in orders],
                   "note": "Drags fit the supplied timer. Inference/search elapsed time is not included; recheck the live timer before execution."},
        "note": "Search estimates select the order; they are not an independent estimate of its playing strength.",
    }
    if search_result:
        result["search_outcomes"] = search_result["search_outcomes"]
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
