#!/usr/bin/env python3
"""Frozen restricted-pool model transfer to current complete opening pools.

This is holdout-only evaluation. No optimizer is called. Every candidate's
continuation uses the same visible-state heuristic, explicit timed commands,
and complete end-of-turn processing before exporting actual board and hand.
"""
from __future__ import annotations

import argparse
from collections import Counter
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
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import numpy as np
from bg_ai.learning import Ranker
from bg_ai.recruit_features import (RECRUIT_V2_FEATURE_NAMES, encode_legal_actions_v2,
                                   action_card, visible_cards)
from bg_ai.turn_budget import TimingProfile
from archive_recruit_runs import read_rows, source_archive_hash

CHECKPOINT_SHA = "b0ae096c38f3c2c04a4c8f70157b305b54f701adcf5bc15d3eccb96adacf787a"
FEATURE_VERSION = "recruit-visible-action-v2-choice-zone"
SCOPE = "frozen recruit-v2 first-action transfer to current complete Tier-1 offering pools, conditional audited play frontier"
SCENARIO_SEED = 202609057
EVALUATION_SEED = 982453021
BOOTSTRAP_SEED = 961748941
TRIBE_ORDER = ("BEAST", "DEMON", "DRAGON", "ELEMENTAL", "MECH",
               "MURLOC", "NAGA", "PIRATE", "QUILBOAR", "UNDEAD")
TRIBE_SCHEDULE = tuple(tuple(TRIBE_ORDER[(i+j) % 10] for j in range(5)) for i in range(10))

LABEL_JS = r'''
import {createInterface} from 'node:readline';
import {OpeningFirestoneCombat} from './simulator/opening-firestone.mjs';
const engine=OpeningFirestoneCombat.fromFiles();
const options=JSON.parse(process.env.BG_OPENING_EVALUATION_OPTIONS);
let processed=0;
for await(const line of createInterface({input:process.stdin,crlfDelay:Infinity})){
 if(!line.trim())continue;
 const row=JSON.parse(line);
 const seed=(options.seed+Math.imul(row.index+1,2654435761))>>>0;
 const results={}; const cache=new Map();
 try {
  for(const [name,index] of Object.entries(row.selections)){
   const candidate=row.candidates[index];
   // The actual hand and enchantments are part of combat state. Never collapse
   // equal-looking boards with different hand, Golden or persistent context.
   const key=JSON.stringify({player:candidate.player,opponent:row.opponent});
   if(!cache.has(key))cache.set(key,engine.evaluate({player:candidate.player,
       opponent:row.opponent,trials:options.trials,seed}));
   results[name]={candidateIndex:index,...cache.get(key)};
  }
  process.stdout.write(JSON.stringify({scenario_id:row.scenario_id,index:row.index,results,
    metadata:{seed,trials:options.trials,uniqueCombats:cache.size*options.trials}})+'\n');
 }catch(error){process.stdout.write(JSON.stringify({scenario_id:row.scenario_id,index:row.index,
     error:String(error)})+'\n');}
 if(++processed%100===0)process.stderr.write(JSON.stringify({stage:'opening_final_combat',processed})+'\n');
}
'''


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def strength(card):
    if not card:
        return 0.0
    a, h = card.get("attack", 0), card.get("health", 0)
    return float(a + h + a*.35 + (a+h*.5)*bool(card.get("divine_shield"))
                 + (a+1)*.7*bool(card.get("reborn")) + a*.25*bool(card.get("windfury")))


def heuristic_value(obs, action):
    """Frozen practical_heuristic_v1 logic, copied to prevent later policy drift."""
    cards, card = visible_cards(obs), action_card(obs, action)
    if action.kind == "choose":
        target = action_card(obs, action, target=True)
        board_ids = {c["entity_id"] for c in cards["board"]}
        return 1000 + (100 if target and target["entity_id"] in board_ids else 0) + strength(target)
    if action.kind == "play":
        return 900 + strength(card) - .001*(action.position or 0)
    if action.kind == "cast_spell":
        return 800 + (20 if cards["board"] else 0)
    if action.kind == "activate":
        return 700
    if action.kind == "buy":
        if str(card.get("card_type", "")).lower() == "minion":
            return 500+strength(card) if obs.action_budget["remaining_ms"] >= 2750 else -50
        cost = card.get("cost", 0)
        if cards["board"] and cost and cost <= obs.private_state["gold"]:
            if "Gain 1 Gold" not in card.get("text", ""):
                return 350
        return -20
    if action.kind == "sell":
        buys = [c for c in cards["shop"] if str(c.get("card_type", "")).lower() == "minion"]
        if obs.private_state["gold"] == 2 and len(cards["board"]) >= 2 and buys:
            gain = max(map(strength, buys)) - strength(card)
            if gain > 2:
                return 250+gain
        return -30-strength(card)
    if action.kind == "refresh":
        return 100 if obs.private_state["gold"] >= 4 else -40
    if action.kind == "end_turn":
        return 0
    return -10


def heuristic_action(obs):
    return max(obs.legal_actions, key=lambda action: heuristic_value(obs, action))


def raw_stats_action(obs):
    def score(action):
        card = action_card(obs, action)
        if action.kind == "choose":
            target = action_card(obs, action, target=True)
            return 1000 + (target.get("attack", 0)+target.get("health", 0) if target else 0)
        if action.kind == "play": return 900
        if action.kind == "cast_spell": return 800
        if action.kind == "buy" and str(card.get("card_type", "")).lower() == "minion":
            return 500+card.get("attack", 0)+card.get("health", 0)
        return 0 if action.kind == "end_turn" else -1
    return max(obs.legal_actions, key=score)


def candidate_actions(obs, limit=16):
    """Same fixed shortlist design as the source first-action curriculum."""
    selected = []
    def add(action):
        if action not in selected:
            selected.append(action)
    add(heuristic_action(obs))
    add(raw_stats_action(obs))
    for action in obs.legal_actions:
        if action.kind in ("end_turn", "refresh", "upgrade", "freeze", "activate", "hero_power", "choose"):
            add(action)
    for action in sorted(obs.legal_actions, key=lambda a: heuristic_value(obs, a), reverse=True):
        if action.kind == "move" or (action.kind == "play" and action.position != 0 and action.target_id is None):
            continue
        add(action)
    return selected[:limit]


def finish_all(timed, first=None):
    """Finish all eight players; all explicit commands use the timing wrapper."""
    trace_start = len(timed.trace)
    actions = []
    obs = timed._visible
    if first is not None:
        actions.append({"player_id": obs.player_id, "action": asdict(first)})
        obs = timed.step(first)
    for _ in range(1600):
        if obs is None:
            return actions, timed.trace[trace_start:]
        action = heuristic_action(obs)
        actions.append({"player_id": obs.player_id, "action": asdict(action)})
        obs = timed.step(action)
    raise RuntimeError("Opening continuation exceeded the finite command guard")


def state_fingerprint(obs, actions):
    vectors = encode_legal_actions_v2(obs, actions)
    # The known training fingerprint uses sorted feature rows. Match that exact
    # representation to exclude any seen source-dataset decision state.
    body = json.dumps(sorted(vectors.tolist()), separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest(), vectors


def source_state_ids(directory):
    seen, train_cards = set(), set()
    for row in read_rows(directory):
        # Source v2 stores its v2 matrices; schema loading is checked separately.
        if any(len(c["features"]) != len(RECRUIT_V2_FEATURE_NAMES) for c in row["candidates"]):
            raise ValueError("Source scenario matrices differ from frozen v2 schema")
        body = json.dumps(sorted(c["features"] for c in row["candidates"]), separators=(",", ":"))
        seen.add(hashlib.sha256(body.encode()).hexdigest())
        if row["split"] == "train":
            private = row["observation"]["private_state"]
            train_cards.update(card["card_id"] for zone in ("board", "hand", "shop")
                               for card in private.get(zone, []))
    return seen, train_cards


def build_scenario(database, ruleset, profile, index, seed):
    from bg_ai.hsbrsim_opening import OpeningFixtureSpec, OpeningRecruitEngine, TimedOpeningEngine
    rng = random.Random(seed+index*15485863)
    tribes = TRIBE_SCHEDULE[index % len(TRIBE_SCHEDULE)]
    available_ms = rng.choice((10000, 14000, 20000, 30000, 45000, 60000))
    spec = OpeningFixtureSpec(valid_tribes=tribes)
    engine = OpeningRecruitEngine(database.db, ruleset, timer_ms=lambda player, turn: available_ms if player == 0 else 60000,
                                  provenance=database.provenance, fixture=spec)
    timed = TimedOpeningEngine.for_fixture(engine, profile, ruleset)
    obs = timed.reset(seed=seed+index*1009)
    for _ in range(rng.randrange(4)):
        if obs is None or obs.player_id != 0:
            raise RuntimeError("Opening ended before requested decision")
        action = heuristic_action(obs)
        if action.kind == "end_turn": break
        branch = timed.fork()
        after = branch.step(action)
        if after is None or after.player_id != 0: break
        timed, obs = branch, after
    actions = candidate_actions(obs)
    if len(actions) < 2:
        raise ValueError("Fewer than two candidate opening decisions")
    fingerprint, vectors = state_fingerprint(obs, actions)
    candidates, opponent = [], None
    for i, action in enumerate(actions):
        branch = timed.fork()
        sequence, timing = finish_all(branch, first=action)
        player = branch.engine.combat_snapshot(0)
        if i == 0:
            # Fixed complete opponent from practical baseline branch. Neither
            # opponent hand nor board is ever an input to the recruit policy.
            opponent = branch.engine.combat_snapshot(1)
        candidates.append({"action": asdict(action), "features": vectors[i].tolist(),
                           "player": player, "continuation": sequence, "timing_trace": timing})
    coverage = dict(engine.coverage())
    coverage.pop("provenance", None)
    return {"scenario_id": f"opening-transfer-{seed}-{index}", "index": index,
            "split": "test", "scope": SCOPE, "valid_tribes": list(tribes),
            "observation": asdict(obs), "visible_state_fingerprint": fingerprint,
            "candidates": candidates, "opponent": opponent,
            "baseline_indices": {"practical_heuristic": 0, "raw_stats": actions.index(raw_stats_action(obs))},
            "metadata": {"initial_available_ms": available_ms, "timing_profile_sha256": profile.fingerprint,
                         "opening_engine_coverage": coverage,
                         "feature_version": FEATURE_VERSION, "engine_provenance_sha256": hashlib.sha256(
                             json.dumps(database.provenance, sort_keys=True).encode()).hexdigest()}}


def paired_interval(values, seed=BOOTSTRAP_SEED, samples=10000):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Paired interval requires finite nonempty scenario differences")
    rng = np.random.default_rng(seed)
    boot = np.empty(samples)
    for start in range(0, samples, 250):
        count = min(250, samples-start)
        boot[start:start+count] = values[rng.integers(0, len(values), (count, len(values)))].mean(axis=1)
    return {"mean_delta": float(values.mean()), "ci95": np.quantile(boot, [.025, .975]).tolist()}


def validate_fresh(rows, fresh, trials):
    if len(fresh) != len(rows) or any("error" in row for row in fresh):
        raise RuntimeError("Every frozen opening test scenario must finish final evaluation")
    for source, result in zip(rows, fresh):
        if result["scenario_id"] != source["scenario_id"] or result["index"] != source["index"]:
            raise ValueError("Final opening result does not match the frozen scenario order")
        expected_seed = (EVALUATION_SEED + (source["index"]+1)*2654435761) & 0xffffffff
        if result["metadata"]["seed"] != expected_seed or result["metadata"]["trials"] != trials:
            raise ValueError("Final opening RNG/trial metadata mismatch")
        for name, index in source["selections"].items():
            measured = result["results"][name]
            if measured["candidateIndex"] != index or not 0 <= measured["score"] <= 1:
                raise ValueError("Final result differs from frozen selection or contains invalid score")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-root", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/20260905-opening-transfer-v1")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/20260905-recruit-v2/selected_model.json")
    parser.add_argument("--source-run", type=Path, default=ROOT / "runs/20260905-recruit-v2")
    parser.add_argument("--target-states", type=int, default=600)
    parser.add_argument("--max-attempts", type=int, default=2400)
    parser.add_argument("--trials", type=int, default=1024)
    parser.add_argument("--allow-historical", action="store_true")
    args = parser.parse_args(argv)
    if args.target_states < 1 or args.max_attempts < args.target_states or args.trials < 1:
        parser.error("Require positive target/trials and sufficient maximum attempts")
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "preregistration.json").exists():
        raise ValueError("Use a fresh run directory; this benchmark must not overwrite a registered evaluation")
    if digest(args.checkpoint) != CHECKPOINT_SHA:
        raise ValueError("Checkpoint differs from the frozen recruit-v2 selected model")
    model = Ranker.load(args.checkpoint, RECRUIT_V2_FEATURE_NAMES)
    if model.metadata.get("recruit_feature_version") != FEATURE_VERSION:
        raise ValueError("Frozen model feature-version mismatch")
    started = time.time()
    if not args.allow_historical:
        subprocess.run([sys.executable, str(ROOT / "scripts/check_live_ruleset.py"), "--report", str(args.out / "live_preflight.json")], cwd=ROOT, check=True)
    ruleset = json.loads((ROOT / "data/ruleset.json").read_text())
    profile = TimingProfile.load(ROOT / "config/turn-budget.json")
    seen, train_cards = source_state_ids(args.source_run)
    from bg_ai.hsbrsim_opening import TIER1_MINIONS, TIER1_SPELLS, UNVERIFIED_SELF_PLAY
    source_files = ["scripts/evaluate_opening_transfer.py", "python/bg_ai/recruit_features.py",
        "python/bg_ai/features.py", "python/bg_ai/learning.py", "python/bg_ai/hsbrsim_opening.py",
        "python/bg_ai/hsbrsim_adapter.py", "python/bg_ai/hsbrsim_data.py", "python/bg_ai/recruiting.py",
        "python/bg_ai/timed_recruiting.py", "python/bg_ai/turn_budget.py", "config/turn-budget.json",
        "simulator/opening-firestone.mjs", "simulator/firestone.mjs"]
    source_hashes = {}
    for relative in source_files:
        source = ROOT / relative
        if not source.exists():
            raise ValueError(f"Missing frozen evaluation dependency: {relative}")
        snapshot = args.out / "source_snapshots" / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(source.read_bytes())
        source_hashes[relative] = digest(snapshot)
    preregistration = {"scope": SCOPE, "full_game_ready": False, "training_performed": False,
        "future_rotation_transfer_measured": False, "scenario_seed": SCENARIO_SEED,
        "evaluation_seed": EVALUATION_SEED, "target_unique_test_states": args.target_states,
        "maximum_attempts": args.max_attempts, "combat_trials_per_selected_candidate": args.trials,
        "model_sha256": CHECKPOINT_SHA, "feature_version": FEATURE_VERSION,
        "feature_names": list(RECRUIT_V2_FEATURE_NAMES), "source_archive_sha256": source_archive_hash(args.source_run),
        "source_training_observed_card_ids": sorted(train_cards), "source_state_exclusions": len(seen),
        "tribe_schedule": [list(x) for x in TRIBE_SCHEDULE],
        "current_offered_minion_ids": sorted(TIER1_MINIONS), "current_offered_spell_ids": sorted(TIER1_SPELLS),
        "blocked_play_minion_ids": sorted(UNVERIFIED_SELF_PLAY),
        "tested_playable_minion_count": len(TIER1_MINIONS-UNVERIFIED_SELF_PLAY),
        "policy": "one model-selected first action, then frozen practical_heuristic_v1",
        "shortlist": "up to16 legal candidates, canonical board insertion0 plus explicit magnetic plays, both baselines included",
        "baselines": "practical_heuristic_v1 and raw-stat first action, identical practical continuation",
        "opponent": "player1 complete board AND actual hand from practical baseline branch, fixed across candidates",
        "inclusion": "Fresh unique visible states, conditional on every shortlisted recruit branch being supported; selection frozen before final combat",
        "decision_rule": "Report paired mean and95CI; positive lower95CI against both baselines supports only this opening curriculum",
        "script_sha256": digest(Path(__file__)), "timing_profile_sha256": profile.fingerprint,
        "source_files_sha256": source_hashes,
        "bootstrap": {"unit": "scenario", "samples": 10000, "seed": BOOTSTRAP_SEED},
        "limitations": ["Current unseen cards, not a future rotation; no new training or transfer-speed measurement",
            "Synthetic available timers and a fixed continuation; no full-game placements or MMR",
            f"All {len(TIER1_MINIONS)} current Tier1 minions are offered; {len(TIER1_MINIONS-UNVERIFIED_SELF_PLAY)} have tested play transitions; blocked IDs are recorded explicitly",
            "Unverified opening-engine interactions remain outside statistical confidence intervals"]}
    write_json(args.out / "preregistration.json", preregistration)
    from bg_ai.hsbrsim_opening import build_opening_database
    databases, rows, failures, provenance = {}, [], [], {}
    path = args.out / "scenarios.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf8") as handle:
        for index in range(args.max_attempts):
            tribes = TRIBE_SCHEDULE[index % len(TRIBE_SCHEDULE)]
            if tribes not in databases:
                databases[tribes] = build_opening_database(args.engine_root, valid_tribes=tribes)
                provenance[",".join(tribes)] = databases[tribes].provenance
            try:
                row = build_scenario(databases[tribes], ruleset, profile, index, SCENARIO_SEED)
                fingerprint = row["visible_state_fingerprint"]
                if fingerprint in seen:
                    raise ValueError("Visible decision state duplicates source or earlier holdout state")
                seen.add(fingerprint)
                private = row["observation"]["private_state"]
                observed = {c["card_id"] for zone in ("board", "hand", "shop") for c in private.get(zone, [])}
                row["unseen_training_card_ids"] = sorted(observed-train_cards)
                rows.append(row)
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            except Exception as error:
                failures.append({"index": index, "split": "test", "stage": "recruit_transition_or_inclusion",
                                 "error": f"{type(error).__name__}: {error}"})
            if (index+1) % 25 == 0:
                print(json.dumps({"stage": "opening_generation", "attempted": index+1, "accepted": len(rows),
                                  "rejected": len(failures), "seconds": round(time.time()-started, 1)}), flush=True)
            if len(rows) >= args.target_states: break
    write_json(args.out / "generation_failures.json", failures)
    write_json(args.out / "scenario_provenance.json", provenance)
    if len(rows) < args.target_states:
        raise RuntimeError(f"Only {len(rows)} supported unique states; retain failed run rather than change scope")
    # Selection is computed only after inclusion is fixed and before any combat
    # result is generated. It receives saved visible features, never snapshots.
    frozen_path = args.out / "test_choices.jsonl"
    with frozen_path.open("w") as handle:
        for row in rows:
            values = model.predict(np.asarray([c["features"] for c in row["candidates"]]))
            row["selections"] = dict(row["baseline_indices"], model=int(np.argmax(values)))
            handle.write(json.dumps({"scenario_id": row["scenario_id"], "index": row["index"],
                "selections": row["selections"], "model_sha256": CHECKPOINT_SHA}) + "\n")
    frozen_hash = digest(frozen_path)
    # JavaScript modules load in the subprocess below, later than Python's
    # imports. Refuse a mixed-version run if a dependency changed meanwhile.
    if any(digest(ROOT / relative) != expected for relative, expected in source_hashes.items()):
        raise ValueError("Evaluation source changed after registration; retain this run and restart cleanly")
    output = args.out / "fresh_evaluation.jsonl"
    with output.open("w") as handle:
        proc = subprocess.Popen(["node", "--input-type=module", "-e", LABEL_JS], cwd=ROOT,
            env=dict(os.environ, BG_OPENING_EVALUATION_OPTIONS=json.dumps({"seed": EVALUATION_SEED, "trials": args.trials})),
            stdin=subprocess.PIPE, stdout=handle, text=True)
        try:
            for row in rows:
                proc.stdin.write(json.dumps(row, separators=(",", ":")) + "\n")
        finally:
            proc.stdin.close()
        if proc.wait(): raise RuntimeError("Opening Firestone evaluation process failed")
    fresh = [json.loads(line) for line in output.read_text().splitlines()]
    validate_fresh(rows, fresh, args.trials)
    if digest(frozen_path) != frozen_hash or digest(args.checkpoint) != CHECKPOINT_SHA:
        raise ValueError("Frozen choices or model changed during final evaluation")
    scores = {name: float(np.mean([r["results"][name]["score"] for r in fresh]))
              for name in ("model", "practical_heuristic", "raw_stats")}
    differences = {name: paired_interval([r["results"]["model"]["score"]-r["results"][name]["score"] for r in fresh])
                   for name in ("practical_heuristic", "raw_stats")}
    observed_counts, selected_counts, command_counts, played_counts = Counter(), Counter(), Counter(), Counter()
    timer_violations, max_actions, hand_states = 0, 0, 0
    for row in rows:
        private = row["observation"]["private_state"]
        observed_counts.update(c["card_id"] for zone in ("board", "hand", "shop") for c in private.get(zone, []))
        hand_states += bool(private.get("hand"))
        selected_counts[row["candidates"][row["selections"]["model"]]["action"]["kind"]] += 1
        for candidate in row["candidates"]:
            command_counts.update(a["action"]["kind"] for a in candidate["continuation"])
            # Final snapshots are sufficient to audit which current minions
            # actually reach combat, separately from mere shop visibility.
            played_counts.update(c["card_id"] for c in candidate["player"]["board"])
            for entry in candidate["timing_trace"]:
                budget = entry.get("after", entry)
                if "actions_used" in budget:
                    max_actions = max(max_actions, budget["actions_used"])
                    timer_violations += int(budget["actions_used"] > budget["action_limit"] or budget["remaining_ms"] < 0)
    if timer_violations: raise RuntimeError("Opening benchmark contains timing-budget violations")
    if not args.allow_historical:
        subprocess.run([sys.executable, str(ROOT / "scripts/check_live_ruleset.py"), "--report", str(args.out / "live_postflight.json")], cwd=ROOT, check=True)
    result = {"scope": SCOPE, "full_game_ready": False, "training_performed": False,
        "future_rotation_transfer_measured": False, "current_snapshot_verified": not args.allow_historical,
        "policy": preregistration["policy"], "baselines": preregistration["baselines"],
        "test_states": len(rows), "attempted_states": len(rows)+len(failures), "rejected_states": len(failures),
        "failure_reasons": dict(Counter(f["error"] for f in failures)), "scores": scores,
        "paired_model_minus_baseline": differences,
        "passes_opening_transfer_benchmark": all(x["ci95"][0] > 0 for x in differences.values()),
        "score_definition": "P(win)+0.5P(tie), first combat only",
        "final_evaluation_combats": sum(r["metadata"]["uniqueCombats"] for r in fresh),
        "observed_card_counts": dict(observed_counts), "source_training_observed_card_ids": sorted(train_cards),
        "terminal_player_board_card_counts": dict(played_counts),
        "current_offered_minion_ids": sorted(TIER1_MINIONS), "current_offered_spell_ids": sorted(TIER1_SPELLS),
        "blocked_play_minion_ids": sorted(UNVERIFIED_SELF_PLAY),
        "tested_playable_minion_count": len(TIER1_MINIONS-UNVERIFIED_SELF_PLAY),
        "observed_unseen_training_ids": sorted(set(observed_counts)-train_cards),
        "states_with_unseen_training_cards": sum(bool(row["unseen_training_card_ids"]) for row in rows),
        "states_with_nonempty_owned_hand": hand_states, "selected_model_action_counts": dict(selected_counts),
        "executed_command_counts": dict(command_counts), "maximum_actions_used": max_actions,
        "timing_budget_violations": timer_violations, "model_sha256": CHECKPOINT_SHA,
        "feature_version": FEATURE_VERSION, "test_choices_sha256": frozen_hash,
        "source_files_sha256": source_hashes,
        "scenario_archive_sha256": digest(path), "elapsed_seconds": time.time()-started,
        "inclusion_scope": preregistration["inclusion"], "limitations": preregistration["limitations"]}
    inspection_order = sorted(range(len(rows)), key=lambda i: fresh[i]["results"]["model"]["score"]
                              - fresh[i]["results"]["practical_heuristic"]["score"])
    inspections = []
    for i in dict.fromkeys(inspection_order[:3] + inspection_order[-3:]):
        row, measured = rows[i], fresh[i]
        inspections.append({"scenario_id": row["scenario_id"], "observation": row["observation"],
            "unseen_training_card_ids": row["unseen_training_card_ids"],
            "model": row["candidates"][row["selections"]["model"]],
            "practical_heuristic": row["candidates"][row["selections"]["practical_heuristic"]],
            "opponent": row["opponent"], "measured_final_combat": measured,
            "selection_note": "Post-evaluation illustrative best/worst differences; never used to tune this frozen policy"})
    write_json(args.out / "inspection_examples.json", inspections)
    write_json(args.out / "results.json", result)
    write_json(ROOT / "reports/opening_transfer_results.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
