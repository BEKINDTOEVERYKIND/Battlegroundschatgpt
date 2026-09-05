#!/usr/bin/env python3
"""Train a finite-budget first recruit-action ranker on real engine rollouts.

This is explicitly a restricted early-turn curriculum. The learned action is
followed by a fixed visible-state heuristic until the current recruit turn ends.
Future refreshed shop cards and terminal boards are never policy inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import gzip
import hashlib
import json
import math
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
from bg_ai.learning import Dataset, Ranker, Scenario, evaluate
from bg_ai.recruit_features import (RECRUIT_FEATURE_NAMES, RECRUIT_FEATURE_VERSION, RECRUIT_V2_FEATURE_NAMES,
    action_card, encode_legal_actions, observation_record, visible_cards)
from bg_ai.recruiting import RecruitAction, RecruitObservation
from bg_ai.turn_budget import TimingProfile
from archive_recruit_runs import read_rows, source_archive_hash, lines as artifact_lines

SCOPE = "restricted-current-pool first recruit action, fixed heuristic continuation, turns 1-2"
TRIBES = ["ELEMENTAL", "NAGA", "PIRATE", "QUILBOAR", "UNDEAD"]

LABEL_JS = r'''
import fs from 'node:fs';
import {createInterface} from 'node:readline';
import {FirestoneCombat,makeEntity} from './simulator/firestone.mjs';
const engine=FirestoneCombat.fromFiles();
const options=JSON.parse(process.env.BG_RECRUIT_LABEL_OPTIONS);
const convert=(board,offset)=>board.map((c,i)=>{
 const base=makeEntity(engine.reference.get(c.card_id),offset+i);
 return {...base,attack:c.attack,health:c.health,golden:c.golden,
   divineShield:c.divine_shield,taunt:c.taunt,reborn:c.reborn,windfury:c.windfury};
});
let processed=0, combats=0;
for await(const line of createInterface({input:process.stdin,crlfDelay:Infinity})){
 if(!line.trim())continue;
 const row=JSON.parse(line);
 const seed=(options.seed+Math.imul(row.index+1,2654435761))>>>0;
 const candidates=options.selectedOnly?Object.entries(row.selections).map(([name,index])=>({name,index})):row.candidates.map((c,index)=>({name:String(index),index}));
 const seen=new Map(); const results={};
 try{
  for(const {name,index} of candidates){
   const board=row.candidates[index].terminal.board;
   const key=JSON.stringify(board.map(c=>[c.card_id,c.attack,c.health,c.divine_shield,c.taunt,c.reborn,c.windfury]));
   if(!seen.has(key)){
    const result=engine.evaluate({board:convert(board,10),opponent:convert(row.opponent.board,100),
     validTribes:row.validTribes,heroId:row.hero_id,turn:row.turn,tavernTier:row.candidates[index].terminal.tavern_tier,
     opponentTavernTier:row.opponent.tavern_tier,trials:options.trials,seed});
    seen.set(key,result);combats+=options.trials;
   }
   results[name]={candidateIndex:index,...seen.get(key)};
  }
  process.stdout.write(JSON.stringify({scenario_id:row.scenario_id,index:row.index,split:row.split,results,
   metadata:{trials:options.trials,combatSeed:seed,uniqueCombats:seen.size*options.trials}})+'\n');
 }catch(error){process.stdout.write(JSON.stringify({scenario_id:row.scenario_id,error:String(error)})+'\n');}
 if(++processed%100===0)process.stderr.write(JSON.stringify({stage:'combat_labels',processed,combats})+'\n');
}
'''


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strength(card):
    if not card:
        return 0.0
    a, h = card.get("attack", 0), card.get("health", 0)
    return float(a + h + a * 0.35 + (a + h * .5) * bool(card.get("divine_shield")) +
                 (a + 1) * .7 * bool(card.get("reborn")) + a * .25 * bool(card.get("windfury")))


def heuristic_value(obs, action):
    """Frozen practical stat heuristic. No candidate rollout or hidden RNG use."""
    cards = visible_cards(obs)
    card = action_card(obs, action)
    if action.kind == "choose":
        target = action_card(obs, action, target=True)
        board_ids = {c["entity_id"] for c in cards["board"]}
        return 1000 + (100 if target and target["entity_id"] in board_ids else 0) + strength(target)
    if action.kind == "play":
        return 900 + strength(card) - .001 * (action.position or 0)
    if action.kind == "cast_spell":
        # Buffs on owned minions have immediate combat value; coin gets explicit
        # cast command too, so the timing wrapper accounts for economic cycles.
        return 800 + (20 if cards["board"] else 0)
    if action.kind == "activate":
        return 700
    if action.kind == "buy":
        minion = str(card.get("card_type", "")).lower() == "minion"
        cost = card.get("cost", 0)
        if minion:
            return 500 + strength(card) if obs.action_budget["remaining_ms"] >= 2750 else -50
        if cards["board"] and cost and cost <= obs.private_state["gold"]:
            # Prefer buffs to neutral coin cycles when no further minion fits.
            if "Gain 1 Gold" not in card.get("text", ""):
                return 350
        return -20
    if action.kind == "sell":
        buys = [c for c in cards["shop"] if str(c.get("card_type", "")).lower() == "minion"]
        if obs.private_state["gold"] == 2 and len(cards["board"]) >= 2 and buys:
            gain = max(map(strength, buys)) - strength(card)
            if gain > 2:
                return 250 + gain
        return -30 - strength(card)
    if action.kind == "refresh":
        return 100 if obs.private_state["gold"] >= 4 else -40
    if action.kind == "end_turn":
        return 0
    # Freeze and upgrade have longer-horizon value outside this objective.
    return -10


def heuristic_action(obs):
    return max(obs.legal_actions, key=lambda a: heuristic_value(obs, a))


def cheap_stat_action(obs):
    """Second fixed baseline: prioritize playing, then highest raw-stat purchase."""
    def score(a):
        c = action_card(obs, a)
        if a.kind == "choose":
            t = action_card(obs, a, target=True)
            return 1000 + (t.get("attack", 0) + t.get("health", 0) if t else 0)
        if a.kind == "play": return 900
        if a.kind == "cast_spell": return 800
        if a.kind == "buy" and str(c.get("card_type", "")).lower() == "minion":
            return 500 + c.get("attack", 0) + c.get("health", 0)
        if a.kind == "end_turn": return 0
        return -1
    return max(obs.legal_actions, key=score)


def candidates_for(obs, limit=16):
    """Fixed shortlist from the live legal mask, including both baselines."""
    selected = []
    def add(action):
        if action not in selected:
            selected.append(action)
    add(heuristic_action(obs))
    add(cheap_stat_action(obs))
    for action in obs.legal_actions:
        if action.kind in ("end_turn", "refresh", "upgrade", "freeze", "activate", "hero_power", "choose"):
            add(action)
    for action in sorted(obs.legal_actions, key=lambda a: heuristic_value(obs, a), reverse=True):
        # One canonical insertion per card keeps this a recruit curriculum,
        # while move actions remain explicitly charged in baseline transitions.
        if action.kind == "play" and action.position != 0:
            continue
        if action.kind == "move":
            continue
        add(action)
    return selected[:limit]


def finish_player(timed, player_id, turn, first=None):
    obs = timed._visible
    actions = []
    if first is not None:
        actions.append(asdict(first))
        obs = timed.step(first)
    for _ in range(200):
        if obs is None or (obs.player_id, obs.turn) != (player_id, turn):
            return timed.engine.player_board(player_id), actions
        action = heuristic_action(obs)
        actions.append(asdict(action))
        obs = timed.step(action)
    raise RuntimeError("Finite recruit continuation exceeded 200 commands")


def build_scenario(database, ruleset, profile, index, seed, split):
    from bg_ai.hsbrsim_adapter import EarlyRecruitFixtureEngine, EarlyRecruitFixtureSpec
    from bg_ai.timed_recruiting import TimedRecruitEngine
    rng = random.Random(seed + index * 15485863)
    target_turn = 1 + index % 2
    available_ms = rng.choice((10000, 14000, 20000, 30000, 45000, 60000))
    engine = EarlyRecruitFixtureEngine(database.db, ruleset,
        timer_ms=lambda player, turn: available_ms if player == 0 and turn == target_turn else 60000,
        provenance=database.provenance, fixture=EarlyRecruitFixtureSpec(max_turns=2))
    timed = TimedRecruitEngine.for_fixture(engine, profile, ruleset)
    obs = timed.reset(seed=seed + index * 1009)
    for _ in range(1000):
        if obs is None:
            raise RuntimeError("Fixture ended before requested state")
        if (obs.player_id, obs.turn) == (0, target_turn):
            break
        obs = timed.step(heuristic_action(obs))
    else:
        raise RuntimeError("Prefix exceeded finite fixture bound")
    # Random-length *legal* baseline prefix yields buy/play/buff/choice states.
    for _ in range(rng.randrange(4)):
        action = heuristic_action(obs)
        if action.kind == "end_turn":
            break
        branch = timed.fork()
        after = branch.step(action)
        if after is None or (after.player_id, after.turn) != (0, target_turn):
            break
        timed, obs = branch, after
    actions = candidates_for(obs)
    if len(actions) < 2:
        raise RuntimeError("Scenario has fewer than two legal candidates")
    features = encode_legal_actions(obs, actions).tolist()
    # One fixed opponent, generated by the same visible baseline in this engine
    # state, is shared by every candidate. Its private state is absent in input.
    opponent_branch = timed.fork()
    finish_player(opponent_branch, 0, target_turn)
    other = opponent_branch._visible
    if other is None or other.turn != target_turn:
        raise RuntimeError("No same-turn opponent available")
    opponent_id = other.player_id
    opponent, _ = finish_player(opponent_branch, opponent_id, target_turn)
    candidates = []
    for action, vector in zip(actions, features):
        branch = timed.fork()
        before_trace = len(branch.trace)
        terminal, sequence = finish_player(branch, 0, target_turn, action)
        candidates.append({"action": asdict(action), "features": vector, "terminal": terminal,
            "continuation": sequence, "timing_trace": branch.trace[before_trace:]})
    return {"scenario_id": f"recruit-{seed}-{index}", "index": index, "split": split,
        "turn": target_turn, "hero_id": "TB_BaconShop_HERO_34", "validTribes": TRIBES,
        "observation": observation_record(obs), "opponent": opponent,
        "candidates": candidates,
        "baseline_indices": {"practical_heuristic": 0,
            "raw_stats": actions.index(cheap_stat_action(obs)),
            "end_if_legal": next((i for i, a in enumerate(actions) if a.kind == "end_turn"), 0)},
        "metadata": {"scope": SCOPE, "feature_version": RECRUIT_FEATURE_VERSION,
            "timer_source": "explicit synthetic remaining-time scenarios; not a claimed live turn-duration table",
            "initial_available_ms": available_ms, "timing_profile_sha256": profile.fingerprint,
            "engine_provenance_sha256": hashlib.sha256(json.dumps(database.provenance, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}}


def score_rows(rows, output, trials, seed, selected_only=False):
    options = {"trials": trials, "seed": seed, "selectedOnly": selected_only}
    with Path(output).open("w") as handle:
        proc = subprocess.Popen(["node", "--input-type=module", "-e", LABEL_JS], cwd=ROOT,
            env=dict(os.environ, BG_RECRUIT_LABEL_OPTIONS=json.dumps(options)), stdin=subprocess.PIPE,
            stdout=handle, text=True)
        try:
            for row in rows:
                proc.stdin.write(json.dumps(row, separators=(",", ":")) + "\n")
        finally:
            proc.stdin.close()
        if proc.wait():
            raise RuntimeError("Firestone labeling process failed")
    return [json.loads(line) for line in Path(output).read_text().splitlines()]


def paired_summary(differences, rng, samples=10000):
    values = np.asarray(differences)
    boot = np.empty(samples)
    for start in range(0, samples, 500):
        n = min(500, samples-start)
        boot[start:start+n] = values[rng.integers(0, len(values), (n, len(values)))].mean(axis=1)
    return {"mean_delta": float(values.mean()), "ci95": np.quantile(boot, [.025, .975]).tolist()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-root", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/20260905-recruit-v1")
    parser.add_argument("--scenarios", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=202609051)
    parser.add_argument("--trials", type=int, default=64)
    parser.add_argument("--evaluation-trials", type=int, default=1024)
    parser.add_argument("--hidden", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--epochs", type=int, nargs="+", default=[10, 20, 40])
    parser.add_argument("--train-validation-attempts", type=int)
    parser.add_argument("--holdout-attempts", type=int)
    parser.add_argument("--exclude-test-from", type=Path, help="Exclude visible v2 test states from every new split")
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--reencode-from", type=Path, help="Reuse prior train/validation real labels with current features; newly generated scenarios are an independent test set")
    parser.add_argument("--allow-historical", action="store_true")
    args = parser.parse_args(argv)
    if args.scenarios < 10 or args.trials < 1 or args.evaluation_trials < 1:
        parser.error("Require at least 10 scenarios and positive combat counts")
    if (args.train_validation_attempts is None) != (args.holdout_attempts is None):
        parser.error("Specify both train-validation-attempts and holdout-attempts")
    if args.train_validation_attempts is not None:
        if args.reencode_from or min(args.train_validation_attempts, args.holdout_attempts) < 10:
            parser.error("Separate fresh split counts require positive counts and cannot reencode old splits")
        args.scenarios = args.train_validation_attempts + args.holdout_attempts
    if not args.hidden or min(args.hidden) < 1 or not args.epochs or args.epochs != sorted(set(args.epochs)) or args.epochs[0] < 1:
        parser.error("Hidden sizes and increasing unique epoch checkpoints must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if not args.allow_historical:
        subprocess.run([sys.executable, str(ROOT / "scripts/check_live_ruleset.py"), "--report", str(args.out / "live_preflight.json")], cwd=ROOT, check=True)
    from bg_ai.hsbrsim_adapter import SUPPORTED_MINIONS, SUPPORTED_SPELLS
    from bg_ai.hsbrsim_data import build_current_database
    ruleset = json.loads((ROOT / "data/ruleset.json").read_text())
    profile = TimingProfile.load(ROOT / "config/turn-budget.json")
    database = build_current_database(args.engine_root, fixture_minion_ids=sorted(SUPPORTED_MINIONS),
                                      fixture_spell_ids=sorted(SUPPORTED_SPELLS))
    preregister = {"scope": SCOPE, "full_game_ready": False, "seed": args.seed,
        "attempted_scenarios": args.scenarios, "split": "seed-index modulo10:0-5 train,6-7 validation,8-9 test",
        "label_trials": args.trials, "fresh_evaluation_trials": args.evaluation_trials,
        "label_seed": 32452843, "fresh_evaluation_seed": 49979687,
        "candidate_models": {"hidden": args.hidden, "epochs": args.epochs, "tie_tolerance": .02},
        "model_selection": "validation mean score only; all selected-model test choices frozen before fresh simulation",
        "continuation_policy": "practical_heuristic_v1", "neural_policy_scope": "one first action followed by same fixed continuation",
        "opponent": "one same-turn actual-engine heuristic opponent, fixed across candidates; no private opponent policy inputs",
        "shortlist": "up to16 actions from timedlegalmask; canonical playposition0; bothbaselinesincluded",
        "feature_version": RECRUIT_FEATURE_VERSION, "feature_count": len(RECRUIT_FEATURE_NAMES),
        "current_snapshot": not args.allow_historical, "provenance": database.provenance,
        "promotion_rule": "positive lower95CI vs practicalheuristic and rawstats; applies only to this curriculum",
        "inclusion_scope": "Conditional on every shortlisted counterfactual transition and combat being supported; failures excluded before model choices",
        "reencode_from": str(args.reencode_from) if args.reencode_from else None,
        "exclude_test_from": str(args.exclude_test_from) if args.exclude_test_from else None,
        "train_validation_attempts": args.train_validation_attempts, "holdout_attempts": args.holdout_attempts}
    if args.reencode_from:
        preregister["split"] = "prior run train/validation only, reencoded; all newly generated seeds are test"
        preregister["base_scenario_archive_sha256"] = source_archive_hash(args.reencode_from)
        preregister["base_labels_sha256"] = sha(args.reencode_from / "labels.jsonl") if (args.reencode_from / "labels.jsonl").exists() else hashlib.sha256("".join(artifact_lines(args.reencode_from / "labels.jsonl.gz")).encode()).hexdigest()
    if args.train_validation_attempts is not None:
        preregister["split"] = "First train_validation_attempts: indexmod5 0-3train/4validation; subsequent holdout_attempts alltest"
    if args.exclude_test_from:
        preregister["excluded_test_archive_sha256"] = source_archive_hash(args.exclude_test_from)
    prepath = args.out / "preregistration.json"
    if not args.reuse:
        write_json(prepath, preregister)
    elif json.loads(prepath.read_text()) != preregister:
        raise ValueError("Cannot reuse a run with changed preregistration")
    rawpath = args.out / "scenarios.jsonl.gz"
    errors = []
    if not args.reuse:
        rows = []
        seen_states = set()
        base_labels = []
        excluded_visible = set()
        v2_indices = [RECRUIT_FEATURE_NAMES.index(name) for name in RECRUIT_V2_FEATURE_NAMES]
        if args.exclude_test_from:
            for previous in read_rows(args.exclude_test_from):
                if previous["split"] == "test":
                    excluded_visible.add(hashlib.sha256(json.dumps(sorted(c["features"] for c in previous["candidates"]), separators=(",", ":")).encode()).hexdigest())
        if args.reencode_from:
            for row in read_rows(args.reencode_from):
                if row["split"] == "test":
                    continue
                record = dict(row["observation"])
                record["legal_actions"] = tuple(RecruitAction(**a) for a in record["legal_actions"])
                observation = RecruitObservation(**record)
                vectors = encode_legal_actions(observation, [RecruitAction(**c["action"]) for c in row["candidates"]])
                for candidate, vector in zip(row["candidates"], vectors):
                    candidate["features"] = vector.tolist()
                row["metadata"]["feature_version"] = RECRUIT_FEATURE_VERSION
                row["visible_state_fingerprint"] = hashlib.sha256(json.dumps(sorted(c["features"] for c in row["candidates"]), separators=(",", ":")).encode()).hexdigest()
                seen_states.add(row["visible_state_fingerprint"])
                rows.append(row)
            retained = {r["scenario_id"] for r in rows}
            base_labels = [r for line in artifact_lines(args.reencode_from / ("labels.jsonl" if (args.reencode_from / "labels.jsonl").exists() else "labels.jsonl.gz"))
                           if (r := json.loads(line))["scenario_id"] in retained]
        with gzip.open(rawpath, "wt", encoding="utf8") as handle:
            for row in rows:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            new_start = len(rows)
            for index in range(args.scenarios):
                split = "test" if args.reencode_from else "train" if index % 10 < 6 else "validation" if index % 10 < 8 else "test"
                if args.train_validation_attempts is not None:
                    split = "test" if index >= args.train_validation_attempts else "validation" if index % 5 == 4 else "train"
                try:
                    row = build_scenario(database, ruleset, profile, index, args.seed, split)
                    prior_fingerprint = hashlib.sha256(json.dumps(sorted([c["features"][i] for i in v2_indices] for c in row["candidates"]), separators=(",", ":")).encode()).hexdigest()
                    if prior_fingerprint in excluded_visible:
                        raise ValueError("Previously evaluated v2 test state excluded from new experiment")
                    fingerprint = hashlib.sha256(json.dumps(sorted(c["features"] for c in row["candidates"]), separators=(",", ":")).encode()).hexdigest()
                    if fingerprint in seen_states:
                        raise ValueError("Duplicate visible candidate state excluded before split leakage")
                    seen_states.add(fingerprint)
                    row["visible_state_fingerprint"] = fingerprint
                    rows.append(row)
                    handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                except Exception as error:
                    errors.append({"index": index, "split": split, "stage": "recruit_rollout", "error": f"{type(error).__name__}: {error}"})
                if (index+1) % 25 == 0:
                    progress = {"stage": "recruit_rollout", "attempted": index+1, "saved": len(rows),
                        "failed": len(errors), "seconds": round(time.time()-started, 1)}
                    print(json.dumps(progress), flush=True)
                    write_json(args.out / "progress.json", progress)
                    handle.flush()
                if (index+1) % 250 == 0:
                    write_json(args.out / "generation_failures.partial.json", errors)
        write_json(args.out / "generation_failures.json", errors)
        write_json(args.out / "progress.json", {"stage": "combat_labels", "saved_scenarios": len(rows), "new_label_scenarios": len(rows)-new_start})
        fresh_labels = score_rows(rows[new_start:], args.out / "labels.jsonl", args.trials, 32452843)
        labels = base_labels + fresh_labels
        if base_labels:
            with (args.out / "labels.jsonl").open("w") as handle:
                for row in labels:
                    handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    else:
        rows = list(read_rows(args.out))
        errors = json.loads((args.out / "generation_failures.json").read_text())
        labels = [json.loads(line) for line in artifact_lines(args.out / ("labels.jsonl" if (args.out / "labels.jsonl").exists() else "labels.jsonl.gz"))]
    labelmap = {r["scenario_id"]: r for r in labels if "error" not in r}
    rows = [r for r in rows if r["scenario_id"] in labelmap]
    scenarios = []
    for row in rows:
        label = labelmap[row["scenario_id"]]
        scenarios.append(Scenario(row["scenario_id"], row["split"],
            np.asarray([c["features"] for c in row["candidates"]]),
            np.asarray([label["results"][str(i)]["score"] for i in range(len(row["candidates"]))], dtype=np.float64),
            {"scope": SCOPE, "source": "actual hsrl2 actions and Firestone terminal combat"}))
    dataset = Dataset(scenarios, RECRUIT_FEATURE_NAMES)
    trainval = Dataset([s for s in scenarios if s.split != "test"], RECRUIT_FEATURE_NAMES)
    if any(not dataset.split(split) for split in ("train", "validation", "test")):
        raise RuntimeError("Failures left an empty required dataset split")
    best_model, best_value, sweep = None, -math.inf, []
    for hidden in args.hidden:
        model = Ranker(RECRUIT_FEATURE_NAMES, hidden=hidden, seed=17)
        prior = 0
        for epochs in args.epochs:
            model.fit(trainval, epochs=epochs-prior, tie_tolerance=.02)
            prior = epochs
            validation = evaluate(model, trainval, split="validation", seed=19, bootstrap_samples=1000)
            model.metadata.update(scope=SCOPE, recruit_feature_version=RECRUIT_FEATURE_VERSION,
                full_game_ready=False, timing_profile_sha256=profile.fingerprint)
            checkpoint = args.out / f"model-h{hidden}-e{epochs}.json"
            model.save(checkpoint)
            sweep.append({"hidden": hidden, "epochs": epochs, "validation_mean": validation["mean_selected_score"],
                "validation_baseline": validation["mean_baseline_score"], "checkpoint": checkpoint.name})
            progress = {"stage": "train", **sweep[-1], "seconds": time.time()-started}
            print(json.dumps(progress), flush=True)
            write_json(args.out / "progress.json", progress)
            if validation["mean_selected_score"] > best_value:
                best_value = validation["mean_selected_score"]
                best_model = Ranker.load(checkpoint, RECRUIT_FEATURE_NAMES)
    write_json(args.out / "validation_sweep.json", sweep)
    best_model.save(args.out / "selected_model.json")
    testrows = [row for row in rows if row["split"] == "test"]
    rng = random.Random(67867967)
    for row in testrows:
        values = best_model.predict(np.asarray([c["features"] for c in row["candidates"]]))
        row["selections"] = dict(row["baseline_indices"], model=int(np.argmax(values)),
                                 random=rng.randrange(len(row["candidates"])))
    frozen = args.out / "test_choices.jsonl"
    with frozen.open("w") as handle:
        for row in testrows:
            handle.write(json.dumps({"scenario_id": row["scenario_id"], "selections": row["selections"],
                "model_sha256": sha(args.out / "selected_model.json")}) + "\n")
    write_json(args.out / "progress.json", {"stage": "fresh_evaluation", "test_scenarios": len(testrows), "seconds": time.time()-started})
    fresh = score_rows(testrows, args.out / "fresh_evaluation.jsonl", args.evaluation_trials, 49979687, True)
    if any("error" in row for row in fresh) or len(fresh) != len(testrows):
        raise RuntimeError("Final evaluation must complete every frozen test scenario")
    stats = {}
    for name in ("model", "practical_heuristic", "raw_stats", "end_if_legal", "random"):
        stats[name] = float(np.mean([row["results"][name]["score"] for row in fresh]))
    bootstrap_rng = np.random.default_rng(86028121)
    differences = {name: paired_summary([r["results"]["model"]["score"] - r["results"][name]["score"] for r in fresh], bootstrap_rng)
                   for name in stats if name != "model"}
    current = None
    if not args.allow_historical:
        subprocess.run([sys.executable, str(ROOT / "scripts/check_live_ruleset.py"), "--report", str(args.out / "live_postflight.json")], cwd=ROOT, check=True)
        current = True
    action_counts = Counter(c["action"]["kind"] for r in rows for c in r["candidates"])
    command_counts = Counter(a["kind"] for r in rows for c in r["candidates"] for a in c["continuation"])
    result = {"scope": SCOPE, "full_game_ready": False, "current_snapshot_verified": current,
        "attempted_new_scenarios": args.scenarios, "generated_scenarios_including_reencoded_base": len(rows),
        "splits": dict(Counter(row["split"] for row in rows)), "generation_failures": len(errors),
        "generation_failure_reasons": dict(Counter(error["error"] for error in errors)),
        "generation_failures_by_split": dict(Counter(error.get("split", "unknown") for error in errors)),
        "inclusion_scope": preregister["inclusion_scope"],
        "reencoded_base": str(args.reencode_from) if args.reencode_from else None,
        "combat_label_failures": len([r for r in labels if "error" in r]),
        "root_action_kind_counts": dict(action_counts), "executed_command_kind_counts": dict(command_counts),
        "label_combats": sum(r.get("metadata", {}).get("uniqueCombats", 0) for r in labels),
        "fresh_evaluation_combats": sum(r["metadata"]["uniqueCombats"] for r in fresh),
        "test_scenarios": len(fresh), "score_definition": "P(win)+0.5P(tie), same-turn combat only",
        "scores": stats, "paired_model_minus_baseline": differences,
        "passes_curriculum_benchmark": all(differences[n]["ci95"][0] > 0 for n in ("practical_heuristic", "raw_stats")),
        "bootstrap": {"unit": "scenario", "samples": 10000, "seed": 86028121,
            "limitations": "Does not cover engine model error, restricted pool distribution shift, or longer-horizon economy"},
        "selected_checkpoint": "selected_model.json", "checkpoint_sha256": sha(args.out / "selected_model.json"),
        "test_choices_sha256": sha(frozen), "scenario_archive_sha256": source_archive_hash(args.out),
        "elapsed_seconds": time.time()-started,
        "limitations": ["Seven current minions and three Tavern spells only, fixed five tribes, Patchwerk",
            "Curriculum timer windows supplied explicitly, not verified client turn durations",
            "Single learned first action followed by fixed heuristic; not an autonomous full-turn neural policy",
            "Same-turn combat objective gives no fair valuation of tavern leveling or saved economic value",
            "One fixed heuristic opponent per scenario; full-lobby placements and MMR unmeasured",
            "Unsupported recruit transitions are recorded failures and never assigned scores"]}
    write_json(args.out / "results.json", result)
    write_json(ROOT / "reports/recruit_curriculum_results.json", result)
    write_json(args.out / "progress.json", {"stage": "completed", "results": "results.json", "seconds": time.time()-started})
    print(json.dumps(result, indent=2), flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
