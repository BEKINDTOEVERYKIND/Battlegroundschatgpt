"""Post-evaluation inspection; never changes inclusion, policy choices or scores."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path


directory = Path(__file__).resolve().parent
choices = {row["scenario_id"]: row for row in map(
    json.loads, (directory / "test_choices.jsonl").read_text().splitlines())}
selected_cards, candidate_casts, selected_casts = Counter(), Counter(), Counter()
owned_hand, tribes, timers, actual_casts, actual_buys = (Counter() for _ in range(5))
identical = freeze_identical = 0
with gzip.open(directory / "scenarios.jsonl.gz", "rt") as handle:
    for line in handle:
        row = json.loads(line)
        selections = choices[row["scenario_id"]]["selections"]
        model = row["candidates"][selections["model"]]
        baseline = row["candidates"][selections["practical_heuristic"]]
        same = model["player"] == baseline["player"]
        identical += same
        freeze_identical += same and model["action"]["kind"] == "freeze"
        tribes[",".join(row["valid_tribes"])] += 1
        timers[row["metadata"]["initial_available_ms"]] += 1
        private = row["observation"]["private_state"]
        cards = {card["entity_id"]: card for zone in ("board", "hand", "shop")
                 for card in private.get(zone, [])}
        selected_cards.update(card["card_id"] for card in model["player"]["board"])
        owned_hand.update(card["card_id"] for card in private.get("hand", []))
        for index, candidate in enumerate(row["candidates"]):
            action = candidate["action"]
            if action["kind"] == "cast_spell":
                card = cards[action["entity_id"]]
                candidate_casts[card["card_id"]] += 1
                if index == selections["model"]:
                    selected_casts[card["card_id"]] += 1
            for entry in candidate["continuation"]:
                action = entry["action"]
                card = cards.get(action["entity_id"])
                if entry["player_id"] != 0 or card is None:
                    continue
                if action["kind"] == "cast_spell":
                    actual_casts[card["card_id"]] += 1
                if action["kind"] == "buy" and card.get("card_type") == "spell":
                    actual_buys[card["card_id"]] += 1

result = {
    "analysis": "Post-evaluation descriptive coverage only; no retraining, selection, or outcome changes",
    "source_results_sha256": hashlib.sha256((directory / "results.json").read_bytes()).hexdigest(),
    "model_practical_exact_equal_combat_snapshots": identical,
    "freeze_then_practical_exact_equal_combat_snapshots": freeze_identical,
    "model_selected_final_board_card_counts": dict(selected_cards),
    "candidate_first_action_cast_spell_card_counts": dict(candidate_casts),
    "model_first_action_cast_spell_card_counts": dict(selected_casts),
    "owned_hand_at_decision_card_counts": dict(owned_hand),
    "accepted_tribe_schedule_counts": dict(tribes),
    "initial_timer_ms_counts": dict(timers),
    "root_visible_spell_ids_actually_cast_in_player0_continuations_lower_bound": dict(actual_casts),
    "root_visible_spell_ids_actually_bought_in_player0_continuations_lower_bound": dict(actual_buys),
}
(directory / "descriptive_coverage.json").write_text(json.dumps(result, indent=2) + "\n")
