"""A board reorder must be executable under its physical drag budget."""
from collections import deque
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from bg_ai.position_actions import affordable_orders, drag_plan, minimum_drag_count

ROOT = Path(__file__).resolve().parents[1]


class PositionActionTests(unittest.TestCase):
    def test_every_five_card_order_matches_independent_shortest_path(self):
        # An independent graph search checks both optimality and action execution.
        start = tuple(range(5))
        distances = {start: 0}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for source in range(5):
                for destination in range(5):
                    changed = list(current)
                    changed.insert(destination, changed.pop(source))
                    target = tuple(changed)
                    if target not in distances:
                        distances[target] = distances[current] + 1
                        queue.append(target)
        for target, expected in distances.items():
            with self.subTest(target=target):
                actions = drag_plan(target)
                self.assertEqual(minimum_drag_count(target), expected)
                self.assertEqual(len(actions), expected)
                actual = list(start)
                for action in actions:
                    card = actual.pop(action["from_index"])
                    self.assertEqual(card, action["original_index"])
                    actual.insert(action["to_index"], card)
                self.assertEqual(tuple(actual), target)

    def test_seven_card_reverse_needs_six_drags(self):
        target = (6, 5, 4, 3, 2, 1, 0)
        self.assertEqual(len(drag_plan(target)), 6)
        self.assertEqual(affordable_orders([tuple(range(7)), target], 5), [tuple(range(7))])

    def test_no_time_keeps_current_order_only(self):
        self.assertEqual(affordable_orders(itertools.permutations(range(4)), 0), [(0, 1, 2, 3)])

    def test_same_card_ids_are_distinguished_by_original_slot(self):
        board = [{"cardId": "duplicate", "attack": 1},
                 {"cardId": "duplicate", "attack": 9},
                 {"cardId": "other", "attack": 3}]
        for action in drag_plan((1, 2, 0)):
            card = board.pop(action["from_index"])
            board.insert(action["to_index"], card)
        self.assertEqual([c["attack"] for c in board], [9, 3, 1])

    def test_nonadjacent_swap_is_two_drags(self):
        self.assertEqual(minimum_drag_count((2, 1, 0, 3)), 2)

    def test_invalid_orders_are_rejected(self):
        for order in ((0, 0), (1, 2), (True, 0), tuple(range(8))):
            with self.subTest(order=order), self.assertRaises(ValueError):
                drag_plan(order)


class PositionRecommendationTimingTests(unittest.TestCase):
    def recommend(self, timer=None, *extra):
        source = json.loads((ROOT / "examples/position.json").read_text())
        source.pop("recruitTimeRemainingMs", None)
        if timer is not None:
            source["recruitTimeRemainingMs"] = timer
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.json"
            path.write_text(json.dumps(source))
            return subprocess.run([
                sys.executable, str(ROOT / "scripts/recommend_position.py"),
                "--input", str(path), "--checkpoint",
                str(ROOT / "runs/20260905-positioning-v2/selected_positioning.json"),
                "--pure-model", *extra,
            ], capture_output=True, text=True, timeout=20, cwd=ROOT)

    def test_missing_timer_fails_and_zero_timer_keeps_current_board(self):
        missing = self.recommend()
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("Supply actual recruitTimeRemainingMs", missing.stderr)
        exhausted = self.recommend(0)
        self.assertEqual(exhausted.returncode, 0, exhausted.stderr)
        result = json.loads(exhausted.stdout)
        self.assertEqual(result["indices"], [0, 1, 2, 3, 4])
        self.assertEqual(result["candidates_ranked"], 1)
        self.assertEqual(result["actions"], [])

    def test_explicit_timer_override_and_reported_actions_agree(self):
        process = self.recommend(0, "--remaining-seconds", "7")
        self.assertEqual(process.returncode, 0, process.stderr)
        result = json.loads(process.stdout)
        current = list(range(5))
        for action in result["actions"]:
            moved = current.pop(action["from_index"])
            self.assertEqual(moved, action["original_index"])
            current.insert(action["to_index"], moved)
        self.assertEqual(current, result["indices"])
        self.assertLessEqual(len(result["actions"]), 2)
        self.assertEqual(len(result["actions"]), minimum_drag_count(result["indices"]))
        self.assertEqual(result["timing"]["timer_source"], "--remaining-seconds")
        self.assertEqual(result["timing"]["after_actions"]["charged_ms"], len(result["actions"]) * 1000)
        self.assertFalse(result["timing"]["inference_elapsed_included"])


if __name__ == "__main__":
    unittest.main()
