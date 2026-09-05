"""Synthetic plumbing fixtures only: these tests make no game-strength claim."""

import json
import hashlib
import itertools
import gzip
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from bg_ai.features import BOARD_FEATURE_NAMES, CARD_FEATURE_NAMES, encode_board, encode_card
from bg_ai.learning import (Dataset, Ranker, Scenario, evaluate, load_dataset,
                            LEGACY_TRAINING_OBJECTIVE)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from select_positions import freeze_selections
from report_fresh_evaluation import summarize


def fixture_dataset(seed=11):
    rng = np.random.default_rng(seed)
    scenarios = []
    for split, count in (("train", 70), ("validation", 12), ("test", 24)):
        for i in range(count):
            x = rng.normal(size=(5, 3))
            # Synthetic ranking relation; unrelated to Battlegrounds results.
            y = x[:, 0] - 0.7 * x[:, 1] + 0.3 * x[:, 2]
            scenarios.append(Scenario(f"synthetic-{split}-{i}", split, x, y,
                                      {"synthetic_fixture": True}))
    return Dataset(scenarios, ("fixture_x", "fixture_y", "fixture_z"))


class FeatureTests(unittest.TestCase):
    def test_id_independence_and_order_sensitivity(self):
        a = {"cardId": "OLD", "attack": 12, "health": 4, "races": ["BEAST"], "deathrattle": True}
        replacement = dict(a, cardId="NEXT_ROTATION", name="Different name")
        np.testing.assert_array_equal(encode_card(a), encode_card(replacement))
        b = {"cardId": "B", "attack": 2, "health": 18, "taunt": True, "divineShield": True}
        forward, reverse = encode_board([a, b], [b]), encode_board([b, a], [b])
        self.assertEqual(forward.shape, (len(BOARD_FEATURE_NAMES),))
        self.assertFalse(np.array_equal(forward, reverse))
        self.assertTrue(np.isfinite(forward).all())
        pair_indices = [i for i, n in enumerate(BOARD_FEATURE_NAMES) if ".pair" in n]
        self.assertFalse(np.array_equal(forward[pair_indices], reverse[pair_indices]))

    def test_aliases_and_validation(self):
        a = encode_card({"attack": 5, "health": 9, "races": ["MECHANICAL"], "divineShield": True})
        b = encode_card({"attack": 5, "health": 9, "tribes": ["MECH"], "mechanics": ["DIVINE_SHIELD"]})
        np.testing.assert_array_equal(a, b)
        all_tribes = encode_card({"attack": 1, "health": 1, "races": ["ALL"]})
        self.assertTrue(all(all_tribes[i] == 1 for i, n in enumerate(CARD_FEATURE_NAMES) if n.startswith("tribe_")))
        with self.assertRaises(ValueError):
            encode_card({"attack": float("nan"), "health": 1})
        with self.assertRaises(ValueError):
            encode_card({"attack": 1, "health": 1, "races": [20]})
        with self.assertRaises(ValueError):
            encode_board([{"attack": 1, "health": 1}] * 8)

    def test_effect_semantics_distinguish_equal_stat_cards_without_ids(self):
        base = {"attack": 4, "health": 4, "deathrattle": True}
        summon = dict(base, cardId="OLD", text="<b>Deathrattle:</b> Summon two 1/1 Beasts.")
        buff = dict(base, cardId="OLD", text="<b>Deathrattle:</b> Give your minions +2/+2.")
        self.assertFalse(np.array_equal(encode_card(summon), encode_card(buff)))
        np.testing.assert_array_equal(encode_card(summon),
                                      encode_card(dict(summon, cardId="NEW_ROTATION", name="Changed name")))
        np.testing.assert_array_equal(encode_card(summon),
                                      encode_card(dict(summon, text="[x]Deathrattle: Summon two 1/1 Beasts.")))


class LearningTests(unittest.TestCase):
    def test_material_gaps_receive_proportionally_larger_gradients(self):
        # Same synthetic features and ordering; only label gap magnitude changes.
        model = Ranker(("fixture_x",), hidden=2, seed=7)
        x = np.array([[0.], [1.], [2.]])
        tiny = Scenario("synthetic-tiny", "train", x, np.array([0., 0.001, 0.002]), {})
        material = Scenario("synthetic-material", "train", x, np.array([0., 0.1, 0.2]), {})
        tiny_loss, tiny_gradient = model._loss_gradient(tiny, 0)
        material_loss, material_gradient = model._loss_gradient(material, 0)
        self.assertAlmostEqual(material_loss, tiny_loss * 100)
        for key in tiny_gradient:
            np.testing.assert_allclose(material_gradient[key], tiny_gradient[key] * 100,
                                       rtol=1e-12, atol=1e-12)
        legacy_tiny_loss, legacy_tiny_gradient = model._loss_gradient(tiny, 0, LEGACY_TRAINING_OBJECTIVE)
        legacy_material_loss, legacy_material_gradient = model._loss_gradient(material, 0, LEGACY_TRAINING_OBJECTIVE)
        self.assertAlmostEqual(legacy_tiny_loss, legacy_material_loss)
        for key in legacy_tiny_gradient:
            np.testing.assert_allclose(legacy_tiny_gradient[key], legacy_material_gradient[key],
                                       rtol=1e-12, atol=1e-12)

    def test_learns_fixture_and_evaluates_paired_scenarios(self):
        data = fixture_dataset()
        model = Ranker(data.feature_names, hidden=8, seed=4)
        history = model.fit(data, epochs=30, learning_rate=0.01)
        self.assertLess(history[-1], history[0] * 0.3)
        report = evaluate(model, data, bootstrap_samples=500)
        self.assertEqual(report["scenario_count"], 24)
        self.assertGreater(report["paired_vs_uniform_candidate"]["ci95"][0], 0)
        self.assertLess(report["mean_regret_to_best_observed"], 0.12)

    def test_checkpoint_exact_resume_and_schema_guard(self):
        data = fixture_dataset()
        model = Ranker(data.feature_names, hidden=6, seed=9)
        model.fit(data, epochs=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.save(path)
            resumed = Ranker.load(path, data.feature_names)
            np.testing.assert_array_equal(model.predict(data.scenarios[0].features),
                                          resumed.predict(data.scenarios[0].features))
            original_mean = model.mean.copy()
            model.fit(data, epochs=2)
            resumed.fit(data, epochs=2)
            for key in model.params:
                np.testing.assert_array_equal(model.params[key], resumed.params[key])
            np.testing.assert_array_equal(resumed.mean, original_mean)
            with self.assertRaisesRegex(ValueError, "schema"):
                Ranker.load(path, ("different",))

    def test_heldout_labels_cannot_affect_training(self):
        data = fixture_dataset()
        changed = Dataset([Scenario(s.scenario_id, s.split, s.features,
                                     s.scores if s.split == "train" else -100 * s.scores,
                                     s.metadata) for s in data.scenarios], data.feature_names)
        first, second = Ranker(data.feature_names, seed=3), Ranker(data.feature_names, seed=3)
        first.fit(data, epochs=3)
        second.fit(changed, epochs=3)
        for key in first.params:
            np.testing.assert_array_equal(first.params[key], second.params[key])
        moved = Dataset([Scenario(data.scenarios[0].scenario_id, "test",
                                  data.scenarios[0].features, data.scenarios[0].scores, {})],
                        data.feature_names)
        with self.assertRaisesRegex(ValueError, "training scenarios"):
            evaluate(first, moved)

    def test_loader_rejects_cross_split_permutations_and_changed_rosters(self):
        a, b = {"attack": 1, "health": 3}, {"attack": 5, "health": 1}
        row = {"scenario_id": "synthetic-a", "split": "train", "opponent": [a],
               "candidates": [{"board": [a, b], "score": 0.1}, {"board": [b, a], "score": 0.3}],
               "metadata": {"synthetic_fixture": True}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text(json.dumps(row) + "\n")
            self.assertEqual(load_dataset(path).scenarios[0].features.shape[1], len(BOARD_FEATURE_NAMES))
            zipped = Path(directory) / "data.jsonl.gz"
            zipped.write_bytes(gzip.compress(path.read_bytes(), mtime=0))
            np.testing.assert_array_equal(load_dataset(zipped).scenarios[0].features,
                                          load_dataset(path).scenarios[0].features)
            # Excluded test rows must not reach candidate/score processing.
            excluded = {"scenario_id": "synthetic-test-excluded", "split": "test", "candidates": "not processed"}
            path.write_text(json.dumps(row) + "\n" + json.dumps(excluded) + "\n")
            self.assertEqual(len(load_dataset(path, allowed_splits=("train",)).scenarios), 1)
            other = dict(row, scenario_id="synthetic-b", split="test")
            path.write_text(json.dumps(row) + "\n" + json.dumps(other) + "\n")
            with self.assertRaisesRegex(ValueError, "data leakage"):
                load_dataset(path)
            row["candidates"][1]["board"] = [a, a]
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "permutations"):
                load_dataset(path)

    def test_ties_and_single_scenario_uncertainty(self):
        data = Dataset([Scenario("synthetic-train", "train", np.array([[0.], [1.]]),
                                 np.array([0.5, 0.5]), {})], ("fixture_x",))
        model = Ranker(data.feature_names)
        with self.assertRaisesRegex(ValueError, "tied"):
            model.fit(data)
        test = Dataset([Scenario("synthetic-test", "test", np.array([[0.], [1.]]),
                                 np.array([0.2, 0.8]), {})], data.feature_names)
        report = evaluate(model, test, bootstrap_samples=100)
        self.assertIsNone(report["paired_vs_baseline"]["ci95"])


class FrozenEvaluationTests(unittest.TestCase):
    def test_freeze_heuristics_and_audit_independent_simulation_fixture(self):
        # Entire scenario and fresh-result payload are synthetic plumbing fixtures.
        board = [{"entityId": 1, "attack": 1, "health": 9, "taunt": False},
                 {"entityId": 2, "attack": 5, "health": 3, "taunt": True},
                 {"entityId": 3, "attack": 8, "health": 1, "taunt": False}]
        row = {"scenario_id": "synthetic-final-test", "split": "test", "opponent": [],
               "candidates": [{"board": list(order), "score": 0.5}
                              for order in itertools.permutations(board)],
               "metadata": {"synthetic_fixture": True, "combatSeed": 1,
                            "cardsSha256": "fixture-cards", "rulesetSha256": "fixture-rules"}}
        with tempfile.TemporaryDirectory() as directory:
            data_path, checkpoint = Path(directory) / "data.jsonl", Path(directory) / "model.json"
            data_path.write_text(json.dumps(row) + "\n")
            Ranker(BOARD_FEATURE_NAMES, hidden=2).save(checkpoint)
            frozen = freeze_selections(data_path, checkpoint)
            zipped = Path(directory) / "data.jsonl.gz"
            zipped.write_bytes(gzip.compress(data_path.read_bytes(), mtime=0))
            zipped_frozen = freeze_selections(zipped, checkpoint)
            self.assertEqual(zipped_frozen[0]["selections"], frozen[0]["selections"])
            self.assertEqual(zipped_frozen[0]["provenance"]["dataset_sha256"],
                             hashlib.sha256(zipped.read_bytes()).hexdigest())
            selections = frozen[0]["selections"]
            self.assertEqual([c["entityId"] for c in row["candidates"][selections["attack"]]["board"]], [3, 2, 1])
            self.assertEqual([c["entityId"] for c in row["candidates"][selections["health"]]["board"]], [1, 2, 3])
            self.assertEqual([c["entityId"] for c in row["candidates"][selections["taunt_last"]]["board"]], [3, 1, 2])
            frozen_path = Path(directory) / "selections.jsonl"
            frozen_path.write_text(json.dumps(frozen[0]) + "\n")
            selection_hash = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
            fresh = {"scenario_id": row["scenario_id"], "split": "test",
                     "scores": {name: {"candidateIndex": index, "score": 0.5,
                                       "simulations": {"won": 10, "tied": 0, "lost": 10, "n": 20, "seed": 2}}
                                for name, index in selections.items()},
                     "metadata": {"trials": 20, "combatSeed": 2, "engine": "synthetic_fixture",
                                  "selectionsSha256": selection_hash,
                                  "cardsSha256": "fixture-cards", "rulesetSha256": "fixture-rules"}}
            fresh_path = Path(directory) / "fresh.jsonl"
            fresh_path.write_text(json.dumps(fresh) + "\n")
            report = summarize(fresh_path, frozen_path, bootstrap_samples=100)
            self.assertFalse(report["results"]["test"]["positioning_benchmark_gate"]["passes"])
            fresh["metadata"]["combatSeed"] = 1
            fresh_path.write_text(json.dumps(fresh) + "\n")
            with self.assertRaisesRegex(ValueError, "reused"):
                summarize(fresh_path, frozen_path, bootstrap_samples=100)


if __name__ == "__main__":
    unittest.main()
