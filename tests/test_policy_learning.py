"""Synthetic objective/optimization tests; no game-strength assertions."""
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from bg_ai.learning import Dataset, Ranker, Scenario
from bg_ai.policy_learning import POLICY_OBJECTIVE, fit_policy, policy_loss_gradient


NAMES = ("synthetic_strength", "synthetic_cost", "synthetic_context")


def groups():
    return [Scenario("multi-positive", "train", np.array([[.2, -.4, .3], [.1, .8, -.1], [1., .5, -.6]]),
                     np.array([1., 0., 1.]), {}),
            Scenario("single-positive", "train", np.array([[-.1, .7, .4], [.8, .2, -.5]]),
                     np.array([0., 1.]), {}),
            Scenario("forced", "train", np.array([[.3, -.9, .2]]), np.array([1.]), {})]


def fixture():
    rng = np.random.default_rng(2907)
    rows = []
    for split, count in (("train", 80), ("validation", 20), ("test", 20)):
        for i in range(count):
            x = rng.normal(size=(2 + i % 5, len(NAMES)))
            y = np.zeros(len(x))
            y[int(np.argmax(x[:, 0] - .6 * x[:, 1]))] = 1
            rows.append(Scenario(f"fixture-{split}-{i}", split, x, y,
                                 {"synthetic": True, "split": split}))
    return Dataset(rows, NAMES)


class PolicyObjectiveTests(unittest.TestCase):
    def test_finite_difference_gradient_variable_groups_and_positive_sets(self):
        model = Ranker(NAMES, hidden=3, seed=23)
        model.mean = np.array([.1, .2, -.2])
        model.scale = np.array([.7, 2., 1.1])
        _, actual = policy_loss_gradient(model, groups())
        epsilon = 1e-6
        for key, array in model.params.items():
            numerical = np.empty_like(array)
            for index in np.ndindex(array.shape):
                saved = array[index]
                array[index] = saved + epsilon
                upper, _ = policy_loss_gradient(model, groups())
                array[index] = saved - epsilon
                lower, _ = policy_loss_gradient(model, groups())
                array[index] = saved
                numerical[index] = (upper - lower) / (2 * epsilon)
            np.testing.assert_allclose(actual[key], numerical, rtol=2e-6, atol=2e-9)

    def test_grouped_loss_weights_decisions_equally_and_is_permutation_invariant(self):
        model = Ranker(NAMES, hidden=4, seed=19)
        rows = groups()
        combined, gradient = policy_loss_gradient(model, rows)
        separate = [policy_loss_gradient(model, [row]) for row in rows]
        self.assertAlmostEqual(combined, np.mean([s[0] for s in separate]), places=14)
        for key in gradient:
            np.testing.assert_allclose(gradient[key], np.mean([s[1][key] for s in separate], axis=0), atol=1e-14)
        shuffled = [Scenario(row.scenario_id, row.split, row.features[::-1], row.scores[::-1], {}) for row in rows[::-1]]
        other, other_gradient = policy_loss_gradient(model, shuffled)
        self.assertAlmostEqual(combined, other, places=14)
        for key in gradient:
            np.testing.assert_allclose(gradient[key], other_gradient[key], atol=1e-14)

    def test_positive_set_probability_not_uniform_target_probability(self):
        model = Ranker(NAMES, hidden=2, seed=5)
        for value in model.params.values():
            value[:] = 0
        multi = groups()[0]
        loss, _ = policy_loss_gradient(model, [multi])
        self.assertAlmostEqual(loss, math.log(3 / 2), places=14)
        all_positive = Scenario("equivalent", "train", multi.features, np.ones(3), {})
        loss, gradient = policy_loss_gradient(model, [all_positive])
        self.assertEqual(loss, 0)
        self.assertTrue(all(np.count_nonzero(g) == 0 for g in gradient.values()))

    def test_duplicate_identical_expert_actions_need_no_arbitrary_order(self):
        model = Ranker(("synthetic",), hidden=1, seed=1)
        model.params["w1"][:] = 1
        model.params["w2"][:] = 8
        row = Scenario("duplicate-teacher", "train", np.array([[1.], [1.], [-1.]]), np.array([1., 1., 0.]), {})
        loss, gradient = policy_loss_gradient(model, [row])
        self.assertLess(loss, 1e-4)
        self.assertEqual(model.predict(row.features)[0], model.predict(row.features)[1])
        self.assertTrue(all(np.isfinite(g).all() for g in gradient.values()))

    def test_very_low_positive_probability_does_not_overflow(self):
        model = Ranker(("synthetic",), hidden=1, seed=1)
        model.params["w1"][:] = 100
        model.params["w2"][:] = 10000
        row = Scenario("extreme", "train", np.array([[-1.], [1.]]), np.array([1., 0.]), {})
        with np.errstate(over="raise", invalid="raise"):
            loss, gradient = policy_loss_gradient(model, [row])
        self.assertAlmostEqual(loss, 20000)
        self.assertTrue(all(np.isfinite(g).all() for g in gradient.values()))

    def test_invalid_or_empty_positive_sets_fail(self):
        model = Ranker(NAMES, hidden=2, seed=5)
        row = groups()[0]
        for scores in ([0., 0., 0.], [1., .5, 0.], [1., float("nan"), 0.], [1., 0.]):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                policy_loss_gradient(model, [Scenario("bad", "train", row.features, np.array(scores), {})])
        with self.assertRaises(ValueError):
            policy_loss_gradient(model, [])


class PolicyFitTests(unittest.TestCase):
    def test_learns_teacher_and_forward_is_once_per_batch(self):
        data = fixture()
        model = Ranker(NAMES, hidden=8, seed=17)
        with patch.object(model, "_forward", wraps=model._forward) as forward:
            losses = fit_policy(model, data, epochs=30, batch_size=16, learning_rate=.01)
        self.assertEqual(forward.call_count, 30 * 5)
        self.assertLess(losses[-1], losses[0] * .15)
        accuracy = np.mean([s.scores[int(np.argmax(model.predict(s.features)))] for s in data.split("validation")])
        self.assertGreaterEqual(accuracy, .9)
        self.assertEqual(model.metadata["training_objective"], POLICY_OBJECTIVE)
        self.assertEqual(len(model.metadata["training_scenario_ids"]), 80)
        self.assertTrue(all(s["split"] == "train" for s in model.metadata["training_sources"]))

    def test_normalization_and_updates_ignore_held_out_values_and_targets(self):
        data = fixture()
        altered = deepcopy(data)
        for row in altered.scenarios:
            if row.split != "train":
                row.features[:] = 1e12
                row.scores[:] = np.nan
        left, right = Ranker(NAMES, hidden=4, seed=31), Ranker(NAMES, hidden=4, seed=31)
        fit_policy(left, data, epochs=2)
        fit_policy(right, altered, epochs=2)
        train_rows = np.concatenate([s.features for s in data.split("train")])
        np.testing.assert_array_equal(left.mean, train_rows.mean(axis=0))
        for key in left.params:
            np.testing.assert_array_equal(left.params[key], right.params[key])
        np.testing.assert_array_equal(left.mean, right.mean)
        np.testing.assert_array_equal(left.scale, right.scale)

    def test_checkpoint_resume_and_copy_leave_source_immutable(self):
        data = fixture()
        initial = Ranker(NAMES, hidden=5, seed=11)
        fit_policy(initial, data, epochs=2)
        initial_mean, initial_scale = initial.mean.copy(), initial.scale.copy()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.json"
            initial.save(path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            copied = deepcopy(initial)
            resumed = Ranker.load(path, NAMES)
            first = fit_policy(copied, data, epochs=3)
            second = fit_policy(resumed, data, epochs=3)
            self.assertEqual(first, second)
            for group in ("params", "m", "v"):
                for key in copied.params:
                    np.testing.assert_array_equal(getattr(copied, group)[key], getattr(resumed, group)[key])
            self.assertEqual(copied.rng.bit_generator.state, resumed.rng.bit_generator.state)
            self.assertEqual(copied.step, resumed.step)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            np.testing.assert_array_equal(resumed.mean, initial_mean)
            np.testing.assert_array_equal(resumed.scale, initial_scale)
            original = Ranker.load(path, NAMES)
            for key in initial.params:
                np.testing.assert_array_equal(initial.params[key], original.params[key])
        uninterrupted = Ranker(NAMES, hidden=5, seed=11)
        fit_policy(uninterrupted, data, epochs=5)
        for key in copied.params:
            np.testing.assert_array_equal(copied.params[key], uninterrupted.params[key])

    def test_warm_pairwise_transition_keeps_normalization_and_optimizer(self):
        data = fixture()
        model = Ranker(NAMES, hidden=4, seed=31)
        model.fit(data, epochs=1)
        mean, scale, step = model.mean.copy(), model.scale.copy(), model.step
        old_objective = model.metadata["training_objective"]
        fit_policy(model, data, epochs=1)
        self.assertGreater(model.step, step)
        np.testing.assert_array_equal(model.mean, mean)
        np.testing.assert_array_equal(model.scale, scale)
        self.assertIn({"from": old_objective, "to": POLICY_OBJECTIVE, "at_optimizer_step": step}, model.metadata["objective_transitions"])

    def test_forced_groups_zero_loss_but_entire_uninformative_training_fails(self):
        rows = groups()
        model = Ranker(NAMES, hidden=3, seed=5)
        with self.assertRaisesRegex(ValueError, "no policy learning signal"):
            fit_policy(model, Dataset([rows[2]], NAMES), epochs=1)
        losses = fit_policy(model, Dataset(rows, NAMES), epochs=1, batch_size=1)
        self.assertEqual(model.step, 2)
        self.assertEqual(len(losses), 1)

    def test_schema_split_leakage_and_invalid_hyperparameters_fail(self):
        rows = groups()
        model = Ranker(NAMES, hidden=3, seed=5)
        with self.assertRaisesRegex(ValueError, "schema"):
            fit_policy(model, Dataset(rows, ("different",)), epochs=1)
        leak = Scenario(rows[0].scenario_id, "validation", rows[0].features, rows[0].scores, {})
        with self.assertRaisesRegex(ValueError, "held-out"):
            fit_policy(model, Dataset(rows + [leak], NAMES), epochs=1)
        fit_policy(model, Dataset(rows, NAMES), epochs=1)
        new = Scenario("new", "train", rows[0].features, rows[0].scores, {})
        with self.assertRaisesRegex(ValueError, "held-out"):
            fit_policy(model, Dataset([new, leak], NAMES), epochs=1)
        for kwargs in ({"epochs": 0}, {"epochs": 1, "batch_size": 0}, {"epochs": 1, "learning_rate": float("nan")}, {"epochs": 1, "l2": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                fit_policy(model, Dataset(rows, NAMES), **kwargs)


if __name__ == "__main__":
    unittest.main()
