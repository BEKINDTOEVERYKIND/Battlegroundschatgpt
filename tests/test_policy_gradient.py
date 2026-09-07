"""Synthetic numerical/recording checks; these do not establish playing strength."""
from copy import deepcopy
from dataclasses import replace
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))
from bg_ai.learning import Ranker
from bg_ai.policy_learning import POLICY_OBJECTIVE
from bg_ai.policy_gradient import (
    GRADIENT_OBJECTIVE, GradientConfig, PolicyDecision, PolicyEpisode,
    fit_policy_gradient, initialize_from_behavior_cloning, log_probabilities,
    policy_gradient_loss, sample_decision,
)
from train_recruit_policy_gradient import gradient_test_promotion, reserve_seeds, update_complete_batch

NAMES = ("synthetic_strength", "synthetic_context", "synthetic_cost")


def initial():
    model = Ranker(NAMES, hidden=3, seed=73)
    model.mean = np.array([.1, -.2, .3])
    model.scale = np.array([.7, 1.2, 1.8])
    model.metadata["training_objective"] = POLICY_OBJECTIVE
    model.step = 11
    for key in model.m:
        model.m[key][:] = .3
        model.v[key][:] = .1
    return model


def episodes(model, *, offset=0, temperature=.5):
    rng = np.random.default_rng(451)
    result = []
    for i, (score, baseline) in enumerate(((1., .25), (.25, .75), (.5, .5))):
        decisions = []
        for turn in range(i + 1):
            features = rng.normal(size=(2 + turn, 3))
            logps = log_probabilities(model.predict(features), temperature)
            decisions.append(PolicyDecision(features, turn, logps, logps))
        result.append(PolicyEpisode(offset+i, score, baseline, tuple(decisions)))
    return result


class GradientNumericalTests(unittest.TestCase):
    def test_finite_difference_with_variable_decisions_normalization_and_kl(self):
        model = initial()
        data = episodes(model)
        model.params["w1"] += .01
        config = GradientConfig(reference_kl=.13)
        _, gradient, stats = policy_gradient_loss(model, data, config)
        self.assertEqual(stats["clipped_fraction"], 0)
        epsilon = 1e-6
        for key, array in model.params.items():
            numerical = np.zeros_like(array)
            for index in np.ndindex(array.shape):
                saved = array[index]
                array[index] = saved + epsilon
                upper = policy_gradient_loss(model, data, config)[0]
                array[index] = saved - epsilon
                lower = policy_gradient_loss(model, data, config)[0]
                array[index] = saved
                numerical[index] = (upper-lower) / (2*epsilon)
            np.testing.assert_allclose(gradient[key], numerical, rtol=3e-6, atol=2e-9)

    def test_clipped_positive_and_negative_advantage_have_zero_gradient(self):
        model = Ranker(("synthetic",), hidden=1, seed=1)
        model.params["w1"][:] = 1
        old = np.log(np.array([.5, .5]))
        decision = PolicyDecision(np.array([[1.], [-1.]]), 0, old, old)
        for weight, score, baseline in ((3., 1., 0.), (-3., 0., 1.)):
            with self.subTest(weight=weight):
                model.params["w2"][:] = weight
                _, gradients, stats = policy_gradient_loss(model,
                    [PolicyEpisode(1, score, baseline, (decision,))], GradientConfig(reference_kl=0))
                self.assertEqual(stats["clipped_fraction"], 1)
                self.assertTrue(all(np.count_nonzero(g) == 0 for g in gradients.values()))

    def test_wrong_direction_outside_clip_still_has_policy_gradient(self):
        model = Ranker(("synthetic",), hidden=1, seed=1)
        model.params["w1"][:] = 1
        old = np.log(np.array([.5, .5]))
        decision = PolicyDecision(np.array([[1.], [-1.]]), 0, old, old)
        for weight, score, baseline in ((-3., 1., 0.), (3., 0., 1.)):
            model.params["w2"][:] = weight
            _, gradient, stats = policy_gradient_loss(model,
                [PolicyEpisode(1, score, baseline, (decision,))], GradientConfig(reference_kl=0))
            self.assertEqual(stats["clipped_fraction"], 0)
            self.assertGreater(sum(np.linalg.norm(g) for g in gradient.values()), 0)

    def test_reference_kl_zero_at_reference_and_pushes_perturbed_actor_back(self):
        reference = initial()
        data = [replace(e, score=.5, practical_score=.5) for e in episodes(reference)]
        config = GradientConfig(reference_kl=.3)
        loss, gradient, diagnostics = policy_gradient_loss(reference, data, config)
        self.assertAlmostEqual(loss, 0, places=13)
        self.assertAlmostEqual(diagnostics["reference_kl_per_decision"], 0, places=13)
        self.assertTrue(all(np.max(np.abs(g)) < 1e-13 for g in gradient.values()))
        actor = deepcopy(reference)
        actor.params["w1"] += .2
        actor.params["w2"] -= .1
        before, gradient, _ = policy_gradient_loss(actor, data, config)
        self.assertGreater(before, 0)
        for key, value in gradient.items():
            actor.params[key] -= .001 * value
        after = policy_gradient_loss(actor, data, config)[0]
        self.assertLess(after, before)

    def test_episode_average_is_sum_of_decisions_without_length_reweighting(self):
        model = initial()
        data = episodes(model)
        loss, gradient, _ = policy_gradient_loss(model, data)
        separate = [policy_gradient_loss(model, [episode]) for episode in data]
        self.assertAlmostEqual(loss, np.mean([row[0] for row in separate]), places=13)
        for key in gradient:
            np.testing.assert_allclose(gradient[key], np.mean([row[1][key] for row in separate], axis=0), atol=1e-13)


class SamplingAndPersistenceTests(unittest.TestCase):
    def test_frozen_features_probabilities_and_seeded_sampling(self):
        model = initial()
        reference = deepcopy(model)
        features = np.random.default_rng(7).normal(size=(5, 3))
        left, right = np.random.default_rng(49), np.random.default_rng(49)
        a = [sample_decision(model, reference, features, left) for _ in range(30)]
        b = [sample_decision(model, reference, features, right) for _ in range(30)]
        self.assertEqual([d.action_index for d in a], [d.action_index for d in b])
        saved = a[0].features.copy()
        features[:] = 100
        model.params["w1"][:] = 0
        np.testing.assert_array_equal(a[0].features, saved)
        with self.assertRaises(ValueError):
            a[0].old_log_probabilities[0] = 10

    def test_checkpoint_resume_frozen_reference_and_normalization(self):
        source = initial()
        with tempfile.TemporaryDirectory() as directory:
            frozen = Path(directory) / "bc.json"
            source.save(frozen)
            digest = hashlib.sha256(frozen.read_bytes()).hexdigest()
            actor = initialize_from_behavior_cloning(source, digest)
            self.assertEqual(actor.step, 0)
            self.assertTrue(all(np.count_nonzero(g) == 0 for g in actor.m.values()))
            data = episodes(actor)
            config = GradientConfig(epochs=2, batch_episodes=2)
            fit_policy_gradient(actor, data, config)
            checkpoint = Path(directory) / "actor.json"
            actor.save(checkpoint)
            resumed = Ranker.load(checkpoint, NAMES)
            new = episodes(actor, offset=100)
            left = fit_policy_gradient(actor, new, config)
            right = fit_policy_gradient(resumed, new, config)
            self.assertEqual(left, right)
            for group in ("params", "m", "v"):
                for key in actor.params:
                    np.testing.assert_array_equal(getattr(actor, group)[key], getattr(resumed, group)[key])
            np.testing.assert_array_equal(actor.mean, source.mean)
            np.testing.assert_array_equal(actor.scale, source.scale)
            self.assertEqual(hashlib.sha256(frozen.read_bytes()).hexdigest(), digest)
            self.assertEqual(actor.metadata["frozen_behavior_checkpoint_sha256"], digest)
            self.assertEqual(actor.metadata["training_objective"], GRADIENT_OBJECTIVE)
            self.assertEqual(actor.metadata["policy_gradient_episode_seeds"], [0, 1, 2, 100, 101, 102])

    def test_reused_or_reserved_episodes_and_non_bc_initialization_rejected(self):
        model = initialize_from_behavior_cloning(initial(), "0"*64)
        data = episodes(model)
        model.metadata["reserved_evaluation_seeds"] = [1]
        with self.assertRaisesRegex(ValueError, "reserved"):
            fit_policy_gradient(model, data)
        model.metadata["reserved_evaluation_seeds"] = []
        fit_policy_gradient(model, data)
        with self.assertRaisesRegex(ValueError, "reused"):
            fit_policy_gradient(model, data)
        source = initial()
        source.metadata["training_objective"] = "old_q_value_ranker"
        with self.assertRaisesRegex(ValueError, "behavior-cloning"):
            initialize_from_behavior_cloning(source, "0"*64)

    def test_zero_advantage_at_reference_preserves_weights_moments_and_step(self):
        actor = initialize_from_behavior_cloning(initial(), "0"*64)
        before = deepcopy(actor)
        data = [replace(episode, score=.5, practical_score=.5) for episode in episodes(actor)]
        history = fit_policy_gradient(actor, data, GradientConfig(epochs=3, batch_episodes=1))
        self.assertEqual(actor.step, 0)
        self.assertEqual(actor.metadata["policy_gradient_updates"], 0)
        self.assertTrue(all(not row["optimizer_updated"] for row in history))
        for group in ("params", "m", "v"):
            for key in actor.params:
                np.testing.assert_array_equal(getattr(actor, group)[key], getattr(before, group)[key])

    def test_gradient_promotion_requires_both_controls_and_never_calls_bc_rl(self):
        selection = {"experimental_learner": "gradient", "full_game_ready": False}
        good = {"failed_episodes": 0, "timing_budget_violations": 0,
                "paired_all_attempts": {"ci95": [.01, .10]}}
        summary = {name: deepcopy(good) for name in ("practical", "bc_reference", "gradient")}
        def promote(comparison, **kwargs):
            return gradient_test_promotion(selection, summary, {"ci95": comparison},
                current_snapshot_verified=True, **kwargs)
        self.assertTrue(promote([.001, .08])["policy_gradient_promoted"])
        self.assertFalse(promote([-.01, .08])["policy_gradient_promoted"])
        kept = promote([-.01, .08], existing_baseline="bc_reference")
        self.assertEqual(kept["deployment_policy"], "bc_reference")
        summary["bc_reference"]["failed_episodes"] = 1
        self.assertFalse(promote([.001, .08])["policy_gradient_promoted"])
        summary["bc_reference"]["failed_episodes"] = 0
        selection["experimental_learner"] = "bc_reference"
        self.assertFalse(promote([.001, .08])["policy_gradient_promoted"])

    def test_inherited_bc_training_and_evaluation_seeds_are_reserved(self):
        source = initial()
        source.metadata.update(training_sources=[{"episode_seed": 5}], reserved_evaluation_seeds=[6])
        self.assertEqual(reserve_seeds({"train": [1], "validation": [2], "test": [3]}, source), [2, 3, 6])
        for collision in (5, 6):
            with self.assertRaises(ValueError):
                reserve_seeds({"train": [1], "validation": [2], "test": [collision]}, source)

    def test_any_unsupported_pair_rejects_entire_update_without_mutation(self):
        actor = initialize_from_behavior_cloning(initial(), "0"*64)
        before = deepcopy(actor)
        complete = episodes(actor)[:1]
        failures = [{"seed": 77, "errors": {"actor": "explicit unsupported tier frontier"}}]
        with patch("train_recruit_policy_gradient.fit_policy_gradient") as fit:
            result = update_complete_batch(actor, complete, failures, 2, GradientConfig())
            fit.assert_not_called()
        self.assertEqual(result, [])
        self.assertEqual(actor.step, before.step)
        self.assertEqual(actor.metadata, before.metadata)
        for group in ("params", "m", "v"):
            for key in actor.params:
                np.testing.assert_array_equal(getattr(actor, group)[key], getattr(before, group)[key])
        with self.assertRaises(ValueError):
            update_complete_batch(actor, complete, [], 2, GradientConfig())

    def test_invalid_partial_heldout_or_unnormalized_records_fail(self):
        model = initial()
        decision = episodes(model)[0].decisions[0]
        for kwargs in ({"complete_combats": 1}, {"split": "test"}, {"score": float("nan")}, {"decisions": ()}):
            values = {"seed": 1, "score": .5, "practical_score": .5, "decisions": (decision,), **kwargs}
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PolicyEpisode(**values)
        with self.assertRaises(ValueError):
            PolicyDecision(decision.features, 0, np.zeros(len(decision.features)), decision.reference_log_probabilities)
        with self.assertRaises(ValueError):
            policy_gradient_loss(model, [])
        with self.assertRaises(ValueError):
            GradientConfig(temperature=0).validate()


if __name__ == "__main__":
    unittest.main()
