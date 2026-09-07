"""Conservative clipped policy gradients on complete, finite opening episodes.

This is an episodic Monte Carlo actor, without a learned value function. A
separate practical rollout supplies an action-independent baseline for the same
fixture seed. Every decision receives the complete two-combat return difference.
Normalization is frozen and sampled visible features are replayed unchanged.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from .learning import Ranker
from .policy_learning import POLICY_OBJECTIVE

GRADIENT_OBJECTIVE = "complete_episode_clipped_policy_gradient_frozen_bc_kl_v1"
NUMERICAL_ZERO_GRADIENT_NORM = 1e-12


@dataclass(frozen=True)
class GradientConfig:
    temperature: float = .5
    clip_epsilon: float = .15
    reference_kl: float = .1
    learning_rate: float = .0001
    epochs: int = 2
    batch_episodes: int = 8
    max_gradient_norm: float = 1.0

    def validate(self):
        for name in ("temperature", "learning_rate", "max_gradient_norm"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(self.clip_epsilon) or not 0 < self.clip_epsilon < 1:
            raise ValueError("clip_epsilon must lie strictly between zero and one")
        if not math.isfinite(self.reference_kl) or self.reference_kl < 0:
            raise ValueError("reference_kl must be finite and nonnegative")
        if any(type(x) is not int or x < 1 for x in (self.epochs, self.batch_episodes)):
            raise ValueError("epochs and batch_episodes must be positive integers")


def log_probabilities(values, temperature=.5):
    values = np.asarray(values, dtype=np.float64)
    if (values.ndim != 1 or not len(values) or not np.isfinite(values).all()
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError("Require finite nonempty logits and positive temperature")
    shifted = (values - values.max()) / temperature
    return shifted - np.log(np.exp(shifted).sum())


def _frozen_array(value):
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PolicyDecision:
    features: np.ndarray
    action_index: int
    old_log_probabilities: np.ndarray
    reference_log_probabilities: np.ndarray

    def __post_init__(self):
        for name in ("features", "old_log_probabilities", "reference_log_probabilities"):
            object.__setattr__(self, name, _frozen_array(getattr(self, name)))
        features = self.features
        if features.ndim != 2 or not len(features) or not np.isfinite(features).all():
            raise ValueError("Decision features must be a finite nonempty matrix")
        if type(self.action_index) is not int or not 0 <= self.action_index < len(features):
            raise ValueError("Sampled action index is outside the legal candidate set")
        for values in (self.old_log_probabilities, self.reference_log_probabilities):
            if (values.shape != (len(features),) or not np.isfinite(values).all()
                    or np.any(values > 1e-12) or not np.isclose(np.exp(values).sum(), 1, atol=1e-12, rtol=0)):
                raise ValueError("Old/reference log probabilities must be normalized and finite")


@dataclass(frozen=True)
class PolicyEpisode:
    seed: int
    score: float
    practical_score: float
    decisions: tuple[PolicyDecision, ...]
    complete_combats: int = 2
    split: str = "train"

    def __post_init__(self):
        object.__setattr__(self, "decisions", tuple(self.decisions))
        if (type(self.seed) is not int or self.seed < 0 or self.split != "train"
                or self.complete_combats != 2 or not self.decisions):
            raise ValueError("Require a complete two-combat training episode and its decisions")
        if any(not math.isfinite(x) or not 0 <= x <= 1 for x in (self.score, self.practical_score)):
            raise ValueError("Complete episode and baseline scores must lie in [0, 1]")


def sample_decision(model, reference, features, rng, temperature=.5):
    if model.feature_names != reference.feature_names:
        raise ValueError("Actor/reference schemas differ")
    old = log_probabilities(model.predict(features), temperature)
    frozen = log_probabilities(reference.predict(features), temperature)
    index = int(rng.choice(len(old), p=np.exp(old)))
    return PolicyDecision(features, index, old, frozen)


def initialize_from_behavior_cloning(source: Ranker, reference_sha256: str) -> Ranker:
    """Copy a BC actor; start Adam anew for the changed objective, keep features."""
    if source.metadata.get("training_objective") != POLICY_OBJECTIVE:
        raise ValueError("Policy gradients require a behavior-cloning checkpoint, not a Q ranker")
    if len(reference_sha256) != 64 or any(c not in "0123456789abcdef" for c in reference_sha256):
        raise ValueError("The immutable BC checkpoint requires a SHA256 fingerprint")
    model = deepcopy(source)
    model.metadata.setdefault("objective_transitions", []).append({
        "from": POLICY_OBJECTIVE, "to": GRADIENT_OBJECTIVE, "at_optimizer_step": source.step,
        "optimizer": "reset Adam moments/step for new objective; retain weights and normalization"})
    model.step = 0
    model.m = {key: np.zeros_like(value) for key, value in model.params.items()}
    model.v = {key: np.zeros_like(value) for key, value in model.params.items()}
    model.metadata.update(training_objective=GRADIENT_OBJECTIVE,
        frozen_behavior_checkpoint_sha256=reference_sha256, policy_gradient_episode_seeds=[],
        full_game_ready=False, normalization="frozen from behavior-cloning checkpoint")
    return model


def policy_gradient_loss(model: Ranker, episodes: Sequence[PolicyEpisode],
                         config=GradientConfig()):
    """Return loss, exact parameter gradients and diagnostics without mutation.

    Sum decisions within an episode, then average episodes. Dividing by each
    sampled episode's action count would change the return objective when the
    policy changes trajectory length. KL is KL(frozen BC || current actor).
    """
    config.validate()
    if not episodes or len({e.seed for e in episodes}) != len(episodes):
        raise ValueError("Require a nonempty batch of independently seeded episodes")
    decisions = [d for episode in episodes for d in episode.decisions]
    features = np.concatenate([d.features for d in decisions])
    counts = np.array([len(d.features) for d in decisions], dtype=np.int64)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    selected = starts + np.array([d.action_index for d in decisions])
    advantages = np.array([e.score - e.practical_score for e in episodes for _ in e.decisions])
    values, hidden, normalized = model._forward(features)
    shifted = (values - np.repeat(np.maximum.reduceat(values, starts), counts)) / config.temperature
    probabilities = np.exp(shifted)
    logps = shifted - np.repeat(np.log(np.add.reduceat(probabilities, starts)), counts)
    probabilities = np.exp(logps)
    old_selected = np.array([d.old_log_probabilities[d.action_index] for d in decisions])
    log_ratios = logps[selected] - old_selected
    if np.any(np.abs(log_ratios) > 80):
        raise ValueError("Policy ratio left the bounded numerical trust region")
    ratios = np.exp(log_ratios)
    clipped = np.clip(ratios, 1 - config.clip_epsilon, 1 + config.clip_epsilon)
    surrogate = np.minimum(ratios * advantages, clipped * advantages)
    active = ((advantages >= 0) & (ratios <= 1 + config.clip_epsilon)
              | (advantages < 0) & (ratios >= 1 - config.clip_epsilon))
    coefficient = -advantages * ratios * active
    grad_values = -np.repeat(coefficient, counts) * probabilities
    grad_values[selected] += coefficient
    reference_logps = np.concatenate([d.reference_log_probabilities for d in decisions])
    reference_probabilities = np.exp(reference_logps)
    kl_terms = reference_probabilities * (reference_logps - logps)
    reference_kl = float(kl_terms.sum())
    grad_values += config.reference_kl * (probabilities - reference_probabilities)
    grad_values /= len(episodes) * config.temperature
    grad_hidden = grad_values[:, None] * model.params["w2"][None, :] * (1 - hidden ** 2)
    gradients = {"w1": normalized.T @ grad_hidden, "b1": grad_hidden.sum(axis=0),
                 "w2": hidden.T @ grad_values}
    loss = float((-surrogate.sum() + config.reference_kl * reference_kl) / len(episodes))
    return loss, gradients, {"loss": loss, "reference_kl_per_decision": max(0., reference_kl / len(decisions)),
        "clipped_fraction": float(np.mean(~active)), "decisions": len(decisions),
        "episodes": len(episodes), "mean_episode_advantage": float(np.mean([e.score-e.practical_score for e in episodes])),
        "max_probability_ratio": float(ratios.max()), "min_probability_ratio": float(ratios.min())}


def fit_policy_gradient(model, episodes, config=GradientConfig()):
    config.validate()
    if model.metadata.get("training_objective") != GRADIENT_OBJECTIVE:
        raise ValueError("Initialize the actor explicitly from frozen behavior cloning first")
    # Validate every frozen row before changing optimizer state.
    policy_gradient_loss(model, episodes, config)
    prior = set(model.metadata.get("policy_gradient_episode_seeds", []))
    seeds = {episode.seed for episode in episodes}
    reserved = set(model.metadata.get("reserved_evaluation_seeds", []))
    if seeds & (prior | reserved):
        raise ValueError("On-policy episodes cannot be reused or overlap reserved evaluation seeds")
    history = []
    for epoch in range(config.epochs):
        order = model.rng.permutation(len(episodes))
        for start in range(0, len(order), config.batch_episodes):
            batch = [episodes[int(i)] for i in order[start:start + config.batch_episodes]]
            loss, gradients, diagnostics = policy_gradient_loss(model, batch, config)
            norm = math.sqrt(sum(float(np.sum(g*g)) for g in gradients.values()))
            if norm <= NUMERICAL_ZERO_GRADIENT_NORM:
                # Fresh Adam can amplify roundoff at the identical reference
                # with zero advantage. Preserve moments and step as well.
                history.append({**diagnostics, "epoch": epoch + 1, "gradient_norm": norm,
                    "gradient_clipped": False, "optimizer_step": model.step,
                    "optimizer_updated": False, "reason": "numerically zero policy/KL gradient"})
                continue
            multiplier = min(1., config.max_gradient_norm / max(norm, 1e-30))
            model.step += 1
            for key, gradient in gradients.items():
                gradient = gradient * multiplier
                model.m[key] = .9 * model.m[key] + .1 * gradient
                model.v[key] = .999 * model.v[key] + .001 * gradient * gradient
                first = model.m[key] / (1 - .9 ** model.step)
                second = model.v[key] / (1 - .999 ** model.step)
                model.params[key] -= config.learning_rate * first / (np.sqrt(second) + 1e-8)
            history.append({**diagnostics, "epoch": epoch + 1, "gradient_norm": norm,
                            "gradient_clipped": multiplier < 1, "optimizer_step": model.step,
                            "optimizer_updated": True})
    model.metadata["policy_gradient_episode_seeds"] = sorted(prior | seeds)
    model.metadata["policy_gradient_updates"] = model.metadata.get("policy_gradient_updates", 0) + sum(
        row["optimizer_updated"] for row in history)
    return history
