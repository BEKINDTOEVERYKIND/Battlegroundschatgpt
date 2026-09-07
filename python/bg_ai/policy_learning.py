"""Grouped supervised action learning using the rotation-stable Ranker network.

Each Scenario contains only currently legal candidate actions. Binary scores
identify a nonempty positive teacher set. The loss is minus log probability of
choosing any positive action, not a requirement to order equivalent positives.
All decisions have equal weight regardless of their number of legal actions.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .learning import Dataset, Ranker, Scenario


POLICY_OBJECTIVE = "positive_action_set_softmax_v1"


def _validated_arrays(model: Ranker, scenarios: Sequence[Scenario]):
    if not scenarios:
        raise ValueError("Policy batch must contain at least one decision")
    arrays, positive, counts = [], [], []
    for scenario in scenarios:
        features = np.asarray(scenario.features, dtype=np.float64)
        scores = np.asarray(scenario.scores, dtype=np.float64)
        if (features.ndim != 2 or features.shape[1] != len(model.feature_names)
                or len(features) < 1 or scores.shape != (len(features),)
                or not np.isfinite(features).all() or not np.isfinite(scores).all()):
            raise ValueError("Invalid policy decision feature/target shape or values")
        if not np.all((scores == 0) | (scores == 1)) or not np.any(scores == 1):
            raise ValueError("Policy targets must be binary with a nonempty positive action set")
        arrays.append(features)
        positive.append(scores == 1)
        counts.append(len(features))
    return np.concatenate(arrays), np.concatenate(positive), np.asarray(counts, dtype=np.int64)


def policy_loss_gradient(model: Ranker, scenarios: Sequence[Scenario]) -> tuple[float, dict[str, np.ndarray]]:
    """One fused forward/backward over variable-sized legal action groups.

    The public helper excludes regularization and does not mutate the model.
    It accepts all-positive groups (zero loss/gradient) for deterministic tests
    and for batches containing forced actions.
    """
    features, positive, counts = _validated_arrays(model, scenarios)
    starts = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts)[:-1]))
    values, hidden, normalized = model._forward(features)
    maxima = np.maximum.reduceat(values, starts)
    exponentials = np.exp(values - np.repeat(maxima, counts))
    denominators = np.add.reduceat(exponentials, starts)
    all_probabilities = exponentials / np.repeat(denominators, counts)

    # A separate stable positive normalizer is needed when all positives have
    # very low logits. Mask BEFORE exponentiation to avoid overflows in rows
    # outside that set.
    positive_values = np.where(positive, values, -np.inf)
    positive_maxima = np.maximum.reduceat(positive_values, starts)
    positive_exponentials = np.exp(positive_values - np.repeat(positive_maxima, counts))
    positive_denominators = np.add.reduceat(positive_exponentials, starts)
    positive_probabilities = positive_exponentials / np.repeat(positive_denominators, counts)
    losses = maxima - positive_maxima + np.log(denominators) - np.log(positive_denominators)
    # Identical normalizers for all-positive decisions yield exact zero. The
    # subtraction can otherwise produce a negligible negative roundoff error.
    loss = float(np.mean(np.maximum(losses, 0.0)))
    grad_values = (all_probabilities - positive_probabilities) / len(scenarios)
    grad_hidden = grad_values[:, None] * model.params["w2"][None, :] * (1 - hidden ** 2)
    grads = {"w1": normalized.T @ grad_hidden,
             "b1": grad_hidden.sum(axis=0),
             "w2": hidden.T @ grad_values}
    return loss, grads


def fit_policy(model: Ranker, dataset: Dataset, epochs: int, batch_size: int = 32,
               learning_rate: float = .003, l2: float = .0001) -> list[float]:
    """Mutate a Ranker in place using train-only supervised action targets.

    Shared weights, Adam moments, optimizer step, normalization and RNG resume
    from a warm checkpoint. Callers wanting an independent candidate must copy
    the source first, just as with Ranker.fit. Validation/test rows contribute
    neither normalization nor gradients nor teacher selection.
    """
    if dataset.feature_names != model.feature_names:
        raise ValueError("Warm-start feature schema mismatch")
    if (type(epochs) is not int or epochs < 1 or type(batch_size) is not int or batch_size < 1
            or not math.isfinite(learning_rate) or learning_rate <= 0
            or not math.isfinite(l2) or l2 < 0):
        raise ValueError("Invalid policy training hyperparameters")
    train = dataset.split("train")
    if not train:
        raise ValueError("No training policy decisions")
    features, positive, counts = _validated_arrays(model, train)
    if bool(np.all(positive)):
        raise ValueError("All training decisions have only positive actions; no policy learning signal")
    ids = [s.scenario_id for s in train]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate training decision identifiers")
    old_ids = set(model.metadata.get("training_scenario_ids", []))
    known_training_ids = old_ids | set(ids)
    if any(s.scenario_id in known_training_ids and s.split != "train" for s in dataset.scenarios):
        raise ValueError("A training decision was moved into a held-out split")
    if model.step == 0:
        model.mean = features.mean(axis=0)
        model.scale = features.std(axis=0)
        model.scale[model.scale < .001] = 1.0
    previous = model.metadata.get("training_objective", "untrained" if model.step == 0 else "unknown")
    if previous != POLICY_OBJECTIVE:
        model.metadata.setdefault("objective_transitions", []).append({
            "from": previous, "to": POLICY_OBJECTIVE, "at_optimizer_step": model.step})
    history = []
    for _ in range(epochs):
        order = model.rng.permutation(len(train))
        total_loss = 0.0
        for start in range(0, len(order), batch_size):
            batch = [train[int(i)] for i in order[start:start + batch_size]]
            loss, grads = policy_loss_gradient(model, batch)
            total_loss += loss * len(batch)
            # Forced/equivalent choices alone contain no learning signal. Do
            # not advance Adam or apply regularization for such a batch.
            if all(np.all(np.asarray(s.scores) == 1) for s in batch):
                continue
            model.step += 1
            for key, grad in grads.items():
                if key != "b1":
                    grad = grad + l2 * model.params[key]
                model.m[key] = .9 * model.m[key] + .1 * grad
                model.v[key] = .999 * model.v[key] + .001 * grad * grad
                first = model.m[key] / (1 - .9 ** model.step)
                second = model.v[key] / (1 - .999 ** model.step)
                model.params[key] -= learning_rate * first / (np.sqrt(second) + 1e-8)
        history.append(total_loss / len(train))
    model.metadata.update({
        "training_objective": POLICY_OBJECTIVE,
        "objective_definition": "Mean over decisions of -log(sum softmax probability of the binary positive teacher action set)",
        "training_scenario_ids": sorted(known_training_ids),
        "epochs_completed": model.metadata.get("epochs_completed", 0) + epochs,
        "last_fit": {"epochs": epochs, "learning_rate": learning_rate,
            "batch_size": batch_size, "l2": l2, "training_scenarios": len(train),
            "final_policy_loss": history[-1], "candidate_rows": int(counts.sum()),
            "positive_candidates": int(positive.sum())},
        "training_sources": [s.metadata for s in train],
    })
    return history
