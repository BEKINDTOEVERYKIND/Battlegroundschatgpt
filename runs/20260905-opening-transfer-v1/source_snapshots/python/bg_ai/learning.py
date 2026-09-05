"""NumPy pairwise ranking for simulator-labelled combat board orders.

JSONL schema (one complete scenario per line)::

    {"scenario_id":"unique-id", "split":"train", "opponent":[...],
     "candidates":[{"board":[...], "score":0.51}, ...],
     "metadata":{"label_source":"Firestone combat simulation", ...}}

``score`` is higher-is-better, normally P(win) + 0.5 P(tie). Every candidate in
one scenario must be a permutation of the same board. Candidate zero is the
declared baseline for paired evaluation. Scenarios, never candidate rows, are
split across train/validation/test. Alternative externally encoded datasets
replace each ``board`` with ``features`` and require top-level ``feature_names``
on every line. Feature names and ordering must match across all rows/checkpoints.

Only real simulator labels establish useful strength. Synthetic fixtures in the
test suite validate plumbing and deliberately make no game-strength claim.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .features import BOARD_FEATURE_NAMES, FEATURE_VERSION, encode_board, feature_schema_id


TRAINING_OBJECTIVE = "gap_weighted_pair_average_v2"
LEGACY_TRAINING_OBJECTIVE = "within_scenario_gap_normalized_v1"


@dataclass
class Scenario:
    scenario_id: str
    split: str
    features: np.ndarray
    scores: np.ndarray
    metadata: dict[str, Any]


@dataclass
class Dataset:
    scenarios: list[Scenario]
    feature_names: tuple[str, ...]

    def split(self, name: str) -> list[Scenario]:
        return [s for s in self.scenarios if s.split == name]


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def jsonl_lines(path: str | Path):
    """Stream UTF-8 JSONL, decompressing .gz artifacts transparently."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        yield from stream


def load_dataset(path: str | Path, allowed_splits: Sequence[str] | None = None) -> Dataset:
    scenarios: list[Scenario] = []
    ids: set[str] = set()
    names: tuple[str, ...] | None = None
    fingerprints: dict[str, str] = {}
    for lineno, line in enumerate(jsonl_lines(path), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            scenario_id = row["scenario_id"]
            split = row["split"]
            if split not in ("train", "validation", "test"):
                raise ValueError("split must be train, validation, or test")
            if allowed_splits is not None and split not in allowed_splits:
                continue
            candidates = row["candidates"]
            if not isinstance(scenario_id, str) or not scenario_id or scenario_id in ids:
                raise ValueError("scenario_id must be a unique nonempty string")
            if len(candidates) < 2:
                raise ValueError("A ranking scenario needs at least two candidate orders")
            board_mode = "board" in candidates[0]
            if any(("board" in c) != board_mode or ("features" in c) == board_mode for c in candidates):
                raise ValueError("Use exactly one of board or features consistently")
            if board_mode:
                row_names = BOARD_FEATURE_NAMES
                opponent = row.get("opponent", [])
                family = sorted(_canonical(c) for c in candidates[0]["board"])
                if any(sorted(_canonical(c) for c in item["board"]) != family for item in candidates):
                    raise ValueError("Candidates must be permutations of the same minions")
                fingerprint = _canonical({"friendly": family, "opponent": opponent})
                features = np.stack([encode_board(c["board"], opponent) for c in candidates])
            else:
                row_names = tuple(row.get("feature_names", ()))
                if not row_names or any(not isinstance(n, str) or not n for n in row_names):
                    raise ValueError("Precomputed features require explicit feature_names")
                if len(set(row_names)) != len(row_names):
                    raise ValueError("feature_names must be unique")
                features = np.asarray([c["features"] for c in candidates], dtype=np.float64)
                fingerprint = _canonical(sorted(_canonical(c["features"]) for c in candidates))
            if names is not None and names != row_names:
                raise ValueError("Feature schema differs between scenarios")
            names = row_names
            scores = np.asarray([c["score"] for c in candidates], dtype=np.float64)
            if features.shape != (len(candidates), len(names)) or scores.shape != (len(candidates),):
                raise ValueError("Invalid feature/score shape")
            if not np.isfinite(features).all() or not np.isfinite(scores).all():
                raise ValueError("Features and scores must be finite")
            if fingerprint in fingerprints and fingerprints[fingerprint] != split:
                raise ValueError("Equivalent scenario appears in multiple splits (data leakage)")
            fingerprints[fingerprint] = split
            ids.add(scenario_id)
            metadata = row.get("metadata", {})
            if not isinstance(metadata, dict):
                raise ValueError("metadata must be an object")
            scenarios.append(Scenario(scenario_id, split, features, scores, metadata))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{lineno}: {exc}") from exc
    if not scenarios or names is None:
        raise ValueError("Dataset is empty")
    return Dataset(scenarios, names)


class Ranker:
    """Small shared MLP; Adam and normalization persist across warm starts."""

    CHECKPOINT_VERSION = 1

    def __init__(self, feature_names: Sequence[str], hidden: int = 32, seed: int = 0):
        if hidden < 1 or not feature_names or len(set(feature_names)) != len(feature_names):
            raise ValueError("Require positive hidden size and unique feature names")
        self.feature_names = tuple(feature_names)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        size = len(feature_names)
        self.mean = np.zeros(size)
        self.scale = np.ones(size)
        self.params = {
            "w1": self.rng.normal(0, 1 / math.sqrt(size), (size, hidden)),
            "b1": np.zeros(hidden),
            "w2": self.rng.normal(0, 1 / math.sqrt(hidden), hidden),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.step = 0
        self.metadata: dict[str, Any] = {"seed": seed, "training_scenario_ids": [],
                                         "training_objective": "untrained"}

    def _forward(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.feature_names) or not np.isfinite(x).all():
            raise ValueError("Prediction features do not match checkpoint schema")
        z = (x - self.mean) / self.scale
        hidden = np.tanh(z @ self.params["w1"] + self.params["b1"])
        return hidden @ self.params["w2"], hidden, z

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self._forward(features)[0]

    def _loss_gradient(self, scenario: Scenario, tie_tolerance: float,
                       objective: str = TRAINING_OBJECTIVE) -> tuple[float, dict[str, np.ndarray]] | None:
        values, hidden, z = self._forward(scenario.features)
        left, right = np.triu_indices(len(values), 1)
        all_pair_count = len(left)
        differences = scenario.scores[left] - scenario.scores[right]
        keep = np.abs(differences) > tie_tolerance
        if not keep.any():
            return None
        left, right, differences = left[keep], right[keep], differences[keep]
        signs = np.sign(differences)
        # Retain effect magnitude across scenarios: small, noisy Monte Carlo
        # gaps must not receive the same total weight as material differences.
        # Tolerance-excluded pairs remain in the denominator as zero loss.
        weights = np.abs(differences)
        weights /= weights.sum() if objective == LEGACY_TRAINING_OBJECTIVE else all_pair_count
        margin = signs * (values[left] - values[right])
        loss = float(np.sum(weights * np.logaddexp(0, -margin)))
        # exp(-logaddexp(0, margin)) is sigmoid(-margin) without overflow.
        grad_margin = -weights * np.exp(-np.logaddexp(0, margin)) * signs
        grad_values = np.zeros(len(values))
        np.add.at(grad_values, left, grad_margin)
        np.add.at(grad_values, right, -grad_margin)
        grad_hidden = grad_values[:, None] * self.params["w2"][None, :] * (1 - hidden ** 2)
        return loss, {"w1": z.T @ grad_hidden, "b1": grad_hidden.sum(axis=0),
                      "w2": hidden.T @ grad_values}

    def fit(self, dataset: Dataset, epochs: int = 40, learning_rate: float = 0.003,
            batch_size: int = 16, l2: float = 0.0001,
            tie_tolerance: float = 0.0, objective: str = TRAINING_OBJECTIVE) -> list[float]:
        """Train on train split only, averaging gap-weighted pair losses.

        Scores inside tie_tolerance are excluded. This is a caller-chosen label
        resolution, not a statistical significance guarantee for Monte Carlo
        labels. Hyperparameters must be selected with validation, not test.
        """
        if dataset.feature_names != self.feature_names:
            raise ValueError("Warm-start feature schema mismatch")
        if epochs < 1 or batch_size < 1 or learning_rate <= 0 or l2 < 0 or tie_tolerance < 0:
            raise ValueError("Invalid training hyperparameters")
        if objective not in (TRAINING_OBJECTIVE, LEGACY_TRAINING_OBJECTIVE):
            raise ValueError("Unknown training objective")
        train = dataset.split("train")
        if not train:
            raise ValueError("No training scenarios")
        old_ids = set(self.metadata.get("training_scenario_ids", []))
        if any(s.scenario_id in old_ids and s.split != "train" for s in dataset.scenarios):
            raise ValueError("A previously trained scenario was moved into a held-out split")
        if self.step == 0:
            rows = np.concatenate([s.features for s in train])
            self.mean = rows.mean(axis=0)
            self.scale = rows.std(axis=0)
            self.scale[self.scale < 0.001] = 1
        previous_objective = self.metadata.get("training_objective",
                                               LEGACY_TRAINING_OBJECTIVE if self.step else "untrained")
        if previous_objective != objective:
            self.metadata.setdefault("objective_transitions", []).append({
                "from": previous_objective, "to": objective,
                "at_optimizer_step": self.step,
            })
        history = []
        for _ in range(epochs):
            order = self.rng.permutation(len(train))
            total_loss = 0.0
            total_used = 0
            for start in range(0, len(order), batch_size):
                grads = {k: np.zeros_like(v) for k, v in self.params.items()}
                used = 0
                batch_indices = order[start:start + batch_size]
                for i in batch_indices:
                    result = self._loss_gradient(train[int(i)], tie_tolerance, objective)
                    if result is None:
                        continue
                    loss, scenario_grads = result
                    total_loss += loss
                    total_used += 1
                    used += 1
                    for k in grads:
                        grads[k] += scenario_grads[k]
                if not used:
                    continue
                self.step += 1
                for k in grads:
                    grad = grads[k] / (used if objective == LEGACY_TRAINING_OBJECTIVE else len(batch_indices))
                    if k != "b1":
                        grad += l2 * self.params[k]
                    self.m[k] = 0.9 * self.m[k] + 0.1 * grad
                    self.v[k] = 0.999 * self.v[k] + 0.001 * grad * grad
                    m_hat = self.m[k] / (1 - 0.9 ** self.step)
                    v_hat = self.v[k] / (1 - 0.999 ** self.step)
                    self.params[k] -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
            if not total_used:
                raise ValueError("All training candidate scores are tied within tie_tolerance")
            history.append(total_loss / (total_used if objective == LEGACY_TRAINING_OBJECTIVE else len(train)))
        self.metadata.update({
            "training_objective": objective,
            "objective_definition": ("Mean over scenarios of sum(abs(label_i-label_j) * pairwise_logistic_loss) / all_candidate_pair_count; pairs within tie_tolerance have zero loss"
                                     if objective == TRAINING_OBJECTIVE else
                                     "Legacy v1: normalize absolute label-gap weights to sum one within each informative scenario; average informative scenarios"),
            "training_scenario_ids": sorted(old_ids | {s.scenario_id for s in train}),
            "epochs_completed": self.metadata.get("epochs_completed", 0) + epochs,
            "last_fit": {"epochs": epochs, "learning_rate": learning_rate,
                         "batch_size": batch_size, "l2": l2, "tie_tolerance": tie_tolerance,
                         "training_scenarios": len(train), "final_pairwise_loss": history[-1]},
            "training_sources": [s.metadata for s in train],
            "scope": "combat board positioning only; no recruiting or full-game policy",
        })
        return history

    def save(self, path: str | Path) -> None:
        payload = {"checkpoint_version": self.CHECKPOINT_VERSION,
                   "feature_version": FEATURE_VERSION,
                   "feature_schema_id": feature_schema_id(self.feature_names),
                   "feature_names": list(self.feature_names), "seed": self.seed,
                   "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                   "params": {k: v.tolist() for k, v in self.params.items()},
                   "optimizer": {"name": "adam", "step": self.step,
                                 "m": {k: v.tolist() for k, v in self.m.items()},
                                 "v": {k: v.tolist() for k, v in self.v.items()}},
                   "rng_state": self.rng.bit_generator.state, "metadata": self.metadata}
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(json.dumps(payload, allow_nan=False) + "\n")
        temporary.replace(target)

    @classmethod
    def load(cls, path: str | Path, feature_names: Sequence[str] | None = None) -> "Ranker":
        payload = json.loads(Path(path).read_text())
        names = tuple(payload["feature_names"])
        if (payload.get("checkpoint_version") != cls.CHECKPOINT_VERSION or
                payload.get("feature_version") != FEATURE_VERSION or
                payload.get("feature_schema_id") != feature_schema_id(names)):
            raise ValueError("Unsupported or inconsistent checkpoint feature schema")
        if feature_names is not None and names != tuple(feature_names):
            raise ValueError("Warm-start feature schema mismatch")
        model = cls(names, hidden=len(payload["params"]["b1"]), seed=payload["seed"])
        for group, raw in ((model.params, payload["params"]),
                           (model.m, payload["optimizer"]["m"]),
                           (model.v, payload["optimizer"]["v"])):
            for k in group:
                arr = np.asarray(raw[k], dtype=np.float64)
                if arr.shape != group[k].shape or not np.isfinite(arr).all():
                    raise ValueError("Invalid checkpoint parameter/optimizer shape")
                group[k] = arr
        for key in ("mean", "scale"):
            arr = np.asarray(payload[key], dtype=np.float64)
            if arr.shape != (len(names),) or not np.isfinite(arr).all():
                raise ValueError("Invalid checkpoint normalization")
            setattr(model, key, arr)
        if np.any(model.scale <= 0) or payload["optimizer"].get("name") != "adam":
            raise ValueError("Invalid checkpoint normalization/optimizer")
        model.step = int(payload["optimizer"]["step"])
        model.rng.bit_generator.state = payload["rng_state"]
        model.metadata = payload.get("metadata", {})
        model.metadata.setdefault("training_objective", LEGACY_TRAINING_OBJECTIVE if model.step else "untrained")
        return model


def _paired_summary(deltas: np.ndarray, rng: np.random.Generator,
                    bootstrap_samples: int) -> dict[str, Any]:
    result: dict[str, Any] = {"mean": float(deltas.mean()), "scenarios": len(deltas)}
    if len(deltas) < 2:
        result["ci95"] = None
        result["note"] = "At least two independent scenarios required for a confidence interval"
        return result
    means = np.empty(bootstrap_samples)
    # Bound memory even for very large held-out datasets.
    for start in range(0, bootstrap_samples, 256):
        count = min(256, bootstrap_samples - start)
        sampled = rng.integers(0, len(deltas), size=(count, len(deltas)))
        means[start:start + count] = deltas[sampled].mean(axis=1)
    result["ci95"] = np.quantile(means, [0.025, 0.975]).tolist()
    return result


def evaluate(model: Ranker, dataset: Dataset, split: str = "test", seed: int = 0,
             bootstrap_samples: int = 5000) -> dict[str, Any]:
    """Paired differences and percentile bootstrap over independent scenarios.

    Intervals measure scenario sampling variation in the supplied simulator
    labels. They do not include Monte Carlo label error or full-game EV. Best
    observed candidate scores are noisy upper references, not proven optima.
    """
    if dataset.feature_names != model.feature_names:
        raise ValueError("Evaluation feature schema mismatch")
    if split not in ("validation", "test", "train") or bootstrap_samples < 100:
        raise ValueError("Invalid evaluation split/bootstrap count")
    scenarios = dataset.split(split)
    if not scenarios:
        raise ValueError(f"No {split} scenarios")
    trained = set(model.metadata.get("training_scenario_ids", []))
    if split != "train" and any(s.scenario_id in trained for s in scenarios):
        raise ValueError("Held-out evaluation contains training scenarios")
    selected, baseline, random_scores, best, details = [], [], [], [], []
    for scenario in scenarios:
        index = int(np.argmax(model.predict(scenario.features)))
        selected.append(float(scenario.scores[index]))
        baseline.append(float(scenario.scores[0]))
        random_scores.append(float(scenario.scores.mean()))
        best.append(float(scenario.scores.max()))
        details.append({"scenario_id": scenario.scenario_id, "chosen_candidate": index,
                        "candidate_count": len(scenario.scores), "chosen_score": selected[-1],
                        "baseline_score": baseline[-1], "random_score": random_scores[-1],
                        "best_observed_score": best[-1]})
    chosen = np.asarray(selected)
    baseline_array = np.asarray(baseline)
    random_array = np.asarray(random_scores)
    rng = np.random.default_rng(seed)
    return {"split": split, "held_out": split != "train", "scenario_count": len(scenarios),
            "score_definition": "higher-is-better labels supplied by the dataset",
            "mean_selected_score": float(chosen.mean()),
            "mean_baseline_score": float(baseline_array.mean()),
            "mean_random_score": float(random_array.mean()),
            "mean_best_observed_score": float(np.mean(best)),
            "mean_regret_to_best_observed": float(np.mean(np.asarray(best) - chosen)),
            "paired_vs_baseline": _paired_summary(chosen - baseline_array, rng, bootstrap_samples),
            "paired_vs_uniform_candidate": _paired_summary(chosen - random_array, rng, bootstrap_samples),
            "bootstrap": {"unit": "scenario", "samples": bootstrap_samples, "seed": seed,
                          "method": "paired percentile bootstrap",
                          "limitations": "Does not include simulator label error, opponent distribution shift, or full-game EV"},
            "scenarios": details}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="Train a simulator-labelled board-order ranker")
    train.add_argument("--data", required=True)
    train.add_argument("--checkpoint", required=True)
    train.add_argument("--warm-start")
    train.add_argument("--epochs", type=int, default=40)
    train.add_argument("--hidden", type=int, default=32)
    train.add_argument("--learning-rate", type=float, default=0.003)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--l2", type=float, default=0.0001)
    train.add_argument("--tie-tolerance", type=float, default=0)
    train.add_argument("--objective", choices=("gap-weighted", "legacy-normalized"), default="gap-weighted")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--report")
    check = sub.add_parser("evaluate", help="Evaluate on complete held-out scenarios")
    check.add_argument("--data", required=True)
    check.add_argument("--checkpoint", required=True)
    check.add_argument("--split", choices=("validation", "test", "train"), default="test")
    check.add_argument("--seed", type=int, default=0)
    check.add_argument("--bootstrap-samples", type=int, default=5000)
    check.add_argument("--report")
    args = parser.parse_args(argv)
    dataset = load_dataset(args.data)
    if args.command == "train":
        model = (Ranker.load(args.warm_start, dataset.feature_names) if args.warm_start
                 else Ranker(dataset.feature_names, args.hidden, args.seed))
        history = model.fit(dataset, args.epochs, args.learning_rate, args.batch_size,
                            args.l2, args.tie_tolerance,
                            TRAINING_OBJECTIVE if args.objective == "gap-weighted" else LEGACY_TRAINING_OBJECTIVE)
        model.save(args.checkpoint)
        report: dict[str, Any] = {"scope": model.metadata["scope"],
                                  "checkpoint": args.checkpoint,
                                  "feature_count": len(model.feature_names),
                                  "warm_start": args.warm_start,
                                  "training_scenarios": len(dataset.split("train")),
                                  "pairwise_loss": history}
        if dataset.split("validation"):
            report["validation"] = evaluate(model, dataset, "validation", args.seed)
    else:
        model = Ranker.load(args.checkpoint, dataset.feature_names)
        report = evaluate(model, dataset, args.split, args.seed, args.bootstrap_samples)
    serialized = json.dumps(report, indent=2, allow_nan=False)
    if args.report:
        target = Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(serialized + "\n")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
