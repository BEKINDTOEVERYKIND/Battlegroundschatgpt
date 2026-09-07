from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("policy_run_audit", ROOT / "scripts/audit_recruit_policy_run.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def trajectory(seed, sample, score=1.0):
    combats = []
    for turn in (1, 2):
        combat_seed = (seed * 2654435761 + turn * 32452843 + sample * 49979687) & 0xFFFFFFFF
        outcomes = [{"player_a": a, "player_b": a + 1, "samples": 1,
                     "winner_id": a if score == 1 else None} for a in (0, 2, 4, 6)]
        combats.append({"request": {"seed": combat_seed},
                        "receipt": {"turn": turn, "seed": combat_seed, "outcomes": outcomes}})
    return {"seed": seed, "sample": sample, "complete": True, "score": score,
            "combats": combats, "commands": [{"player_id": 0, "action": {"kind": "buy"}},
                                                {"player_id": 1, "action": {"kind": "freeze"}}],
            "timing_trace": [{"player_id": 0, "event": "action", "after": {"remaining_ms": 1, "remaining_actions": 0}}]}


class MemoryEvidence:
    def __init__(self, records, report):
        self.records, self.report = records, report

    def json(self, name):
        return self.report

    def rows(self, name):
        return iter(self.records)


class RecruitPolicyAuditTests(unittest.TestCase):
    def fixture(self):
        records, episodes = [], []
        for seed in (11, 22):
            episode = {"seed": seed, "policies": {}}
            for name in ("practical", "model"):
                failed = name == "model" and seed == 11
                complete = not failed
                penalty = 0.0 if failed else 1.0
                rollouts = [trajectory(seed, sample) for sample in range(1 if failed else 2)]
                failures = [{"sample": 1, "error": "unsupported", "partial": {
                    "seed": seed, "sample": 1, "commands": [{"player_id": 0, "action": {"kind": "play"}}],
                    "timing_trace": []}}] if failed else []
                records.append({"seed": seed, "policy": name, "complete": complete,
                                "penalized_score": penalty, "rollouts": rollouts, "failures": failures})
                episode["policies"][name] = {"complete": complete, "penalized_score": penalty,
                    "failed_samples": int(failed), "failure_reasons": {"unsupported": 1} if failed else {},
                    "action_counts": {"buy": len(rollouts)}, "timing_budget_violations": 0}
            episodes.append(episode)
        report = {"episodes": episodes, "policies": {
            "practical": {"attempted_episodes": 2, "complete_episodes": 2, "failed_episodes": 0,
                "failure_penalized_mean": 1.0, "timing_budget_violations": 0,
                "paired_all_attempts": {"mean_delta": 0.0, "ci95": [0.0, 0.0]},
                "support_intersection_episodes": 2,
                "paired_supported_intersection": {"mean_delta": 0.0, "ci95": [0.0, 0.0]}},
            "model": {"attempted_episodes": 2, "complete_episodes": 1, "failed_episodes": 1,
                "failure_penalized_mean": .5, "timing_budget_violations": 0,
                "paired_all_attempts": {"mean_delta": -.5, "ci95": [-1.0, 0.0]},
                "support_intersection_episodes": 1,
                "paired_supported_intersection": {"mean_delta": 0.0, "ci95": [0.0, 0.0]}}}}
        return MemoryEvidence(records, report)

    def test_unfinished_run_never_reads_holdout(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(audit.Evidence, "json") as reader:
            with self.assertRaisesRegex(ValueError, "unfinished"):
                audit.audit_run(Path(directory))
            reader.assert_not_called()

    def test_actual_receipts_determine_score_and_sample_identity(self):
        value = trajectory(11, 1)
        self.assertEqual(audit.actual_score(value, 11, 1), 1.0)
        for changed in ("score", "receipt", "missing_turn", "sample"):
            with self.subTest(changed=changed):
                corrupt = deepcopy(value)
                if changed == "score":
                    corrupt["score"] = .5
                elif changed == "receipt":
                    corrupt["combats"][0]["receipt"]["outcomes"][0]["winner_id"] = 1
                elif changed == "missing_turn":
                    corrupt["combats"].pop()
                else:
                    corrupt["sample"] = 0
                with self.assertRaises(ValueError):
                    audit.actual_score(corrupt, 11, 1)

    def test_failure_penalty_is_separate_from_actual_completed_score(self):
        _, diagnostics, _ = audit.audit_evaluation(self.fixture(), "test", [11, 22], ["practical", "model"], 2)
        model = diagnostics["model"]
        self.assertEqual(model["actual_complete_sample_mean"], 1.0)
        self.assertEqual(model["failure_penalized_episode_mean"], .5)
        self.assertEqual(model["completed_samples"], 3)
        self.assertEqual(model["completed_sample_action_counts"], {"buy": 3})
        self.assertEqual(model["failed_partial_action_counts"], {"play": 1})
        self.assertNotIn("freeze", model["completed_sample_action_counts"])

    def test_missing_duplicate_samples_or_rows_are_rejected(self):
        for change in ("missing_sample", "duplicate_sample", "missing_row", "duplicate_row", "imputed_loss"):
            with self.subTest(change=change):
                evidence = self.fixture()
                if change == "missing_sample":
                    evidence.records[0]["rollouts"].pop()
                elif change == "duplicate_sample":
                    evidence.records[0]["rollouts"][1] = evidence.records[0]["rollouts"][0]
                elif change == "missing_row":
                    evidence.records.pop()
                elif change == "duplicate_row":
                    evidence.records.append(evidence.records[0])
                else:
                    evidence.records[1]["penalized_score"] = 1.0
                with self.assertRaises(ValueError):
                    audit.audit_evaluation(evidence, "test", [11, 22], ["practical", "model"], 2)

    def test_corrupt_interval_and_hidden_timing_violation_are_detected(self):
        for change in ("interval", "timing"):
            evidence = self.fixture()
            if change == "interval":
                evidence.report["policies"]["model"]["paired_all_attempts"]["ci95"] = [-.5, -.5]
            else:
                evidence.records[0]["rollouts"][0]["timing_trace"][0]["after"]["remaining_ms"] = -1
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit.audit_evaluation(evidence, "test", [11, 22], ["practical", "model"], 2)


if __name__ == "__main__":
    unittest.main()
