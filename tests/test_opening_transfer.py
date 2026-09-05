"""Scientific-contract checks for the frozen opening transfer evaluation."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("evaluate_opening_transfer", ROOT / "scripts/evaluate_opening_transfer.py")
transfer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transfer)


class TestOpeningTransferContract(unittest.TestCase):
    def valid_measurement(self):
        row = {"scenario_id": "heldout-9", "index": 9,
               "selections": {"model": 2, "practical_heuristic": 0, "raw_stats": 1}}
        result = {"scenario_id": row["scenario_id"], "index": row["index"],
                  "metadata": {"seed": (transfer.EVALUATION_SEED + 10*2654435761) & 0xffffffff,
                               "trials": 1024},
                  "results": {name: {"candidateIndex": index, "score": .5}
                              for name, index in row["selections"].items()}}
        return [row], [result]

    def test_frozen_selection_and_independent_rng_metadata_are_required(self):
        rows, fresh = self.valid_measurement()
        transfer.validate_fresh(rows, fresh, 1024)
        bad = deepcopy(fresh)
        bad[0]["results"]["model"]["candidateIndex"] = 1
        with self.assertRaisesRegex(ValueError, "frozen selection"):
            transfer.validate_fresh(rows, bad, 1024)
        bad = deepcopy(fresh)
        bad[0]["metadata"]["seed"] = transfer.SCENARIO_SEED
        with self.assertRaisesRegex(ValueError, "RNG"):
            transfer.validate_fresh(rows, bad, 1024)

    def test_final_errors_cannot_be_filtered_out_to_improve_results(self):
        rows, fresh = self.valid_measurement()
        fresh[0]["error"] = "unsupported combat"
        with self.assertRaisesRegex(RuntimeError, "Every frozen"):
            transfer.validate_fresh(rows, fresh, 1024)
        with self.assertRaisesRegex(RuntimeError, "Every frozen"):
            transfer.validate_fresh(rows, [], 1024)

    def test_result_ids_order_and_trial_counts_are_checked(self):
        rows, fresh = self.valid_measurement()
        fresh[0]["scenario_id"] = "a-different-game"
        with self.assertRaisesRegex(ValueError, "scenario order"):
            transfer.validate_fresh(rows, fresh, 1024)
        rows, fresh = self.valid_measurement()
        fresh[0]["metadata"]["trials"] = 32
        with self.assertRaisesRegex(ValueError, "trial"):
            transfer.validate_fresh(rows, fresh, 1024)

    def test_bootstrap_pairs_whole_scenarios_and_preserves_constant_effect(self):
        result = transfer.paired_interval([.125]*20, samples=1000)
        self.assertEqual(result, {"mean_delta": .125, "ci95": [.125, .125]})
        with self.assertRaises(ValueError):
            transfer.paired_interval([])

    def test_schedule_has_five_tribes_per_lobby_and_equal_tribe_coverage(self):
        from collections import Counter
        counts = Counter(t for lobby in transfer.TRIBE_SCHEDULE for t in lobby)
        self.assertEqual(set(counts), set(transfer.TRIBE_ORDER))
        self.assertEqual(set(counts.values()), {5})
        self.assertTrue(all(len(set(lobby)) == 5 for lobby in transfer.TRIBE_SCHEDULE))

    @unittest.skipUnless((ROOT / "runs/20260905-recruit-v2/selected_model.json").exists(), "frozen checkpoint absent")
    def test_frozen_model_loads_with_exact_v2_schema(self):
        checkpoint = ROOT / "runs/20260905-recruit-v2/selected_model.json"
        self.assertEqual(transfer.digest(checkpoint), transfer.CHECKPOINT_SHA)
        model = transfer.Ranker.load(checkpoint, transfer.RECRUIT_V2_FEATURE_NAMES)
        self.assertEqual(model.metadata["recruit_feature_version"], transfer.FEATURE_VERSION)
        with self.assertRaisesRegex(ValueError, "schema mismatch"):
            transfer.Ranker.load(checkpoint, (*transfer.RECRUIT_V2_FEATURE_NAMES, "invented-v3-feature"))


if __name__ == "__main__":
    unittest.main()
