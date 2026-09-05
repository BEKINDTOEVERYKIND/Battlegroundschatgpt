"""Reject replay labels from a different engine even when the ruleset matches."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from bg_ai.provenance import file_sha256


_SPEC = importlib.util.spec_from_file_location(
    "position_training_pipeline", Path(__file__).resolve().parents[1] / "scripts/train_positioning.py")
_PIPELINE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PIPELINE)


class GenerationProvenanceTests(unittest.TestCase):
    def test_reference_or_engine_changes_reject_reused_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "data").mkdir()
            (root / "data/ruleset.json").write_text("{}")
            (root / "data/reference_cards.json").write_text("[]")
            data = root / "positions.jsonl"
            data.write_text("{}\n")
            for package in ("@firestone-hs/simulate-bgs-battle", "@firestone-hs/reference-data"):
                package_dir = root / "node_modules" / package
                package_dir.mkdir(parents=True)
                (package_dir / "package.json").write_text('{"version":"1.0.0"}')
            generation = {
                "datasetSha256": file_sha256(data),
                "rulesetSha256": file_sha256(root / "data/ruleset.json"),
                "cardsSha256": file_sha256(root / "data/reference_cards.json"),
                "engine": "1.0.0", "referencePackage": "1.0.0", "rejected": 0,
            }
            self.assertEqual(_PIPELINE.validate_generation_metadata(generation, data, root), "legacy-v1")
            with self.assertRaisesRegex(ValueError, "Requested adapter"):
                _PIPELINE.validate_generation_metadata(generation, data, root,
                    expected_adapter_version="firestone-combat-v2-tribes")
            with self.assertRaisesRegex(ValueError, "schema version"):
                _PIPELINE.validate_generation_metadata(dict(generation, adapterVersion="firestone-combat-v2-tribes"), data, root)
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                _PIPELINE.validate_generation_metadata(dict(generation, adapterVersion="future-unknown"), data, root)
            current = dict(generation, adapterVersion="firestone-combat-v2-tribes", datasetSchemaVersion=2)
            with self.assertRaisesRegex(ValueError, "row 1 adapter"):
                _PIPELINE.validate_generation_metadata(current, data, root)
            data.write_text(json.dumps({"metadata": {"adapterVersion": current["adapterVersion"], "datasetSchemaVersion": 2}}) + "\n")
            current["datasetSha256"] = file_sha256(data)
            self.assertEqual(_PIPELINE.validate_generation_metadata(current, data, root), current["adapterVersion"])
            # Both manifest and row versions must agree; source hashes cannot relabel legacy rows.
            data.write_text("{}\n")
            for field in ("cardsSha256", "engine", "referencePackage"):
                with self.subTest(changed_field=field):
                    changed = dict(generation, **{field: "old-labels"})
                    with self.assertRaises(ValueError):
                        _PIPELINE.validate_generation_metadata(changed, data, root)


if __name__ == "__main__":
    unittest.main()
