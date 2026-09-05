"""Saved simulation traces must remain exact without duplicated provenance."""
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from archive_recruit_runs import archive, read_rows, restored_lines


class RecruitArchiveTests(unittest.TestCase):
    def test_deterministic_exact_round_trip_and_registry_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [{"scenario_id": f"game-{i}", "metadata": {"engine_provenance": {"ruleset": "test", "version": 1}},
                     "features": [0.5, 1, 2.25]} for i in range(3)]
            raw = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows).encode()
            with gzip.open(root / "scenarios.jsonl.gz", "wb") as handle:
                handle.write(raw)
            first = archive(root)
            second = archive(root)
            self.assertEqual(first, second)
            self.assertEqual(first["original_uncompressed_sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual("".join(restored_lines(root)).encode(), raw)
            (root / "scenarios.jsonl.gz").unlink()
            self.assertEqual(list(read_rows(root)), rows)
            compact_file = root / first["compact_file"]
            compact = compact_file.read_bytes()
            first["compact_parts"] = []
            for index, offset in enumerate(range(0, len(compact), 17)):
                content = compact[offset:offset+17]
                part = root / f"fixture.part-{index:03d}"
                part.write_bytes(content)
                first["compact_parts"].append({"file": part.name, "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest()})
            (root / "scenario_archive.json").write_text(json.dumps(first))
            compact_file.unlink()
            self.assertEqual(list(read_rows(root)), rows)
            part = root / first["compact_parts"][0]["file"]
            original_part = part.read_bytes()
            part.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "part checksum mismatch"):
                list(read_rows(root))
            part.write_bytes(original_part)
            (root / "scenario_provenance.json").write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                list(read_rows(root))


if __name__ == "__main__":
    unittest.main()
