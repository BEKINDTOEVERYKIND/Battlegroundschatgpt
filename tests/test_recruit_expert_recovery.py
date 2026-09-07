import gzip
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("recruit_expert_recovery", ROOT / "scripts/recover_recruit_expert_archive.py")
recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)


class RecruitExpertRecoveryTests(unittest.TestCase):
    def test_sync_flushed_prefix_is_preserved_exactly_without_inventing_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original, preserved = root / "original.gz", root / "prefix.gz"
            raw = b'{"seed":11}\n{"seed":22}\n'
            compressor = zlib.compressobj(wbits=31)
            original.write_bytes(compressor.compress(raw) + compressor.flush(zlib.Z_SYNC_FLUSH))
            original_bytes = original.read_bytes()
            with self.assertRaises(EOFError):
                gzip.decompress(original_bytes)
            report = recovery.extract_prefix(original, preserved)
            self.assertEqual(report["complete_original_rows"], 2)
            self.assertFalse(report["original_stream_complete"])
            self.assertEqual(report["decoded_prefix_sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(gzip.decompress(preserved.read_bytes()), raw)
            self.assertEqual(original.read_bytes(), original_bytes)

    def test_partial_json_cannot_be_silently_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original.gz"
            compressor = zlib.compressobj(wbits=31)
            original.write_bytes(compressor.compress(b'{"seed":11}\n{"seed":') + compressor.flush(zlib.Z_SYNC_FLUSH))
            with self.assertRaisesRegex(ValueError, "complete-JSONL prefix"):
                recovery.extract_prefix(original, root / "prefix.gz")


if __name__ == "__main__":
    unittest.main()
