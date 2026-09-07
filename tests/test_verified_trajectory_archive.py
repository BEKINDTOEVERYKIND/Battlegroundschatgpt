import gzip
import json
from pathlib import Path
import tempfile
import unittest

from bg_ai.verified_trajectory_archive import (
    ArchiveIntegrityError, VerifiedTrajectoryWriter, verify_trajectory_archive,
)


def trajectory(seed):
    return {"seed": seed, "complete": True,
            "combats": [{"receipt": {"turn": 1}}, {"receipt": {"turn": 2}}]}


class VerifiedTrajectoryArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.destination = Path(self.temp.name) / "expert.jsonl.gz"

    def test_complete_footer_seed_plan_and_publication_are_verified(self):
        with VerifiedTrajectoryWriter(self.destination) as writer:
            writer.write(trajectory(11))
            writer.write(trajectory(22))
            self.assertFalse(self.destination.exists())
            result = writer.finish([11, 22])
            self.assertFalse(writer.temporary_path.exists())
        self.assertEqual(result["records"], 2)
        self.assertTrue(result["gzip_footer_verified"])
        self.assertEqual(verify_trajectory_archive(self.destination, [11, 22])["archive_sha256"], result["archive_sha256"])

    def test_replaced_open_path_is_detected_without_publishing_or_overwriting(self):
        with self.assertRaisesRegex(ArchiveIntegrityError, "replaced"):
            with VerifiedTrajectoryWriter(self.destination) as writer:
                writer.write(trajectory(1))
                old = writer.temporary_path.with_suffix(".retained")
                writer.temporary_path.rename(old)
                writer.temporary_path.write_bytes(b"an externally substituted stale snapshot")
                writer.finish([1])
        self.assertFalse(self.destination.exists())
        self.assertEqual(writer.temporary_path.read_bytes(), b"an externally substituted stale snapshot")

    def test_valid_footer_with_missing_trajectory_still_fails(self):
        with gzip.open(self.destination, "wt") as stream:
            stream.write(json.dumps(trajectory(1)) + "\n")
        with self.assertRaisesRegex(ArchiveIntegrityError, "count/order/seeds"):
            verify_trajectory_archive(self.destination, [1, 2])
        with self.assertRaisesRegex(ArchiveIntegrityError, "generation plan"):
            with VerifiedTrajectoryWriter(self.destination.with_name("new.jsonl.gz")) as writer:
                writer.write(trajectory(1))
                writer.finish([1, 2])

    def test_missing_footer_even_after_complete_jsonl_lines_fails(self):
        body = (json.dumps(trajectory(1)) + "\n").encode()
        self.destination.write_bytes(gzip.compress(body)[:-8])
        with self.assertRaisesRegex(ArchiveIntegrityError, "Strict trajectory archive"):
            verify_trajectory_archive(self.destination, [1])

    def test_same_seeds_with_changed_record_contents_cannot_be_published(self):
        with self.assertRaisesRegex(ArchiveIntegrityError, "record contents"):
            with VerifiedTrajectoryWriter(self.destination) as writer:
                writer.write(dict(trajectory(1), score=1.0))
                writer._text.close()
                changed = (json.dumps(dict(trajectory(1), score=0.0), separators=(",", ":")) + "\n").encode()
                writer._raw.seek(0)
                writer._raw.write(gzip.compress(changed))
                writer._raw.truncate()
                writer.finish([1])
        self.assertFalse(self.destination.exists())

    def test_existing_destination_and_unfinalized_context_cannot_succeed(self):
        self.destination.write_bytes(b"prior evidence")
        with self.assertRaises(FileExistsError):
            VerifiedTrajectoryWriter(self.destination)
        self.assertEqual(self.destination.read_bytes(), b"prior evidence")
        with self.assertRaisesRegex(ArchiveIntegrityError, "explicit finish"):
            with VerifiedTrajectoryWriter(self.destination.with_name("new.jsonl.gz")) as writer:
                writer.write(trajectory(1))
        self.assertFalse(writer.destination.exists())
